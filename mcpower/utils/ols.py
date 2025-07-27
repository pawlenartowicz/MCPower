import os
import numpy as np
import pickle

# P-value lookup table ranges and resolutions
F_X_MIN, F_X_MAX, F_RESOLUTION = 0.0, 10.0, 512
T_X_MIN, T_X_MAX, T_RESOLUTION = 0.0, 6.0, 1024
Z_X_MIN, Z_X_MAX, Z_RESOLUTION = 0.0, 6.0, 1024
T_MAX_DF = 30
FLOAT_NEAR_ZERO = 1e-15

# Global p-value lookup tables
F_PVAL_TABLE = None  # F-distribution: [dfn, dfd, x] -> p-value
T_PVAL_TABLE = None  # t-distribution: [df, x] -> p-value
Z_PVAL_TABLE = None  # Normal: [x] -> p-value


def _init_tables():
    """Initialize/load cached p-value lookup tables for fast statistical tests."""
    global F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE

    cache_file = os.path.join(os.path.dirname(__file__), ".ols_lookup_tables.pkl")

    try:
        with open(cache_file, "rb") as f:
            F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE = pickle.load(f)
    except Exception:
        from scipy.stats import norm, f as f_dist, t as t_dist

        # F-distribution table: [dfn=1-30, dfd=10-500, x=0-10]
        f_x = np.linspace(F_X_MIN, F_X_MAX, F_RESOLUTION)
        f_dfn_range = np.arange(1, 31)

        # Adaptive dfd range for better resolution at small df
        f_dfd_range = np.concatenate(
            [
                np.arange(10, 31),  # 10-30 by 1
                np.arange(35, 101, 5),  # 35-100 by 5
                np.arange(110, 201, 10),  # 110-200 by 10
                np.arange(220, 501, 20),  # 220-500 by 20
            ]
        )

        F_PVAL_TABLE = np.zeros((30, len(f_dfd_range), F_RESOLUTION))
        for i, dfn in enumerate(f_dfn_range):
            for j, dfd in enumerate(f_dfd_range):
                F_PVAL_TABLE[i, j, :] = 1 - f_dist.cdf(f_x, dfn, dfd)

        # t-distribution table: [df=1-30, x=0-6]
        t_x = np.linspace(T_X_MIN, T_X_MAX, T_RESOLUTION)
        t_df_range = np.arange(1, T_MAX_DF + 1)

        T_PVAL_TABLE = np.zeros((T_MAX_DF, T_RESOLUTION))
        for i, df in enumerate(t_df_range):
            T_PVAL_TABLE[i, :] = 2.0 * (1.0 - t_dist.cdf(t_x, df))  # Two-tailed

        # Normal distribution table: [x=0-6]
        x_norm = np.linspace(Z_X_MIN, Z_X_MAX, Z_RESOLUTION)
        Z_PVAL_TABLE = 2 * (1 - norm.cdf(x_norm))  # Two-tailed

        # Cache tables
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE), f)
        except:
            pass


_init_tables()

# AOT compilation setup (build-time only)
if os.environ.get("NUMBA_AOT_BUILD"):
    try:
        from numba.pycc import CC

        cc = CC("ols_compiled")
        compile_function = lambda sig: lambda func: cc.export(func.__name__, sig)(func)  # type: ignore
    except ImportError as e:
        print(f"Warning: AOT compilation not available: {e}")
        cc = None
        compile_function = None
else:
    cc = None
    compile_function = None


# REMOVED: _find_dfd_index function to avoid Numba compilation issues
# The logic is now inlined in _f_to_pval function


def _ols_core(
    X_expanded,
    y,
    target_indices,
    f_pval_table,
    t_pval_table,
    z_pval_table,
    correction_method=0,
    alpha=0.05,
):
    """
    Core OLS implementation with F/t-tests and multiple comparison corrections.

    ALGORITHM:
    1. QR decomposition of [1, X] for numerical stability
    2. F-test for overall model significance
    3. t-tests for individual coefficients (specified by target_indices)
    4. Apply multiple comparison correction if requested

    Args:
        X_expanded: (n, p) design matrix (no intercept column)
        y: (n,) response vector
        target_indices: Coefficient indices to test
        *_pval_table: Pre-computed p-value lookup tables
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
        alpha: Significance level

    Returns:
        results: [F_significant, uncorrected_significances..., corrected_significances...]
    """

    def _z_to_pval(x: float) -> float:
        """Fast normal p-value via linear interpolation."""
        if x >= Z_X_MAX:
            return 0.0

        x_idx = x / Z_X_MAX * (Z_RESOLUTION - 1)
        x_idx_int = int(x_idx)

        if x_idx_int >= Z_RESOLUTION - 1:
            return z_pval_table[Z_RESOLUTION - 1]

        x_frac = x_idx - x_idx_int
        return (
            z_pval_table[x_idx_int] * (1 - x_frac)
            + z_pval_table[x_idx_int + 1] * x_frac
        )

    def _z_to_pval_1tail(x: float) -> float:
        """ADDED: Fast normal p-value 1-tailed upper tail."""
        # Get 2-tailed p-value and divide by 2 for upper tail
        two_tailed = _z_to_pval(x)
        return two_tailed / 2.0

    def _chi2_to_pval(x: float, df: int) -> float:
        """FIXED: Chi-squared p-value using Wilson-Hilferty approximation for large df (1-tailed)."""
        if x <= 0:
            return 1.0

        if df == 1:
            # FIXED: 1-tailed instead of 2-tailed
            return _z_to_pval_1tail(x**0.5)
        elif df == 2:
            return np.exp(-x / 2)
        else:
            # Wilson-Hilferty normal approximation
            h = 2.0 / (9.0 * df)
            z = ((x / df) ** (1.0 / 3.0) - 1.0 + h) / (h**0.5)
            # FIXED: 1-tailed instead of 2-tailed
            return _z_to_pval_1tail(abs(z))

    def _f_to_pval(x: float, dfn: int, dfd: int) -> float:
        """FIXED: Fast F p-value via lookup table with fallbacks for extreme cases."""
        if x <= F_X_MIN:
            return 1.0
        if x >= F_X_MAX:
            return 0.0

        # For large dfd, use chi-squared approximation
        if dfd > 500:
            chi2_stat = x * dfn
            if dfn <= 30:
                return _chi2_to_pval(chi2_stat, dfn)
            else:
                z = (chi2_stat - dfn) / ((2 * dfn) ** 0.5)
                # FIXED: 1-tailed instead of 2-tailed
                return _z_to_pval_1tail(abs(z))

        # FIXED: Correct lookup table indexing
        dfn_idx = min(max(dfn - 1, 0), 29)

        # FIXED: Inline dfd index calculation for Numba compatibility
        # f_dfd_range structure: [10-30, 35-100 by 5, 110-200 by 10, 220-500 by 20]
        if dfd <= 10:
            dfd_idx = 0
        elif dfd <= 30:
            dfd_idx = dfd - 10  # indices 0-20 for dfd 10-30
        elif dfd < 35:
            dfd_idx = 20  # Use dfd=30 (closest)
        elif dfd <= 100:
            # Map to 35, 40, 45, ..., 100 range (indices 21-34)
            rounded_dfd = ((dfd - 35) // 5) * 5 + 35
            dfd_idx = 21 + (rounded_dfd - 35) // 5
        elif dfd < 110:
            dfd_idx = 34  # Use dfd=100 (closest)
        elif dfd <= 200:
            # Map to 110, 120, 130, ..., 200 range (indices 35-44)
            rounded_dfd = ((dfd - 110) // 10) * 10 + 110
            dfd_idx = 35 + (rounded_dfd - 110) // 10
        elif dfd < 220:
            dfd_idx = 44  # Use dfd=200 (closest)
        elif dfd <= 500:
            # Map to 220, 240, 260, ..., 500 range (indices 45-59)
            rounded_dfd = ((dfd - 220) // 20) * 20 + 220
            dfd_idx = 45 + (rounded_dfd - 220) // 20
        else:
            dfd_idx = 59  # Use dfd=500 (highest in table)

        x_idx = (x - F_X_MIN) / (F_X_MAX - F_X_MIN) * (F_RESOLUTION - 1)
        x_idx_int = int(x_idx)

        if x_idx_int >= F_RESOLUTION - 1:
            return f_pval_table[dfn_idx, dfd_idx, F_RESOLUTION - 1]

        # Linear interpolation
        x_frac = x_idx - x_idx_int
        return (
            f_pval_table[dfn_idx, dfd_idx, x_idx_int] * (1 - x_frac)
            + f_pval_table[dfn_idx, dfd_idx, x_idx_int + 1] * x_frac
        )

    def _t_to_pval(x: float, df: int) -> float:
        """Fast t p-value via lookup table."""
        if x >= T_X_MAX:
            return 0.0

        df_idx = min(max(df - 1, 0), T_MAX_DF - 1)
        x_idx = x / T_X_MAX * (T_RESOLUTION - 1)
        x_idx_int = int(x_idx)

        if x_idx_int >= T_RESOLUTION - 1:
            return t_pval_table[df_idx, T_RESOLUTION - 1]

        x_frac = x_idx - x_idx_int
        return (
            t_pval_table[df_idx, x_idx_int] * (1 - x_frac)
            + t_pval_table[df_idx, x_idx_int + 1] * x_frac
        )

    def _correction_bonferroni(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Bonferroni correction: reject if p < α/m."""
        n = len(p_values)
        return (p_values < (alpha / n)).astype(np.float64)

    def _correction_fdr(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Benjamini-Hochberg FDR correction: step-up procedure."""
        n = len(p_values)
        if n == 0:
            return np.zeros(0)

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        critical_vals = np.arange(1, n + 1) / n * alpha

        result = np.zeros(n)
        last_sig = -1
        for i in range(n):
            if sorted_p[i] <= critical_vals[i]:
                last_sig = i

        if last_sig >= 0:
            for i in range(last_sig + 1):
                result[sorted_indices[i]] = 1.0

        return result

    def _correction_holm(p_values, alpha):
        """Holm correction: step-down procedure."""
        n = len(p_values)
        if n == 0:
            return np.zeros(0)

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        result = np.zeros(n)
        for i in range(n):
            if sorted_p[i] < alpha / (n - i):
                result[sorted_indices[i]] = 1.0
            else:
                break

        return result

    # MAIN OLS ALGORITHM
    n, p = X_expanded.shape
    X_int = np.column_stack((np.ones(n), X_expanded))  # Add intercept

    # QR decomposition for numerical stability
    Q, R = np.linalg.qr(X_int)
    QTy = np.ascontiguousarray(Q.T) @ np.ascontiguousarray(y)
    beta_all = np.linalg.solve(R, QTy)  # [intercept, coefficients...]
    beta = beta_all[1:]  # Exclude intercept

    # Residuals and MSE
    y_pred = X_int @ beta_all
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    dof = n - (p + 1)  # Degrees of freedom

    if dof <= 0:
        return np.zeros(len(target_indices) * 2 + 1)

    mse = ss_res / dof

    # F-test for overall model significance
    if p > 0:
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot > 1e-10:
            f_stat = ((ss_tot - ss_res) / p) / mse
            f_p = _f_to_pval(f_stat, p, dof)
        else:
            f_p = 1.0
    else:
        f_p = 1.0

    f_significant = 1.0 if f_p < alpha else 0.0

    # Individual coefficient tests
    n_targets = len(target_indices)
    results = np.zeros(1 + 2 * n_targets)  # [F_test, uncorrected..., corrected...]
    results[0] = f_significant

    if n_targets > 0 and mse > FLOAT_NEAR_ZERO:
        target_p_values = np.ones(n_targets)

        # t-tests for target coefficients
        for idx_pos, coef_idx in enumerate(target_indices):
            if coef_idx < p:
                param_idx = coef_idx + 1  # Account for intercept

                # Standard error via inverse of R'R (QR-based)
                ei = np.zeros(p + 1)
                ei[param_idx] = 1.0

                # Solve R^T xi = ei for standard error computation
                xi = np.zeros(p + 1)
                for i in range(p, -1, -1):
                    xi[i] = ei[i]
                    for j in range(i + 1, p + 1):
                        xi[i] -= R[i, j] * xi[j]
                    xi[i] /= R[i, i]

                var_coef = mse * np.sum(xi**2)
                std_err = (var_coef) ** 0.5

                if std_err > FLOAT_NEAR_ZERO:
                    t_stat = abs(beta[coef_idx] / std_err)
                    target_p_values[idx_pos] = _t_to_pval(t_stat, dof)

        # Uncorrected significances
        uncorrected = (target_p_values < alpha).astype(np.float64)
        results[1 : 1 + n_targets] = uncorrected

        # Apply multiple comparison correction
        if correction_method == 1:
            corrected = _correction_bonferroni(target_p_values, alpha)
        elif correction_method == 2:
            corrected = _correction_fdr(target_p_values, alpha)
        elif correction_method == 3:
            corrected = _correction_holm(target_p_values, alpha)
        else:
            corrected = uncorrected

        results[1 + n_targets : 1 + 2 * n_targets] = corrected

    return results


def _generate_y_core(
    X_expanded, effect_sizes, heterogeneity, heteroskedasticity, sim_seed
):
    """
    Generate response variable with heterogeneity and heteroskedasticity.

    ALGORITHM:
    1. Linear predictor: X @ β (with heterogeneity = varying effects)
    2. Error generation with heteroskedasticity
    3. Y = linear_predictor + error

    Args:
        X_expanded: (n, p) design matrix
        effect_sizes: (p,) true effect sizes
        heterogeneity: SD of effect size variation across observations
        heteroskedasticity: Correlation between linear predictor and error variance
        sim_seed: Random seed

    Returns:
        y: (n,) response vector
    """

    def _generate_error(n_samples, sim_seed):
        """Generate standard normal errors."""
        if sim_seed >= 0:
            np.random.seed(sim_seed)
        return np.random.normal(0.0, 1.0, n_samples)

    def _apply_heterogeneity(X_matrix, effects, het_sd, sim_seed):
        """Apply varying effect sizes across observations."""
        n_samples, n_features = X_matrix.shape

        if abs(het_sd) < FLOAT_NEAR_ZERO:
            return np.sum(X_matrix * effects, axis=1)

        if sim_seed >= 0:
            np.random.seed(sim_seed + 1)

        linear_pred = np.zeros(n_samples)

        # Each observation gets slightly different effect sizes
        for i in range(n_features):
            base_effect = effects[i]
            noise_scale = het_sd * abs(base_effect)
            noise = np.random.normal(0.0, noise_scale, n_samples)
            het_effects_i = base_effect + noise
            linear_pred += X_matrix[:, i] * het_effects_i

        return linear_pred

    def _apply_heteroskedasticity(linear_pred, error, heterosk):
        """Make error variance depend on linear predictor."""
        if abs(heterosk) < FLOAT_NEAR_ZERO:
            return error

        n_samples = len(linear_pred)
        lp_mean = np.mean(linear_pred)
        lp_std = np.std(linear_pred)

        # Standardize linear predictor
        if lp_std > FLOAT_NEAR_ZERO:
            standardized = (linear_pred - lp_mean) / lp_std
        else:
            standardized = np.zeros(n_samples)

        # Error variance = baseline + correlation with predictor
        error_variance = (1.0 - heterosk) + np.abs(heterosk * standardized)
        error_variance = np.maximum(error_variance, 0.1)  # Avoid zero variance

        adjusted_error = error * (error_variance) ** 0.5

        # Re-standardize to maintain unit variance
        error_std = np.std(adjusted_error)
        if error_std > FLOAT_NEAR_ZERO:
            adjusted_error = adjusted_error / error_std

        return adjusted_error

    # Main Y generation
    n_samples = X_expanded.shape[0]
    linear_predictor = _apply_heterogeneity(
        X_expanded, effect_sizes, heterogeneity, sim_seed
    )
    error = _generate_error(n_samples, sim_seed + 2 if sim_seed >= 0 else -1)
    final_error = _apply_heteroskedasticity(linear_predictor, error, heteroskedasticity)

    return linear_predictor + final_error


# COMPILATION SETUP: AOT → JIT → Python fallback
if os.environ.get("NUMBA_AOT_BUILD"):
    if compile_function and cc:
        _ols_core = compile_function(
            "f8[:](f8[:,:], f8[:], i8[:], f8[:,:,:], f8[:,:], f8[:], i4, f8)"
        )(_ols_core)
        _generate_y_core = compile_function("f8[:](f8[:,:], f8[:], f8, f8, i8)")(
            _generate_y_core
        )
        cc.compile()
    _ols_runtime = _ols_core
    _generate_y_runtime = _generate_y_core
else:
    try:
        # Try AOT compiled version first
        from .ols_compiled import (
            _ols_core as _ols_core_aot,
            _generate_y_core as _generate_y_core_aot,
        )

        _ols_runtime = _ols_core_aot
        _generate_y_runtime = _generate_y_core_aot
        print("Using AOT OLS")
    except (ImportError, AttributeError):
        try:
            # Fall back to JIT compilation
            from numba import njit

            _ols_runtime = njit(
                "f8[:](f8[:,:], f8[:], i8[:], f8[:,:,:], f8[:,:], f8[:], i4, f8)",
                cache=True,
            )(_ols_core)
            _generate_y_runtime = njit("f8[:](f8[:,:], f8[:], f8, f8, i8)", cache=True)(
                _generate_y_core
            )
            print("Using JIT OLS")
        except ImportError:
            # Pure Python fallback
            _ols_runtime = _ols_core
            _generate_y_runtime = _generate_y_core
            print("Using Python OLS")
            print("Install numba for faster performance: pip install numba")


def _ols_analysis(X_expanded, y, target_indices, correction_method=0, alpha=0.05):
    """
    Public interface to OLS analysis.

    Args:
        X_expanded: (n, p) design matrix
        y: (n,) response vector
        target_indices: Coefficient indices to test
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
        alpha: Significance level

    Returns:
        [F_significant, individual_significances_uncorrected, individual_significances_corrected]
    """
    return _ols_runtime(
        X_expanded,
        y,  # type: ignore
        target_indices,
        F_PVAL_TABLE,
        T_PVAL_TABLE,
        Z_PVAL_TABLE,
        correction_method,
        alpha,
    )


def _generate_y(X_expanded, effect_sizes, heterogeneity, heteroskedasticity, sim_seed):
    """
    Public interface to Y generation.

    Args:
        X_expanded: (n, p) design matrix
        effect_sizes: (p,) true effect sizes
        heterogeneity: Effect size variation across observations (0 = constant)
        heteroskedasticity: Error variance correlation with predictor (-1 to 1)
        sim_seed: Random seed

    Returns:
        y: (n,) response vector
    """
    return _generate_y_runtime(X_expanded, effect_sizes, heterogeneity, heteroskedasticity, sim_seed)  # type: ignore
