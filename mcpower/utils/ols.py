"""
Compiled statistical utilities for Monte Carlo Power Analysis with Numba optimization.
Provides optimized OLS regression analysis and dependent variable generation with 
lookup tables for F, T, and Z distributions for enhanced performance.
"""

import os

if os.environ.get('NUMBA_AOT_BUILD'):
    from numba.pycc import CC
    cc = CC('mcpower_compiled')
    compile_function = lambda sig: lambda func: cc.export(func.__name__, sig)(func) # type: ignore
else:
    from numba import njit
    cc = None
    compile_function = lambda sig: njit(sig, cache=True)


import numpy as np
import pickle

# Constants
F_X_MIN, F_X_MAX, F_RESOLUTION = 0.0, 10.0, 1024
T_X_MIN, T_X_MAX, T_RESOLUTION = 0.0, 6.0, 1024
Z_X_MIN, Z_X_MAX, Z_RESOLUTION = 0.0, 6.0, 1024

FLOAT_NEAR_ZERO = 1e-15 # for division by 0 protection

# Global tables (initialized at import) with p-values
F_PVAL_TABLE = None
T_PVAL_TABLE = None
Z_PVAL_TABLE = None


def _init_tables():
    """
    Initialize or load cached CDF tables.
    Creates 25.5 MB cached file
    """
    global F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE
    
    cache_file = os.path.join(os.path.dirname(__file__), '.ols_lookup_tables.pkl')
    
    try:
        with open(cache_file, 'rb') as f:
            F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE = pickle.load(f)
    except Exception:
        # Create tables
        from scipy.stats import norm, f as f_dist, t as t_dist

        
        # F-distribution table (dfn: 1-30, dfd: 10-500 step 10)
        f_x = np.linspace(F_X_MIN, F_X_MAX, F_RESOLUTION)
        f_dfn_range = np.arange(1, 31)  # 30 values
        f_dfd_range = np.arange(10, 501, 5)  # 99 values
        
        F_PVAL_TABLE = np.zeros((30, 99, F_RESOLUTION))
        for i, dfn in enumerate(f_dfn_range):
            for j, dfd in enumerate(f_dfd_range):
                F_PVAL_TABLE[i, j, :] = 1 - f_dist.cdf(f_x, dfn, dfd)

        # T-distribution table (df: 1-300)
        t_x = np.linspace(T_X_MIN, T_X_MAX, T_RESOLUTION)
        t_df_range = np.arange(1, 101)  # 100 values
        
        T_PVAL_TABLE = np.zeros((100, T_RESOLUTION))
        for i, df in enumerate(t_df_range):
            T_PVAL_TABLE[i, :] =  2.0 * (1.0 - t_dist.cdf(t_x, df))
        
        x_norm = np.linspace(Z_X_MIN, Z_X_MAX, Z_RESOLUTION)
        Z_PVAL_TABLE = 2 * (1 - norm.cdf(x_norm))


        # Cache tables
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE), f)
        except:
            pass  # Ignore cache write errors

# Initialize tables at import
_init_tables()

# Compiled OLS method
# f8[:,:] - X_expanded
# f8[:] - y
# i8[:] - target_indices
# f8[:,:,:] - f_pval_table (3D F-table)
# f8[:,:] - t_pval_table (2D T-table)
# f8[:] - z_pval_table (1D Z-table)
# i4 - correction_method
# f8 - alpha
@compile_function('f8[:](f8[:,:], f8[:], i8[:], f8[:,:,:], f8[:,:], f8[:], i4, f8)')
def _ols(X_expanded, y, target_indices, f_pval_table, t_pval_table, z_pval_table, correction_method=0, alpha=0.05):
    """
     OLS analysis for power calculations.
    
    Args:
        X_expanded: Design matrix (n_samples, n_features)
        y: Response vector  
        target_indices: Indices of coefficients to test
        correction_method: 0=none, 1=bonferroni, 2=benjamini_hochberg
        alpha: Significance level
        
    Returns:
        results: Array [f_significant, target_coef_significances...]
    """

    def _z_to_pval(x: float) -> float:
        """T-distribution CDF lookup for positive x."""
        if x >= Z_X_MAX:
            return 0.0
        
        # Linear interpolation in x
        x_idx = x / Z_X_MAX * (Z_RESOLUTION - 1)
        x_idx_int = int(x_idx)
        
        if x_idx_int >= Z_RESOLUTION - 1:
            return z_pval_table[Z_RESOLUTION - 1]
        
        x_frac = x_idx - x_idx_int
        return (z_pval_table[x_idx_int] * (1 - x_frac) + 
                z_pval_table[x_idx_int + 1] * x_frac)

    def _chi2_to_pval(x: float, df: int) -> float:
        if df == 1:
            return 2 * (1 - _z_to_pval(x**0.5))  # was np.sqrt(x)
        elif df == 2:
            return np.exp(-x/2)  # keep np.exp
        else:
            h = 2.0 / (9.0 * df)
            z = ((x/df)**(1.0/3.0) - 1.0 + h) / (h**0.5)  # was np.power and np.sqrt
            return 2 * (1 - _z_to_pval(abs(z)))
    
    def _f_to_pval(x: float, dfn: int, dfd: int) -> float:
        """F-distribution lookup."""
        if x <= F_X_MIN:
            return 1.0
        if x >= F_X_MAX:
            return 0.0
        
        # Use chi-squared approximation for large dfd
        if dfd > 500:
            chi2_stat = x * dfn
            if dfn <= 30:
                return _chi2_to_pval(chi2_stat, dfn)
            else:
                z = (chi2_stat - dfn) / np.sqrt(2 * dfn)
                return 2 * (1 - _z_to_pval(abs(z)))
        
        # Use lookup table (clamp dfn if needed)
        dfn_idx = min(max(dfn - 1, 0), 29)
        dfd_idx = min(max((dfd - 10) // 5, 0), 98)
        
        x_idx = (x - F_X_MIN) / (F_X_MAX - F_X_MIN) * (F_RESOLUTION - 1)
        x_idx_int = int(x_idx)
        
        if x_idx_int >= F_RESOLUTION - 1:
            return f_pval_table[dfn_idx, dfd_idx, F_RESOLUTION - 1]
        
        x_frac = x_idx - x_idx_int
        return (f_pval_table[dfn_idx, dfd_idx, x_idx_int] * (1 - x_frac) + 
                f_pval_table[dfn_idx, dfd_idx, x_idx_int + 1] * x_frac)

    def _t_to_pval(x: float, df: int) -> float:
        """T-distribution lookup for positive x."""
        if x >= T_X_MAX:
            return 0.0
        
        # Map to table index
        df_idx = min(max(df - 1, 0), 99)
        
        # Linear interpolation in x
        x_idx = x / T_X_MAX * (T_RESOLUTION - 1)
        x_idx_int = int(x_idx)
        
        if x_idx_int >= T_RESOLUTION - 1:
            return t_pval_table[df_idx, T_RESOLUTION - 1]
        
        x_frac = x_idx - x_idx_int
        return (t_pval_table[df_idx, x_idx_int] * (1 - x_frac) + 
                t_pval_table[df_idx, x_idx_int + 1] * x_frac)

    def _correction_bonferroni(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply Bonferroni correction."""
        n = len(p_values)
        return (p_values < (alpha / n)).astype(np.float64)

    def _correction_fdr(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
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
        """Apply Holm step-down correction."""
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
                break  # Stop at first non-significant
        
        return result

    n, p = X_expanded.shape
    
    # Add intercept
    X_int = np.column_stack((np.ones(n), X_expanded))
    
    # QR decomposition
    Q, R = np.linalg.qr(X_int)
    QTy = np.ascontiguousarray(Q.T) @ np.ascontiguousarray(y)
    beta_all = np.linalg.solve(R, QTy)
    beta = beta_all[1:]  # Skip intercept
    
    # Residuals and df-s (dof)
    y_pred = X_int @ beta_all
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    dof = n - (p + 1)
    
    if dof <= 0:
        return np.zeros(len(target_indices) + 1)
    
    mse = ss_res / dof
    
    # F-statistic (overall test)
    if p > 0:
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot > 1e-10: # Error protection
            f_stat = ((ss_tot - ss_res) / p) / mse
            f_p = _f_to_pval(f_stat, p, dof)
        else:
            f_p = 1.0
    else:
        f_p = 1.0
    
    f_significant = 1.0 if f_p < alpha else 0.0
    
    # T-statistics for target coefficients
    n_targets = len(target_indices)
    results = np.zeros(1 + 2 * n_targets)
    results[0] = f_significant
    
    if n_targets > 0 and mse > 1e-15:  # Division by zero protection
        target_p_values = np.ones(n_targets)
        
        for idx_pos, coef_idx in enumerate(target_indices):
            if coef_idx < p:
                # Standard error computation
                param_idx = coef_idx + 1  # +1 for intercept
                ei = np.zeros(p + 1)
                ei[param_idx] = 1.0
                
                # Solve R * x = ei (back substitution)
                xi = np.zeros(p + 1)
                for i in range(p, -1, -1):
                    xi[i] = ei[i]
                    for j in range(i + 1, p + 1):
                        xi[i] -= R[i, j] * xi[j]
                    xi[i] /= R[i, i]
                
                var_coef = mse * np.sum(xi ** 2)
                std_err = (var_coef)**0.5
                
                if std_err > 1e-15: # Division by zero protection
                    t_stat = abs(beta[coef_idx] / std_err)
                    target_p_values[idx_pos] = _t_to_pval(t_stat, dof)
        
        # Uncorrected results
        uncorrected = (target_p_values < alpha).astype(np.float64)
        results[1:1+n_targets] = uncorrected
        
        # Corrected results  
        if correction_method == 1:
            corrected = _correction_bonferroni(target_p_values, alpha)
        elif correction_method == 2:
            corrected = _correction_fdr(target_p_values, alpha)
        elif correction_method == 3:
            corrected = _correction_holm(target_p_values, alpha)
        else:
            corrected = uncorrected  # No correction
        
        results[1+n_targets:1+2*n_targets] = corrected
    
    return results

def _ols_analysis(X_expanded, y, target_indices, correction_method=0, alpha=0.05):
    return _ols(X_expanded, y, target_indices, F_PVAL_TABLE, T_PVAL_TABLE, Z_PVAL_TABLE, correction_method, alpha)

@compile_function('f8[:](f8[:,:], f8[:], f8, f8, i8)')
def _generate_y(X_expanded, effect_sizes, heterogeneity, heteroskedasticity, sim_seed):
    """
    Generate dependent variable for linear regression with numba optimization.
    
    Args:
        X_expanded: Design matrix including interactions (n_samples, n_features)
        effect_sizes: Effect sizes for each column in X_expanded (n_features,)
        heterogeneity_sd: Standard deviation for effect heterogeneity
        heteroskedasticity: Correlation between linear predictor and error variance
        sim_seed: Random seed for reproducibility
        
    Returns:
        y: Generated dependent variable array
    """
    def _generate_error(n_samples, sim_seed):
        """Generate standard normal errors."""
        if sim_seed >= 0:
            np.random.seed(sim_seed)
        return np.random.normal(0.0, 1.0, n_samples)
    
    def _apply_heterogeneity(X_matrix, effects, het_sd, sim_seed):
        """Apply effect heterogeneity across observations."""
        n_samples, n_features = X_matrix.shape
        
        if abs(het_sd) < FLOAT_NEAR_ZERO:
            # No heterogeneity - vectorized multiplication
            return np.sum(X_matrix * effects, axis=1)
        
        # With heterogeneity - generate random effects per feature
        if sim_seed >= 0:
            np.random.seed(sim_seed + 1)
        
        linear_pred = np.zeros(n_samples)
        
        # Generate heterogeneous effects - Numba-compatible version
        for i in range(n_features):
            base_effect = effects[i]
            noise_scale = het_sd * abs(base_effect)
            # Generate random variations for this feature
            noise = np.random.normal(0.0, noise_scale, n_samples)
            het_effects_i = base_effect + noise
            linear_pred += X_matrix[:, i] * het_effects_i
        
        return linear_pred
    
    def _apply_heteroskedasticity(linear_pred, error, heterosk):
        """Apply heteroskedasticity to error term."""
        if abs(heterosk) < FLOAT_NEAR_ZERO:
            return error
        
        n_samples = len(linear_pred)
        
        # Standardize linear predictor
        lp_mean = np.mean(linear_pred)
        lp_std = np.std(linear_pred)
        
        if lp_std > FLOAT_NEAR_ZERO:
            standardized = (linear_pred - lp_mean) / lp_std
        else:
            standardized = np.zeros(n_samples)
        
        # Error variance depends on predictor values
        error_variance = (1.0 - heterosk) + np.abs(heterosk * standardized)
        error_variance = np.maximum(error_variance, 0.1)
        
        # Scale error term
        adjusted_error = error * np.sqrt(error_variance)
        
        # Standardize to maintain unit variance
        error_std = np.std(adjusted_error)
        if error_std > FLOAT_NEAR_ZERO:
            adjusted_error = adjusted_error / error_std
            
        return adjusted_error
    
    n_samples = X_expanded.shape[0]
    
    # Generate linear predictor with optional heterogeneity
    linear_predictor = _apply_heterogeneity(X_expanded, effect_sizes, heterogeneity, sim_seed)
    
    # Generate error term
    error = _generate_error(n_samples, sim_seed + 2 if sim_seed >= 0 else -1)
    
    # Apply heteroskedasticity
    final_error = _apply_heteroskedasticity(linear_predictor, error, heteroskedasticity)
    
    # Combine
    y = linear_predictor + final_error
    
    return y