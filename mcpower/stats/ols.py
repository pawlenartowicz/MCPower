"""
OLS Analysis for Monte Carlo Power Analysis.

Performs OLS regression with F-tests, t-tests, and multiple comparison corrections.

Performance: JIT compiled via numba, falls back to pure Python.
Future: C++ backend via pybind11 (see mcpower/backends/native.py)
"""

import numpy as np

FLOAT_NEAR_ZERO = 1e-15


def compute_critical_values(alpha, dfn, dfd, n_targets, correction_method):
    """Pre-compute critical F and t values for OLS significance testing.

    Called once before the simulation loop so that each iteration only
    needs to compare test statistics against these thresholds.

    Args:
        alpha: Significance level.
        dfn: Numerator degrees of freedom (number of predictors).
        dfd: Denominator degrees of freedom (``n - p - 1``).
        n_targets: Number of individual effects being tested.
        correction_method: Encoded correction (0=none, 1=Bonferroni,
            2=Benjamini-Hochberg, 3=Holm).

    Returns:
        Tuple of ``(f_crit, t_crit, correction_t_crits)`` where
        *correction_t_crits* is an array of length *n_targets* with
        the per-rank critical t-values for the chosen correction.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist

    if dfd <= 0:
        return np.inf, np.inf, np.full(max(n_targets, 1), np.inf)

    f_crit = f_dist.ppf(1 - alpha, dfn, dfd) if dfn > 0 else np.inf
    t_crit = t_dist.ppf(1 - alpha / 2, dfd)

    m = n_targets
    if m == 0:
        return f_crit, t_crit, np.empty(0)

    if correction_method == 0:  # None
        correction_t_crits = np.full(m, t_crit)
    elif correction_method == 1:  # Bonferroni
        bonf_crit = t_dist.ppf(1 - alpha / (2 * m), dfd)
        correction_t_crits = np.full(m, bonf_crit)
    elif correction_method == 2:  # FDR (Benjamini-Hochberg)
        correction_t_crits = np.array([t_dist.ppf(1 - (k + 1) / m * alpha / 2, dfd) for k in range(m)])
    elif correction_method == 3:  # Holm
        correction_t_crits = np.array([t_dist.ppf(1 - alpha / (2 * (m - k)), dfd) for k in range(m)])
    else:
        correction_t_crits = np.full(m, t_crit)

    return f_crit, t_crit, correction_t_crits


def _ols_core(
    X_expanded,
    y,
    target_indices,
    f_crit,
    t_crit,
    correction_t_crits,
    correction_method=0,
):
    """
    Core OLS with F/t-tests and multiple comparison corrections.

    Args:
        X_expanded: (n, p) design matrix (no intercept)
        y: (n,) response vector
        target_indices: Coefficient indices to test
        f_crit: Precomputed F critical value
        t_crit: Precomputed t critical value (two-tailed, uncorrected)
        correction_t_crits: Precomputed correction critical values array
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm

    Returns:
        [F_significant, uncorrected..., corrected...]
    """

    # Main OLS
    n, p = X_expanded.shape
    X_int = np.column_stack((np.ones(n), X_expanded))

    Q, R = np.linalg.qr(X_int)
    QTy = np.ascontiguousarray(Q.T) @ np.ascontiguousarray(y)
    beta_all = np.linalg.solve(R, QTy)
    beta = beta_all[1:]

    y_pred = X_int @ beta_all
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    dof = n - (p + 1)

    n_targets = len(target_indices)

    if dof <= 0:
        return np.zeros(n_targets * 2 + 1)

    mse = ss_res / dof

    # F-test
    f_significant = 0.0
    if p > 0:
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot > 1e-10:
            f_stat = ((ss_tot - ss_res) / p) / mse
            f_significant = 1.0 if f_stat > f_crit else 0.0

    # t-tests
    results = np.zeros(1 + 2 * n_targets)
    results[0] = f_significant

    if n_targets > 0 and mse > FLOAT_NEAR_ZERO:
        t_abs_values = np.zeros(n_targets)

        for idx_pos in range(n_targets):
            coef_idx = target_indices[idx_pos]
            if coef_idx < p:
                param_idx = coef_idx + 1
                ei = np.zeros(p + 1)
                ei[param_idx] = 1.0

                xi = np.zeros(p + 1)
                for i in range(p + 1):
                    xi[i] = ei[i]
                    for j in range(i):
                        xi[i] -= R[j, i] * xi[j]
                    xi[i] /= R[i, i]

                var_coef = mse * np.sum(xi**2)
                std_err = var_coef**0.5
                if std_err > FLOAT_NEAR_ZERO:
                    t_abs_values[idx_pos] = abs(beta[coef_idx] / std_err)

        # Uncorrected: compare each |t| against t_crit
        for i in range(n_targets):
            results[1 + i] = 1.0 if t_abs_values[i] > t_crit else 0.0

        # Correction logic
        if correction_method == 0 or correction_method == 1:
            # No correction or Bonferroni: direct comparison per target
            # (all thresholds in correction_t_crits are equal)
            for i in range(n_targets):
                results[1 + n_targets + i] = 1.0 if t_abs_values[i] > correction_t_crits[i] else 0.0
        elif correction_method == 2:
            # FDR (Benjamini-Hochberg): step-up procedure
            # Sort |t| descending, compare against correction_t_crits
            sorted_indices = np.zeros(n_targets, dtype=np.int64)
            for i in range(n_targets):
                sorted_indices[i] = i
            # Sort indices by |t| descending
            for i in range(n_targets):
                for j in range(i + 1, n_targets):
                    if t_abs_values[sorted_indices[j]] > t_abs_values[sorted_indices[i]]:
                        tmp = sorted_indices[i]
                        sorted_indices[i] = sorted_indices[j]
                        sorted_indices[j] = tmp

            # Find last k where |t|_(k) >= correction_t_crits[k]
            corrected = np.zeros(n_targets)
            last_sig = -1
            for k in range(n_targets):
                if t_abs_values[sorted_indices[k]] > correction_t_crits[k]:
                    last_sig = k
            if last_sig >= 0:
                for k in range(last_sig + 1):
                    corrected[sorted_indices[k]] = 1.0
            results[1 + n_targets : 1 + 2 * n_targets] = corrected
        elif correction_method == 3:
            # Holm: step-down procedure
            sorted_indices = np.zeros(n_targets, dtype=np.int64)
            for i in range(n_targets):
                sorted_indices[i] = i
            # Sort indices by |t| descending
            for i in range(n_targets):
                for j in range(i + 1, n_targets):
                    if t_abs_values[sorted_indices[j]] > t_abs_values[sorted_indices[i]]:
                        tmp = sorted_indices[i]
                        sorted_indices[i] = sorted_indices[j]
                        sorted_indices[j] = tmp

            corrected = np.zeros(n_targets)
            for k in range(n_targets):
                if t_abs_values[sorted_indices[k]] > correction_t_crits[k]:
                    corrected[sorted_indices[k]] = 1.0
                else:
                    break
            results[1 + n_targets : 1 + 2 * n_targets] = corrected
        else:
            # Unknown correction: same as uncorrected
            results[1 + n_targets : 1 + 2 * n_targets] = results[1 : 1 + n_targets]

    return results


def _generate_y_core(X_expanded, effect_sizes, heterogeneity, heteroskedasticity, sim_seed):
    """Generate ``y = X @ beta + error`` with optional assumption violations.

    Args:
        X_expanded: Design matrix of shape ``(n, p)``.
        effect_sizes: Standardised beta weights of shape ``(p,)``.
        heterogeneity: SD of per-observation effect-size noise
            (0 = homogeneous).
        heteroskedasticity: Correlation between the linear predictor
            and error SD (0 = homoskedastic).
        sim_seed: Random seed (``-1`` for unseeded).

    Returns:
        Response vector of shape ``(n,)``.
    """

    def _generate_error(n_samples, sim_seed):
        """Draw i.i.d. N(0,1) errors."""
        if sim_seed >= 0:
            np.random.seed(sim_seed)
        return np.random.normal(0.0, 1.0, n_samples)

    def _apply_heterogeneity(X_matrix, effects, het_sd, sim_seed):
        """Add per-observation Gaussian noise to each effect size."""
        n_samples, n_features = X_matrix.shape
        if abs(het_sd) < FLOAT_NEAR_ZERO:
            return np.sum(X_matrix * effects, axis=1)
        if sim_seed >= 0:
            np.random.seed(sim_seed + 1)
        linear_pred = np.zeros(n_samples)
        for i in range(n_features):
            base_effect = effects[i]
            noise_scale = het_sd * abs(base_effect)
            noise = np.random.normal(0.0, noise_scale, n_samples)
            het_effects_i = base_effect + noise
            linear_pred += X_matrix[:, i] * het_effects_i
        return linear_pred

    def _apply_heteroskedasticity(linear_pred, error, heterosk):
        """Scale error variance proportionally to the linear predictor."""
        if abs(heterosk) < FLOAT_NEAR_ZERO:
            return error
        n_samples = len(linear_pred)
        lp_mean = np.mean(linear_pred)
        lp_std = np.std(linear_pred)
        if lp_std > FLOAT_NEAR_ZERO:
            standardized = (linear_pred - lp_mean) / lp_std
        else:
            standardized = np.zeros(n_samples)
        error_variance = (1.0 - heterosk) + np.abs(heterosk * standardized)
        error_variance = np.maximum(error_variance, 0.1)
        adjusted_error = error * error_variance**0.5
        error_std = np.std(adjusted_error)
        if error_std > FLOAT_NEAR_ZERO:
            adjusted_error = adjusted_error / error_std
        return adjusted_error

    n_samples = X_expanded.shape[0]
    linear_predictor = _apply_heterogeneity(X_expanded, effect_sizes, heterogeneity, sim_seed)
    error = _generate_error(n_samples, sim_seed + 2 if sim_seed >= 0 else -1)
    final_error = _apply_heteroskedasticity(linear_predictor, error, heteroskedasticity)
    return linear_predictor + final_error


def compute_tukey_critical_value(alpha, n_levels, dfd):
    """Compute Tukey HSD critical value for pairwise comparisons.

    Uses the Studentized Range distribution. The critical value for
    comparing means is ``q_{alpha, k, df} / sqrt(2)``.

    Args:
        alpha: Significance level.
        n_levels: Number of factor levels (k).
        dfd: Denominator degrees of freedom (n - p - 1).

    Returns:
        Tukey critical t-value threshold.
    """
    from scipy.stats import studentized_range

    if dfd <= 0:
        return np.inf
    q_crit = studentized_range.ppf(1 - alpha, n_levels, dfd)
    return q_crit / np.sqrt(2)


def compute_posthoc_contrasts(
    X_expanded,
    y,
    posthoc_specs,
    method,
    t_crit,
    tukey_crits,
    target_indices=None,
    correction_method=0,
    correction_t_crits_combined=None,
):
    """Compute post-hoc pairwise contrast significance after OLS.

    Performs its own QR decomposition (keeps existing ``_ols_core`` clean).
    When correction is FDR or Holm, also re-derives the regular target
    t-statistics so that the correction ranking spans ALL tests (regular +
    post-hoc) in one combined family.

    Args:
        X_expanded: ``(n, p)`` design matrix (no intercept).
        y: ``(n,)`` response vector.
        posthoc_specs: List of ``PostHocSpec`` objects.
        method: ``"t-test"`` or ``"tukey"``.
        t_crit: Uncorrected t critical value.
        tukey_crits: Dict mapping factor_name to Tukey critical value.
        target_indices: Regular target indices (needed for combined correction).
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm.
        correction_t_crits_combined: Combined critical values array
            (length ``n_regular + n_posthoc``) for the correction family.

    Returns:
        Tuple ``(posthoc_uncorrected, posthoc_corrected,
        regular_corrected_override)``.  ``regular_corrected_override`` is
        a bool array of length ``n_regular`` when FDR/Holm combined
        correction was applied, or ``None`` otherwise.
    """
    n_specs = len(posthoc_specs)
    if n_specs == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), None

    n, p = X_expanded.shape
    X_int = np.column_stack((np.ones(n), X_expanded))

    Q, R = np.linalg.qr(X_int)
    QTy = np.ascontiguousarray(Q.T) @ np.ascontiguousarray(y)
    beta_all = np.linalg.solve(R, QTy)
    beta = beta_all[1:]

    y_pred = X_int @ beta_all
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    dof = n - (p + 1)

    if dof <= 0:
        return np.zeros(n_specs, dtype=bool), np.zeros(n_specs, dtype=bool), None

    mse = ss_res / dof

    # Compute (R^T R)^{-1} for variance of contrasts
    R_inv = np.linalg.solve(R, np.eye(p + 1))
    cov_unscaled = R_inv @ R_inv.T  # (X'X)^{-1}

    # Compute post-hoc t-statistics
    posthoc_t_abs = np.zeros(n_specs)
    for i, spec in enumerate(posthoc_specs):
        c = np.zeros(p + 1)
        if spec.col_idx_a is None and spec.col_idx_b is None:
            continue
        elif spec.col_idx_a is None:
            c[spec.col_idx_b + 1] = 1.0
        elif spec.col_idx_b is None:
            c[spec.col_idx_a + 1] = 1.0
        else:
            c[spec.col_idx_a + 1] = 1.0
            c[spec.col_idx_b + 1] = -1.0

        contrast_est = c @ beta_all
        se_sq = mse * (c @ cov_unscaled @ c)
        if se_sq > FLOAT_NEAR_ZERO:
            posthoc_t_abs[i] = abs(contrast_est) / np.sqrt(se_sq)

    # Uncorrected post-hoc significance
    ph_uncorrected = np.zeros(n_specs, dtype=bool)
    ph_corrected = np.zeros(n_specs, dtype=bool)

    if method == "tukey":
        for i, spec in enumerate(posthoc_specs):
            tukey_crit = tukey_crits.get(spec.factor_name, np.inf)
            sig = posthoc_t_abs[i] > tukey_crit
            ph_uncorrected[i] = sig
            ph_corrected[i] = sig  # Tukey IS the correction
        return ph_uncorrected, ph_corrected, None

    # t-test method: uncorrected uses plain t_crit
    ph_uncorrected = posthoc_t_abs > t_crit

    # Corrected column
    n_regular = len(target_indices) if target_indices is not None else 0
    needs_combined_ranking = correction_method in (2, 3) and n_regular > 0

    if needs_combined_ranking:
        # FDR or Holm: need combined ranking of regular + posthoc t-statistics
        # Re-derive regular target t-statistics from the same QR decomposition
        regular_t_abs = np.zeros(n_regular)
        for idx_pos in range(n_regular):
            coef_idx = target_indices[idx_pos]
            if coef_idx < p:
                param_idx = coef_idx + 1
                ei = np.zeros(p + 1)
                ei[param_idx] = 1.0
                xi = np.linalg.solve(R, ei)
                var_coef = mse * np.sum(xi**2)
                if var_coef > FLOAT_NEAR_ZERO:
                    regular_t_abs[idx_pos] = abs(beta[coef_idx] / np.sqrt(var_coef))

        # Combine all t-statistics: [regular..., posthoc...]
        all_t_abs = np.concatenate([regular_t_abs, posthoc_t_abs])
        n_combined = len(all_t_abs)

        if correction_t_crits_combined is None or len(correction_t_crits_combined) != n_combined:
            # Fallback: no combined correction
            ph_corrected = ph_uncorrected.copy()
            return ph_uncorrected, ph_corrected, None

        # Apply combined correction
        all_corrected = np.zeros(n_combined, dtype=bool)

        if correction_method == 2:
            # FDR (BH): step-up procedure
            sorted_indices = np.argsort(-all_t_abs)
            last_sig = -1
            for k in range(n_combined):
                if all_t_abs[sorted_indices[k]] > correction_t_crits_combined[k]:
                    last_sig = k
            if last_sig >= 0:
                for k in range(last_sig + 1):
                    all_corrected[sorted_indices[k]] = True
        elif correction_method == 3:
            # Holm: step-down procedure
            sorted_indices = np.argsort(-all_t_abs)
            for k in range(n_combined):
                if all_t_abs[sorted_indices[k]] > correction_t_crits_combined[k]:
                    all_corrected[sorted_indices[k]] = True
                else:
                    break

        regular_corrected_override = all_corrected[:n_regular]
        ph_corrected = all_corrected[n_regular:]
        return ph_uncorrected, ph_corrected, regular_corrected_override

    elif correction_method == 1 and correction_t_crits_combined is not None:
        # Bonferroni: same threshold for all (already computed with combined m)
        bonf_crit = correction_t_crits_combined[0]
        ph_corrected = posthoc_t_abs > bonf_crit
    else:
        # No correction
        ph_corrected = ph_uncorrected.copy()

    return ph_uncorrected, ph_corrected, None


# Try JIT compilation, fall back to pure Python
try:
    from numba import njit

    _ols_jit = njit(
        "f8[:](f8[:,:], f8[:], i8[:], f8, f8, f8[:], i4)",
        cache=True,
    )(_ols_core)
    _generate_y_jit = njit("f8[:](f8[:,:], f8[:], f8, f8, i8)", cache=True)(_generate_y_core)
    _USE_JIT = True
except ImportError:
    _ols_jit = _ols_core
    _generate_y_jit = _generate_y_core
    _USE_JIT = False
