"""
OLS Analysis for Monte Carlo Power Analysis.

Performs OLS regression with F-tests, t-tests, and multiple comparison corrections.
Core computation is handled by the C++ backend (see mcpower/backends/native.py).
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
    from mcpower.stats.distributions import compute_critical_values_ols

    return compute_critical_values_ols(alpha, dfn, dfd, n_targets, correction_method)


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
    from mcpower.stats.distributions import compute_tukey_critical_value as _compute_tukey

    return _compute_tukey(alpha, n_levels, dfd)


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

    Performs its own QR decomposition (separate from the C++ OLS path).
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
