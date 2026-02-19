"""Linear Mixed-Effects (LME) analysis for Monte Carlo power analysis.

Routes LME analysis to the C++ native backend (custom profiled-deviance
solver) or statsmodels MixedLM (fallback).  Includes likelihood-ratio
tests, Wald z-tests, and multiple-comparison corrections (Bonferroni,
Benjamini-Hochberg, Holm).

The C++ backend handles convergence efficiently internally.  The
statsmodels fallback path uses warm-start parameter caching for speedup.
"""

import threading
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Suppress statsmodels convergence warnings (expected with small samples/low ICC).
# Module-level filterwarnings with module= is unreliable for statsmodels internals,
# so we also use catch_warnings() context managers around .fit() calls below.
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")
warnings.filterwarnings("ignore", category=Warning, module="numpy")

# Thread-local cache for warm start parameters (safe under parallel execution)
_lme_thread_local = threading.local()


def _lme_analysis_wrapper(
    X_expanded: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_column_indices: List[int],
    correction_method: int,
    alpha: float,
    backend: str = "custom",
    verbose: bool = False,
    chi2_crit: Optional[float] = None,
    z_crit: Optional[float] = None,
    correction_z_crits: Optional[np.ndarray] = None,
    re_result: Optional[Any] = None,
) -> Optional[Union[np.ndarray, Dict]]:
    """
    Route LME analysis to specific backend.

    This is the public API for mixed model analysis. Supports the custom
    profiled-deviance solver (fast) and statsmodels MixedLM (fallback).

    Args:
        X_expanded: (n, p) design matrix (excludes cluster effect columns)
        y: (n,) response vector
        target_indices: Coefficient indices to test (fixed effects only)
        cluster_ids: (n,) cluster membership array [0,0,0, 1,1,1, ...]
        cluster_column_indices: Indices of cluster effect columns (unused)
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
        alpha: Significance level
        backend: "custom" (default) or "statsmodels" (fallback)
        verbose: Return detailed diagnostics
        chi2_crit: Precomputed chi2 critical value (custom backend)
        z_crit: Precomputed z critical value (custom backend)
        correction_z_crits: Precomputed correction z critical values (custom backend)
        re_result: RandomEffectsResult from _generate_random_effects
            (Phase 2: slopes/nesting). Contains Z matrices, nested
            cluster IDs, and child_to_parent mappings.

    Returns:
        When verbose=False (backward compatible):
            np.ndarray: [F_significant, uncorrected..., corrected..., wald_flag] or None if failed
        When verbose=True:
            Dict with:
                - 'results': np.ndarray (same as above) or None if failed
                - 'diagnostics': Dict with convergence, fit stats, etc. (if succeeded)
                - 'failure_reason': str (if failed)
    """
    if backend == "custom":
        # Route to the appropriate solver based on model type
        if re_result is not None and re_result.Z_matrices:
            # Random slopes path: q > 1
            return _lme_analysis_custom_general(
                X_expanded,
                y,
                target_indices,
                cluster_ids,
                re_result,
                correction_method,
                alpha,
                chi2_crit=chi2_crit,
                z_crit=z_crit,
                correction_z_crits=correction_z_crits,
                verbose=verbose,
            )
        elif re_result is not None and re_result.child_to_parent is not None:
            # Nested random intercepts path
            return _lme_analysis_custom_nested(
                X_expanded,
                y,
                target_indices,
                re_result,
                correction_method,
                alpha,
                chi2_crit=chi2_crit,
                z_crit=z_crit,
                correction_z_crits=correction_z_crits,
                verbose=verbose,
            )
        else:
            # Standard q=1 random intercept path
            return _lme_analysis_custom(
                X_expanded,
                y,
                target_indices,
                cluster_ids,
                correction_method,
                alpha,
                chi2_crit=chi2_crit,
                z_crit=z_crit,
                correction_z_crits=correction_z_crits,
                verbose=verbose,
            )
    elif backend == "statsmodels":
        return _lme_analysis_statsmodels(
            X_expanded, y, target_indices, cluster_ids, cluster_column_indices, correction_method, alpha, verbose
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _lme_analysis_statsmodels(
    X_expanded: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_column_indices: List[int],
    correction_method: int,
    alpha: float,
    verbose: bool = False,
) -> Optional[Union[np.ndarray, Dict]]:
    """
    Statsmodels MixedLM implementation with REML estimation.

    Optimizations:
    - Warm start with exponential moving average (30-50% speedup)
    - Direct MixedLM initialization (5-10% speedup vs from_formula)
    - Optimized REML settings (10-20% speedup)
    - Convergence retry strategy (allows ≤3% failures)

    Args:
        X_expanded: (n, p) design matrix (includes cluster effect columns)
        y: (n,) response vector
        target_indices: Coefficient indices to test (fixed effects only)
        cluster_ids: (n,) cluster membership array
        cluster_column_indices: Indices of cluster effect columns to remove
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
        alpha: Significance level
        verbose: Return detailed diagnostics

    Returns:
        When verbose=False:
            [F_significant, uncorrected..., corrected...] or None if convergence failed
        When verbose=True:
            Dict with 'results', 'diagnostics', and optionally 'failure_reason'
    """
    _lme_warm_start_params = getattr(_lme_thread_local, "warm_start_params", None)

    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        from mcpower.stats.distributions import chi2_cdf
    except ImportError as e:
        raise ImportError("statsmodels required for mixed models: pip install mcpower[lme]") from e

    n, p = X_expanded.shape
    n_targets = len(target_indices)

    # Note: X_expanded already excludes cluster effects (they're not in the design matrix)
    # cluster_column_indices is now unused in this function but kept for API compatibility
    X_fixed = X_expanded

    # Step 1: Add intercept to fixed effects
    X_with_intercept = np.column_stack([np.ones(n), X_fixed])

    # Step 2: Fit LME with REML (random intercept only)
    # Use direct initialization (faster than from_formula)
    model = MixedLM(endog=y, exog=X_with_intercept, groups=cluster_ids)

    # Convergence retry strategy: warm start → cold start → cold start (more iters)
    # Cold start is tried early because a bad warm start point won't improve
    # with more iterations — the optimizer needs a fresh starting point.
    result = None
    convergence_level = -1
    failure_reason = None

    if _lme_warm_start_params is not None:
        attempts = [
            (50, _lme_warm_start_params),  # 1. Warm start (fast path)
            (100, None),  # 2. Cold start (fresh starting point)
            (200, None),  # 3. Cold start with more iterations
        ]
    else:
        attempts = [
            (100, None),  # 1. Cold start
            (200, None),  # 2. Cold start with more iterations
            (500, None),  # 3. Cold start with max iterations
        ]

    for iter_idx, (max_iter, start_params) in enumerate(attempts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if start_params is not None:
                    result = model.fit(reml=True, method="lbfgs", start_params=start_params, full_output=False, maxiter=max_iter)
                else:
                    result = model.fit(reml=True, method="lbfgs", full_output=False, maxiter=max_iter)

            # Check convergence flag before accepting the result
            if not getattr(result, "converged", True):
                failure_reason = "Model did not converge"
                if iter_idx == len(attempts) - 1:
                    if verbose:
                        return {"results": None, "failure_reason": failure_reason}
                    else:
                        return None
                continue

            # Successful fit — update warm start cache only with converged results
            _lme_thread_local.warm_start_params = result.params.copy()
            convergence_level = iter_idx
            break

        except Exception as e:
            failure_reason = f"{type(e).__name__}: {str(e)}"
            if iter_idx == len(attempts) - 1:
                if verbose:
                    return {"results": None, "failure_reason": failure_reason}
                else:
                    return None
            continue

    if result is None:
        if verbose:
            return {"results": None, "failure_reason": failure_reason or "Unknown convergence failure"}
        else:
            return None

    # Step 3: Extract fixed effects results (exclude intercept)
    fixed_params = result.params[1:]  # Skip intercept
    fixed_pvalues = result.pvalues[1:]  # Skip intercept

    # Step 4: Likelihood ratio test for overall model fit (analog to F-test)
    # Compare full model to null model (intercept + random effects only)
    # NOTE: Must use ML (not REML) for LR tests comparing models with different
    # fixed effects. REML likelihoods are not comparable across such models.

    def _fit_ml_with_retry(model, start_params=None):
        """Fit ML model with progressive retry strategy.

        Returns:
            (result, success, llf) where llf is the log-likelihood computed
            via model.loglike() rather than result.llf, because statsmodels
            returns inf for .llf when the random effects variance is at the
            boundary (zero).
        """
        # Use lower iteration limits when warm start available
        if start_params is not None:
            ml_max_iters = [100, 300, 800]
        else:
            ml_max_iters = [200, 500, 1000]

        for max_iter in ml_max_iters:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if start_params is not None:
                        ml_result = model.fit(reml=False, method="lbfgs", start_params=start_params, maxiter=max_iter, full_output=False)
                    else:
                        ml_result = model.fit(reml=False, method="lbfgs", maxiter=max_iter, full_output=False)

                # Use model.loglike() instead of .llf — statsmodels returns
                # inf for .llf when the random effects variance is at the
                # boundary (cov_re=0), but model.loglike() computes correctly.
                llf = model.loglike(ml_result.params)
                if not np.isnan(llf) and np.isfinite(llf):
                    return ml_result, True, llf
            except Exception:
                continue

        return None, False, np.nan

    def _compute_wald_test(result, alpha):
        """
        Compute Wald test for overall model significance.

        H0: all fixed effects = 0
        Test statistic: β' * Cov(β)^-1 * β ~ χ²(df)

        Used as fallback when LR test produces NaN.
        """
        try:
            # Extract fixed effects (excluding intercept)
            # Use fe_params which contains ONLY fixed effects (not random effect variances)
            if hasattr(result, "fe_params"):
                beta = result.fe_params[1:]  # Exclude intercept
            else:
                # Fallback: assume first k params are fixed effects
                n_fixed = result.model.exog.shape[1]
                beta = result.params[1:n_fixed]

            # Extract covariance matrix for fixed effects (excluding intercept)
            cov_params = result.cov_params()
            # Only take the fixed effects portion (first n_fixed rows/cols)
            n_fixed = result.model.exog.shape[1]
            if hasattr(cov_params, "iloc"):
                cov_beta = cov_params.iloc[1:n_fixed, 1:n_fixed].values
            else:
                cov_beta = cov_params[1:n_fixed, 1:n_fixed]

            # Compute Wald statistic
            wald_stat = beta @ np.linalg.inv(cov_beta) @ beta
            df = len(beta)

            # Compute p-value
            p_value = 1 - chi2_cdf(wald_stat, df)

            return 1.0 if p_value < alpha else 0.0
        except Exception:
            return 0.0  # Conservative fallback

    # Try likelihood ratio test first
    test_method = "unknown"
    full_ml_llf = np.nan
    null_ml_llf = np.nan
    try:
        # Check if random effects variance is at the boundary (zero).
        # When cov_re ≈ 0, MixedLM reduces to OLS and statsmodels' profiled
        # log-likelihood becomes unreliable (returns inf for .llf, and
        # model.loglike() can produce inconsistent values across models).
        # In this case, compute LRT using OLS log-likelihoods directly.
        re_var = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, "iloc") else float(np.asarray(result.cov_re).flat[0])
        at_boundary = re_var < 1e-10

        if at_boundary:
            # Random effects variance ≈ 0: model is effectively OLS.
            # Use OLS log-likelihoods for a reliable LRT.
            try:
                import statsmodels.api as sm
            except ImportError as e:
                raise ImportError("statsmodels required for mixed models: pip install mcpower[lme]") from e

            ols_full = sm.OLS(y, X_with_intercept).fit()
            ols_null = sm.OLS(y, np.ones((n, 1))).fit()
            full_ml_llf = ols_full.llf
            null_ml_llf = ols_null.llf
            lr_stat = 2 * (full_ml_llf - null_ml_llf)
            df = p  # Degrees of freedom = number of fixed effects (excluding intercept)

            if np.isnan(lr_stat) or lr_stat < 0 or not np.isfinite(lr_stat):
                f_significant = _compute_wald_test(result, alpha)
                test_method = "wald_fallback"
            else:
                lr_pvalue = 1 - chi2_cdf(lr_stat, df) if df > 0 else 1.0
                f_significant = 1.0 if lr_pvalue < alpha else 0.0
                test_method = "likelihood_ratio"
        else:
            # Random effects variance > 0: use ML refits with model.loglike()
            full_ml_result, full_success, full_ml_llf = _fit_ml_with_retry(model, start_params=result.params)

            if not full_success:
                f_significant = _compute_wald_test(result, alpha)
                test_method = "wald_fallback"
            else:
                null_model = MixedLM(endog=y, exog=np.ones((n, 1)), groups=cluster_ids)
                null_start_params = result.params[:1]
                null_result, null_success, null_ml_llf = _fit_ml_with_retry(null_model, start_params=null_start_params)

                if null_success:
                    lr_stat = 2 * (full_ml_llf - null_ml_llf)
                    df = p

                    if np.isnan(lr_stat) or lr_stat < 0 or not np.isfinite(lr_stat):
                        f_significant = _compute_wald_test(result, alpha)
                        test_method = "wald_fallback"
                    else:
                        lr_pvalue = 1 - chi2_cdf(lr_stat, df) if df > 0 else 1.0
                        f_significant = 1.0 if lr_pvalue < alpha else 0.0
                        test_method = "likelihood_ratio"
                else:
                    f_significant = _compute_wald_test(result, alpha)
                    test_method = "wald_fallback"
    except Exception:
        # If everything fails, use Wald test
        f_significant = _compute_wald_test(result, alpha)
        test_method = "wald_fallback"

    # Step 5: t-tests for individual fixed effects
    # Uncorrected t-tests
    uncorrected = np.zeros(n_targets)

    for idx_pos in range(n_targets):
        coef_idx = target_indices[idx_pos]
        if coef_idx < p:
            # Two-tailed test
            uncorrected[idx_pos] = 1.0 if fixed_pvalues[coef_idx] < alpha else 0.0

    # Step 6: Multiple comparison corrections
    corrected = np.zeros(n_targets)

    if correction_method == 0:
        # No correction
        corrected = uncorrected.copy()

    elif correction_method == 1:
        # Bonferroni correction
        bonf_alpha = alpha / n_targets if n_targets > 0 else alpha
        for idx_pos in range(n_targets):
            coef_idx = target_indices[idx_pos]
            if coef_idx < p:
                corrected[idx_pos] = 1.0 if fixed_pvalues[coef_idx] < bonf_alpha else 0.0

    elif correction_method == 2:
        # FDR (Benjamini-Hochberg): step-up procedure
        # Extract p-values for target effects
        target_pvalues = np.zeros(n_targets)
        for idx_pos in range(n_targets):
            coef_idx = target_indices[idx_pos]
            if coef_idx < p:
                target_pvalues[idx_pos] = fixed_pvalues[coef_idx]
            else:
                target_pvalues[idx_pos] = 1.0

        # Sort p-values
        sorted_indices = np.argsort(target_pvalues)
        sorted_pvalues = target_pvalues[sorted_indices]

        # Find largest k where p(k) <= (k+1)/m * alpha
        reject = np.zeros(n_targets, dtype=bool)
        for k in range(n_targets - 1, -1, -1):
            if sorted_pvalues[k] <= (k + 1) / n_targets * alpha:
                reject[sorted_indices[: k + 1]] = True
                break

        corrected = reject.astype(float)

    elif correction_method == 3:
        # Holm correction: step-down procedure
        # Extract p-values for target effects
        target_pvalues = np.zeros(n_targets)
        for idx_pos in range(n_targets):
            coef_idx = target_indices[idx_pos]
            if coef_idx < p:
                target_pvalues[idx_pos] = fixed_pvalues[coef_idx]
            else:
                target_pvalues[idx_pos] = 1.0

        # Sort p-values
        sorted_indices = np.argsort(target_pvalues)
        sorted_pvalues = target_pvalues[sorted_indices]

        # Find smallest k where p(k) > alpha / (m - k)
        reject = np.zeros(n_targets, dtype=bool)
        for k in range(n_targets):
            if sorted_pvalues[k] <= alpha / (n_targets - k):
                reject[sorted_indices[k]] = True
            else:
                break

        corrected = reject.astype(float)

    # Step 7: Compute diagnostics if requested
    diagnostics = None
    if verbose:
        # Estimate ICC from fitted model
        # ICC = tau^2 / (tau^2 + sigma^2)
        try:
            # Get random effects variance (tau^2)
            random_effects_var = result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else 0.0
            # Get residual variance (sigma^2)
            residual_var = result.scale
            # Compute ICC
            icc_estimated = random_effects_var / (random_effects_var + residual_var) if (random_effects_var + residual_var) > 0 else 0.0
        except Exception:
            random_effects_var = np.nan
            residual_var = np.nan
            icc_estimated = np.nan

        # Store LR test statistics (use model.loglike() values, not .llf)
        try:
            if not np.isnan(null_ml_llf) and not np.isnan(full_ml_llf):
                lr_statistic = 2 * (full_ml_llf - null_ml_llf)
                null_log_likelihood = null_ml_llf
                ml_fit_success = True
                lr_stat_is_nan = np.isnan(lr_statistic)
            else:
                lr_statistic = np.nan
                null_log_likelihood = null_ml_llf
                ml_fit_success = False
                lr_stat_is_nan = True
        except Exception:
            lr_statistic = np.nan
            null_log_likelihood = np.nan
            ml_fit_success = False
            lr_stat_is_nan = True

        diagnostics = {
            "converged": True,
            "convergence_level": convergence_level,
            "iterations": attempts[convergence_level][0] if convergence_level >= 0 else -1,
            "test_method": test_method,
            "log_likelihood": result.llf,
            "null_log_likelihood": null_log_likelihood,
            "lr_statistic": lr_statistic,
            "lr_stat_is_nan": lr_stat_is_nan,
            "ml_fit_success": ml_fit_success,
            "fixed_effects": fixed_params.copy(),
            "fixed_effects_se": result.bse[1:].copy(),  # Skip intercept
            "fixed_effects_pvalues": fixed_pvalues.copy(),
            "random_effects_variance": random_effects_var,
            "residual_variance": residual_var,
            "icc_estimated": icc_estimated,
        }

    # Step 8: Return results
    # Append a Wald-fallback flag (1.0 if Wald was used, 0.0 if LR) so the
    # simulation runner can track fallback rates without verbose mode.
    wald_flag = 1.0 if test_method == "wald_fallback" else 0.0
    results_array = np.concatenate([[f_significant], uncorrected, corrected, [wald_flag]])

    if verbose:
        return {"results": results_array, "diagnostics": diagnostics}
    else:
        return results_array


def _lme_analysis_custom(
    X_expanded: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    cluster_ids: np.ndarray,
    correction_method: int,
    alpha: float,
    chi2_crit: Optional[float] = None,
    z_crit: Optional[float] = None,
    correction_z_crits: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Optional[Union[np.ndarray, Dict]]:
    """LME analysis for random-intercept models via C++ backend.

    Uses precomputed critical values (chi2_crit, z_crit) to avoid
    per-simulation scipy calls. Falls back to computing them if not provided.
    """
    n, p = X_expanded.shape
    n_targets = len(target_indices)
    K = int(cluster_ids.max()) + 1

    if z_crit is None or chi2_crit is None or correction_z_crits is None:
        from .lme_solver import compute_lme_critical_values

        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(alpha, p, n_targets, correction_method)

    from mcpower.backends import mcpower_native as _native  # type: ignore[attr-defined]

    result = _native.lme_analysis(
        np.ascontiguousarray(X_expanded, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(cluster_ids, dtype=np.int32),
        K,
        np.ascontiguousarray(target_indices, dtype=np.int32),
        float(chi2_crit),
        float(z_crit),
        np.ascontiguousarray(correction_z_crits, dtype=np.float64),
        int(correction_method),
        float(-1.0),
    )
    if len(result) > 0:
        if verbose:
            return {"results": result, "diagnostics": {"solver": "native_q1"}}
        return result  # type: ignore[no-any-return]

    if verbose:
        return {"results": None, "failure_reason": "C++ solver returned empty result"}
    return None


def _lme_analysis_custom_general(
    X_expanded: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    cluster_ids: np.ndarray,
    re_result: Any,
    correction_method: int,
    alpha: float,
    chi2_crit: Optional[float] = None,
    z_crit: Optional[float] = None,
    correction_z_crits: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Optional[Union[np.ndarray, Dict]]:
    """LME analysis for random slopes (q > 1) via C++ backend."""
    from .lme_solver import compute_lme_critical_values

    n, p = X_expanded.shape
    n_targets = len(target_indices)

    first_gv = next(iter(re_result.Z_matrices))
    Z = re_result.Z_matrices[first_gv]
    q = Z.shape[1]
    K = int(cluster_ids.max()) + 1

    if z_crit is None or chi2_crit is None or correction_z_crits is None:
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(alpha, p, n_targets, correction_method)

    from mcpower.backends import mcpower_native as _native  # type: ignore[attr-defined]

    warm_theta_arr = np.empty(0, dtype=np.float64)
    result = _native.lme_analysis_general(
        np.ascontiguousarray(X_expanded, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(Z, dtype=np.float64),
        np.ascontiguousarray(cluster_ids, dtype=np.int32),
        K,
        q,
        np.ascontiguousarray(target_indices, dtype=np.int32),
        float(chi2_crit),
        float(z_crit),
        np.ascontiguousarray(correction_z_crits, dtype=np.float64),
        int(correction_method),
        warm_theta_arr,
    )
    if len(result) > 0:
        if verbose:
            return {"results": result, "diagnostics": {"solver": "native_general", "q": q}}
        return result  # type: ignore[no-any-return]

    if verbose:
        return {"results": None, "failure_reason": "C++ general solver returned empty result"}
    return None


def _lme_analysis_custom_nested(
    X_expanded: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    re_result: Any,
    correction_method: int,
    alpha: float,
    chi2_crit: Optional[float] = None,
    z_crit: Optional[float] = None,
    correction_z_crits: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Optional[Union[np.ndarray, Dict]]:
    """LME analysis for nested random intercepts via C++ backend."""
    from .lme_solver import compute_lme_critical_values

    n, p = X_expanded.shape
    n_targets = len(target_indices)

    parent_ids = re_result.cluster_ids_dict[re_result.parent_var]
    child_ids = re_result.cluster_ids_dict[re_result.child_var]
    K_parent = re_result.K_parent
    K_child = re_result.K_child
    child_to_parent = re_result.child_to_parent

    if z_crit is None or chi2_crit is None or correction_z_crits is None:
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(alpha, p, n_targets, correction_method)

    from mcpower.backends import mcpower_native as _native  # type: ignore[attr-defined]

    warm_theta_arr = np.empty(0, dtype=np.float64)
    result = _native.lme_analysis_nested(
        np.ascontiguousarray(X_expanded, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(parent_ids, dtype=np.int32),
        np.ascontiguousarray(child_ids, dtype=np.int32),
        K_parent,
        K_child,
        np.ascontiguousarray(child_to_parent, dtype=np.int32),
        np.ascontiguousarray(target_indices, dtype=np.int32),
        float(chi2_crit),
        float(z_crit),
        np.ascontiguousarray(correction_z_crits, dtype=np.float64),
        int(correction_method),
        warm_theta_arr,
    )
    if len(result) > 0:
        if verbose:
            return {"results": result, "diagnostics": {"solver": "native_nested", "K_parent": K_parent, "K_child": K_child}}
        return result  # type: ignore[no-any-return]

    if verbose:
        return {"results": None, "failure_reason": "C++ nested solver returned empty result"}
    return None


def reset_warm_start_cache():
    """
    Reset the warm start parameter cache.

    Useful for testing or when switching to a very different model structure.
    Only the statsmodels fallback path uses warm-start; the C++ backend
    handles convergence internally.
    """
    _lme_thread_local.warm_start_params = None
