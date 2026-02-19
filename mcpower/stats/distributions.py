"""Statistical distribution functions for MCPower.

Provides F, t, chi2, normal, and studentized range distribution
functions plus batch critical-value computation and table generation.

Backend priority:
  1. C++ native (Boost.Math + R Tukey port) via mcpower_native
  2. scipy (optional shim, for when C++ is not compiled)

Usage:
    from mcpower.stats.distributions import norm_ppf, compute_critical_values_ols
"""

import numpy as np

# ============================================================================
# Backend selection
# ============================================================================

_BACKEND = None

try:
    from mcpower.backends.mcpower_native import (  # type: ignore[import]
        chi2_cdf,
        chi2_ppf,
        compute_critical_values_lme,
        compute_critical_values_ols,
        compute_tukey_critical_value,
        f_ppf,
        generate_norm_cdf_table,
        generate_t3_ppf_table,
        norm_cdf,
        norm_ppf,
        norm_ppf_array,
        studentized_range_ppf,
        t_ppf,
    )

    _BACKEND = "native"

except ImportError:
    # -------------------------------------------------------------------
    # scipy shim -- temporary fallback for when C++ is not compiled.
    # Will be removed when Python fallback backends are fully dropped.
    # -------------------------------------------------------------------
    try:
        from scipy.stats import (  # isort: skip
            chi2 as _chi2_dist,
            f as _f_dist,
            norm as _norm_dist,
            studentized_range as _sr_dist,
            t as _t_dist,
        )

        def norm_ppf(p):  # noqa: F811
            """Standard normal quantile function (inverse CDF)."""
            return float(_norm_dist.ppf(p))

        def norm_cdf(x):  # noqa: F811
            """Standard normal CDF."""
            return float(_norm_dist.cdf(x))

        def t_ppf(p, df):  # noqa: F811
            """Student's t quantile function."""
            return float(_t_dist.ppf(p, df))

        def f_ppf(p, dfn, dfd):  # noqa: F811
            """Fisher F quantile function."""
            return float(_f_dist.ppf(p, dfn, dfd))

        def chi2_ppf(p, df):  # noqa: F811
            """Chi-squared quantile function."""
            return float(_chi2_dist.ppf(p, df))

        def chi2_cdf(x, df):  # noqa: F811
            """Chi-squared CDF."""
            return float(_chi2_dist.cdf(x, df))

        def studentized_range_ppf(p, k, df):  # noqa: F811
            """Studentized range quantile (Tukey). k=groups, df=denom df."""
            if df < 2 or k < 2 or k > 200 or p <= 0.0 or p >= 1.0:
                return float("inf")
            return float(_sr_dist.ppf(p, k, df))

        def compute_critical_values_ols(alpha, dfn, dfd, n_targets, correction_method):  # noqa: F811
            """Compute OLS critical values using scipy (fallback).

            Args:
                alpha: Significance level.
                dfn: Numerator degrees of freedom (number of predictors).
                dfd: Denominator degrees of freedom (n - p - 1).
                n_targets: Number of individual effects being tested.
                correction_method: 0=none, 1=Bonferroni, 2=FDR (BH), 3=Holm.

            Returns:
                Tuple of (f_crit, t_crit, correction_t_crits) where
                correction_t_crits is an ndarray of length n_targets.
            """
            if dfd <= 0:
                return np.inf, np.inf, np.full(max(n_targets, 1), np.inf)

            f_crit = _f_dist.ppf(1 - alpha, dfn, dfd) if dfn > 0 else np.inf
            t_crit = _t_dist.ppf(1 - alpha / 2, dfd)

            m = n_targets
            if m == 0:
                return f_crit, t_crit, np.empty(0)

            if correction_method == 0:  # None
                correction_t_crits = np.full(m, t_crit)
            elif correction_method == 1:  # Bonferroni
                bonf_crit = _t_dist.ppf(1 - alpha / (2 * m), dfd)
                correction_t_crits = np.full(m, bonf_crit)
            elif correction_method == 2:  # FDR (Benjamini-Hochberg)
                correction_t_crits = np.array(
                    [_t_dist.ppf(1 - (k + 1) / m * alpha / 2, dfd) if (k + 1) / m * alpha / 2 >= 1e-12 else np.inf for k in range(m)]
                )
            elif correction_method == 3:  # Holm
                correction_t_crits = np.array(
                    [_t_dist.ppf(1 - alpha / (2 * (m - k)), dfd) if alpha / (2 * (m - k)) >= 1e-12 else np.inf for k in range(m)]
                )
            else:
                correction_t_crits = np.full(m, t_crit)

            return f_crit, t_crit, correction_t_crits

        def compute_tukey_critical_value(alpha, n_levels, dfd):  # noqa: F811
            """Compute Tukey HSD critical value (q / sqrt(2))."""
            if dfd <= 0:
                return np.inf
            q_crit = _sr_dist.ppf(1 - alpha, n_levels, dfd)
            return q_crit / np.sqrt(2)

        def compute_critical_values_lme(alpha, n_fixed, n_targets, correction_method):  # noqa: F811
            """Compute LME critical values using scipy (fallback).

            Args:
                alpha: Significance level.
                n_fixed: Number of fixed effects (excluding intercept).
                n_targets: Number of individual effects being tested.
                correction_method: 0=none, 1=Bonferroni, 2=FDR (BH), 3=Holm.

            Returns:
                Tuple of (chi2_crit, z_crit, correction_z_crits) where
                correction_z_crits is an ndarray of length n_targets.
            """
            chi2_crit = _chi2_dist.ppf(1 - alpha, n_fixed) if n_fixed > 0 else np.inf
            z_crit = _norm_dist.ppf(1 - alpha / 2)

            m = n_targets
            if m == 0:
                return chi2_crit, z_crit, np.empty(0)

            if correction_method == 0:  # None
                correction_z_crits = np.full(m, z_crit)
            elif correction_method == 1:  # Bonferroni
                bonf = _norm_dist.ppf(1 - alpha / (2 * m))
                correction_z_crits = np.full(m, bonf)
            elif correction_method == 2:  # FDR (Benjamini-Hochberg)
                correction_z_crits = np.array(
                    [_norm_dist.ppf(1 - (k + 1) / m * alpha / 2) if (k + 1) / m * alpha / 2 >= 1e-12 else np.inf for k in range(m)]
                )
            elif correction_method == 3:  # Holm
                correction_z_crits = np.array(
                    [_norm_dist.ppf(1 - alpha / (2 * (m - k))) if alpha / (2 * (m - k)) >= 1e-12 else np.inf for k in range(m)]
                )
            else:
                correction_z_crits = np.full(m, z_crit)

            return chi2_crit, z_crit, correction_z_crits

        def generate_norm_cdf_table(x_min, x_max, resolution):  # noqa: F811
            """Generate normal CDF lookup table."""
            x = np.linspace(x_min, x_max, resolution)
            return _norm_dist.cdf(x).astype(np.float64)

        def generate_t3_ppf_table(perc_min, perc_max, resolution):  # noqa: F811
            """Generate t(3) PPF lookup table (divided by sqrt(3))."""
            p = np.linspace(perc_min, perc_max, resolution)
            return (_t_dist.ppf(p, 3) / np.sqrt(3)).astype(np.float64)

        def norm_ppf_array(percentiles):  # noqa: F811
            """Vectorized normal PPF for percentile array."""
            return _norm_dist.ppf(np.asarray(percentiles)).astype(np.float64)

        _BACKEND = "scipy"

    except ImportError as exc:
        raise ImportError(
            "No distribution backend available. "
            "Install from PyPI for prebuilt C++ wheels: pip install MCPower\n"
            "Or install scipy as fallback: pip install scipy"
        ) from exc


# ============================================================================
# Also re-export scipy optimizer shims for lme_solver.py
# These replace scipy.optimize.minimize and minimize_scalar
# ============================================================================


def minimize_lbfgsb(objective, x0, bounds, maxiter=200, ftol=1e-10, gtol=1e-6):
    """L-BFGS-B minimization -- C++ native or scipy fallback.

    Args:
        objective: Callable f(x) -> float
        x0: Initial guess (numpy array)
        bounds: List of (lower, upper) tuples
        maxiter: Max iterations
        ftol: Function tolerance
        gtol: Gradient tolerance

    Returns:
        Object with .x (optimal point), .fun (optimal value), .converged (bool)
    """
    if _BACKEND == "native":
        try:
            from mcpower.backends.mcpower_native import lbfgsb_minimize_fd  # type: ignore[import]

            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
            return lbfgsb_minimize_fd(objective, np.asarray(x0, dtype=np.float64), lb, ub, maxiter, ftol, gtol)
        except ImportError:
            import warnings

            warnings.warn(
                "Native L-BFGS-B optimizer not available despite native backend being loaded. Falling back to scipy.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as e:
            import warnings

            warnings.warn(
                f"Native L-BFGS-B optimizer failed ({type(e).__name__}: {e}), falling back to scipy.",
                RuntimeWarning,
                stacklevel=2,
            )

    # scipy fallback
    from scipy.optimize import minimize

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )

    class _Result:
        pass

    r = _Result()
    r.x = result.x
    r.fun = result.fun
    r.converged = result.success
    return r


def minimize_scalar_brent(objective, bounds, tol=1e-8, maxiter=150):
    """Brent 1D minimization -- C++ native or scipy fallback.

    Args:
        objective: Callable f(x) -> float
        bounds: (lower, upper) tuple
        tol: Absolute tolerance
        maxiter: Max iterations

    Returns:
        Object with .x (optimal point), .fun (optimal value), .converged (bool)
    """
    if _BACKEND == "native":
        try:
            from mcpower.backends.mcpower_native import brent_minimize_scalar  # type: ignore[import]

            return brent_minimize_scalar(objective, bounds[0], bounds[1], tol, maxiter)
        except ImportError:
            import warnings

            warnings.warn(
                "Native Brent optimizer not available despite native backend being loaded. Falling back to scipy.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as e:
            import warnings

            warnings.warn(
                f"Native Brent optimizer failed ({type(e).__name__}: {e}), falling back to scipy.",
                RuntimeWarning,
                stacklevel=2,
            )

    # scipy fallback
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": tol, "maxiter": maxiter},
    )

    class _Result:
        pass

    r = _Result()
    r.x = result.x
    r.fun = result.fun
    r.converged = bool(getattr(result, "success", True))
    return r
