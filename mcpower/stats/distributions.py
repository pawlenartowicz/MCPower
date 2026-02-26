"""Statistical distribution functions for MCPower.

Provides F, t, chi2, normal, and studentized range distribution
functions plus batch critical-value computation and table generation.

All functions are provided by the C++ native backend (Boost.Math + R Tukey port).

Usage:
    from mcpower.stats.distributions import norm_ppf, compute_critical_values_ols
"""

import numpy as np

# ============================================================================
# Backend — native C++ only
# ============================================================================

try:
    from mcpower.backends.mcpower_native import (  # type: ignore[import]  # noqa: F401
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

except ImportError as exc:
    raise ImportError("Native C++ backend not available. Install from PyPI for prebuilt wheels: pip install MCPower") from exc


# ============================================================================
# Optimizer wrappers for lme_solver.py
# ============================================================================


def minimize_lbfgsb(objective, x0, bounds, maxiter=200, ftol=1e-10, gtol=1e-6):
    """L-BFGS-B minimization via native C++ backend.

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
    from mcpower.backends.mcpower_native import lbfgsb_minimize_fd  # type: ignore[import]

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    return lbfgsb_minimize_fd(objective, np.asarray(x0, dtype=np.float64), lb, ub, maxiter, ftol, gtol)


def minimize_scalar_brent(objective, bounds, tol=1e-8, maxiter=150):
    """Brent 1D minimization via native C++ backend.

    Args:
        objective: Callable f(x) -> float
        bounds: (lower, upper) tuple
        tol: Absolute tolerance
        maxiter: Max iterations

    Returns:
        Object with .x (optimal point), .fun (optimal value), .converged (bool)
    """
    from mcpower.backends.mcpower_native import brent_minimize_scalar  # type: ignore[import]

    return brent_minimize_scalar(objective, bounds[0], bounds[1], tol, maxiter)
