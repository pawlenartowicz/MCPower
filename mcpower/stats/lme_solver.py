"""LME solver utilities for Monte Carlo power analysis.

Provides critical value precomputation for LME significance testing.
Core LME fitting is handled by the C++ backend (see mcpower/backends/native.py).
"""


# ---------------------------------------------------------------------------
# Critical value precomputation (mirrors ols.compute_critical_values)
# ---------------------------------------------------------------------------


def compute_lme_critical_values(alpha, n_fixed, n_targets, correction_method):
    """Pre-compute critical z and chi2 values for LME significance testing.

    Called once before the simulation loop. LME uses Wald z-tests
    (normal approximation) rather than t-tests.

    Args:
        alpha: Significance level.
        n_fixed: Number of fixed effects (excluding intercept).
        n_targets: Number of individual effects being tested.
        correction_method: 0=none, 1=Bonferroni, 2=BH, 3=Holm.

    Returns:
        Tuple of (chi2_crit, z_crit, correction_z_crits).
    """
    from mcpower.stats.distributions import compute_critical_values_lme

    return compute_critical_values_lme(alpha, n_fixed, n_targets, correction_method)
