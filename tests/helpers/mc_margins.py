"""
Monte Carlo margin-of-error calculations.

Single source of truth for all MC tolerance computations.
"""

import numpy as np

from tests.config import ALLOWED_BIAS, MC_Z


def mc_margin(alpha, n_sims, z=MC_Z):
    """
    MC margin of error for a rejection rate (in %).

    Returns half-width of an approximate (1 - 2*Phi(-z)) CI
    for a binomial proportion.
    """
    return z * np.sqrt(alpha * (1 - alpha) / n_sims) * 100 + ALLOWED_BIAS


def mc_accuracy_margin(true_power_pct, n_sims, z=MC_Z):
    """
    MC margin for a known true power (in %).

    Used by accuracy tests that compare MC estimates to exact analytical power.
    """
    p = true_power_pct / 100
    return z * np.sqrt(p * (1 - p) / n_sims) * 100 + ALLOWED_BIAS


def mc_proportion_margin(p, n, z=MC_Z):
    """
    MC margin of error for a raw proportion (0-1 scale).

    Used by cross-backend comparison tests.
    """
    return z * np.sqrt(p * (1 - p) / n) + ALLOWED_BIAS
