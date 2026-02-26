"""
Power analysis helpers for tests.

Shared utilities for model creation, power extraction, and critical values.
"""

from tests.config import DEFAULT_ALPHA, N_SIMS, SEED


def make_null_model(equation="y = x1", n_sims=N_SIMS, alpha=DEFAULT_ALPHA, seed=SEED):
    """Create a model with all effects set to zero."""
    from mcpower import MCPower

    m = MCPower(equation)
    m.set_simulations(n_sims)
    m.set_seed(seed)
    m.set_alpha(alpha)
    effects = ", ".join(f"{e}=0" for e in m._registry.effect_names)
    m.set_effects(effects)
    return m


def get_power(result, test_name="overall"):
    """Extract power for a test from a result dict."""
    return result["results"]["individual_powers"][test_name]


def get_power_corrected(result, test_name="overall"):
    """Extract corrected power for a test from a result dict."""
    return result["results"]["individual_powers_corrected"][test_name]


def compute_crits(X, target_indices, alpha=DEFAULT_ALPHA, correction_method=0):
    """Helper to compute critical values for a given X matrix."""
    from mcpower.stats.ols import compute_critical_values

    n, p = X.shape
    dof = n - p - 1
    n_targets = len(target_indices)
    return compute_critical_values(alpha, p, dof, n_targets, correction_method)


