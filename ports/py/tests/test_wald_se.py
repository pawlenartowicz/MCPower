"""wald_se="hessian"|"rx" per-call kwarg — surface test.

Mirrors the correction= pattern: validated at find_power entry, normalised in
spec_builder, forwarded into LinearSpec → project_contract. Only affects the
clustered-binary GLMM estimator (WaldSe::affects); the OLS/LMM paths ignore it.
"""

from __future__ import annotations

import warnings

import pytest

from mcpower import MCPower


def _glmm():
    """Clustered GLMM fixture — mirrors test_glmm_cluster.py setup."""
    return (
        MCPower("y ~ x1 + (1|grp)", family="logit")
        .set_effects("x1=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("grp", ICC=0.3, n_clusters=20)
    )


def test_wald_se_default_is_hessian_and_rx_less_conservative():
    m = _glmm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Default (no kwarg) must equal explicit "hessian".
        r_default = m.find_power(200, n_sims=400, seed=2137, verbose=False)
        r_h = m.find_power(200, n_sims=400, wald_se="hessian", seed=2137, verbose=False)
        r_rx = m.find_power(200, n_sims=400, wald_se="rx", seed=2137, verbose=False)
    p_default = r_default["power_uncorrected"][0][0]
    p_h = r_h["power_uncorrected"][0][0]
    p_rx = r_rx["power_uncorrected"][0][0]
    # Default is hessian: same seed/data ⇒ identical power.
    assert p_default == p_h
    # Hessian (default) is at least as conservative as rx (equal or lower power);
    # allow 0.05 tolerance for Monte Carlo noise at n_sims=400.
    assert p_h <= p_rx + 0.05


def test_wald_se_rejects_garbage():
    m = _glmm()
    with pytest.raises((ValueError, Exception)):
        m.find_power(200, n_sims=100, wald_se="asymp", verbose=False)
    with pytest.raises((ValueError, Exception)):
        m.find_power(200, n_sims=100, wald_se="nonsense", verbose=False)
