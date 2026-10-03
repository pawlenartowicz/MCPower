"""df_method="between_within"|"asymptotic" per-call kwarg — surface test.

Mirrors the wald_se= pattern: validated at find_power entry, normalised in
spec_builder, forwarded into LinearSpec → project_contract. Only affects
mixed-model (Mle, or clustered Glm) fixed-effect tests; OLS and unclustered
GLM ignore it.
"""

from __future__ import annotations

import pytest

from mcpower import MCPower
from mcpower.config import get_estimation_defaults


def _few_cluster_lmm():
    """Few-cluster LMM fixture with a cluster-level predictor — where
    between-within df diverges most sharply from the asymptotic Wald z."""
    return (
        MCPower("outcome = treat + baseline_score + (1|clinic)", family="lme")
        .set_variable_type("treat=binary")
        .set_effects("treat=0.40, baseline_score=0.25")
        .set_cluster("clinic", ICC=0.20, n_clusters=6, cluster_level_vars=["treat"])
    )


def _ols():
    return MCPower("y = x1").set_effects("x1=0.30")


def test_df_method_rejects_garbage():
    m = _few_cluster_lmm()
    with pytest.raises(ValueError, match="df_method"):
        m.find_power(120, n_sims=100, df_method="wald", verbose=False)
    with pytest.raises(ValueError, match="df_method"):
        m.find_power(120, n_sims=100, df_method="nonsense", verbose=False)


def test_df_method_default_is_between_within():
    assert get_estimation_defaults()["df_method"] == "between_within"
    m = _few_cluster_lmm()
    r_default = m.find_power(120, target_test="treat", n_sims=300, seed=2137, verbose=False)
    r_bw = m.find_power(
        120,
        target_test="treat",
        df_method="between_within",
        n_sims=300,
        seed=2137,
        verbose=False,
    )
    p_default = r_default["power_uncorrected"][0][0]
    p_bw = r_bw["power_uncorrected"][0][0]
    # Default (no kwarg) must equal explicit "between_within" — same seed/data.
    assert p_default == p_bw


def test_df_method_reaches_engine():
    """The value threads through to the engine: asymptotic vs between_within
    at the same seed produce different power for the few-cluster LMM."""
    m = _few_cluster_lmm()
    r_asym = m.find_power(
        120,
        target_test="treat",
        df_method="asymptotic",
        n_sims=500,
        seed=2137,
        verbose=False,
    )
    r_bw = m.find_power(
        120,
        target_test="treat",
        df_method="between_within",
        n_sims=500,
        seed=2137,
        verbose=False,
    )
    p_asym = r_asym["power_uncorrected"][0][0]
    p_bw = r_bw["power_uncorrected"][0][0]
    assert p_asym != p_bw


def test_df_method_ols_unchanged():
    """OLS ignores df_method: asymptotic vs between_within give identical
    power at the same seed."""
    m = _ols()
    r_asym = m.find_power(200, df_method="asymptotic", n_sims=300, seed=2137, verbose=False)
    r_bw = m.find_power(200, df_method="between_within", n_sims=300, seed=2137, verbose=False)
    p_asym = r_asym["power_uncorrected"][0][0]
    p_bw = r_bw["power_uncorrected"][0][0]
    assert p_asym == p_bw
