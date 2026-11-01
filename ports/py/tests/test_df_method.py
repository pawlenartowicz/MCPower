"""df_method="auto"|"between_within"|"satterthwaite"|"asymptotic" per-call
kwarg — surface test.

Mirrors the wald_se= pattern: validated by name at find_power entry,
normalised in spec_builder, forwarded into LinearSpec → project_contract,
where a named method the design does not support is refused. Only affects
mixed-model (Mle, or clustered Glm) fixed-effect tests; OLS and unclustered
GLM ignore it.
"""

from __future__ import annotations

import warnings

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


def _slope_lmm():
    """Few-cluster random-slope LMM: the slope variable's between-within df
    is the within-cluster df, its Satterthwaite df rest on the 8 clusters."""
    return (
        MCPower("y ~ x1 + (1 + x1|g)", family="lme")
        .set_effects("x1=0.30")
        .set_cluster("g", ICC=0.20, n_clusters=8, random_slopes=["x1"],
                     slope_variance=0.10)
    )


def _crossed_lmm():
    m = MCPower("y ~ x1 + (1|school) + (1|rater)", family="lme")
    m.set_effects("x1=0.4")
    m.set_cluster("school", ICC=0.2, n_clusters=10)
    m.set_cluster("rater", ICC=0.1, n_clusters=6)
    return m


def _glmm():
    return (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )


def _ols():
    return MCPower("y = x1").set_effects("x1=0.30")


def _power(m, df_method, n=120, n_sims=300, **kw):
    r = m.find_power(n, df_method=df_method, n_sims=n_sims, seed=2137, verbose=False, **kw)
    return r["power_uncorrected"][0][0]


def test_df_method_rejects_garbage():
    m = _few_cluster_lmm()
    for bad in ("wald", "nonsense", "kenward_roger"):
        with pytest.raises(ValueError, match="df_method"):
            m.find_power(120, n_sims=100, df_method=bad, verbose=False)


def test_df_method_default_is_auto():
    assert get_estimation_defaults()["df_method"] == "auto"
    m = _few_cluster_lmm()
    r_default = m.find_power(120, target_test="treat", n_sims=300, seed=2137, verbose=False)
    assert r_default["power_uncorrected"][0][0] == _power(m, "auto", target_test="treat")


def test_auto_is_between_within_without_a_slope():
    """Single grouping, no random slope: auto is the between-within test."""
    m = _few_cluster_lmm()
    assert _power(m, "auto", target_test="treat") == _power(
        m, "between_within", target_test="treat"
    )


def test_auto_is_satterthwaite_with_a_slope():
    """Single grouping with a random slope: auto is the Satterthwaite test,
    which at 8 clusters rejects less often than between-within."""
    m = _slope_lmm()
    p_auto = _power(m, "auto", n=160, n_sims=400)
    assert p_auto == _power(m, "satterthwaite", n=160, n_sims=400)
    assert p_auto < _power(m, "between_within", n=160, n_sims=400)


def test_df_method_reaches_engine():
    """The value threads through to the engine: asymptotic vs between_within
    at the same seed produce different power for the few-cluster LMM."""
    m = _few_cluster_lmm()
    p_asym = _power(m, "asymptotic", n_sims=500, target_test="treat")
    p_bw = _power(m, "between_within", n_sims=500, target_test="treat")
    assert p_asym != p_bw


def test_between_within_refused_on_crossed_groupings():
    m = _crossed_lmm()
    with pytest.raises(ValueError, match="'between_within' needs a single grouping.*'auto'"):
        m.find_power(120, df_method="between_within", n_sims=20, verbose=False)
    # auto runs the same design, at Satterthwaite df
    assert _power(m, "auto", n=120, n_sims=40) == _power(m, "satterthwaite", n=120, n_sims=40)


def test_satterthwaite_refused_on_glmm():
    m = _glmm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="'satterthwaite' needs a continuous outcome.*'auto'"):
            m.find_power(100, df_method="satterthwaite", n_sims=20, verbose=False)


def test_df_method_ols_unchanged():
    """OLS ignores df_method: every value gives identical power at the same
    seed."""
    m = _ols()
    powers = {
        d: _power(m, d, n=200)
        for d in ("auto", "asymptotic", "between_within", "satterthwaite")
    }
    assert len(set(powers.values())) == 1, powers


def test_empty_df_method_is_not_recognised():
    m = MCPower("y ~ x + (1|g)", family="lme")
    m.set_cluster("g", ICC=0.2, n_clusters=20)
    m.set_effects("x=0.3")
    with pytest.raises(ValueError, match="df_method='' is not recognised"):
        m.find_power(200, n_sims=20, seed=1, df_method="")
