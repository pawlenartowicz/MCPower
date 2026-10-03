"""Python frontend TDDs — spec construction, progress callback wiring."""

from __future__ import annotations

import pytest

from mcpower import MCPower, progress
from mcpower.spec.scenario_config import get_default_scenario_config as _get_sc

DEFAULT_SCENARIO_CONFIG = _get_sc()


# ---------------------------------------------------------------------------
# MCPower frontend
# ---------------------------------------------------------------------------


def test_f1_set_parallel_raises_attribute_error():
    m = MCPower("y = x1")
    with pytest.raises(AttributeError, match="set_n_threads"):
        m.set_parallel(True)


def test_f1_set_cluster_without_random_effect_in_formula_raises():
    # set_cluster on a formula without (1|group) raises a validator error
    # ("grouping variable not found in formula random effects").
    m = MCPower("y = x1")
    with pytest.raises(ValueError, match="not found in formula"):
        m.set_cluster("school", ICC=0.2, n_clusters=10)


def test_f1_set_baseline_probability_accepted():
    # set_baseline_probability does not raise at call time.
    # For logit models it stores the pending probability; for OLS models
    # the value is stored but effectively unused (find_power validates at
    # runtime if the user accidentally calls it on an OLS model).
    m_logit = MCPower("y = x1", family="logit")
    m_logit.set_baseline_probability(0.2)  # must not raise

    # Invalid probability still raises ValueError.
    with pytest.raises(ValueError):
        m_logit.set_baseline_probability(1.5)  # out of (0, 1)


def test_f1_model_constructs_spec():
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    spec = m.to_simulation_spec(scenario_name="optimistic")
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "ols"
    # intercept placeholder + two effects.
    assert len(spec["outcome"]["coefficients"]) == 3
    assert spec["outcome"]["coefficients"][0] == 0.0  # intercept placeholder
    # Two continuous predictors: two generation columns, no factor columns.
    assert len(spec["generation"]["columns"]) == 2
    assert all(col["Synthetic"]["kind"] == "Normal" for col in spec["generation"]["columns"])
    # Correlation is identity (string enum) by default.
    assert spec["generation"]["correlations"] == "Identity"
    # Scenario fields match the engine layout.
    sc = spec["scenario"]
    assert sc["name"] == "optimistic"
    assert sc["heterogeneity"] == 0.0
    assert sc["correlation_noise_sd"] == 0.0
    # optimistic carries the same distribution lists as other scenarios; what
    # makes it optimistic is the zero probabilities, not the list contents.
    assert sc["distribution_change_prob"] == 0.0
    assert sc["residual_change_prob"] == 0.0
    assert isinstance(sc["new_distributions"], list)
    assert isinstance(sc["residual_dists"], list)
    assert sc["lme"] is None
    # Target terms skip the intercept: marginal terms 1 and 2.
    marginal_terms = [t["term"] for t in spec["test"]["targets"] if t["kind"] == "marginal"]
    assert marginal_terms == [1, 2]
    # Engine encoding fields present.
    assert spec["test"]["correction"] == "none"
    # heteroskedasticity_driver is a flat field on OutcomeSpec (None = LP-based).
    assert spec["outcome"]["heteroskedasticity_driver"] is None
    assert spec["outcome"]["residual"]["distribution"] == "normal"
    # No df field on contract ResidualSpec; df lives in the active scenario.


def test_f1_set_scenario_configs_merges():
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    m.set_scenario_configs(
        {
            "realistic": {"correlation_noise_sd": 0.20},
            "stress": {"correlation_noise_sd": 0.40},
        }
    )
    # Realistic preset: only the overridden key changes.
    assert m._scenario_configs["realistic"]["correlation_noise_sd"] == 0.20
    # Other realistic defaults survive.
    assert m._scenario_configs["realistic"]["heterogeneity"] == DEFAULT_SCENARIO_CONFIG["realistic"]["heterogeneity"]
    assert m._scenario_configs["realistic"]["distribution_change_prob"] == DEFAULT_SCENARIO_CONFIG["realistic"]["distribution_change_prob"]
    # Custom scenario inherits from optimistic, applies override.
    assert m._scenario_configs["stress"]["correlation_noise_sd"] == 0.40
    assert m._scenario_configs["stress"]["heterogeneity"] == DEFAULT_SCENARIO_CONFIG["optimistic"]["heterogeneity"]
    assert m._scenario_configs["stress"]["distribution_change_prob"] == DEFAULT_SCENARIO_CONFIG["optimistic"]["distribution_change_prob"]


def test_f1_set_scenario_configs_rejects_unknown_key():
    # A typo'd knob must raise, naming the key — silently no-oping would
    # produce unperturbed "scenarios".
    m = MCPower("y = x1 + x2")
    with pytest.raises(ValueError, match="heterogenity"):
        m.set_scenario_configs({"realistic": {"heterogenity": 0.2}})


def test_f1_set_scenario_configs_rejects_old_doc_key():
    # `heteroskedasticity` was the formerly documented name of
    # `heteroskedasticity_ratio`; anyone who learned the old key must get a
    # loud error, not a silent no-op.
    m = MCPower("y = x1 + x2")
    with pytest.raises(ValueError, match="heteroskedasticity"):
        m.set_scenario_configs({"realistic": {"heteroskedasticity": 2.0}})


def test_f1_set_scenario_configs_accepts_all_live_keys():
    lme_keys = {"icc_noise_sd", "random_effect_dist", "random_effect_df"}
    live_cfg = {k: v for k, v in DEFAULT_SCENARIO_CONFIG["doomer"].items() if k not in lme_keys}
    assert live_cfg, "live key set must be non-empty"
    m = MCPower("y = x1 + x2")
    m.set_scenario_configs({"custom": live_cfg})
    for k, v in live_cfg.items():
        assert m._scenario_configs["custom"][k] == v


def test_f1_unknown_scenario_in_to_spec_raises():
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="Unknown scenario"):
        m.to_simulation_spec(scenario_name="not_a_real_scenario")


# ---------------------------------------------------------------------------
# Scenario resolution (MCPower._resolve_scenarios_arg)
# ---------------------------------------------------------------------------


def test_f2_resolve_scenarios_false_is_optimistic_only():
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    assert m._resolve_scenarios_arg(False) == ["optimistic"]


def test_f2_resolve_scenarios_true_lists_all_configured():
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    names = m._resolve_scenarios_arg(True)
    assert set(names) == set(DEFAULT_SCENARIO_CONFIG.keys())
    assert names[0] == "optimistic"  # optimistic pinned first


def test_f2_resolve_scenarios_unknown_raises():
    m = MCPower("y = x1")
    m.set_effects("x1=0.5")
    with pytest.raises(ValueError, match="Unknown scenario"):
        m._resolve_scenarios_arg(["nope"])


# ---------------------------------------------------------------------------
# Progress callback wrapper
# ---------------------------------------------------------------------------


def test_f3_progress_callback_true_closes_bar_at_completion():
    cb = progress.resolve_progress_callback(True)
    assert cb(0, 100) is True
    assert cb._bar is not None
    # Reaching the total releases the bar without an explicit close().
    assert cb(100, 100) is True
    assert cb._bar is None


def test_f3_progress_callback_custom_wrapped():
    calls = []

    def user_cb(cur, tot):
        calls.append((cur, tot))
        return True

    wrapped = progress.resolve_progress_callback(user_cb)
    assert wrapped is not None
    assert wrapped(1, 10) is True
    assert calls == [(1, 10)]


def test_f3_progress_callback_user_false_cancels():
    def user_cb(cur, tot):
        return False

    wrapped = progress.resolve_progress_callback(user_cb)
    assert wrapped is not None
    assert wrapped(1, 10) is False


def test_f3_progress_callback_user_none_continues():
    def user_cb(cur, tot):
        return None  # treated as "keep going"

    wrapped = progress.resolve_progress_callback(user_cb)
    assert wrapped is not None
    assert wrapped(1, 10) is True


def test_f3_progress_callback_non_callable_raises():
    with pytest.raises(TypeError, match="progress_callback"):
        progress.resolve_progress_callback(42)


# ---------------------------------------------------------------------------
# Integration smoke (real engine)
# ---------------------------------------------------------------------------


def test_f5_smoke_ols_runs_end_to_end():
    """Real engine call through the default scenarios=False: the port unwraps
    the engine's envelope and returns the single scenario's flat result."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.5, x2=0.3")
    result = m.find_power(
        sample_size=100,
        n_sims=200,
        seed=42,
        progress_callback=False,
    )
    # Single-scenario flat result (not the envelope).
    assert "scenarios" not in result
    assert result["n_sims"] == 200
    assert result["n_targets"] == 2
    assert result["sample_sizes"] == [100]
    # One power row (one N) with one entry per target.
    assert [len(row) for row in result["power_uncorrected"]] == [2]
    assert [len(row) for row in result["power_corrected"]] == [2]
    # Well-conditioned OLS converges on every sim.
    assert result["convergence_rate"] == [1.0]


def test_f5_smoke_scenarios_envelope_runs_end_to_end():
    """Multi-scenario real engine call returns the envelope shape."""
    m = MCPower("y = x1")
    m.set_effects("x1=0.4")
    result = m.find_power(
        sample_size=80,
        n_sims=100,
        seed=7,
        scenarios=True,
        progress_callback=False,
    )
    assert set(result.keys()) == {"scenarios", "comparison"}
    assert "optimistic" in result["scenarios"]
