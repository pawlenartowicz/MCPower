"""Spec-build tests — construction, set_baseline_probability, _validate_logit_runtime,
and to_simulation_spec without calling find_power end-to-end.
"""

from __future__ import annotations

import math
import warnings

import pytest

from mcpower import MCPower

# ---------------------------------------------------------------------------
# Construction gates
# ---------------------------------------------------------------------------


def test_logit_constructs_without_raising() -> None:
    """MCPower('y = x', family='logit') must not raise."""
    m = MCPower("y = x", family="logit")
    assert m.family == "logit"


def test_ols_still_constructs() -> None:
    """OLS construction is unaffected by logit changes."""
    m = MCPower("y = x", family="ols")
    assert m.family == "ols"


def test_lme_now_constructs() -> None:
    """family='lme' is accepted at construction time."""
    m = MCPower("y ~ x + (1|cluster)", family="lme")
    assert m.family == "lme"


def test_logit_init_intercept_default() -> None:
    """self.intercept starts at 0.0 and _pending_baseline_probability at None."""
    m = MCPower("y = x", family="logit")
    assert m.intercept == 0.0
    assert m._pending_baseline_probability is None


# ---------------------------------------------------------------------------
# set_baseline_probability + _apply_baseline_probability
# ---------------------------------------------------------------------------


def test_set_baseline_probability_valid() -> None:
    """set_baseline_probability(0.3) stores the pending value."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    assert m._pending_baseline_probability == pytest.approx(0.3)


def test_set_baseline_probability_returns_self() -> None:
    """set_baseline_probability is chainable."""
    m = MCPower("y = x", family="logit")
    result = m.set_baseline_probability(0.3)
    assert result is m


def test_set_baseline_probability_rejects_zero() -> None:
    m = MCPower("y = x", family="logit")
    with pytest.raises(ValueError, match="open interval"):
        m.set_baseline_probability(0.0)


def test_set_baseline_probability_rejects_one() -> None:
    m = MCPower("y = x", family="logit")
    with pytest.raises(ValueError, match="open interval"):
        m.set_baseline_probability(1.0)


def test_set_baseline_probability_rejects_negative() -> None:
    m = MCPower("y = x", family="logit")
    with pytest.raises(ValueError):
        m.set_baseline_probability(-0.1)


def test_apply_baseline_probability_sets_intercept() -> None:
    """After _apply, self.intercept == logit(p) exactly."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m._apply()
    expected = math.log(0.3 / 0.7)
    assert abs(m.intercept - expected) < 1e-12


def test_pending_baseline_stays_after_apply() -> None:
    """_pending_baseline_probability is NOT cleared after apply."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m._apply()
    assert m._pending_baseline_probability == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _validate_logit_runtime gates (called from find_power before engine)
# ---------------------------------------------------------------------------


def test_find_power_without_baseline_raises_for_logit() -> None:
    """find_power without set_baseline_probability raises ValueError for logit.

    The validator runs before the engine call.
    """
    m = MCPower("y = x", family="logit")
    m.set_effects("x=0.5")
    with pytest.raises(ValueError, match="baseline probability required"):
        m.find_power(sample_size=100)


def test_find_power_ols_without_baseline_does_not_raise() -> None:
    """OLS does not require set_baseline_probability (smoke: just checks no ValueError)."""
    m = MCPower("y = x", family="ols")
    m.set_effects("x=0.5")
    # OLS has no logit gate — no raise from _validate_logit_runtime.
    # We don't call the engine here; just verify the validator is silent.
    m._apply()
    m._validate_logit_runtime(["optimistic"])  # should not raise


def test_find_power_logit_with_baseline_and_scenarios_runs() -> None:
    """find_power with scenarios=True returns the 3-scenario envelope for logit.

    Gate-removal regression: scenario analysis is supported for binary
    outcomes (β-jitter applies as log-odds heterogeneity).
    """
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    env = m.find_power(
        sample_size=100, n_sims=50, scenarios=True, progress_callback=False
    )
    assert set(env["scenarios"].keys()) == {"optimistic", "realistic", "doomer"}


def test_validate_logit_runtime_silent_for_ols() -> None:
    """_validate_logit_runtime is a no-op for family='ols'."""
    m = MCPower("y = x", family="ols")
    m.set_effects("x=0.5")
    m._apply()
    # Should not raise for any scenario_filter value.
    m._validate_logit_runtime(["realistic", "doomer"])


def test_validate_logit_runtime_optimistic_ok() -> None:
    """scenario_filter=['optimistic'] is the one allowed case for logit."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m._apply()
    m._validate_logit_runtime(["optimistic"])  # must not raise


def test_validate_logit_runtime_none_ok() -> None:
    """scenario_filter=None is also allowed (no scenario block)."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    m._apply()
    m._validate_logit_runtime(None)  # must not raise


# ---------------------------------------------------------------------------
# to_simulation_spec emits family + intercept correctly
# ---------------------------------------------------------------------------


def test_to_simulation_spec_ols_unchanged() -> None:
    """OLS spec carries outcome.kind='continuous', estimator='ols', and intercept=0.0; coefficients[0]==0.0."""
    m = MCPower("y = x")
    m.set_effects("x=0.5")
    spec = m.to_simulation_spec()
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "ols"
    assert spec["outcome"]["intercept"] == 0.0
    assert spec["outcome"]["coefficients"][0] == 0.0


def test_to_simulation_spec_logit_writes_intercept() -> None:
    """Logit spec carries outcome.kind='binary', estimator='glm', outcome.intercept=logit(p), and coefficients[0]=0.0."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    spec = m.to_simulation_spec()
    assert spec["outcome"]["kind"] == "binary"
    assert spec["estimator"] == "glm"
    expected = math.log(0.3 / 0.7)
    assert abs(spec["outcome"]["intercept"] - expected) < 1e-12
    # intercept is now a separate field; coefficients[0] is the placeholder (0.0)
    assert spec["outcome"]["coefficients"][0] == 0.0


def test_to_simulation_spec_logit_x_effect_not_in_slot0() -> None:
    """The predictor beta is not overwritten by the intercept injection."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    m.set_effects("x=0.5")
    spec = m.to_simulation_spec()
    # coefficients has shape [placeholder(0.0), x_beta], so slot 1 must be 0.5.
    assert abs(spec["outcome"]["coefficients"][1] - 0.5) < 1e-12


def test_to_simulation_spec_ols_two_predictors_shape() -> None:
    """OLS coefficients layout is correct with two predictors."""
    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.4")
    spec = m.to_simulation_spec()
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "ols"
    assert spec["outcome"]["coefficients"][0] == 0.0          # intercept placeholder slot
    assert abs(spec["outcome"]["coefficients"][1] - 0.3) < 1e-12  # x1
    assert abs(spec["outcome"]["coefficients"][2] - 0.4) < 1e-12  # x2


# ---------------------------------------------------------------------------
# estimator= override
# ---------------------------------------------------------------------------


def test_headline_clustered_ols_spec() -> None:
    """MCPower('y ~ x + (1|group)', family='lme', estimator='ols') must emit
    outcome.kind='continuous', estimator='ols', AND a non-empty cluster block
    (the 'cost of ignoring clustering' study: DGP is clustered but fitted with OLS).
    """
    m = MCPower("y ~ x + (1|group)", family="lme", estimator="ols")
    m.set_effects("x=0.4")
    m.set_cluster("group", ICC=0.2, n_clusters=15)
    spec = m.to_simulation_spec()
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "ols"
    # Cluster block must be present — DGP is still clustered.
    cluster = spec["generation"]["cluster"]
    assert cluster is not None
    assert cluster["sizing"]["FixedClusters"]["n_clusters"] == 15
    assert abs(cluster["tau_squared"] - 0.2 / 0.8) < 1e-12


def test_estimator_override_without_changing_dgp() -> None:
    """estimator='ols' on a default OLS formula: still continuous+ols, no cluster."""
    m = MCPower("y = x", estimator="ols")
    m.set_effects("x=0.3")
    spec = m.to_simulation_spec()
    assert spec["outcome"]["kind"] == "continuous"
    assert spec["estimator"] == "ols"
    assert spec["generation"].get("cluster") is None


def test_invalid_estimator_raises() -> None:
    """An unrecognised estimator string must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="estimator must be"):
        MCPower("y = x", estimator="poisson")


def test_solve_as_alias_accepted() -> None:
    """solve_as= is an alias for estimator=; must not raise."""
    m = MCPower("y = x", solve_as="ols")
    assert m.estimator == "ols"


def test_estimator_kwarg_accepted_none() -> None:
    """estimator=None is the default; must not raise and must derive correctly."""
    m_ols = MCPower("y = x", estimator=None)
    assert m_ols.estimator == "ols"
    m_glm = MCPower("y = x", family="logit", estimator=None)
    assert m_glm.estimator == "glm"
    m_mle = MCPower("y ~ x + (1|g)", family="lme", estimator=None)
    assert m_mle.estimator == "mle"


# ---------------------------------------------------------------------------
# _warn_logit_effect_scale wiring
# ---------------------------------------------------------------------------


def test_warn_logit_large_effect_emits_warning() -> None:
    """Large logit effect (|β| > 3) triggers UserWarning via warnings.warn."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.set_effects("x=4.0")    # |β| > 3 → scale-mismatch warning
        m._apply()
    assert any(
        issubclass(warning.category, UserWarning)
        and "4.0" in str(warning.message)
        for warning in w
    ), f"Expected UserWarning mentioning 4.0; got: {[str(x.message) for x in w]}"


def test_warn_logit_small_effect_no_warning() -> None:
    """Small logit effect (|β| <= 3) does NOT trigger the scale warning."""
    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.set_effects("x=0.5")
        m._apply()
    scale_warnings = [
        warning for warning in w
        if issubclass(warning.category, UserWarning) and "|β|" in str(warning.message)
    ]
    assert len(scale_warnings) == 0


def test_warn_logit_not_emitted_for_ols() -> None:
    """OLS never triggers the logit scale warning, even for large effects."""
    m = MCPower("y = x", family="ols")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.set_effects("x=4.0")
        m._apply()
    logit_warnings = [
        warning for warning in w
        if issubclass(warning.category, UserWarning) and "|β|" in str(warning.message)
    ]
    assert len(logit_warnings) == 0


# ---------------------------------------------------------------------------
# set_heteroskedasticity family guard
# ---------------------------------------------------------------------------


def test_set_heteroskedasticity_driver_warns_for_logit() -> None:
    """set_heteroskedasticity_driver is OLS-only; a logit model must warn the user."""
    m = MCPower("y ~ x", family="logit")
    m.set_effects("x=0.5")
    m.set_baseline_probability(0.3)
    with pytest.warns(UserWarning, match="heteroskedasticity"):
        m.set_heteroskedasticity_driver(var=None)


def test_set_heteroskedasticity_driver_warns_for_lme() -> None:
    """set_heteroskedasticity_driver is OLS-only; an LME model must warn the user."""
    m = MCPower("y = x + (1|g)", family="lme")
    m.set_effects("x=0.3")
    m.set_cluster("g", n_clusters=20, ICC=0.2)
    with pytest.warns(UserWarning, match="heteroskedasticity"):
        m.set_heteroskedasticity_driver(var=None)


def test_set_heteroskedasticity_driver_no_warning_for_ols() -> None:
    """OLS must not trigger the family warning from set_heteroskedasticity_driver."""
    m = MCPower("y = x", family="ols")
    m.set_effects("x=0.3")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.set_heteroskedasticity_driver(var=None)
    het_warns = [x for x in w if "heteroskedasticity" in str(x.message).lower()
                 and issubclass(x.category, UserWarning)]
    assert len(het_warns) == 0


def test_cluster_level_vars_emitted_in_linear_spec():
    """cluster_level_vars stored in set_cluster appear in the LinearSpec payload."""
    from mcpower.spec.spec_builder import build_linear_spec

    m = MCPower("y ~ x1 + x2 + (1|school)", family="lme")
    m.set_effects("x1=0.3, x2=0.2")
    m.set_cluster("school", ICC=0.2, n_clusters=20, cluster_level_vars=["x2"])
    m._apply()

    # Collect cluster_level_vars across all pending clusters
    clv: list = []
    for cfg in m._pending_clusters.values():
        clv.extend(cfg.get("cluster_level_vars", []))

    payload = build_linear_spec(
        m._registry,
        ["optimistic"],
        heteroskedasticity=m._heteroskedasticity,
        residual_dist_name=m._residual_dist_name,
        residual_pinned=m._residual_pinned,
        alpha=m.alpha,
        correction=None,
        target_test=None,
        test_formula=None,
        pending_data=None,
        equation=m.equation,
        scenario_configs=m._scenario_configs,
        max_failed_simulations=m.max_failed_simulations,
        cluster_level_vars=clv,
    )
    assert "cluster_level_vars" in payload
    assert payload["cluster_level_vars"] == ["x2"]


def test_cluster_level_vars_absent_when_empty():
    """When no cluster_level_vars are set the key is absent from the payload."""
    from mcpower.spec.spec_builder import build_linear_spec

    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m._apply()

    payload = build_linear_spec(
        m._registry,
        ["optimistic"],
        heteroskedasticity=m._heteroskedasticity,
        residual_dist_name=m._residual_dist_name,
        residual_pinned=m._residual_pinned,
        alpha=m.alpha,
        correction=None,
        target_test=None,
        test_formula=None,
        pending_data=None,
        equation=m.equation,
        scenario_configs=m._scenario_configs,
        max_failed_simulations=m.max_failed_simulations,
    )
    assert "cluster_level_vars" not in payload


def test_build_linear_spec_wald_se_default_reads_config():
    """M10: ``build_linear_spec(wald_se=None)`` must resolve to the config
    ``estimation.wald_se`` default, not a hardcoded literal — matches R's
    ``.wald_se_for_rust(NULL)``, which reads the same config key. The two
    golden-payload tests above call ``build_linear_spec`` directly (no
    ``wald_se=``), so this pins the default they now get."""
    from mcpower.config import get_estimation_defaults
    from mcpower.spec.spec_builder import build_linear_spec

    m = MCPower("y ~ x1 + (1|school)", family="lme")
    m.set_effects("x1=0.3")
    m.set_cluster("school", ICC=0.2, n_clusters=20)
    m._apply()

    payload = build_linear_spec(
        m._registry,
        ["optimistic"],
        heteroskedasticity=m._heteroskedasticity,
        residual_dist_name=m._residual_dist_name,
        residual_pinned=m._residual_pinned,
        alpha=m.alpha,
        correction=None,
        target_test=None,
        test_formula=None,
        pending_data=None,
        equation=m.equation,
        scenario_configs=m._scenario_configs,
        max_failed_simulations=m.max_failed_simulations,
    )
    assert payload["wald_se"] == get_estimation_defaults()["wald_se"]
