import pytest

from mcpower.output.tables import diagnostic_warnings, fmt_pct
from mcpower.output.report import Report
from mcpower.output.results import _check_failure_threshold
from mcpower.config import get_report_config
from mcpower import MCPower


def test_convergence_warning_trips_below_threshold():
    inner = {
        "convergence_rate": [0.912],
        "boundary_hit_rate_tau_zero": [0.0],
        "boundary_hit_rate_high_tau": [0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    warns = diagnostic_warnings(inner)
    assert any("convergence" in w and "91.2%" in w for w in warns)


def test_no_warning_when_clean():
    inner = {
        "convergence_rate": [0.999],
        "boundary_hit_rate_tau_zero": [0.0],
        "boundary_hit_rate_high_tau": [0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    assert diagnostic_warnings(inner) == []


# ---------------------------------------------------------------------------
# Task 3 additions: convergence_rate ABSENT and LIST paths
# ---------------------------------------------------------------------------


def test_convergence_rate_absent_defaults_to_one_no_warning():
    """When convergence_rate is absent, inner.get("convergence_rate", 1.0)
    defaults to 1.0 (a scalar), the isinstance branch is NOT taken, and no
    convergence warning is emitted."""
    inner = {
        # convergence_rate deliberately omitted
        "boundary_hit_rate_tau_zero": [0.0],
        "boundary_hit_rate_high_tau": [0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    warns = diagnostic_warnings(inner)
    convergence_warns = [w for w in warns if "convergence" in w]
    assert convergence_warns == [], (
        f"Expected no convergence warning when key absent; got {convergence_warns}"
    )


def test_convergence_rate_list_uses_worst_element():
    """convergence_rate=[0.6, 0.9] → min=0.6 → below threshold → warning emitted;
    the warning text must contain the exact percentage string that fmt_pct(0.6, 1)
    produces."""
    inner = {
        "convergence_rate": [0.6, 0.9],
        "boundary_hit_rate_tau_zero": [0.0, 0.0],
        "boundary_hit_rate_high_tau": [0.0, 0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    warns = diagnostic_warnings(inner)
    expected_pct = fmt_pct(0.6, 1)  # "60.0%"
    convergence_warns = [w for w in warns if "convergence" in w]
    assert len(convergence_warns) == 1, (
        f"Expected exactly one convergence warning; got {warns}"
    )
    assert expected_pct in convergence_warns[0], (
        f"Expected '{expected_pct}' in '{convergence_warns[0]}'"
    )


def test_convergence_rate_list_best_below_threshold_no_spurious_warning():
    """[0.99, 0.99] → worst=0.99, well above threshold → no convergence warning."""
    inner = {
        "convergence_rate": [0.99, 0.99],
        "boundary_hit_rate_tau_zero": [0.0, 0.0],
        "boundary_hit_rate_high_tau": [0.0, 0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    warns = diagnostic_warnings(inner)
    convergence_warns = [w for w in warns if "convergence" in w]
    assert convergence_warns == [], (
        f"Expected no convergence warning for high rates; got {convergence_warns}"
    )


# ---------------------------------------------------------------------------
# Task 3: _check_result_failure_threshold None-guard (model.py ~1748)
# ---------------------------------------------------------------------------


def test_check_result_failure_threshold_none_cr_returns_without_raising():
    """When convergence_rate is absent (None), _check_result_failure_threshold
    returns silently — the OLS / non-LME no-op guard (model.py ~1748)."""
    model = MCPower("y ~ x1 + x2")
    model.set_effects("x1=0.5, x2=0.3")
    # OLS result: no convergence_rate key → cr is None → early return
    mock_result = {
        "power": [0.8],
        # deliberately no convergence_rate
    }
    model._check_result_failure_threshold(mock_result)  # must not raise


def test_check_result_failure_threshold_list_cr_below_threshold_raises():
    """When convergence_rate is a list with a bad value, _check_result_failure_threshold
    raises RuntimeError — exercises the non-None path on a non-LME model object."""
    model = MCPower("y ~ x1 + (1|c)", family="lme")
    model.set_effects("x1=0.5").set_cluster("c", ICC=0.2, n_clusters=10)
    model.set_max_failed_simulations(0.1)  # allow 10% failures
    mock_result = {
        "convergence_rate": [0.5, 0.5],  # 50% failure >> 10% threshold
        "boundary_hit_rate_tau_zero": [0.4, 0.4],
        "boundary_hit_rate_high_tau": [0.1, 0.1],
    }
    with pytest.raises(RuntimeError, match="failure rate"):
        model._check_result_failure_threshold(mock_result)


# ---------------------------------------------------------------------------
# Task 11: factor exclusion / separation-drop diagnostic warnings
# ---------------------------------------------------------------------------

_CLEAN_BASE = {
    "convergence_rate": [1.0],
    "boundary_hit_rate_tau_zero": [0.0],
    "boundary_hit_rate_high_tau": [0.0],
    "estimator_extras": {"estimator": "ols"},
}


def test_factor_exclusion_warning_trips():
    """factor_exclusion_counts=[200,0], factor_separation_counts=[0,3], n_sims=200:
    dose excluded 100.0% → trips; site separation-dropped 1.5% → trips (> 0.0 threshold)."""
    inner = {
        **_CLEAN_BASE,
        "n_sims": 200,
        "factor_exclusion_counts": [200, 0],
        "factor_separation_counts": [0, 3],
    }
    warns = diagnostic_warnings(inner, factor_names=["dose", "site"])
    assert any("dose excluded 100.0%" in w for w in warns), (
        f"Expected 'dose excluded 100.0%' in {warns}"
    )
    assert any("site separation-dropped 1.5%" in w for w in warns), (
        f"Expected 'site separation-dropped 1.5%' in {warns}"
    )


def test_factor_exclusion_silent_when_healthy():
    """All-zero counts produce no exclusion or separation-drop warnings."""
    inner = {
        **_CLEAN_BASE,
        "n_sims": 200,
        "factor_exclusion_counts": [0, 0],
        "factor_separation_counts": [0, 0],
    }
    warns = diagnostic_warnings(inner)
    assert not [w for w in warns if "excluded" in w or "separation-dropped" in w], (
        f"Expected no exclusion/separation warnings; got {warns}"
    )


def test_factor_exclusion_fallback_name():
    """Without factor_names, warning uses 'factor N' positional fallback."""
    inner = {
        **_CLEAN_BASE,
        "n_sims": 100,
        "factor_exclusion_counts": [100],
        "factor_separation_counts": [0],
    }
    warns = diagnostic_warnings(inner)
    assert any("factor 1 excluded 100.0%" in w for w in warns), (
        f"Expected positional fallback name 'factor 1' in {warns}"
    )


def test_factor_exclusion_per_grid_point_worst_case():
    """find_sample_size shape: counts is list-of-lists (one per grid point).
    Rate reduces to the worst grid point, not the average."""
    inner = {
        **_CLEAN_BASE,
        "boundary_hit_rate_tau_zero": [0.0, 0.0],
        "boundary_hit_rate_high_tau": [0.0, 0.0],
        "n_sims": 100,
        # Grid points: N=20 → 80 excluded, N=40 → 10 excluded.
        # Worst rate = 80/100 = 80%.
        "factor_exclusion_counts": [[80], [10]],
        "factor_separation_counts": [[0], [0]],
    }
    warns = diagnostic_warnings(inner, factor_names=["grp"])
    assert any("grp excluded 80.0%" in w for w in warns), (
        f"Expected 'grp excluded 80.0%' (worst grid point) in {warns}"
    )


# ---------------------------------------------------------------------------
# Boundary-hit gate → high-τ̂ only (decision 2.2). Benign τ̂=0 must NOT trip.
# ---------------------------------------------------------------------------


def test_high_tau_boundary_trips():
    """boundary_hit_rate_high_tau over lme_boundary_hit_max → 'high-τ̂ boundary'."""
    inner = {
        **_CLEAN_BASE,
        "boundary_hit_rate_tau_zero": [0.0],
        "boundary_hit_rate_high_tau": [0.05],  # 5% > 1% threshold
    }
    warns = diagnostic_warnings(inner)
    assert any("high-τ̂ boundary" in w and "5.0%" in w for w in warns), (
        f"Expected 'high-τ̂ boundary 5.0%' in {warns}"
    )


def test_benign_tau_zero_alone_does_not_trip():
    """A high benign τ̂=0 rate with zero high-τ̂ must NOT raise a boundary warning
    (decision 2.2 — τ̂=0 is common for small ICC and stays informational)."""
    inner = {
        **_CLEAN_BASE,
        "boundary_hit_rate_tau_zero": [0.40],  # 40% singular fits — benign
        "boundary_hit_rate_high_tau": [0.0],
    }
    warns = diagnostic_warnings(inner)
    assert not [w for w in warns if "boundary" in w], (
        f"Expected no boundary warning for benign τ̂=0; got {warns}"
    )


# ---------------------------------------------------------------------------
# GLM baseline drift → live gate against the requested baseline (§3.2).
# ---------------------------------------------------------------------------


def _glm_inner(realized):
    return {
        **_CLEAN_BASE,
        "estimator_extras": {"estimator": "glm", "baseline_prob_realized": realized},
    }


def test_glm_drift_trips_when_over_threshold():
    """|realized − requested| over glm_baseline_drift_max trips with the 3-dp drift."""
    warns = diagnostic_warnings(_glm_inner(0.50), baseline_prob_requested=0.30)
    assert any("GLM baseline drift 0.200" in w for w in warns), (
        f"Expected 'GLM baseline drift 0.200' in {warns}"
    )


def test_glm_drift_clean_within_threshold():
    """Realized within glm_baseline_drift_max of requested → no drift warning."""
    warns = diagnostic_warnings(_glm_inner(0.32), baseline_prob_requested=0.30)
    assert not [w for w in warns if "drift" in w], (
        f"Expected no drift warning within threshold; got {warns}"
    )


def test_glm_drift_silent_when_requested_none():
    """requested is None (the dead-branch guard): no drift warning even with a
    realized value far from any baseline."""
    warns = diagnostic_warnings(_glm_inner(0.90), baseline_prob_requested=None)
    assert not [w for w in warns if "drift" in w], (
        f"Expected no drift warning when requested is None; got {warns}"
    )


def test_glm_drift_silent_for_ols():
    """OLS carries no baseline_prob_realized → no drift warning even if a requested
    value is threaded through."""
    warns = diagnostic_warnings(
        {**_CLEAN_BASE, "estimator_extras": {"estimator": "ols"}},
        baseline_prob_requested=0.30,
    )
    assert not [w for w in warns if "drift" in w], (
        f"Expected no drift warning for OLS; got {warns}"
    )


# ---------------------------------------------------------------------------
# Multi-scenario diagnostics + Laplace-in-report (rendered via Report).
# ---------------------------------------------------------------------------


def _clean_scen(**over):
    base = {
        "convergence_rate": [1.0],
        "boundary_hit_rate_tau_zero": [0.0],
        "boundary_hit_rate_high_tau": [0.0],
        "estimator_extras": {"estimator": "ols"},
    }
    base.update(over)
    return base


def test_multi_scenario_degraded_scenario_surfaces_with_prefix():
    """A clean scenario 0 + a degraded scenario 1 → only scenario-1 warns, and the
    message carries the '{scenario}: ' prefix (multi-scenario attribution)."""
    result = {"scenarios": {
        "baseline": _clean_scen(),
        "stress": _clean_scen(convergence_rate=[0.50]),
    }}
    rep = Report(result, {"factors": {}}, kind="find_power")
    out = rep._diagnostics(get_report_config())
    assert "⚠ Diagnostics" in out
    assert "stress: convergence" in out, f"Expected 'stress: convergence' in:\n{out}"
    assert "baseline:" not in out, f"Clean scenario should not appear:\n{out}"


def test_single_scenario_has_no_prefix():
    """A single-scenario result must keep the bare (un-prefixed) message."""
    result = {"scenarios": {"default": _clean_scen(convergence_rate=[0.50])}}
    rep = Report(result, {"factors": {}}, kind="find_power")
    out = rep._diagnostics(get_report_config())
    assert "! convergence" in out
    assert "default:" not in out, f"Single scenario should not be prefixed:\n{out}"


def test_laplace_bias_surfaces_in_report():
    """τ̂² over glmm_tau_sq_warn with min_cluster_size below the recommendation →
    the persistent Laplace line appears in the rendered diagnostics block."""
    inner = _clean_scen(estimator_extras={
        "estimator": "glm", "baseline_prob_realized": 0.30, "tau_squared_hat_mean": 2.0,
    })
    result = {"scenarios": {"default": inner}}
    meta = {"factors": {}, "baseline_prob_requested": 0.30, "min_cluster_size": 4}
    rep = Report(result, meta, kind="find_power")
    out = rep._diagnostics(get_report_config())
    assert "Laplace-approximation bias likely" in out, (
        f"Expected the Laplace line in:\n{out}"
    )


def test_laplace_bias_silent_when_clusters_large():
    """min_cluster_size at/above the recommendation → no Laplace line even with
    large τ̂²."""
    from mcpower.config import get_config
    recommended = get_config()["limits"]["recommended_rows_per_cluster"]
    inner = _clean_scen(estimator_extras={
        "estimator": "glm", "baseline_prob_realized": 0.30, "tau_squared_hat_mean": 2.0,
    })
    result = {"scenarios": {"default": inner}}
    meta = {"factors": {}, "baseline_prob_requested": 0.30, "min_cluster_size": recommended}
    rep = Report(result, meta, kind="find_power")
    out = rep._diagnostics(get_report_config())
    assert "Laplace" not in out, f"Expected no Laplace line for large clusters:\n{out}"
