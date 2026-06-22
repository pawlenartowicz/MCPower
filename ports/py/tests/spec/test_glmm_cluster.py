"""GLMM (binary+cluster) Python port — latent-scale ICC, Laplace warning, smoke."""

from __future__ import annotations

import math
import warnings

import pytest

from mcpower import MCPower

_PI_SQ_OVER_3 = math.pi ** 2 / 3.0


# ---------------------------------------------------------------------------
# Task 1.1 — latent-scale ICC→τ² for binary outcome
# ---------------------------------------------------------------------------


def _tau_from_icc_gaussian(icc: float) -> float:
    denom = 1.0 - icc
    return icc / denom if denom > 0 else 0.0


def _tau_from_icc_logistic(icc: float) -> float:
    return _tau_from_icc_gaussian(icc) * _PI_SQ_OVER_3


def _cluster_spec(model: MCPower) -> dict:
    """Extract the first ClusterSpec dict from _encode_outcome_and_clusters."""
    import json as _json
    _, _, _, clusters_json = model._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    assert specs, "expected non-empty clusters_json"
    return specs[0]


def test_logit_cluster_tau_squared_is_latent_scale():
    """Binary+cluster path uses τ² = ICC/(1−ICC) · π²/3 (latent-scale logistic)."""
    icc = 0.3
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=icc, n_clusters=20)
    )
    m._apply()
    # Expected low-obs warning (5 obs/cluster) is incidental here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m._validate_lme_runtime(100, ["optimistic"])

    spec = _cluster_spec(m)
    expected = _tau_from_icc_logistic(icc)
    assert abs(spec["tau_squared"] - expected) < 1e-12, (
        f"tau_squared={spec['tau_squared']!r}, expected latent-scale {expected!r}"
    )


def test_continuous_cluster_tau_squared_unchanged():
    """Gaussian (lme) path uses τ² = ICC/(1−ICC); π²/3 factor must NOT apply."""
    icc = 0.3
    m = (
        MCPower("y ~ x + (1|group)", family="lme")
        .set_effects("x=0.5")
        .set_cluster("group", ICC=icc, n_clusters=20)
    )
    m._apply()
    # Expected low-obs warning (5 obs/cluster) is incidental here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m._validate_lme_runtime(100, ["optimistic"])

    spec = _cluster_spec(m)
    expected_gaussian = _tau_from_icc_gaussian(icc)
    assert abs(spec["tau_squared"] - expected_gaussian) < 1e-12, (
        f"tau_squared={spec['tau_squared']!r}, expected Gaussian {expected_gaussian!r}"
    )
    # Confirm latent-scale was NOT applied.
    assert abs(spec["tau_squared"] - _tau_from_icc_logistic(icc)) > 1e-6


def test_logit_cluster_extra_groupings_tau_squared_latent_scale():
    """Extra groupings on the binary+cluster path also get π²/3 scaling."""
    icc_primary = 0.2
    icc_extra = 0.15
    m = (
        MCPower("y ~ x + (1|a) + (1|b)", family="logit")
        .set_effects("x=0.4")
        .set_baseline_probability(0.25)
        .set_cluster("a", ICC=icc_primary, n_clusters=15)
        .set_cluster("b", ICC=icc_extra, n_clusters=10)
    )
    m._apply()
    m._validate_lme_runtime(150, ["optimistic"])

    import json as _json
    _, _, _, clusters_json = m._encode_outcome_and_clusters()
    specs = _json.loads(clusters_json)
    primary_spec = specs[0]
    assert abs(primary_spec["tau_squared"] - _tau_from_icc_logistic(icc_primary)) < 1e-12

    extra = primary_spec.get("extra_groupings", [])
    assert len(extra) == 1
    assert abs(extra[0]["tau_squared"] - _tau_from_icc_logistic(icc_extra)) < 1e-12


def test_logit_cluster_icc_zero_tau_zero():
    """ICC=0 → τ²=0 regardless of latent scale."""
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=0.0, n_clusters=20)
    )
    m._apply()
    # Expected low-obs warning (5 obs/cluster) is incidental here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m._validate_lme_runtime(100, ["optimistic"])
    spec = _cluster_spec(m)
    assert spec["tau_squared"] == 0.0


# ---------------------------------------------------------------------------
# Task 1.2 — Laplace-bias warning helper
# ---------------------------------------------------------------------------


def test_laplace_bias_warning_fires_when_tau_large_and_cluster_small():
    """GLMM result with τ̂²>threshold AND small clusters emits a warning string."""
    from mcpower.model import _glmm_laplace_bias_warning
    from mcpower.config import get_config

    cfg = get_config()
    thr = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]
    min_size = cfg["limits"]["recommended_rows_per_cluster"]

    w = _glmm_laplace_bias_warning(
        {"estimator": "glm", "tau_squared_hat_mean": thr + 0.1},
        min_cluster_size=min_size - 1,
        cfg=cfg,
    )
    assert w is not None, "expected a warning string, got None"
    assert "τ̂²" in w or "tau" in w.lower()


def test_laplace_bias_warning_silent_below_threshold():
    """τ̂² at or below the threshold → no warning."""
    from mcpower.model import _glmm_laplace_bias_warning
    from mcpower.config import get_config

    cfg = get_config()
    thr = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]
    min_size = cfg["limits"]["recommended_rows_per_cluster"]

    w = _glmm_laplace_bias_warning(
        {"estimator": "glm", "tau_squared_hat_mean": thr},
        min_cluster_size=min_size - 1,
        cfg=cfg,
    )
    assert w is None


def test_laplace_bias_warning_silent_large_clusters():
    """τ̂² above threshold but large clusters → no warning."""
    from mcpower.model import _glmm_laplace_bias_warning
    from mcpower.config import get_config

    cfg = get_config()
    thr = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]
    min_size = cfg["limits"]["recommended_rows_per_cluster"]

    w = _glmm_laplace_bias_warning(
        {"estimator": "glm", "tau_squared_hat_mean": thr + 0.5},
        min_cluster_size=min_size,  # equal to min_size = not small
        cfg=cfg,
    )
    assert w is None


def test_laplace_bias_warning_silent_for_ols_extras():
    """Non-GLMM estimator_extras (no tau_squared_hat_mean) never fires."""
    from mcpower.model import _glmm_laplace_bias_warning
    from mcpower.config import get_config

    cfg = get_config()
    min_size = cfg["limits"]["recommended_rows_per_cluster"]

    w = _glmm_laplace_bias_warning(
        {"estimator": "ols"},  # no tau_squared_hat_mean
        min_cluster_size=min_size - 1,
        cfg=cfg,
    )
    assert w is None


def test_find_power_glmm_emits_laplace_warning(monkeypatch):
    """find_power wires the Laplace-bias warning: patched engine result with
    high τ̂² + small clusters triggers UserWarning at call site."""
    from mcpower import _engine as eng
    from mcpower.config import get_config

    cfg = get_config()
    thr = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]

    def fake_find_power(*args, **kwargs):
        return {
            "scenarios": {
                "optimistic": {
                    "power_uncorrected": [[0.7]], "power_corrected": [[0.7]],
                    "ci_uncorrected": [[[0.6, 0.8]]], "ci_corrected": [[[0.6, 0.8]]],
                    "target_indices": [0], "convergence_rate": [1.0],
                    "n_sims": 50, "sample_sizes": [120], "grid_warnings": [],
                    "estimator_extras": {
                        "estimator": "glm", "tau_squared_hat_mean": thr + 0.5,
                    },
                }
            }
        }

    monkeypatch.setattr(eng, "find_power", fake_find_power)

    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        # Reachable warning band: 120//20 = 6 obs/cluster ∈ [reliable=5, recommended=10),
        # so the validator allows it (with a Low-observations note) while the Laplace
        # warning stays reachable — below reliable the validator hard-rejects.
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )

    # Record-form capture: the incidental low-observations warning also fires here,
    # and pytest.warns(match=...) re-emits non-matching warnings into the run summary.
    with pytest.warns(UserWarning) as record:
        m.find_power(120, n_sims=50, verbose=False)
    assert any("Laplace-approximation bias" in str(w.message) for w in record)


def test_find_power_glmm_no_warning_when_tau_low(monkeypatch):
    """find_power does not warn when τ̂² is below threshold."""
    from mcpower import _engine as eng
    from mcpower.config import get_config

    cfg = get_config()
    thr = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]

    def fake_find_power(*args, **kwargs):
        return {
            "scenarios": {
                "optimistic": {
                    "power_uncorrected": [[0.7]], "power_corrected": [[0.7]],
                    "ci_uncorrected": [[[0.6, 0.8]]], "ci_corrected": [[[0.6, 0.8]]],
                    "target_indices": [0], "convergence_rate": [1.0],
                    "n_sims": 50, "sample_sizes": [120], "grid_warnings": [],
                    "estimator_extras": {
                        "estimator": "glm", "tau_squared_hat_mean": thr - 0.1,
                    },
                }
            }
        }

    monkeypatch.setattr(eng, "find_power", fake_find_power)

    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        # Same reachable band (120//20 = 6 obs/cluster ∈ [5,10)) as the emit test,
        # but τ̂² is below threshold so no Laplace warning is expected.
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.find_power(120, n_sims=50, verbose=False)

    laplace_warns = [w for w in caught if "Laplace" in str(w.message)]
    assert not laplace_warns, f"Unexpected Laplace warning: {laplace_warns}"


# ---------------------------------------------------------------------------
# Task 1.3 — scenario RE knobs on the GLMM path
# ---------------------------------------------------------------------------


def test_set_scenario_configs_accepts_re_knobs_on_glmm_path():
    """set_scenario_configs accepts icc_noise_sd on binary+cluster without error."""
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )
    m.set_scenario_configs({"optimistic": {"icc_noise_sd": 0.05}})
    assert m._scenario_configs["optimistic"]["icc_noise_sd"] == pytest.approx(0.05)


def test_glmm_scenario_re_knob_reaches_engine():
    """A logit+cluster model with a scenario icc_noise_sd override builds a valid
    contract and runs find_power without the LmeScenarioRequiresMle error
    (requires Phase 0 Task 0.1 — invariant_13 relaxation)."""
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )
    m.set_scenario_configs({"optimistic": {"icc_noise_sd": 0.05}})
    # Expected low-obs/Laplace warnings are incidental here (mirrors R's suppressWarnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = m.find_power(100, n_sims=50, seed=2137, verbose=False)
    assert "power_uncorrected" in result
    pw = result["power_uncorrected"][0][0]
    assert 0.0 <= pw <= 1.0
    # Floor guards against a broken GLMM RE-knob path returning all-zeros/NaN→0.
    # Observed: 0.42 at seed=2137 (x=0.5, N=100, logit+cluster, icc_noise_sd=0.05).
    assert pw > 0.15, f"glmm re-knob power {pw} too low — GLMM scenario path may be broken"


# ---------------------------------------------------------------------------
# End-to-end GLMM smoke + binary+cluster port audit
# ---------------------------------------------------------------------------


def test_glmm_end_to_end_find_power():
    """audit: family='logit' + (1|group) + set_cluster reaches the GLMM
    kernel; result carries estimator_extras.estimator=='glm' and tau_squared_hat_mean."""
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=0.3, n_clusters=20)
    )
    # Expected low-obs/Laplace warnings are incidental here (mirrors R's suppressWarnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = m.find_power(100, n_sims=50, verbose=False)

    assert "power_uncorrected" in result
    pw = result["power_uncorrected"][0][0]
    # x effect 0.5 at N=100/50 sims (default seed 2137) is detected with power ≈0.4;
    # pin a floor so a broken GLMM path (no effect, NaN→0) fails where `0<=pw<=1` passed.
    assert pw > 0.15, f"glmm power too low: {pw}"

    extras = result.get("estimator_extras", {})
    assert extras.get("estimator") == "glm", (
        f"Expected estimator_extras.estimator=='glm', got: {extras!r}"
    )
    assert "tau_squared_hat_mean" in extras, (
        f"tau_squared_hat_mean missing from estimator_extras: {extras!r}"
    )
    tau_hat = extras["tau_squared_hat_mean"]
    assert isinstance(tau_hat, float) and tau_hat >= 0.0


def test_glmm_to_simulation_spec_uses_latent_scale_tau():
    """to_simulation_spec for a logit+cluster model encodes τ² on the latent scale."""
    icc = 0.25
    m = (
        MCPower("y ~ x + (1|group)", family="logit")
        .set_effects("x=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("group", ICC=icc, n_clusters=20)
    )
    spec = m.to_simulation_spec()
    # Key paths confirmed against live msgpack shape (output verified in session).
    assert spec["outcome"]["kind"] == "binary"
    assert spec["estimator"] == "glm"
    cluster = spec["generation"]["cluster"]
    expected_latent = (icc / (1.0 - icc)) * _PI_SQ_OVER_3
    assert abs(cluster["tau_squared"] - expected_latent) < 1e-12
