"""Python API parity tests — Logit and LME contracts built via MCPower, run through find_power.

Asserts the result envelope matches the Rust integration test shape: exact equality on
enums/counts/booleans, sensible numeric ranges. Does NOT assert bit-equal floats across
the FFI boundary because Python may derive base_seed differently from Rust.
"""

import math

import pytest

from mcpower import MCPower


def _logit_model():
    # Note: set_baseline_probability is required for logit family validation.
    # It sets self.intercept via log(p/(1-p)) during _apply(); a manual
    # intercept assignment here would be overwritten, so we use baseline only.
    m = (
        MCPower("y = x1 + x2 + x3 + b", family="logit")
        .set_variable_type("b=binary")
        .set_baseline_probability(0.4)
        .set_effects({"x1": 0.3, "x2": 0.2, "x3": 0.1, "b": 0.4})
        .set_simulations(800)
    )
    m.seed = 2137
    return m


def _lme_model():
    m = (
        MCPower("y = x1 + x2 + (1|group)", family="lme")
        .set_effects({"x1": 0.3, "x2": 0.2})
        .set_cluster("group", ICC=0.3 / 1.3, n_clusters=20)
        .set_simulations(800)
    )
    m.seed = 2137
    return m


def test_logit_parity():
    m = _logit_model()
    result = m.find_power(sample_size=200, scenarios=False)
    # Single-scenario call returns the inner (unwrapped) dict.
    assert result["n_sims"] == 800
    fx = result["estimator_extras"]
    assert fx["estimator"] == "glm"
    # The kernel does not yet surface a realized baseline probability;
    # the engine carries NaN as a placeholder (aggregation.rs:96).
    assert "baseline_prob_realized" in fx
    assert math.isnan(fx["baseline_prob_realized"])


def test_lme_parity():
    m = _lme_model()
    result = m.find_power(sample_size=200, scenarios=False)
    assert result["n_sims"] == 800
    fx = result["estimator_extras"]
    assert fx["estimator"] == "mle"
    # joint_*_rate fields are present, valid, and consistent: x1+x2 effects give a
    # high joint detection rate (≈0.99, seed 2137); correction can only lower it.
    # (Was: both bounds tautological `0<=x<=1`.)
    assert fx["joint_uncorrected_rate"] > 0.5, fx["joint_uncorrected_rate"]
    assert 0.0 <= fx["joint_corrected_rate"] <= fx["joint_uncorrected_rate"]
    # boundary_rate_per_component: one entry per variance component (random
    # intercept → len 1); each rate in [0, 1].
    brc = fx["boundary_rate_per_component"]
    assert isinstance(brc, list), f"expected list, got {type(brc)}"
    assert len(brc) >= 1, "expected at least one component for random-intercept LME"
    assert all(0.0 <= r <= 1.0 for r in brc), f"rates out of [0,1]: {brc}"


def test_ols_parity():
    # The OLS entry must surface estimator_extras.estimator == "ols" on a live
    # find_power round-trip — the only estimator tag with no dedicated integration
    # guard (logit/lme are covered above); a Rust-side tag rename would slip past.
    m = MCPower("y = x1 + x2").set_effects({"x1": 0.5, "x2": 0.3}).set_simulations(400)
    m.seed = 2137
    result = m.find_power(sample_size=100, scenarios=False)
    assert result["estimator_extras"]["estimator"] == "ols"
    # x1 effect 0.5 at N=100 is strongly detected — not a degenerate zero-power result.
    assert result["power_uncorrected"][0][0] > 0.5
