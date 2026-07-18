"""agq= per-call kwarg + poisson baseline_rate — surface tests.

Mirrors the wald_se= pattern in test_wald_se.py: agq is validated at
find_power entry (model._resolve_estimation), an ineligible design
warn-and-strips to Laplace (agq=1) rather than erroring, and
set_baseline_rate is the poisson counterpart of set_baseline_probability
(intercept = ln(lambda0) on the log link).
"""

from __future__ import annotations

import math
import warnings

import pytest

from mcpower import MCPower


def _glmm():
    """Clustered GLMM fixture, AGQ-eligible: single grouping factor, 1 RE
    (intercept only). Mirrors test_wald_se.py's _glmm()."""
    return (
        MCPower("y ~ x1 + (1|grp)", family="logit")
        .set_effects("x1=0.5")
        .set_baseline_probability(0.3)
        .set_cluster("grp", ICC=0.3, n_clusters=20)
    )


def test_agq_rejects_even_value():
    m = _glmm()
    with pytest.raises(ValueError):
        m.find_power(50, n_sims=50, agq=4, verbose=False)


def test_agq_rejects_out_of_range_value():
    m = _glmm()
    with pytest.raises(ValueError):
        m.find_power(50, n_sims=50, agq=27, verbose=False)


def test_agq_ineligible_design_warns_and_falls_back_to_laplace():
    # Unclustered logit: _pending_clusters is empty, so _agq_eligible() is
    # False regardless of agq — this must warn, not raise, and still return
    # a usable result computed at agq=1 (Laplace).
    m = MCPower("y ~ x1", family="logit").set_effects("x1=0.5").set_baseline_probability(0.3)
    with pytest.warns(UserWarning, match="is not available for this design"):
        result = m.find_power(50, n_sims=50, agq=5, verbose=False)
    assert result["power_uncorrected"][0][0] is not None


def test_agq_eligible_reaches_contract_and_changes_result():
    """agq=5 on an eligible clustered logit GLMM (single grouping factor, one
    intercept-only RE) must survive _agq_eligible(), land on the contract as
    nagq=5 (not silently dropped or stripped to 1 en route to
    build_linear_spec), and change the fitted random-effect variance estimate
    relative to Laplace (agq=1) — Laplace and 5-point AGQ are different
    quadrature schemes, not aliases, so a real fit at each must diverge.
    (power_uncorrected itself is not used here: at this effect size most sims
    are far from the significance boundary, so the two schemes can tie on the
    rounded significant/not-significant count even though the underlying fits
    differ — tau_squared_hat_mean is the per-sim numeric that actually moves.)"""
    import json

    import msgpack

    from mcpower import _engine

    m = _glmm()
    payload = m._to_linear_spec_dict(["optimistic"], nagq=5)
    outcome_kind_wire, link_wire, estimator_wire, intercept_arg, clusters_json = (
        m._encode_outcome_and_clusters()
    )
    names, contracts_bytes, _skeleton_json = _engine.build_contract_from_spec(
        json.dumps(payload),
        outcome_kind_wire,
        link_wire,
        estimator_wire,
        intercept_arg,
        clusters_json,
    )
    contracts = msgpack.unpackb(contracts_bytes, raw=False)
    contract = contracts[names.index("optimistic")]
    assert contract["nagq"] == 5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r_laplace = m.find_power(200, n_sims=400, seed=2137, agq=1, verbose=False)
        r_agq5 = m.find_power(200, n_sims=400, seed=2137, agq=5, verbose=False)
    tau_laplace = r_laplace["estimator_extras"]["tau_squared_hat_mean"]
    tau_agq5 = r_agq5["estimator_extras"]["tau_squared_hat_mean"]
    assert tau_laplace != tau_agq5


def test_baseline_rate_sets_log_intercept():
    m = MCPower("y ~ x1", family="poisson").set_effects("x1=0.3").set_baseline_rate(2.0)
    m._apply()
    assert m.intercept == pytest.approx(math.log(2.0))


def test_poisson_without_baseline_rate_raises_at_find_power():
    m = MCPower("y ~ x1", family="poisson").set_effects("x1=0.3")
    with pytest.raises(ValueError, match="baseline rate required"):
        m.find_power(50, n_sims=50, verbose=False)


def test_baseline_rate_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        MCPower("y ~ x1", family="poisson").set_baseline_rate(0)
    with pytest.raises(ValueError):
        MCPower("y ~ x1", family="poisson").set_baseline_rate(-1)
