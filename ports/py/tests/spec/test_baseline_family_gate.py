"""set_baseline_rate / set_baseline_probability family gate (B4).

Before this gate, calling both setters on the same model silently let the
second call's intercept win (``_apply_baseline_probability`` applies whichever
pending value is non-``None``, probability first) with no error and no
warning. Each setter now rejects the family it does not apply to, mirroring
the ``set_cluster`` ICC/tau_squared gate.
"""

from __future__ import annotations

import pytest

from mcpower import MCPower


def test_set_baseline_rate_rejected_for_logit():
    m = MCPower("y ~ x1", family="logit")
    with pytest.raises(ValueError, match="set_baseline_rate is only for family='poisson'"):
        m.set_baseline_rate(2.0)


def test_set_baseline_rate_rejected_for_probit():
    m = MCPower("y ~ x1", family="probit")
    with pytest.raises(ValueError, match="set_baseline_rate is only for family='poisson'"):
        m.set_baseline_rate(2.0)


def test_set_baseline_probability_rejected_for_poisson():
    m = MCPower("y ~ x1", family="poisson")
    with pytest.raises(
        ValueError, match="set_baseline_probability is only for family='logit'/'probit'"
    ):
        m.set_baseline_probability(0.3)


def test_set_baseline_probability_accepted_for_logit_and_probit():
    """Correct pairings are unaffected by the gate."""
    MCPower("y ~ x1", family="logit").set_baseline_probability(0.3)
    MCPower("y ~ x1", family="probit").set_baseline_probability(0.3)


def test_set_baseline_rate_accepted_for_poisson():
    MCPower("y ~ x1", family="poisson").set_baseline_rate(2.0)


def test_set_baseline_probability_rejected_for_ols_and_lme():
    """Non-binary families other than the pairwise mismatch are also gated."""
    with pytest.raises(ValueError, match="set_baseline_probability is only for"):
        MCPower("y ~ x1").set_baseline_probability(0.3)
    with pytest.raises(ValueError, match="set_baseline_probability is only for"):
        MCPower("y ~ x1 + (1|g)", family="lme").set_baseline_probability(0.3)


def test_set_baseline_rate_rejected_for_ols_and_lme():
    with pytest.raises(ValueError, match="set_baseline_rate is only for"):
        MCPower("y ~ x1").set_baseline_rate(2.0)
    with pytest.raises(ValueError, match="set_baseline_rate is only for"):
        MCPower("y ~ x1 + (1|g)", family="lme").set_baseline_rate(2.0)
