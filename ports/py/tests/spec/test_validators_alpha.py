"""Alpha gate: the hard ``(0, 1)`` range is enforced by the engine (contract
``invariant_15``) at run; the Python setter only soft-warns above ``max_alpha``."""

import warnings

import pytest

import mcpower


def test_alpha_small_positive_accepted_without_warning():
    m = mcpower.MCPower("y = x1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        m.set_alpha(0.001)
    assert m.alpha == 0.001


def test_alpha_above_quarter_warns_but_is_accepted():
    # The 0.25 ceiling is now a soft warning, not a hard reject.
    m = mcpower.MCPower("y = x1")
    with pytest.warns(UserWarning, match="(?i)alpha"):
        m.set_alpha(0.3)
    assert m.alpha == 0.3


def test_alpha_just_below_one_accepted_at_run():
    # 0.999 is in (0, 1): warns at the setter, accepted by the engine.
    m = mcpower.MCPower("y = x1")
    m.set_effects("x1=0.5")
    m.set_simulations(50)
    with pytest.warns(UserWarning, match="(?i)alpha"):
        m.set_alpha(0.999)
    m.find_power(sample_size=60, verbose=False)  # must not raise


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0])
def test_alpha_out_of_open_unit_interval_rejected_at_run_not_setter(bad_alpha):
    # The hard (0, 1) range moved to the engine: the setter accepts these, the
    # run rejects them.
    m = mcpower.MCPower("y = x1")
    m.set_effects("x1=0.5")
    m.set_simulations(50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 1.0 soft-warns; not what we assert here
        m.set_alpha(bad_alpha)  # no raise at the setter
    with pytest.raises(Exception, match="(?i)alpha"):
        m.find_power(sample_size=60, verbose=False)
