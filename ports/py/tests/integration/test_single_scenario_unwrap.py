"""Single-scenario unwrap happens in Python, not Rust."""
from mcpower import MCPower


def test_engine_returns_envelope_for_single_scenario():
    """Python unwraps single-scenario envelopes before returning to the caller.

    The user-visible shape is identical to the pre-unwrap contract — this test
    locks that the unwrap is transparent: scenarios=False yields the inner dict.
    """
    m = MCPower("y = x1")
    m.set_effects("x1=0.3")
    m.set_simulations(64)
    result = m.find_power(sample_size=80, scenarios=False)
    # When scenarios=False, the user gets the inner dict (NOT the envelope).
    assert "n_sims" in result
    assert "scenarios" not in result
