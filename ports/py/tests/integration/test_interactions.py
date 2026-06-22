"""End-to-end interaction power tests — `a:b` and `a*b` expansion target the interaction term."""

from mcpower import MCPower


def test_continuous_interaction_runs_and_targets():
    m = (
        MCPower("y = x1 * x2")  # expands to x1 + x2 + x1:x2
        .set_effects({"x1": 0.3, "x2": 0.2, "x1:x2": 0.4})
        .set_simulations(400)
    )
    m.seed = 2137
    # power_uncorrected is [[...]] — one list per N, inner list per target.
    # target_test="x1:x2" restricts power reporting to the interaction term.
    result = m.find_power(sample_size=300, target_test="x1:x2", scenarios=False)
    assert result["n_sims"] == 400
    # power_uncorrected[0] = values for N=300; at least one target (x1:x2)
    power_row = result["power_uncorrected"][0]
    assert len(power_row) >= 1
    # x1:x2 effect 0.4 at N=300/400 sims (seed 2137) is fully powered (≈1.0); pin a
    # floor so a broken interaction target (wrong column, dropped effect) fails where
    # the old tautological `0<=p<=1` passed.
    assert 0.9 < power_row[0] <= 1.0, f"interaction power: {power_row[0]}"


def test_interaction_effect_name_is_listed():
    m = MCPower("y = x1 * x2").set_effects({"x1": 0.3, "x2": 0.2, "x1:x2": 0.4})
    # the colon name is a first-class effect name
    names = m._registry.effect_names  # introspection used by validation/reporting
    assert "x1:x2" in names
