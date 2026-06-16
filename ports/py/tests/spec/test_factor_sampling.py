"""Tests that sampled_proportions flows from set_variable_type through to the wire dict."""

import mcpower


def test_sampled_proportions_reaches_wire_predictor():
    m = mcpower.MCPower("y ~ g")
    m._registry.set_variable_type("g", "factor", n_levels=2, proportions=[0.5, 0.5],
                                   sampled_proportions=True)
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert g["sampled_proportions"] is True


def test_sampled_proportions_false_flows_through():
    m = mcpower.MCPower("y ~ g")
    m._registry.set_variable_type("g", "factor", n_levels=2, proportions=[0.5, 0.5],
                                   sampled_proportions=False)
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert g["sampled_proportions"] is False


def test_sampled_proportions_omitted_when_inherit():
    # Default (kwarg absent) => key omitted => serde-default None (inherit scenario).
    m = mcpower.MCPower("y ~ g")
    m._registry.set_variable_type("g", "factor", n_levels=2, proportions=[0.5, 0.5])
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert "sampled_proportions" not in g
