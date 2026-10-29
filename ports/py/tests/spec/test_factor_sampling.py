"""Tests that sampled_proportions flows from set_factor_sampling through to the wire dict."""

import mcpower


def test_sampled_proportions_reaches_wire_predictor():
    m = mcpower.MCPower("y ~ g")
    m.set_variable_type("g=(factor,0.5,0.5)").set_factor_sampling("g", "sampled")
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert g["sampled_proportions"] is True


def test_sampled_proportions_false_flows_through():
    m = mcpower.MCPower("y ~ g")
    m.set_variable_type("g=(factor,0.5,0.5)").set_factor_sampling("g", "exact")
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert g["sampled_proportions"] is False


def test_sampled_proportions_omitted_when_inherit():
    # No override => key omitted => serde-default None (inherit scenario).
    m = mcpower.MCPower("y ~ g")
    m.set_variable_type("g=(factor,0.5,0.5)")
    payload = m._to_linear_spec_dict(["optimistic"])
    g = next(p for p in payload["predictors"] if p["name"] == "g")
    assert "sampled_proportions" not in g
