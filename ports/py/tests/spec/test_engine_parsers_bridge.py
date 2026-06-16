from mcpower import _engine


def test_parse_formula_returns_dict():
    out = _engine.parse_formula("y ~ x1 + x2")
    assert isinstance(out, dict)
    assert out["dependent"] == "y"
    assert out["predictors"] == ["x1", "x2"]
    assert out["terms"][0]["kind"] == "main"
    assert out["terms"][0]["name"] == "x1"
    assert out["random_effects"] == []


def test_parse_formula_random_intercept():
    out = _engine.parse_formula("y ~ x + (1|g)")
    assert out["random_effects"] == [
        {"kind": "intercept", "group": "g", "parent": None},
    ]


def test_implicit_intercept_slope_matches_explicit():
    """(x|g) parses identically to (1+x|g) through the Rust bridge."""
    imp = _engine.parse_formula("y ~ x + (x|g)")
    exp = _engine.parse_formula("y ~ x + (1+x|g)")
    assert imp["random_effects"] == exp["random_effects"]
    assert imp["random_effects"][0] == {"kind": "slope", "group": "g", "vars": ["x"]}


def test_parse_formula_raises_on_term_removal():
    import pytest
    with pytest.raises(ValueError, match=r"term removal with '-'"):
        _engine.parse_formula("y ~ x1 - x2")


def test_parse_assignments_returns_dict():
    out = _engine.parse_assignments(
        "x1=0.5, x2=0.3", "effect", {"predictors": ["x1", "x2"], "interaction_terms": []}
    )
    assert isinstance(out, dict)
    assert out["items"] == [
        {"key": {"name": "x1"}, "value": {"effect": 0.5}},
        {"key": {"name": "x2"}, "value": {"effect": 0.3}},
    ]
    assert out["errors"] == []


def test_parse_assignments_soft_error_unknown_name():
    out = _engine.parse_assignments(
        "xnone=0.5, x1=0.3", "effect", {"predictors": ["x1"], "interaction_terms": []}
    )
    assert len(out["items"]) == 1
    assert len(out["errors"]) == 1
