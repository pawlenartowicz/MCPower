"""
Tests for parsing utilities.
"""

import pytest


class TestParseEquation:
    """Test _parse_equation function."""

    def test_equals_format(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("y = x1 + x2")
        assert dep == "y"
        assert "x1" in formula
        assert "x2" in formula
        assert random_effects == []

    def test_tilde_format(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("outcome ~ predictor1 + predictor2")
        assert dep == "outcome"
        assert "predictor1" in formula
        assert random_effects == []

    def test_spaces_handling(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("  y   =   x1  +  x2  ")
        assert dep == "y"
        assert random_effects == []

    def test_no_separator(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("x1 + x2")
        assert dep == "explained_variable"
        assert random_effects == []

    def test_single_random_intercept(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("y ~ x1 + x2 + (1|school)")
        assert dep == "y"
        assert "x1" in formula
        assert "x2" in formula
        assert "(1|school)" not in formula  # Should be removed
        assert len(random_effects) == 1
        assert random_effects[0]["type"] == "random_intercept"
        assert random_effects[0]["grouping_var"] == "school"

    def test_multiple_random_intercepts(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("y ~ x + (1|school) + (1|classroom)")
        assert len(random_effects) == 2
        assert random_effects[0]["grouping_var"] == "school"
        assert random_effects[1]["grouping_var"] == "classroom"

    def test_random_slope_parses(self):
        from mcpower.utils.parsers import _parse_equation

        dep, formula, random_effects = _parse_equation("y ~ x1 + (1 + x1|school)")
        assert len(random_effects) == 1
        assert random_effects[0]["type"] == "random_slope"
        assert random_effects[0]["grouping_var"] == "school"
        assert random_effects[0]["slope_vars"] == ["x1"]

    def test_duplicate_grouping_var_raises_error(self):
        from mcpower.utils.parsers import _parse_equation

        with pytest.raises(ValueError, match="Duplicate random effect"):
            _parse_equation("y ~ x + (1|school) + (1|school)")


class TestParseIndependentVariables:
    """Test _parse_independent_variables function."""

    def test_single_variable(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("x1")
        var_names = [v["name"] for v in variables.values()]
        assert "x1" in var_names

    def test_multiple_variables(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("x1 + x2 + x3")
        var_names = [v["name"] for v in variables.values()]
        assert len(var_names) == 3
        assert set(var_names) == {"x1", "x2", "x3"}

    def test_colon_interaction(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("x1 + x2 + x1:x2")
        effect_names = [e["name"] for e in effects.values()]
        assert "x1" in effect_names
        assert "x2" in effect_names
        assert "x1:x2" in effect_names

    def test_star_interaction(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("a * b")
        effect_names = [e["name"] for e in effects.values()]
        assert "a" in effect_names
        assert "b" in effect_names
        assert "a:b" in effect_names

    def test_three_way_interaction(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("a * b * c")
        effect_names = [e["name"] for e in effects.values()]
        assert "a:b" in effect_names
        assert "a:c" in effect_names
        assert "b:c" in effect_names
        assert "a:b:c" in effect_names

    def test_effect_types(self):
        from mcpower.utils.parsers import _parse_independent_variables

        variables, effects = _parse_independent_variables("x1 + x2 + x1:x2")
        for e in effects.values():
            if e["name"] in ["x1", "x2"]:
                assert e["type"] == "main"
            else:
                assert e["type"] == "interaction"


class TestAssignmentParser:
    """Test _AssignmentParser class."""

    def test_split_simple(self):
        from mcpower.utils.parsers import _parser

        result = _parser._split_assignments("x1=0.3, x2=0.2")
        assert result == ["x1=0.3", "x2=0.2"]

    def test_split_with_parentheses(self):
        from mcpower.utils.parsers import _parser

        result = _parser._split_assignments("x=(factor,3), y=0.5")
        assert len(result) == 2
        assert result[0] == "x=(factor,3)"
        assert result[1] == "y=0.5"

    def test_split_nested_parentheses(self):
        from mcpower.utils.parsers import _parser

        result = _parser._split_assignments("x=(factor,0.3,0.4,0.3), y=normal")
        assert len(result) == 2

    def test_parse_effect_value(self):
        from mcpower.utils.parsers import _parser

        value, error = _parser._parse_effect_value("0.5")
        assert value == 0.5
        assert error is None

    def test_parse_effect_value_invalid(self):
        from mcpower.utils.parsers import _parser

        value, error = _parser._parse_effect_value("invalid")
        assert error is not None

    def test_parse_correlation_value(self):
        from mcpower.utils.parsers import _parser

        value, error = _parser._parse_correlation_value("0.5")
        assert value == 0.5
        assert error is None

    def test_parse_correlation_value_out_of_range(self):
        from mcpower.utils.parsers import _parser

        value, error = _parser._parse_correlation_value("1.5")
        assert error is not None


class TestVariableTypeParsing:
    """Test variable type parsing."""

    def test_parse_normal(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("normal")
        assert result["type"] == "normal"
        assert error is None

    def test_parse_binary_simple(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("binary")
        assert result["type"] == "binary"
        assert result["proportion"] == 0.5

    def test_parse_binary_with_proportion(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("(binary,0.3)")
        assert result["type"] == "binary"
        assert result["proportion"] == 0.3

    def test_parse_factor_simple(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("(factor,3)")
        assert result["type"] == "factor"
        assert result["n_levels"] == 3
        assert len(result["proportions"]) == 3

    def test_parse_factor_with_proportions(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("(factor,0.5,0.3,0.2)")
        assert result["type"] == "factor"
        assert result["n_levels"] == 3
        assert abs(sum(result["proportions"]) - 1.0) < 1e-6

    def test_parse_invalid_type(self):
        from mcpower.utils.parsers import _parser

        result, error = _parser._parse_variable_type_value("invalid_type")
        assert error is not None


class TestCorrelationParsing:
    """Test correlation parsing."""

    def test_parse_correlation_pair(self):
        from mcpower.utils.parsers import _parser

        parsed, errors = _parser._parse("(x1,x2)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) == 0
        key = tuple(sorted(["x1", "x2"]))
        assert key in parsed
        assert parsed[key] == 0.5

    def test_parse_corr_function_format(self):
        from mcpower.utils.parsers import _parser

        parsed, errors = _parser._parse("corr(x1,x2)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) == 0

    def test_parse_multiple_correlations(self):
        from mcpower.utils.parsers import _parser

        parsed, errors = _parser._parse("(x1,x2)=0.5, (x1,x3)=0.3", "correlation", ["x1", "x2", "x3"])
        assert len(errors) == 0
        assert len(parsed) == 2

    def test_invalid_variable_in_correlation(self):
        from mcpower.utils.parsers import _parser

        parsed, errors = _parser._parse("(x1,invalid)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) > 0

    def test_self_correlation_error(self):
        from mcpower.utils.parsers import _parser

        parsed, errors = _parser._parse("(x1,x1)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) > 0
