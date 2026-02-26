"""Tests for parser error paths and edge cases."""

import pytest

from mcpower.utils.parsers import _AssignmentParser, _parse_equation


_parser = _AssignmentParser()


class TestAssignmentParserErrors:
    """Error paths in _AssignmentParser._parse."""

    def test_missing_equals_sign(self):
        parsed, errors = _parser._parse("x1 0.5", "effect", ["x1"])
        assert len(errors) == 1
        assert "Invalid format" in errors[0]

    def test_unknown_parse_type(self):
        parsed, errors = _parser._parse("x1=0.5", "unknown_type", ["x1"])
        assert len(errors) == 1
        assert "Unknown parse type" in errors[0]

    def test_unavailable_variable(self):
        parsed, errors = _parser._parse("x_missing=0.5", "effect", ["x1", "x2"])
        assert len(errors) == 1
        assert "not found" in errors[0]
        assert "x_missing" in errors[0]

    def test_invalid_effect_value(self):
        parsed, errors = _parser._parse("x1=abc", "effect", ["x1"])
        assert len(errors) == 1
        assert "Invalid effect size" in errors[0]

    def test_multiple_errors(self):
        parsed, errors = _parser._parse("x_bad=abc, x_also_bad=xyz", "effect", ["x1"])
        assert len(errors) == 2


class TestCorrelationParserErrors:
    """Error paths for correlation parsing."""

    def test_invalid_correlation_format(self):
        parsed, errors = _parser._parse("x1_x2=0.5", "correlation", ["x1", "x2"])
        assert len(errors) == 1
        assert "Invalid format" in errors[0] or "Invalid correlation" in errors[0]

    def test_correlation_var_not_found(self):
        parsed, errors = _parser._parse("corr(x1, x_missing)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_self_correlation(self):
        parsed, errors = _parser._parse("corr(x1, x1)=0.5", "correlation", ["x1", "x2"])
        assert len(errors) == 1
        assert "Cannot correlate variable with itself" in errors[0]

    def test_correlation_value_out_of_range(self):
        parsed, errors = _parser._parse("corr(x1, x2)=1.5", "correlation", ["x1", "x2"])
        assert len(errors) == 1
        assert "between -1 and 1" in errors[0]

    def test_invalid_correlation_value(self):
        parsed, errors = _parser._parse("corr(x1, x2)=abc", "correlation", ["x1", "x2"])
        assert len(errors) == 1
        assert "Invalid correlation value" in errors[0]


class TestVariableTypeErrors:
    """Error paths for variable type parsing."""

    def test_unsupported_type(self):
        parsed, errors = _parser._parse("x1=crazy_type", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "Unsupported type" in errors[0]

    def test_binary_proportion_out_of_range(self):
        parsed, errors = _parser._parse("x1=(binary,1.5)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "between 0 and 1" in errors[0]

    def test_binary_non_numeric_proportion(self):
        parsed, errors = _parser._parse("x1=(binary,abc)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "Invalid proportion" in errors[0]

    def test_binary_wrong_param_count(self):
        parsed, errors = _parser._parse("x1=(binary,0.3,0.4)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "exactly 2 values" in errors[0]

    def test_factor_less_than_2_levels(self):
        parsed, errors = _parser._parse("x1=(factor,1)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "at least 2 levels" in errors[0]

    def test_factor_more_than_20_levels(self):
        parsed, errors = _parser._parse("x1=(factor,21)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "more than 20 levels" in errors[0]

    def test_factor_non_integer_levels(self):
        parsed, errors = _parser._parse("x1=(factor,abc)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "Must be integer" in errors[0]

    def test_factor_proportions_more_than_20(self):
        props = ",".join(["0.04"] * 21)
        parsed, errors = _parser._parse(f"x1=(factor,{props})", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "more than 20 levels" in errors[0]

    def test_factor_zero_proportion(self):
        parsed, errors = _parser._parse("x1=(factor,0.5,0.0,0.5)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "positive" in errors[0]

    def test_factor_non_numeric_proportions(self):
        parsed, errors = _parser._parse("x1=(factor,abc,def)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "numeric" in errors[0]

    def test_tuple_no_comma(self):
        parsed, errors = _parser._parse("x1=(binary)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "Invalid tuple format" in errors[0]

    def test_tuple_unsupported_type_in_tuple(self):
        parsed, errors = _parser._parse("x1=(normal,0.5)", "variable_type", ["x1"])
        assert len(errors) == 1
        assert "only supported for binary and factor" in errors[0]


class TestEquationParsing:
    """Edge cases in _parse_equation."""

    def test_nested_random_effects(self):
        dep, formula, ranefs = _parse_equation("y ~ x1 + (1|A/B)")
        assert dep == "y"
        assert len(ranefs) == 2
        group_vars = {r["grouping_var"] for r in ranefs}
        assert "A" in group_vars
        assert "A:B" in group_vars

    def test_duplicate_grouping_var_raises(self):
        with pytest.raises(ValueError, match="Duplicate random effect grouping variable"):
            _parse_equation("y ~ x1 + (1|school) + (1|school)")

    def test_random_slopes(self):
        dep, formula, ranefs = _parse_equation("y ~ x1 + (1 + x1|school)")
        assert len(ranefs) == 1
        assert ranefs[0]["type"] == "random_slope"
        assert ranefs[0]["slope_vars"] == ["x1"]
        assert ranefs[0]["grouping_var"] == "school"

    def test_random_slope_duplicate_grouping_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _parse_equation("y ~ x1 + (1|school) + (1 + x1|school)")

    def test_no_separator_uses_default_dep(self):
        dep, formula, ranefs = _parse_equation("x1+x2")
        assert dep == "explained_variable"
        assert "x1" in formula
        assert "x2" in formula

    def test_nested_duplicate_parent_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _parse_equation("y ~ (1|A) + (1|A/B)")
