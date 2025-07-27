"""
Tests for parsing utilities.
"""

import pytest
import numpy as np
from mcpower.utils.parsers import (
    _parse_equation,
    _parse_independent_variables,
    _validate_and_parse_effects,
    _parser,
)


class TestEquationParsing:
    """Test equation format parsing."""

    def test_equals_format(self):
        """Test 'y = x1 + x2' format."""
        dep_var, formula = _parse_equation("y = x1 + x2")
        assert dep_var == "y"
        assert formula == "x1+x2"

    def test_tilde_format(self):
        """Test 'y ~ x1 + x2' format."""
        dep_var, formula = _parse_equation("outcome ~ treatment + age")
        assert dep_var == "outcome"
        assert formula == "treatment+age"

    def test_no_separator(self):
        """Test formula without separator."""
        dep_var, formula = _parse_equation("x1 + x2")
        assert dep_var == "explained_variable"
        assert formula == "x1+x2"

    def test_whitespace_handling(self):
        """Test whitespace removal."""
        dep_var, formula = _parse_equation("  y  =  x1  +  x2  ")
        assert dep_var == "y"
        assert formula == "x1+x2"


class TestIndependentVariableParsing:
    """Test independent variable parsing."""

    def test_simple_main_effects(self):
        """Test simple main effects parsing."""
        variables, effects = _parse_independent_variables("x1 + x2")

        var_names = [info["name"] for info in variables.values()]
        assert "x1" in var_names
        assert "x2" in var_names

        effect_names = [info["name"] for info in effects.values()]
        assert "x1" in effect_names
        assert "x2" in effect_names

    def test_interaction_colon(self):
        """Test interaction with colon syntax."""
        variables, effects = _parse_independent_variables("x1 + x2 + x1:x2")

        effect_names = [info["name"] for info in effects.values()]
        assert "x1" in effect_names
        assert "x2" in effect_names
        assert "x1:x2" in effect_names

        # Check interaction details
        interaction_effect = None
        for effect in effects.values():
            if effect["name"] == "x1:x2":
                interaction_effect = effect
                break

        assert interaction_effect["type"] == "interaction"
        assert interaction_effect["var_names"] == ["x1", "x2"]

    def test_interaction_star(self):
        """Test interaction with star syntax."""
        variables, effects = _parse_independent_variables("x1*x2")

        effect_names = [info["name"] for info in effects.values()]
        # Star should create main effects + interaction
        assert "x1" in effect_names
        assert "x2" in effect_names
        assert "x1:x2" in effect_names

    def test_three_way_interaction(self):
        """Test three-way interaction."""
        variables, effects = _parse_independent_variables("x1*x2*x3")

        effect_names = [info["name"] for info in effects.values()]
        # Should have all main effects
        assert "x1" in effect_names
        assert "x2" in effect_names
        assert "x3" in effect_names

        # Should have all 2-way interactions
        assert "x1:x2" in effect_names
        assert "x1:x3" in effect_names
        assert "x2:x3" in effect_names

        # Should have 3-way interaction
        assert "x1:x2:x3" in effect_names

    def test_column_indices(self):
        """Test column index assignment."""
        variables, effects = _parse_independent_variables("age + treatment")

        # Check main effect indices
        for effect in effects.values():
            if effect["type"] == "main":
                assert "column_index" in effect
                assert isinstance(effect["column_index"], int)


class TestAssignmentParser:
    """Test generic assignment parser."""

    def test_effect_parsing(self):
        """Test effect assignment parsing."""
        available_effects = {"effect_1": {"name": "x1"}, "effect_2": {"name": "x2"}}

        parsed, errors = _parser._parse("x1=0.5, x2=0.3", "effect", ["x1", "x2"])

        assert len(errors) == 0
        assert "x1" in parsed
        assert "x2" in parsed
        assert parsed["x1"] == 0.5
        assert parsed["x2"] == 0.3

    def test_variable_type_parsing(self):
        """Test variable type parsing."""
        parsed, errors = _parser._parse(
            "x1=binary, x2=normal", "variable_type", ["x1", "x2"]
        )

        assert len(errors) == 0
        assert parsed["x1"]["type"] == "binary"
        assert parsed["x1"]["proportion"] == 0.5
        assert parsed["x2"]["type"] == "normal"

    def test_binary_with_proportion(self):
        """Test binary variable with custom proportion."""
        parsed, errors = _parser._parse(
            "treatment=(binary,0.3)", "variable_type", ["treatment"]
        )

        assert len(errors) == 0
        assert parsed["treatment"]["type"] == "binary"
        assert parsed["treatment"]["proportion"] == 0.3

    def test_correlation_parsing(self):
        """Test correlation parsing."""
        parsed, errors = _parser._parse(
            "corr(x1,x2)=0.5, corr(x1,x3)=-0.3", "correlation", ["x1", "x2", "x3"]
        )

        assert len(errors) == 0
        assert ("x1", "x2") in parsed or ("x2", "x1") in parsed

        # Check correlation values
        if ("x1", "x2") in parsed:
            assert parsed[("x1", "x2")] == 0.5
        else:
            assert parsed[("x2", "x1")] == 0.5

    def test_correlation_shorthand(self):
        """Test correlation shorthand syntax."""
        parsed, errors = _parser._parse("(x1,x2)=0.4", "correlation", ["x1", "x2"])

        assert len(errors) == 0
        assert len(parsed) == 1

    def test_invalid_assignments(self):
        """Test invalid assignment formats."""
        # Missing equals
        parsed, errors = _parser._parse("x1 0.5", "effect", ["x1"])
        assert len(errors) > 0

        # Invalid value
        parsed, errors = _parser._parse("x1=invalid", "effect", ["x1"])
        assert len(errors) > 0

        # Unknown variable
        parsed, errors = _parser._parse("unknown=0.5", "effect", ["x1"])
        assert len(errors) > 0

    def test_parentheses_splitting(self):
        """Test assignment splitting respects parentheses."""
        assignments = _parser._split_assignments(
            "x1=binary, x2=(binary,0.3), x3=normal"
        )

        assert len(assignments) == 3
        assert "x2=(binary,0.3)" in assignments


class TestValidateAndParseEffects:
    """Test effect validation and parsing."""

    def test_string_input(self):
        """Test string effect assignments."""
        available_effects = {
            "effect_1": {"name": "x1"},
            "effect_2": {"name": "x2"},
            "effect_3": {"name": "x1:x2"},
        }

        def find_by_name(name):
            for key, effect in available_effects.items():
                if effect["name"] == name:
                    return key, effect
            return None, None

        valid_items, find_func = _validate_and_parse_effects(
            "x1=0.5, x2=0.3", available_effects, "effect"
        )

        assert len(valid_items) == 2
        assert valid_items[0]["name"] == "x1"
        assert valid_items[0]["value"] == 0.5

    def test_dict_input(self):
        """Test dict effect assignments."""
        available_effects = {"effect_1": {"name": "x1"}}

        valid_items, find_func = _validate_and_parse_effects(
            {"x1": 0.5}, available_effects, "effect"
        )

        assert len(valid_items) == 1
        assert valid_items[0]["name"] == "x1"
        assert valid_items[0]["value"] == 0.5

    def test_list_input(self):
        """Test list input."""
        available_effects = {"effect_1": {"name": "x1"}}

        valid_items, find_func = _validate_and_parse_effects(
            ["x1"], available_effects, "effect"
        )

        assert len(valid_items) == 1
        assert valid_items[0]["name"] == "x1"

    def test_invalid_effects(self):
        """Test validation errors."""
        available_effects = {"effect_1": {"name": "x1"}}

        with pytest.raises(ValueError, match="Validation failed"):
            _validate_and_parse_effects("unknown=0.5", available_effects, "effect")


class TestVariableTypeValues:
    """Test variable type value parsing."""

    def test_normal_type(self):
        """Test normal distribution type."""
        result, error = _parser._parse_variable_type_value("normal")
        assert error is None
        assert result["type"] == "normal"

    def test_all_supported_types(self):
        """Test all supported distribution types."""
        types = [
            "normal",
            "binary",
            "right_skewed",
            "left_skewed",
            "high_kurtosis",
            "uniform",
        ]

        for var_type in types:
            result, error = _parser._parse_variable_type_value(var_type)
            assert error is None
            assert result["type"] == var_type

    def test_binary_default_proportion(self):
        """Test binary type gets default proportion."""
        result, error = _parser._parse_variable_type_value("binary")
        assert error is None
        assert result["type"] == "binary"
        assert result["proportion"] == 0.5

    def test_binary_custom_proportion(self):
        """Test binary with custom proportion."""
        result, error = _parser._parse_variable_type_value("(binary,0.3)")
        assert error is None
        assert result["type"] == "binary"
        assert result["proportion"] == 0.3

    def test_invalid_proportion_range(self):
        """Test invalid proportion values."""
        result, error = _parser._parse_variable_type_value("(binary,1.5)")
        assert error is not None
        assert "between 0 and 1" in error

        result, error = _parser._parse_variable_type_value("(binary,-0.1)")
        assert error is not None

    def test_unsupported_type(self):
        """Test unsupported distribution type."""
        result, error = _parser._parse_variable_type_value("unknown_dist")
        assert error is not None
        assert "Unsupported type" in error

    def test_tuple_format_validation(self):
        """Test tuple format validation."""
        # Wrong number of values
        result, error = _parser._parse_variable_type_value("(binary)")
        assert error is not None

        result, error = _parser._parse_variable_type_value("(binary,0.3,extra)")
        assert error is not None

        # Non-binary with tuple
        result, error = _parser._parse_variable_type_value("(normal,0.5)")
        assert error is not None


class TestCorrelationValues:
    """Test correlation value parsing."""

    def test_valid_correlations(self):
        """Test valid correlation values."""
        valid_values = [0.0, 0.5, -0.5, 1.0, -1.0, 0.99, -0.99]

        for value in valid_values:
            result, error = _parser._parse_correlation_value(str(value))
            assert error is None
            assert result == value

    def test_invalid_correlation_range(self):
        """Test correlation values outside [-1, 1]."""
        invalid_values = [1.5, -1.5, 2.0, -2.0]

        for value in invalid_values:
            result, error = _parser._parse_correlation_value(str(value))
            assert error is not None
            assert "between -1 and 1" in error

    def test_non_numeric_correlation(self):
        """Test non-numeric correlation values."""
        result, error = _parser._parse_correlation_value("not_a_number")
        assert error is not None
        assert "Invalid correlation value" in error


class TestCorrelationAssignmentParsing:
    """Test correlation assignment parsing."""

    def test_corr_format(self):
        """Test 'corr(x1,x2)=0.5' format."""
        name, value = _parser._parse_correlation_assignment("corr(x1, x2)=0.5")
        assert name == ("x1", "x2")
        assert value == "0.5"

    def test_shorthand_format(self):
        """Test '(x1,x2)=0.5' format."""
        name, value = _parser._parse_correlation_assignment("(x1, x2)=0.3")
        assert name == ("x1", "x2")
        assert value == "0.3"

    def test_whitespace_handling(self):
        """Test whitespace in correlation assignments."""
        name, value = _parser._parse_correlation_assignment("corr( x1 , x2 ) = 0.7")
        assert name == ("x1", "x2")
        assert value == "0.7"

    def test_invalid_correlation_format(self):
        """Test invalid correlation formats."""
        with pytest.raises(ValueError):
            _parser._parse_correlation_assignment("invalid_format")

        with pytest.raises(ValueError):
            _parser._parse_correlation_assignment(
                "corr(x1)=0.5"
            )  # Missing second variable

        with pytest.raises(ValueError):
            _parser._parse_correlation_assignment(
                "corr(x1, x2, x3)=0.5"
            )  # Too many variables


class TestCorrelationValidation:
    """Test correlation pair validation."""

    def test_valid_pairs(self):
        """Test valid variable pairs."""
        available_vars = ["x1", "x2", "x3"]

        valid, error = _parser._validate_correlation_pair(("x1", "x2"), available_vars)
        assert valid is True
        assert error is None

    def test_missing_variables(self):
        """Test pairs with missing variables."""
        available_vars = ["x1", "x2"]

        valid, error = _parser._validate_correlation_pair(("x1", "x3"), available_vars)
        assert valid is False
        assert "not found" in error

    def test_self_correlation(self):
        """Test correlation of variable with itself."""
        available_vars = ["x1", "x2"]

        valid, error = _parser._validate_correlation_pair(("x1", "x1"), available_vars)
        assert valid is False
        assert "correlate variable with itself" in error


class TestCorrelationMatrixHandling:
    """Test correlation matrix creation and validation."""

    def test_well_defined_matrix_2x2(self):
        """Test valid 2x2 correlation matrix."""
        parsed, errors = _parser._parse("corr(x1,x2)=0.5", "correlation", ["x1", "x2"])

        assert len(errors) == 0
        assert len(parsed) == 1

        # Should be symmetric pair
        key = list(parsed.keys())[0]
        assert key == ("x1", "x2") or key == ("x2", "x1")
        assert abs(parsed[key] - 0.5) < 1e-10

    def test_well_defined_matrix_3x3(self):
        """Test valid 3x3 correlation matrix specification."""
        corr_string = "corr(x1,x2)=0.3, corr(x1,x3)=0.5, corr(x2,x3)=0.2"
        parsed, errors = _parser._parse(corr_string, "correlation", ["x1", "x2", "x3"])

        assert len(errors) == 0
        assert len(parsed) == 3

        # Check all pairs present
        pairs = {tuple(sorted(k)) for k in parsed.keys()}
        expected = {("x1", "x2"), ("x1", "x3"), ("x2", "x3")}
        assert pairs == expected

    def test_duplicate_correlations(self):
        """Test duplicate correlation specifications."""
        corr_string = "corr(x1,x2)=0.3, corr(x2,x1)=0.5"
        parsed, errors = _parser._parse(corr_string, "correlation", ["x1", "x2"])

        # Should handle duplicates (implementation dependent)
        assert len(parsed) == 1

    def test_ill_defined_correlation_values(self):
        """Test invalid correlation values."""
        invalid_correlations = [
            "corr(x1,x2)=1.5",  # > 1
            "corr(x1,x2)=-1.1",  # < -1
            "corr(x1,x2)=2.0",  # > 1
            "corr(x1,x2)=-2.0",  # < -1
        ]

        for corr in invalid_correlations:
            parsed, errors = _parser._parse(corr, "correlation", ["x1", "x2"])
            assert len(errors) > 0

    def test_boundary_correlation_values(self):
        """Test boundary correlation values."""
        boundary_values = [
            ("corr(x1,x2)=1.0", 1.0),
            ("corr(x1,x2)=-1.0", -1.0),
            ("corr(x1,x2)=0.0", 0.0),
            ("corr(x1,x2)=0.999", 0.999),
            ("corr(x1,x2)=-0.999", -0.999),
        ]

        for corr_string, expected in boundary_values:
            parsed, errors = _parser._parse(corr_string, "correlation", ["x1", "x2"])
            assert len(errors) == 0
            key = list(parsed.keys())[0]
            assert abs(parsed[key] - expected) < 1e-10

    def test_potentially_non_positive_definite(self):
        """Test correlation patterns that could create non-PD matrices."""
        # High correlations that might violate PD constraint
        corr_string = "corr(x1,x2)=0.9, corr(x1,x3)=0.9, corr(x2,x3)=-0.9"
        parsed, errors = _parser._parse(corr_string, "correlation", ["x1", "x2", "x3"])

        # Parser should accept these (validation happens elsewhere)
        assert len(errors) == 0
        assert len(parsed) == 3

    def test_mixed_correlation_formats(self):
        """Test mixing corr() and shorthand formats."""
        corr_string = "corr(x1,x2)=0.3, (x1,x3)=0.5"
        parsed, errors = _parser._parse(corr_string, "correlation", ["x1", "x2", "x3"])

        assert len(errors) == 0
        assert len(parsed) == 2

    def test_large_correlation_matrix(self):
        """Test specification for larger correlation matrix."""
        variables = ["x1", "x2", "x3", "x4", "x5"]
        correlations = []

        # Create all pairs
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                corr_val = 0.1 * (i + j)  # Vary correlations
                correlations.append(f"corr({variables[i]},{variables[j]})={corr_val}")

        corr_string = ", ".join(correlations)
        parsed, errors = _parser._parse(corr_string, "correlation", variables)

        assert len(errors) == 0
        expected_pairs = len(variables) * (len(variables) - 1) // 2
        assert len(parsed) == expected_pairs


class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_empty_strings(self):
        """Test empty string inputs."""
        dep_var, formula = _parse_equation("")
        assert dep_var == "explained_variable"  # Default when no separator
        assert formula == ""

        variables, effects = _parse_independent_variables("")
        assert len(variables) == 0
        assert len(effects) == 0

    def test_complex_interaction_formula(self):
        """Test complex formula with multiple interactions."""
        variables, effects = _parse_independent_variables(
            "a + b + c + a:b + a:c + b:c + a:b:c"
        )

        effect_names = [info["name"] for info in effects.values()]

        # All main effects
        assert "a" in effect_names
        assert "b" in effect_names
        assert "c" in effect_names

        # All 2-way interactions
        assert "a:b" in effect_names
        assert "a:c" in effect_names
        assert "b:c" in effect_names

        # 3-way interaction
        assert "a:b:c" in effect_names

    def test_variable_name_validation(self):
        """Test variable name patterns."""
        # Valid names
        valid_names = ["x1", "treatment", "age_group", "var123"]

        for name in valid_names:
            variables, effects = _parse_independent_variables(name)
            assert len(variables) > 0
            assert len(effects) > 0

    def test_mixed_assignment_types(self):
        """Test mixed valid and invalid assignments."""
        parsed, errors = _parser._parse(
            "x1=0.5, invalid, x2=0.3", "effect", ["x1", "x2"]
        )

        # Should have some errors but also some valid results
        assert len(errors) > 0
        assert "x1" in parsed or "x2" in parsed
