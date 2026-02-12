"""
Tests for VariableRegistry.
"""

import numpy as np


class TestVariableRegistryInit:
    """Test VariableRegistry initialization."""

    def test_simple_equation(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        assert registry.dependent == "y"
        assert set(registry.predictor_names) == {"x1", "x2"}

    def test_interaction_equation(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = a + b + a:b")
        assert "a" in registry.effect_names
        assert "b" in registry.effect_names
        assert "a:b" in registry.effect_names

    def test_equation_property(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        assert registry.equation == "y = x1 + x2"


class TestEffectManagement:
    """Test effect size management."""

    def test_set_effect_size(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        registry.set_effect_size("x1", 0.5)
        registry.set_effect_size("x2", 0.3)
        sizes = registry.get_effect_sizes()
        assert sizes[0] == 0.5
        assert sizes[1] == 0.3

    def test_effect_names(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = a + b + a:b")
        names = registry.effect_names
        assert len(names) == 3
        assert "a" in names
        assert "b" in names
        assert "a:b" in names

    def test_get_target_indices(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2 + x3")
        indices = registry.get_target_indices(["x1", "x3"])
        assert len(indices) == 2


class TestFactorVariables:
    """Test factor variable handling."""

    def test_set_factor_type(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1")
        registry.set_variable_type("group", "factor", n_levels=3)
        assert "group" in registry.factor_names

    def test_expand_factors(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1")
        registry.set_variable_type("group", "factor", n_levels=3)
        registry.expand_factors()
        dummies = registry.dummy_names
        assert len(dummies) == 2  # n_levels - 1
        assert "group[2]" in dummies
        assert "group[3]" in dummies

    def test_factor_effect_names(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1")
        registry.set_variable_type("group", "factor", n_levels=3)
        registry.expand_factors()
        effects = registry.effect_names
        assert "group[2]" in effects
        assert "group[3]" in effects
        assert "x1" in effects
        assert "group" not in effects  # Original should be replaced

    def test_get_factor_specs(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1")
        registry.set_variable_type("group", "factor", n_levels=3, proportions=[0.5, 0.3, 0.2])
        registry.expand_factors()
        specs = registry.get_factor_specs()
        assert len(specs) == 1
        assert specs[0]["n_levels"] == 3

    def test_non_factor_names(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1 + x2")
        registry.set_variable_type("group", "factor", n_levels=3)
        registry.expand_factors()
        non_factors = registry.non_factor_names
        assert "x1" in non_factors
        assert "x2" in non_factors
        assert "group" not in non_factors


class TestCorrelationMatrix:
    """Test correlation matrix handling."""

    def test_set_correlation_matrix(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        registry.set_correlation_matrix(matrix)
        result = registry.get_correlation_matrix()
        assert np.array_equal(result, matrix)

    def test_set_correlation_pair(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        registry.set_correlation_matrix(np.eye(2))
        registry.set_correlation("x1", "x2", 0.5)
        matrix = registry.get_correlation_matrix()
        assert matrix[0, 1] == 0.5
        assert matrix[1, 0] == 0.5


class TestVariableTypes:
    """Test variable type handling."""

    def test_set_binary_type(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = treatment + x1")
        registry.set_variable_type("treatment", "binary")
        pred = registry.get_predictor("treatment")
        assert pred.var_type == "binary"

    def test_get_var_types(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        # Set defaults
        for name in registry.predictor_names:
            pred = registry.get_predictor(name)
            pred.var_type = "normal"
        types = registry.get_var_types()
        assert len(types) == 2
        assert all(t == 0 for t in types)  # 0 = normal

    def test_type_codes(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2 + x3")
        p1 = registry.get_predictor("x1")
        p2 = registry.get_predictor("x2")
        p3 = registry.get_predictor("x3")
        p1.var_type = "normal"
        p2.var_type = "binary"
        p3.var_type = "right_skewed"
        types = registry.get_var_types()
        assert types[0] == 0  # normal
        assert types[1] == 1  # binary
        assert types[2] == 2  # right_skewed


class TestIsDummyVariable:
    """Test dummy variable detection."""

    def test_regular_variable(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = x1 + x2")
        assert not registry.is_dummy_variable("x1")

    def test_dummy_variable(self):
        from mcpower.core.variables import VariableRegistry

        registry = VariableRegistry("y = group + x1")
        registry.set_variable_type("group", "factor", n_levels=3)
        registry.expand_factors()
        assert registry.is_dummy_variable("group[2]")
        assert registry.is_dummy_variable("group[3]")
        assert not registry.is_dummy_variable("x1")
