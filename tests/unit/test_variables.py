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


class TestNamedFactorLevels:
    """Tests for named (string/numeric) factor level labels."""

    def test_predictor_var_accepts_string_factor_level(self):
        """PredictorVar.factor_level can be a string."""
        from mcpower.core.variables import PredictorVar

        p = PredictorVar(name="cyl[6]", factor_level="6", is_dummy=True, factor_source="cyl")
        assert p.factor_level == "6"

    def test_predictor_var_accepts_int_factor_level(self):
        """PredictorVar.factor_level still accepts int (backward compat)."""
        from mcpower.core.variables import PredictorVar

        p = PredictorVar(name="cyl[2]", factor_level=2, is_dummy=True, factor_source="cyl")
        assert p.factor_level == 2

    def test_effect_accepts_string_factor_level(self):
        """Effect.factor_level can be a string."""
        from mcpower.core.variables import Effect

        e = Effect(name="cyl[6]", effect_type="main", factor_level="6", factor_source="cyl")
        assert e.factor_level == "6"

    def test_posthoc_spec_accepts_string_levels(self):
        """PostHocSpec.level_a/level_b can be strings."""
        from mcpower.core.variables import PostHocSpec

        spec = PostHocSpec(
            factor_name="origin",
            level_a="Europe",
            level_b="Japan",
            col_idx_a=None,
            col_idx_b=0,
            n_levels=3,
            label="origin[Europe] vs origin[Japan]",
        )
        assert spec.level_a == "Europe"
        assert spec.level_b == "Japan"

    def test_expand_factors_with_labels(self):
        """expand_factors uses level_labels for dummy names when present."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = cyl")
        reg.set_variable_type("cyl", "factor", n_levels=3, level_labels=["4", "6", "8"])
        reg.expand_factors()
        assert "cyl[6]" in reg.dummy_names
        assert "cyl[8]" in reg.dummy_names
        assert "cyl[4]" not in reg.dummy_names  # reference level, dropped
        assert "cyl[2]" not in reg.dummy_names
        assert "cyl[3]" not in reg.dummy_names

    def test_expand_factors_with_string_labels(self):
        """expand_factors uses string level labels."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = origin")
        reg.set_variable_type("origin", "factor", n_levels=3,
                              level_labels=["Europe", "Japan", "USA"])
        reg.expand_factors()
        assert "origin[Japan]" in reg.dummy_names
        assert "origin[USA]" in reg.dummy_names
        assert "origin[Europe]" not in reg.dummy_names

    def test_expand_factors_without_labels_uses_integers(self):
        """expand_factors falls back to integer indices when no labels."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = group")
        reg.set_variable_type("group", "factor", n_levels=3)
        reg.expand_factors()
        assert "group[2]" in reg.dummy_names
        assert "group[3]" in reg.dummy_names

    def test_expand_factors_with_custom_reference(self):
        """expand_factors respects custom reference_level from level_labels."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = cyl")
        reg.set_variable_type("cyl", "factor", n_levels=3,
                              level_labels=["4", "6", "8"], reference_level="6")
        reg.expand_factors()
        assert "cyl[4]" in reg.dummy_names
        assert "cyl[8]" in reg.dummy_names
        assert "cyl[6]" not in reg.dummy_names

    def test_expand_factors_interaction_with_labels(self):
        """Interactions expand using level labels."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = origin + x1 + origin:x1")
        reg.set_variable_type("origin", "factor", n_levels=3,
                              level_labels=["Europe", "Japan", "USA"])
        reg.expand_factors()
        effect_names = reg.effect_names
        assert "origin[Japan]:x1" in effect_names
        assert "origin[USA]:x1" in effect_names

    def test_get_factor_specs_includes_labels(self):
        """get_factor_specs returns level_labels when present."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = cyl")
        reg.set_variable_type("cyl", "factor", n_levels=3,
                              level_labels=["4", "6", "8"])
        specs = reg.get_factor_specs()
        assert specs[0]["level_labels"] == ["4", "6", "8"]

    def test_get_factor_specs_no_labels(self):
        """get_factor_specs returns None for level_labels when not set."""
        from mcpower.core.variables import VariableRegistry

        reg = VariableRegistry("y = group")
        reg.set_variable_type("group", "factor", n_levels=3)
        specs = reg.get_factor_specs()
        assert specs[0].get("level_labels") is None


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
