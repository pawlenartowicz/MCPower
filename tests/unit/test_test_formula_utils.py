"""Tests for test_formula parsing utilities."""

from collections import OrderedDict
from unittest.mock import MagicMock

import numpy as np


class TestExtractTestFormulaEffects:
    """Test _extract_test_formula_effects helper."""

    def _make_registry(
        self,
        effect_names,
        factor_names=None,
        factor_dummies=None,
        cluster_effect_names=None,
    ):
        """Create a minimal mock registry for testing."""
        reg = MagicMock()
        reg.effect_names = effect_names
        reg.factor_names = factor_names or []
        reg.cluster_effect_names = cluster_effect_names or []

        # Build _effects dict with correct ordering
        effects = OrderedDict()
        for name in effect_names:
            eff = MagicMock()
            eff.effect_type = "interaction" if ":" in name else "main"
            effects[name] = eff
        reg._effects = effects

        # Factor dummies
        reg._factor_dummies = factor_dummies or {}
        return reg

    def test_simple_subset(self):
        """y ~ x1 + x2 from generation y ~ x1 + x2 + x3."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3"])
        effects, random_effects = _extract_test_formula_effects("y ~ x1 + x2", registry)
        assert effects == ["x1", "x2"]
        assert random_effects == []

    def test_single_variable(self):
        """y ~ x1 from generation y ~ x1 + x2 + x3."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3"])
        effects, random_effects = _extract_test_formula_effects("y ~ x1", registry)
        assert effects == ["x1"]

    def test_with_interaction(self):
        """y ~ x1 + x2 + x1:x2 from generation y ~ x1 + x2 + x3 + x1:x2."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3", "x1:x2"])
        effects, _ = _extract_test_formula_effects("y ~ x1 + x2 + x1:x2", registry)
        assert effects == ["x1", "x2", "x1:x2"]

    def test_interaction_omitted(self):
        """y ~ x1 + x2 from generation y ~ x1 + x2 + x1:x2."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x1:x2"])
        effects, _ = _extract_test_formula_effects("y ~ x1 + x2", registry)
        assert effects == ["x1", "x2"]

    def test_factor_expands_to_dummies(self):
        """y ~ x1 + gender from generation y ~ x1 + x2 + gender."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(
            ["x1", "x2", "gender[F]", "gender[Other]"],
            factor_names=["gender"],
            factor_dummies={
                "gender[F]": {"factor_name": "gender", "level": "F"},
                "gender[Other]": {"factor_name": "gender", "level": "Other"},
            },
        )
        effects, _ = _extract_test_formula_effects("y ~ x1 + gender", registry)
        assert effects == ["x1", "gender[F]", "gender[Other]"]

    def test_factor_omitted(self):
        """y ~ x1 from generation y ~ x1 + gender."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(
            ["x1", "gender[F]", "gender[Other]"],
            factor_names=["gender"],
            factor_dummies={
                "gender[F]": {"factor_name": "gender", "level": "F"},
                "gender[Other]": {"factor_name": "gender", "level": "Other"},
            },
        )
        effects, _ = _extract_test_formula_effects("y ~ x1", registry)
        assert effects == ["x1"]

    def test_with_random_effects(self):
        """y ~ x1 + (1|school) extracts random effects."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2"])
        effects, random_effects = _extract_test_formula_effects(
            "y ~ x1 + (1|school)", registry
        )
        assert effects == ["x1"]
        assert len(random_effects) == 1
        assert random_effects[0]["grouping_var"] == "school"

    def test_star_operator_expands(self):
        """y ~ x1*x2 expands to x1 + x2 + x1:x2."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3", "x1:x2"])
        effects, _ = _extract_test_formula_effects("y ~ x1*x2", registry)
        assert effects == ["x1", "x2", "x1:x2"]

    def test_equals_sign_formula(self):
        """y = x1 + x2 works same as y ~ x1 + x2."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3"])
        effects, _ = _extract_test_formula_effects("y = x1 + x2", registry)
        assert effects == ["x1", "x2"]

    def test_preserves_registry_order(self):
        """Effects returned in registry order, not formula order."""
        from mcpower.utils.test_formula_utils import _extract_test_formula_effects

        registry = self._make_registry(["x1", "x2", "x3", "x1:x2"])
        # Formula lists x2 before x1
        effects, _ = _extract_test_formula_effects("y ~ x2 + x1", registry)
        assert effects == ["x1", "x2"]  # registry order preserved


class TestComputeTestColumnIndices:
    """Test _compute_test_column_indices helper."""

    def test_subset_two_of_three(self):
        """Selecting 2 of 3 effects gives correct indices."""
        from mcpower.utils.test_formula_utils import _compute_test_column_indices

        all_effect_names = ["x1", "x2", "x3"]
        test_effect_names = ["x1", "x2"]
        result = _compute_test_column_indices(all_effect_names, test_effect_names)
        assert list(result) == [0, 1]

    def test_skip_middle(self):
        """Selecting first and last of 3 effects."""
        from mcpower.utils.test_formula_utils import _compute_test_column_indices

        all_effect_names = ["x1", "x2", "x3"]
        test_effect_names = ["x1", "x3"]
        result = _compute_test_column_indices(all_effect_names, test_effect_names)
        assert list(result) == [0, 2]

    def test_single_effect(self):
        """Single effect selected."""
        from mcpower.utils.test_formula_utils import _compute_test_column_indices

        all_effect_names = ["x1", "x2", "x3"]
        test_effect_names = ["x2"]
        result = _compute_test_column_indices(all_effect_names, test_effect_names)
        assert list(result) == [1]

    def test_all_effects_returns_all_indices(self):
        """Selecting all effects returns full range."""
        from mcpower.utils.test_formula_utils import _compute_test_column_indices

        all_effect_names = ["x1", "x2", "x3"]
        test_effect_names = ["x1", "x2", "x3"]
        result = _compute_test_column_indices(all_effect_names, test_effect_names)
        assert list(result) == [0, 1, 2]

    def test_with_interactions(self):
        """Interaction effects have correct indices."""
        from mcpower.utils.test_formula_utils import _compute_test_column_indices

        all_effect_names = ["x1", "x2", "x3", "x1:x2"]
        test_effect_names = ["x1", "x2", "x1:x2"]
        result = _compute_test_column_indices(all_effect_names, test_effect_names)
        assert list(result) == [0, 1, 3]


class TestRemapTargetIndices:
    """Test _remap_target_indices helper."""

    def test_simple_remap(self):
        """Target indices remapped to positions within test columns."""
        from mcpower.utils.test_formula_utils import _remap_target_indices

        # Original target_indices: [0, 1] (x1, x2 in full model)
        # test_column_indices: [0, 1] (x1, x2 at positions 0, 1 in X_expanded)
        # In X_test, x1 is at 0, x2 is at 1 -> remapped: [0, 1]
        original = np.array([0, 1])
        test_cols = np.array([0, 1])
        result = _remap_target_indices(original, test_cols)
        assert list(result) == [0, 1]

    def test_remap_with_gap(self):
        """Target indices remapped when test columns skip positions."""
        from mcpower.utils.test_formula_utils import _remap_target_indices

        # Full model: [x1=0, x2=1, x3=2, x1:x2=3]
        # Test model: [x1=0, x1:x2=3] -> X_test columns at [0, 3]
        # target_test="x1" -> original target_indices=[0]
        # In X_test, x1 is at position 0 -> remapped: [0]
        original = np.array([0])
        test_cols = np.array([0, 3])
        result = _remap_target_indices(original, test_cols)
        assert list(result) == [0]

    def test_remap_target_at_end(self):
        """Target index that moves to different position in X_test."""
        from mcpower.utils.test_formula_utils import _remap_target_indices

        # Full model: [x1=0, x2=1, x3=2]
        # Test model: [x2=1, x3=2] -> test_column_indices=[1, 2]
        # target_test="x3" -> original target_indices=[2]
        # In X_test, x3 is at position 1 (second column) -> remapped: [1]
        original = np.array([2])
        test_cols = np.array([1, 2])
        result = _remap_target_indices(original, test_cols)
        assert list(result) == [1]


class TestPrepareMetadataWithTestFormula:
    """Integration test: prepare_metadata with test_formula_effects."""

    def test_metadata_has_test_indices_when_provided(self):
        from mcpower import MCPower
        from mcpower.core.simulation import prepare_metadata

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        metadata = prepare_metadata(model, ["x1", "x2"], test_formula_effects=["x1", "x2"])
        assert metadata.test_column_indices is not None
        assert list(metadata.test_column_indices) == [0, 1]
        assert metadata.test_target_indices is not None
        assert metadata.test_effect_count == 2

    def test_metadata_no_test_indices_by_default(self):
        from mcpower import MCPower
        from mcpower.core.simulation import prepare_metadata

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        metadata = prepare_metadata(model, ["x1", "x2"])
        assert metadata.test_column_indices is None

    def test_remap_skips_targets_not_in_test_formula(self):
        from mcpower import MCPower
        from mcpower.core.simulation import prepare_metadata

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        # target_tests = all 3, but test formula only has x1, x2
        metadata = prepare_metadata(model, ["x1", "x2", "x3"], test_formula_effects=["x1", "x2"])
        # test_target_indices should only have indices for x1 and x2 in X_test
        assert len(metadata.test_target_indices) == 2


class TestParseTargetTestsWithTestFormula:
    """Test _parse_target_tests limits 'all' when test_formula is active."""

    def test_all_expands_to_test_formula_effects_only(self):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        result = model._parse_target_tests("all", test_formula_effects=["x1", "x2"])
        assert "x3" not in result
        assert "x1" in result
        assert "x2" in result
        assert "overall" in result

    def test_explicit_target_not_in_test_formula_raises(self):
        from mcpower import MCPower

        import pytest

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        with pytest.raises(ValueError, match="x3"):
            model._parse_target_tests("x3", test_formula_effects=["x1", "x2"])

    def test_overall_always_allowed(self):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        result = model._parse_target_tests("overall", test_formula_effects=["x1", "x2"])
        assert "overall" in result

    def test_no_test_formula_uses_all_effects(self):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.2")
        model._apply()

        result = model._parse_target_tests("all")
        assert "x1" in result
        assert "x2" in result
        assert "x3" in result
