"""Integration tests for Phase 2: random slopes and nested random effects.

Tests the full pipeline: model setup -> data generation -> fitting -> power.
"""

import numpy as np
import pytest

from mcpower import MCPower

pytestmark = pytest.mark.lme


class TestRandomSlopesIntegration:
    """End-to-end tests for random slopes through the full pipeline."""

    def test_slope_model_setup(self):
        """Model with (1+x|school) should configure without errors."""
        model = MCPower("y ~ x1 + (1 + x1|school)")
        model.set_cluster(
            "school", ICC=0.2, n_clusters=20,
            random_slopes=["x1"], slope_variance=0.1,
            slope_intercept_corr=0.3,
        )
        model.set_effects("x1=0.5")
        model._apply()

        # Verify cluster spec was configured correctly
        spec = model._registry._cluster_specs["school"]
        assert spec.q == 2
        assert spec.random_slope_vars == ["x1"]
        assert spec.slope_variance == 0.1
        assert spec.slope_intercept_corr == 0.3
        assert spec.G_matrix is not None
        assert spec.G_matrix.shape == (2, 2)

    def test_slope_model_data_generation(self):
        """Random slopes data generation should produce correct shapes."""
        from mcpower.stats.data_generation import _generate_random_effects

        cluster_specs = {
            "school": {
                "n_clusters": 20,
                "cluster_size": 50,
                "tau_squared": 0.25,
                "q": 2,
                "G_matrix": np.array([[0.25, 0.06], [0.06, 0.1]]),
                "random_slope_vars": ["x1"],
                "slope_variance": 0.1,
                "slope_intercept_corr": 0.3,
            }
        }
        N = 1000
        X_nf = np.random.randn(N, 2)
        re = _generate_random_effects(N, cluster_specs, X_nf, ["x1", "x2"], sim_seed=42)

        assert re.intercept_columns.shape == (N, 1)
        assert re.slope_contribution.shape == (N,)
        assert not np.allclose(re.slope_contribution, 0)
        assert "school" in re.Z_matrices
        assert re.Z_matrices["school"].shape == (N, 2)
        assert "school" in re.cluster_ids_dict

    def test_slope_model_find_power(self):
        """find_power should work for a random slope model."""
        model = MCPower("y ~ x1 + (1 + x1|school)")
        model.set_cluster(
            "school", ICC=0.15, n_clusters=20,
            random_slopes=["x1"], slope_variance=0.05,
            slope_intercept_corr=0.2,
        )
        model.set_effects("x1=0.5")
        model.set_simulations(30)
        model.set_max_failed_simulations(0.30)

        result = model.find_power(sample_size=1000, return_results=True)

        assert result is not None
        assert "results" in result
        assert "individual_powers" in result["results"]

    def test_slope_model_power_reasonable(self):
        """Power with random slopes should be between 0 and 100."""
        model = MCPower("y ~ x1 + (1 + x1|school)")
        model.set_cluster(
            "school", ICC=0.15, n_clusters=30,
            random_slopes=["x1"], slope_variance=0.05,
            slope_intercept_corr=0.0,
        )
        model.set_effects("x1=0.5")
        model.set_simulations(50)
        model.set_max_failed_simulations(0.30)

        result = model.find_power(sample_size=1500, return_results=True)

        power_overall = result["results"]["individual_powers"]["overall"]
        assert 0 <= power_overall <= 100


class TestNestedIntegration:
    """End-to-end tests for nested random effects through the full pipeline."""

    def test_nested_model_setup(self):
        """Model with (1|school/classroom) should configure without errors."""
        model = MCPower("y ~ treatment + (1|school/classroom)")
        model.set_cluster("school", ICC=0.15, n_clusters=10)
        model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
        model.set_effects("treatment=0.5")
        model._apply()

        assert "school" in model._registry._cluster_specs
        assert "school:classroom" in model._registry._cluster_specs

        school_spec = model._registry._cluster_specs["school"]
        classroom_spec = model._registry._cluster_specs["school:classroom"]
        assert school_spec.n_clusters == 10
        assert classroom_spec.n_clusters == 30
        assert classroom_spec.parent_var == "school"
        assert classroom_spec.n_per_parent == 3

    def test_nested_data_generation(self):
        """Nested data generation should produce hierarchical structure."""
        from mcpower.stats.data_generation import _generate_random_effects

        cluster_specs = {
            "school": {
                "n_clusters": 10,
                "cluster_size": 150,
                "tau_squared": 0.15,
                "q": 1,
            },
            "school:classroom": {
                "n_clusters": 30,
                "cluster_size": 50,
                "tau_squared": 0.10,
                "q": 1,
                "parent_var": "school",
                "n_per_parent": 3,
            },
        }
        N = 1500
        X_nf = np.random.randn(N, 1)
        re = _generate_random_effects(N, cluster_specs, X_nf, ["treatment"], sim_seed=42)

        assert re.intercept_columns.shape == (N, 2)
        assert "school" in re.cluster_ids_dict
        assert "school:classroom" in re.cluster_ids_dict
        assert re.child_to_parent is not None
        assert re.K_parent == 10
        assert re.K_child == 30
        assert len(re.child_to_parent) == 30
        assert len(np.unique(re.child_to_parent)) == 10

    def test_nested_model_find_power(self):
        """find_power should work for a nested random effects model."""
        model = MCPower("y ~ treatment + (1|school/classroom)")
        model.set_cluster("school", ICC=0.15, n_clusters=10)
        model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
        model.set_effects("treatment=0.5")
        model.set_simulations(30)
        model.set_max_failed_simulations(0.30)

        result = model.find_power(sample_size=1500, return_results=True)

        assert result is not None
        assert "results" in result
        assert "individual_powers" in result["results"]

    def test_nested_model_power_reasonable(self):
        """Power for nested model should be reasonable for moderate effect."""
        model = MCPower("y ~ treatment + (1|school/classroom)")
        model.set_cluster("school", ICC=0.10, n_clusters=15)
        model.set_cluster("classroom", ICC=0.10, n_per_parent=4)
        model.set_effects("treatment=0.5")
        model.set_simulations(50)
        model.set_max_failed_simulations(0.30)

        result = model.find_power(sample_size=3000, return_results=True)

        power = result["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100


class TestFormulaParsingIntegration:
    """Test that formulas with slopes/nesting are parsed and configured correctly."""

    def test_formula_with_slopes_parsed(self):
        """(1 + x1 | school) should be parsed as random_slope."""
        model = MCPower("y ~ x1 + x2 + (1 + x1|school)")
        parsed = model._registry._random_effects_parsed
        assert len(parsed) == 1
        assert parsed[0]["type"] == "random_slope"
        assert parsed[0]["slope_vars"] == ["x1"]
        assert parsed[0]["grouping_var"] == "school"

    def test_formula_with_nested_parsed(self):
        """(1|school/classroom) should expand to two intercept terms."""
        model = MCPower("y ~ x1 + (1|school/classroom)")
        parsed = model._registry._random_effects_parsed
        assert len(parsed) == 2
        parent = [p for p in parsed if p.get("parent_var") is None]
        assert len(parent) == 1
        assert parent[0]["grouping_var"] == "school"
        child = [p for p in parsed if p.get("parent_var") is not None]
        assert len(child) == 1
        assert child[0]["grouping_var"] == "school:classroom"
        assert child[0]["parent_var"] == "school"

    def test_simple_intercept_still_works(self):
        """(1|school) should still work as before."""
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(20)

        result = model.find_power(sample_size=1000, return_results=True)
        assert result is not None
        assert "results" in result
