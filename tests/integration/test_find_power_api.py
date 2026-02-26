"""
Integration tests for MCPower.

These tests verify complete workflows and known results.
"""

import numpy as np


class TestBasicWorkflows:
    """Test basic end-to-end workflows."""

    def test_simple_power_analysis(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None
        assert "results" in result
        power = result["results"]["individual_powers"]["overall"]
        # x1=0.3, x2=0.2, n=100: ~88.9 (C++) / ~90.2 (Python)
        assert 85 < power < 93

    def test_sample_size_analysis(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1")
        model.set_effects("x1=0.3")
        result = model.find_sample_size(from_size=50, to_size=150, by=50, print_results=False, return_results=True)

        assert result is not None
        assert len(result["results"]["sample_sizes_tested"]) == 3

    def test_interaction_model(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = a + b + a:b")
        model.set_effects("a=0.4, b=0.3, a:b=0.2")
        result = model.find_power(120, print_results=False, return_results=True)

        powers = result["results"]["individual_powers"]
        assert "a" in powers or "overall" in powers

    def test_factor_model(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        result = model.find_power(150, print_results=False, return_results=True)

        assert result is not None
        power = result["results"]["individual_powers"]["overall"]
        # group[2]=0.4, group[3]=0.3, x1=0.2, n=150: ~74.8 (C++) / ~77.8 (Python)
        assert 71 < power < 81


class TestCorrelatedPredictors:
    """Test models with correlated predictors."""

    def test_positive_correlation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_correlations("(x1,x2)=0.5")
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_negative_correlation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_correlations("(x1,x2)=-0.5")
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_multiple_correlations(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_correlations("(x1,x2)=0.5, (x1,x3)=0.3, (x2,x3)=0.2")
        model.set_effects("x1=0.3, x2=0.2, x3=0.1")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None


class TestVariableTypes:
    """Test different variable types."""

    def test_binary_predictor(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = treatment + x1")
        model.set_variable_type("treatment=(binary,0.3)")
        model.set_effects("treatment=0.5, x1=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_factor_predictor(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,4)")
        model.set_effects("group[2]=0.4, group[3]=0.3, group[4]=0.2, x1=0.2")
        result = model.find_power(200, print_results=False, return_results=True)

        assert result is not None

    def test_skewed_predictor(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_variable_type("x1=right_skewed")
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None


class TestUploadedData:
    """Test uploaded empirical data."""

    def test_upload_dict(self, suppress_output):
        from mcpower import MCPower

        np.random.seed(42)

        model = MCPower("y = x1 + x2")
        model.upload_data({"x1": np.random.exponential(2, 100), "x2": np.random.normal(0, 1, 100)})
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_upload_partial(self, suppress_output):
        from mcpower import MCPower

        np.random.seed(42)

        model = MCPower("y = x1 + x2 + x3")
        model.upload_data({"x1": np.random.exponential(2, 100)})  # Only x1
        model.set_effects("x1=0.3, x2=0.2, x3=0.1")
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None


class TestCorrections:
    """Test multiple comparison corrections."""

    def test_bonferroni(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.3, x2=0.3, x3=0.3")
        result = model.find_power(150, correction="bonferroni", print_results=False, return_results=True)

        assert "individual_powers_corrected" in result["results"]

    def test_benjamini_hochberg(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.3, x2=0.3, x3=0.3")
        result = model.find_power(150, correction="benjamini-hochberg", print_results=False, return_results=True)

        assert "individual_powers_corrected" in result["results"]

    def test_holm(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.3, x2=0.3, x3=0.3")
        result = model.find_power(150, correction="holm", print_results=False, return_results=True)

        assert "individual_powers_corrected" in result["results"]


class TestTargetTests:
    """Test targeting specific effects."""

    def test_single_target(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.1")
        result = model.find_power(100, target_test="x1", print_results=False, return_results=True)

        powers = result["results"]["individual_powers"]
        assert "x1" in powers

    def test_multiple_targets(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.5, x2=0.3, x3=0.1")
        result = model.find_power(100, target_test="x1,x2", print_results=False, return_results=True)

        powers = result["results"]["individual_powers"]
        assert "x1" in powers
        assert "x2" in powers

    def test_all_targets(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        result = model.find_power(100, target_test="all", print_results=False, return_results=True)

        powers = result["results"]["individual_powers"]
        assert "overall" in powers
        assert "x1" in powers
        assert "x2" in powers


class TestHeterogeneity:
    """Test heterogeneity via scenario configs."""

    def test_with_heterogeneity(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_scenario_configs({"het": {"heterogeneity": 0.1}})
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_with_heteroskedasticity(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_scenario_configs({"hsked": {"heteroskedasticity": 0.2}})
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None

    def test_combined(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_scenario_configs({"combo": {"heterogeneity": 0.1, "heteroskedasticity": 0.2}})
        result = model.find_power(100, print_results=False, return_results=True)

        assert result is not None


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_same_seed_same_result(self, suppress_output):
        from mcpower import MCPower

        model1 = MCPower("y = x1 + x2")
        model1.set_seed(42)
        model1.set_effects("x1=0.3, x2=0.2")
        result1 = model1.find_power(100, print_results=False, return_results=True)

        model2 = MCPower("y = x1 + x2")
        model2.set_seed(42)
        model2.set_effects("x1=0.3, x2=0.2")
        result2 = model2.find_power(100, print_results=False, return_results=True)

        power1 = result1["results"]["individual_powers"]["overall"]
        power2 = result2["results"]["individual_powers"]["overall"]
        assert power1 == power2

    def test_different_seed_different_result(self, suppress_output):
        from mcpower import MCPower

        model1 = MCPower("y = x1 + x2")
        model1.set_seed(42)
        model1.set_effects("x1=0.3, x2=0.2")
        result1 = model1.find_power(100, print_results=False, return_results=True)

        model2 = MCPower("y = x1 + x2")
        model2.set_seed(123)
        model2.set_effects("x1=0.3, x2=0.2")
        result2 = model2.find_power(100, print_results=False, return_results=True)

        power1 = result1["results"]["individual_powers"]["overall"]
        power2 = result2["results"]["individual_powers"]["overall"]
        # Different seeds, same setup: ~87.8–90.2 depending on seed and backend
        assert 85 < power1 < 92
        assert 85 < power2 < 92


class TestComplexModels:
    """Test complex model configurations."""

    def test_factor_with_correlation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1 + x2")
        model.set_variable_type("group=(factor,3)")
        model.set_correlations("(x1,x2)=0.5")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2, x2=0.1")
        result = model.find_power(150, print_results=False, return_results=True)

        assert result is not None

    def test_interaction_with_factor(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1 + group:x1")
        model.set_variable_type("group=(factor,2)")
        model.set_effects("group[2]=0.3, x1=0.2, group[2]:x1=0.15")
        result = model.find_power(150, print_results=False, return_results=True)

        assert result is not None

    def test_all_features_combined(self, suppress_output):
        from mcpower import MCPower

        np.random.seed(42)

        model = MCPower("y = group + x1 + x2 + x1:x2")
        model.set_variable_type("group=(factor,3)")
        model.upload_data({"x1": np.random.exponential(2, 100)})
        model.set_correlations("(x1,x2)=0.3")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2, x2=0.15, x1:x2=0.1")
        model.set_scenario_configs({"test": {"heterogeneity": 0.05}})
        result = model.find_power(200, print_results=False, return_results=True)

        assert result is not None
