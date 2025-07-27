"""
Integration tests for MCPower package.
Tests full workflows end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
import mcpower


class TestBasicWorkflows:
    """Test complete basic workflows."""

    def test_simple_power_workflow(self):
        """Test complete simple power analysis."""
        model = (
            mcpower.LinearRegression("outcome = treatment + age")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.5, age=0.3")
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=100,
            target_test="treatment",
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "treatment" in result["results"]["individual_powers"]
        assert 0 <= result["results"]["individual_powers"]["treatment"] <= 100

    def test_sample_size_workflow(self):
        """Test complete sample size analysis."""
        model = (
            mcpower.LinearRegression("y = x1 + x2")
            .set_variable_type("x1=binary")
            .set_effects("x1=0.4, x2=0.3")
            .set_simulations(150)
        )

        result = model.find_sample_size(
            target_test="x1",
            from_size=50,
            to_size=150,
            by=50,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "x1" in result["results"]["first_achieved"]

    def test_interaction_workflow(self):
        """Test interaction analysis workflow."""
        model = (
            mcpower.LinearRegression("response = treatment*age + baseline")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.4, age=0.2, treatment:age=0.3, baseline=0.5")
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=200,
            target_test="treatment:age",
            print_results=False,
            return_results=True,
        )

        assert "treatment:age" in result["results"]["individual_powers"]


class TestMethodChaining:
    """Test method chaining workflows."""

    def test_full_chain(self):
        """Test complete method chaining."""
        result = (
            mcpower.LinearRegression("satisfaction = treatment + motivation")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.6, motivation=0.4")
            .set_power(85)
            .set_alpha(0.01)
            .set_simulations(250)
            .set_heterogeneity(0.1)
            .find_power(
                sample_size=120,
                target_test="all",
                print_results=False,
                return_results=True,
            )
        )

        assert result["model"]["target_power"] == 85
        assert result["model"]["alpha"] == 0.01
        assert len(result["results"]["individual_powers"]) >= 2

    def test_correlations_chain(self):
        """Test chaining with correlations."""
        model = (
            mcpower.LinearRegression("y = x1 + x2 + x3")
            .set_correlations("corr(x1,x2)=0.4, corr(x1,x3)=0.3")
            .set_effects("x1=0.4, x2=0.3, x3=0.5")
            .set_simulations(200)
        )

        result = model.find_sample_size(
            target_test="all",
            from_size=80,
            to_size=160,
            by=40,
            print_results=False,
            return_results=True,
        )

        assert len(result["results"]["first_achieved"]) >= 3


class TestCorrections:
    """Test multiple testing corrections."""

    def test_bonferroni_workflow(self):
        """Test Bonferroni correction workflow."""
        model = (
            mcpower.LinearRegression("outcome = x1 + x2 + x3 + x4")
            .set_effects("x1=0.4, x2=0.3, x3=0.0, x4=0.0")
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=150,
            target_test="all",
            correction="Bonferroni",
            print_results=False,
            return_results=True,
        )

        assert "individual_powers_corrected" in result["results"]
        assert result["model"]["correction"] == "Bonferroni"

    def test_bh_correction_workflow(self):
        """Test Benjamini-Hochberg workflow."""
        model = (
            mcpower.LinearRegression("y = a + b + c")
            .set_effects("a=0.5, b=0.3, c=0.2")
            .set_simulations(200)
        )

        result = model.find_sample_size(
            target_test="a",
            from_size=60,
            to_size=120,
            by=30,
            correction="BH",
            print_results=False,
            return_results=True,
        )

        assert "first_achieved_corrected" in result["results"]


class TestScenarios:
    """Test scenario analysis workflows."""

    def test_scenario_power(self):
        """Test scenario power analysis."""
        model = (
            mcpower.LinearRegression("outcome = treatment + covariate")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.4, covariate=0.3")
            .set_simulations(150)
        )

        result = model.find_power(
            sample_size=100,
            target_test="treatment",
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert result["analysis_type"] == "power"
        assert "optimistic" in result["scenarios"]
        assert "realistic" in result["scenarios"]
        assert "doomer" in result["scenarios"]

    def test_scenario_sample_size(self):
        """Test scenario sample size analysis."""
        model = (
            mcpower.LinearRegression("y = treatment")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.5")
            .set_simulations(120)
        )

        result = model.find_sample_size(
            target_test="treatment",
            from_size=50,
            to_size=150,
            by=50,
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert result["analysis_type"] == "sample_size"
        assert "scenarios" in result

    def test_custom_scenarios(self):
        """Test custom scenario configurations."""
        model = (
            mcpower.LinearRegression("y = x")
            .set_effects("x=0.4")
            .set_scenario_configs(
                {"realistic": {"heterogeneity": 0.3}, "doomer": {"heterogeneity": 0.6}}
            )
            .set_simulations(120)
        )

        result = model.find_power(
            sample_size=80, scenarios=True, print_results=False, return_results=True
        )

        assert "scenarios" in result


class TestDataUpload:
    """Test uploaded data workflows."""

    def test_cars_data_workflow(self, cars_data):
        """Test workflow with cars dataset."""
        model = (
            mcpower.LinearRegression("mpg = hp + wt + am")
            .upload_own_data(cars_data)
            .set_effects("hp=-0.3, wt=-0.5, am=0.4")
            .set_simulations(150)
        )

        result = model.find_power(
            sample_size=50, target_test="hp", print_results=False, return_results=True
        )

        assert "hp" in result["results"]["individual_powers"]

    def test_partial_data_upload(self, cars_data):
        """Test workflow with partial data matching."""
        # Use only subset of variables
        partial_data = cars_data[["hp", "wt"]].copy()

        model = (
            mcpower.LinearRegression("mpg = hp + wt + am")
            .upload_own_data(partial_data)
            .set_variable_type("am=binary")
            .set_effects("hp=-0.3, wt=-0.5, am=0.4")
            .set_simulations(150)
        )

        result = model.find_sample_size(
            target_test="am",
            from_size=40,
            to_size=80,
            by=20,
            print_results=False,
            return_results=True,
        )

        assert "am" in result["results"]["first_achieved"]


class TestComplexModels:
    """Test complex model workflows."""

    def test_many_predictors(self):
        """Test model with many predictors."""
        vars_str = " + ".join([f"x{i}" for i in range(1, 8)])
        effects_str = ", ".join([f"x{i}=0.{i+2}" for i in range(1, 8)])

        model = (
            mcpower.LinearRegression(f"y = {vars_str}")
            .set_variable_type("x1=binary, x2=binary")
            .set_effects(effects_str)
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=300, target_test="x1", print_results=False, return_results=True
        )

        assert "x1" in result["results"]["individual_powers"]

    def test_three_way_interaction(self):
        """Test three-way interaction."""
        model = (
            mcpower.LinearRegression("outcome = a*b*c")
            .set_variable_type("a=binary, b=binary")
            .set_effects("a=0.4, b=0.3, c=0.2, a:b=0.2, a:c=0.1, b:c=0.1, a:b:c=0.3")
            .set_simulations(150)
        )

        result = model.find_sample_size(
            target_test="a:b:c",
            from_size=200,
            to_size=400,
            by=100,
            print_results=False,
            return_results=True,
        )

        assert "a:b:c" in result["results"]["first_achieved"]

    def test_mixed_distributions(self):
        """Test model with mixed variable distributions."""
        model = (
            mcpower.LinearRegression(
                "response = normal_var + binary_var + skewed_var + uniform_var"
            )
            .set_variable_type(
                "binary_var=binary, skewed_var=right_skewed, uniform_var=uniform"
            )
            .set_effects(
                "normal_var=0.3, binary_var=0.4, skewed_var=0.3, uniform_var=0.2"
            )
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=150, target_test="all", print_results=False, return_results=True
        )

        assert len(result["results"]["individual_powers"]) >= 4


class TestErrorHandling:
    """Test error handling in workflows."""

    def test_missing_effects_error(self):
        """Test error when effects not set."""
        model = mcpower.LinearRegression("y = x")

        with pytest.raises(ValueError, match="Effect sizes must be set"):
            model.find_power(sample_size=100, print_results=False)

    def test_invalid_target_error(self):
        """Test error with invalid target test."""
        model = mcpower.LinearRegression("y = x").set_effects("x=0.5")

        with pytest.raises(ValueError):
            model.find_power(
                sample_size=100, target_test="nonexistent", print_results=False
            )

    def test_invalid_sample_size_range(self):
        """Test error with invalid sample size range."""
        model = mcpower.LinearRegression("y = x").set_effects("x=0.5")

        with pytest.raises(ValueError):
            model.find_sample_size(from_size=200, to_size=100, print_results=False)

    def test_invalid_correlation_matrix(self):
        """Test error with invalid correlation matrix."""
        model = mcpower.LinearRegression("y = x1 + x2")

        invalid_matrix = np.array([[1.0, 1.5], [1.5, 1.0]])

        with pytest.raises(ValueError):
            model.set_correlations(invalid_matrix)


class TestReproducibility:
    """Test reproducibility across workflows."""

    def test_seed_reproducibility(self):
        """Test same results with same seed."""

        def create_and_run():
            return (
                mcpower.LinearRegression("y = x")
                .set_effects("x=0.5")
                .set_seed(42)
                .set_simulations(100)
                .find_power(
                    sample_size=80,
                    target_test="x",
                    print_results=False,
                    return_results=True,
                )
            )

        result1 = create_and_run()
        result2 = create_and_run()

        assert (
            result1["results"]["individual_powers"]["x"]
            == result2["results"]["individual_powers"]["x"]
        )

    def test_different_seeds(self):
        """Test different results with different seeds."""
        model1 = (
            mcpower.LinearRegression("y = x")
            .set_effects("x=0.4")
            .set_seed(123)
            .set_simulations(150)
        )

        model2 = (
            mcpower.LinearRegression("y = x")
            .set_effects("x=0.4")
            .set_seed(456)
            .set_simulations(150)
        )

        result1 = model1.find_power(
            sample_size=100, print_results=False, return_results=True
        )
        result2 = model2.find_power(
            sample_size=100, print_results=False, return_results=True
        )

        # Results should be different (with high probability)
        assert (
            result1["results"]["individual_powers"]["overall"]
            != result2["results"]["individual_powers"]["overall"]
        )


class TestPerformance:
    """Test performance with realistic sizes."""

    def test_large_sample_size(self):
        """Test with large sample size."""
        model = (
            mcpower.LinearRegression("y = x1 + x2")
            .set_effects("x1=0.3, x2=0.2")
            .set_simulations(100)
        )  # Reduced for speed

        result = model.find_power(
            sample_size=5000, target_test="x1", print_results=False, return_results=True
        )

        assert result is not None

    def test_many_simulations(self):
        """Test with many simulations."""
        model = (
            mcpower.LinearRegression("y = treatment")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.4")
            .set_simulations(2000)
        )

        result = model.find_power(
            sample_size=100,
            target_test="treatment",
            print_results=False,
            return_results=True,
        )

        assert result["model"]["n_simulations"] == 2000


class TestRealWorldScenarios:
    """Test scenarios based on package examples."""

    def test_rct_scenario(self):
        """Test RCT scenario from examples."""
        model = (
            mcpower.LinearRegression("outcome = treatment + age + baseline_score")
            .set_variable_type("treatment=binary")
            .set_effects("treatment=0.5, age=0.2, baseline_score=0.7")
            .set_simulations(200)
        )

        # Power analysis
        power_result = model.find_power(
            sample_size=160,
            target_test="treatment",
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        # Sample size analysis
        size_result = model.find_sample_size(
            target_test="treatment",
            from_size=100,
            to_size=200,
            by=25,
            print_results=False,
            return_results=True,
        )

        assert (
            "treatment"
            in power_result["scenarios"]["optimistic"]["results"]["individual_powers"]
        )
        assert "treatment" in size_result["results"]["first_achieved"]

    def test_correlation_study(self):
        """Test correlated predictors scenario."""
        model = (
            mcpower.LinearRegression("wellbeing = income + education + social_support")
            .set_correlations(
                "corr(income, education)=0.5, corr(income, social_support)=0.3"
            )
            .set_effects("income=0.4, education=0.3, social_support=0.6")
            .set_simulations(200)
        )

        result = model.find_sample_size(
            target_test="all",
            from_size=150,
            to_size=300,
            by=50,
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert len(result["scenarios"]) >= 3
        assert (
            "income" in result["scenarios"]["optimistic"]["results"]["first_achieved"]
        )

    def test_marketing_ab_test(self):
        """Test A/B test with interaction scenario."""
        model = (
            mcpower.LinearRegression(
                "conversion = treatment + user_type + treatment*user_type"
            )
            .set_variable_type("treatment=binary, user_type=binary")
            .set_effects("treatment=0.4, user_type=0.3, treatment:user_type=0.5")
            .set_simulations(200)
        )

        result = model.find_power(
            sample_size=400,
            target_test="treatment:user_type",
            correction="BH",
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert (
            "treatment:user_type"
            in result["scenarios"]["optimistic"]["results"]["individual_powers"]
        )
