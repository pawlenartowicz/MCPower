"""
Tests for LinearRegression class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import mcpower


class TestLinearRegressionBasics:
    """Test basic LinearRegression functionality."""

    def test_model_type(self):
        """Test model_type property."""
        model = mcpower.LinearRegression("y = x")
        assert model.model_type == "Linear Regression"

    def test_inheritance(self):
        """Test LinearRegression inherits from MCPowerBase."""
        model = mcpower.LinearRegression("y = x")
        assert hasattr(model, "set_effects")
        assert hasattr(model, "set_variable_type")
        assert hasattr(model, "find_power")
        assert hasattr(model, "find_sample_size")

    def test_basic_initialization(self):
        """Test basic model creation."""
        model = mcpower.LinearRegression("outcome = treatment + age")

        assert model.equation == "outcome = treatment + age"
        assert model.model_type == "Linear Regression"
        assert len(model.effects) == 2  # treatment, age


class TestSimplePowerAnalysis:
    """Test basic power analysis workflows."""

    def test_simple_power_calculation(self, simple_model, small_simulation_settings):
        """Test basic power calculation."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])

        simple_model.set_effects("treatment=0.4")

        result = simple_model.find_power(
            sample_size=100,
            target_test="treatment",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "model" in result
        assert "results" in result
        assert result["model"]["sample_size"] == 100
        assert "individual_powers" in result["results"]
        assert "treatment" in result["results"]["individual_powers"]

    def test_sample_size_calculation(self, simple_model, small_simulation_settings):
        """Test basic sample size calculation."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])

        simple_model.set_effects("treatment=0.4")

        result = simple_model.find_sample_size(
            target_test="treatment",
            from_size=50,
            to_size=150,
            by=50,
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "model" in result
        assert "results" in result
        assert "first_achieved" in result["results"]
        assert "treatment" in result["results"]["first_achieved"]


class TestMultipleEffects:
    """Test models with multiple effects."""

    def test_multiple_main_effects(self, small_simulation_settings):
        """Test model with multiple main effects."""
        model = mcpower.LinearRegression("y = x1 + x2 + x3")
        model.set_variable_type("x1=binary, x2=binary")
        model.set_effects("x1=0.4, x2=0.3, x3=0.5")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=120,
            target_test="all",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        powers = result["results"]["individual_powers"]
        assert "overall" in powers
        assert "x1" in powers
        assert "x2" in powers
        assert "x3" in powers

    def test_interaction_effects(self, small_simulation_settings):
        """Test model with interaction effects."""
        model = mcpower.LinearRegression("y = x1 + x2 + x1:x2")
        model.set_variable_type("x1=binary")
        model.set_effects("x1=0.4, x2=0.3, x1:x2=0.2")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=150,
            target_test="x1:x2",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "x1:x2" in result["results"]["individual_powers"]


class TestCorrections:
    """Test multiple testing corrections."""

    def test_bonferroni_correction(self, sample_model, small_simulation_settings):
        """Test Bonferroni correction."""
        sample_model.set_simulations(small_simulation_settings["n_simulations"])
        sample_model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")

        result = sample_model.find_power(
            sample_size=200,
            target_test="all",
            correction="Bonferroni",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert result["model"]["correction"] == "Bonferroni"
        assert "individual_powers_corrected" in result["results"]

    def test_bh_correction(self, sample_model, small_simulation_settings):
        """Test Benjamini-Hochberg correction."""
        sample_model.set_simulations(small_simulation_settings["n_simulations"])
        sample_model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")

        result = sample_model.find_power(
            sample_size=200,
            target_test="all",
            correction="Benjamini-Hochberg",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert result["model"]["correction"] == "Benjamini-Hochberg"
        assert "individual_powers_corrected" in result["results"]

    def test_invalid_correction(self, simple_model):
        """Test invalid correction method."""
        with pytest.raises(ValueError):
            simple_model.find_power(
                sample_size=100, correction="InvalidMethod", print_results=False
            )


class TestCorrelatedPredictors:
    """Test models with correlated predictors."""

    def test_string_correlations(self, small_simulation_settings):
        """Test power with string-specified correlations."""
        model = mcpower.LinearRegression("y = x1 + x2 + x3")
        model.set_effects("x1=0.4, x2=0.3, x3=0.5")
        model.set_correlations("corr(x1, x2)=0.5, corr(x1, x3)=0.3")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=150,
            target_test="x1",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "x1" in result["results"]["individual_powers"]

    def test_matrix_correlations(
        self, correlation_matrix_3x3, small_simulation_settings
    ):
        """Test power with matrix-specified correlations."""
        model = mcpower.LinearRegression("y = x1 + x2 + x3")
        model.set_effects("x1=0.4, x2=0.3, x3=0.5")
        model.set_correlations(correlation_matrix_3x3)
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=150,
            target_test="all",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        powers = result["results"]["individual_powers"]
        assert len(powers) == 4  # overall + 3 predictors


class TestVariableTypes:
    """Test different variable distributions."""

    def test_binary_variables(self, small_simulation_settings):
        """Test model with binary variables."""
        model = mcpower.LinearRegression("y = treatment + group")
        model.set_variable_type("treatment=binary, group=(binary,0.3)")
        model.set_effects("treatment=0.5, group=0.3")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=100,
            target_test="all",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "treatment" in result["results"]["individual_powers"]
        assert "group" in result["results"]["individual_powers"]

    def test_skewed_variables(self, small_simulation_settings):
        """Test model with skewed distributions."""
        model = mcpower.LinearRegression("y = income + age")
        model.set_variable_type("income=right_skewed, age=left_skewed")
        model.set_effects("income=0.4, age=0.2")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=120,
            target_test="income",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "income" in result["results"]["individual_powers"]


class TestHeterogeneityEffects:
    """Test heterogeneity and heteroskedasticity."""

    def test_heterogeneity(self, simple_model, small_simulation_settings):
        """Test effect heterogeneity."""
        simple_model.set_heterogeneity(0.2)
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        result = simple_model.find_power(
            sample_size=100,
            target_test="treatment",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "treatment" in result["results"]["individual_powers"]

    def test_heteroskedasticity(self, simple_model, small_simulation_settings):
        """Test heteroskedasticity."""
        simple_model.set_heteroskedasticity(0.3)
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        result = simple_model.find_power(
            sample_size=100,
            target_test="treatment",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "treatment" in result["results"]["individual_powers"]


class TestScenarioAnalysis:
    """Test scenario-based analysis."""

    def test_scenario_power(self, simple_model, small_simulation_settings):
        """Test power analysis with scenarios."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        result = simple_model.find_power(
            sample_size=120,
            target_test="treatment",
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert result["analysis_type"] == "power"
        assert "scenarios" in result
        assert "optimistic" in result["scenarios"]
        assert "realistic" in result["scenarios"]
        assert "doomer" in result["scenarios"]

    def test_scenario_sample_size(self, simple_model, small_simulation_settings):
        """Test sample size analysis with scenarios."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        result = simple_model.find_sample_size(
            target_test="treatment",
            from_size=80,
            to_size=120,
            by=20,
            scenarios=True,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert result["analysis_type"] == "sample_size"
        assert "scenarios" in result
        assert "optimistic" in result["scenarios"]


class TestDataUploadIntegration:
    """Test uploaded data integration."""

    def test_power_with_uploaded_data(self, cars_data, small_simulation_settings):
        """Test power analysis with uploaded data."""
        model = mcpower.LinearRegression("mpg = hp + wt")
        model.upload_own_data(cars_data)
        model.set_effects("hp=-0.3, wt=-0.5")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=50,
            target_test="hp",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "hp" in result["results"]["individual_powers"]

    def test_sample_size_with_uploaded_data(self, cars_data, small_simulation_settings):
        """Test sample size calculation with uploaded data."""
        model = mcpower.LinearRegression("mpg = hp + wt + am")
        model.upload_own_data(cars_data)
        model.set_effects("hp=-0.3, wt=-0.5, am=0.4")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_sample_size(
            target_test="am",
            from_size=40,
            to_size=80,
            by=20,
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "am" in result["results"]["first_achieved"]


class TestTestFormula:
    """Test test_formula functionality."""

    def test_subset_testing(self, small_simulation_settings):
        """Test testing subset of effects."""
        model = mcpower.LinearRegression("y = x1 + x2 + x3 + x1:x2")
        model.set_effects("x1=0.4, x2=0.3, x3=0.5, x1:x2=0.2")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=150,
            target_test="x1",
            test_formula="x1 + x2",  # Only test x1 and x2
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "x1" in result["results"]["individual_powers"]

    def test_interaction_only_testing(self, sample_model, small_simulation_settings):
        """Test testing only interaction effects."""
        sample_model.set_simulations(small_simulation_settings["n_simulations"])
        sample_model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")

        result = sample_model.find_power(
            sample_size=200,
            target_test="x1:x2",
            test_formula="x1:x2",  # Only test interaction
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        assert "x1:x2" in result["results"]["individual_powers"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_effect_sizes_set(self):
        """Test error when effect sizes not set."""
        model = mcpower.LinearRegression("y = x")

        with pytest.raises(ValueError, match="Effect sizes must be set"):
            model.find_power(sample_size=100, print_results=False)

    def test_invalid_target_test(self, simple_model):
        """Test invalid target test specification."""
        with pytest.raises(ValueError):
            simple_model.set_effects("treatment=0.4")
            simple_model.find_power(
                sample_size=100, target_test="nonexistent_effect", print_results=False
            )

    def test_zero_effect_sizes(self, small_simulation_settings):
        """Test model with zero effect sizes."""
        model = mcpower.LinearRegression("y = x")
        model.set_effects("x=0.0")
        model.set_simulations(small_simulation_settings["n_simulations"])

        result = model.find_power(
            sample_size=100,
            target_test="x",
            scenarios=False,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        # Power should be low with zero effect
        assert result["results"]["individual_powers"]["x"] < 10


class TestSummaryFormats:
    """Test different summary formats."""

    def test_short_summary(self, simple_model, small_simulation_settings):
        """Test short summary format."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        # Should not raise errors
        simple_model.find_power(
            sample_size=100,
            target_test="treatment",
            summary="short",
            scenarios=False,
            print_results=True,
        )

    def test_long_summary(self, simple_model, small_simulation_settings):
        """Test long summary format."""
        simple_model.set_simulations(small_simulation_settings["n_simulations"])
        simple_model.set_effects("treatment=0.4")
        # Should not raise errors
        simple_model.find_power(
            sample_size=100,
            target_test="treatment",
            summary="long",
            scenarios=False,
            print_results=True,
        )
