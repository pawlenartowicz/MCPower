"""
Tests for scenario analysis module.
"""

from unittest.mock import MagicMock

import numpy as np

from mcpower.core.scenarios import (
    ScenarioRunner,
    apply_per_simulation_perturbations,
)


class TestScenarioRunner:
    """Test ScenarioRunner class."""

    def test_get_configs_default(self):
        model = MagicMock()
        runner = ScenarioRunner(model)
        configs = runner.get_configs()
        assert "realistic" in configs
        assert "doomer" in configs

    def test_get_configs_custom(self):
        model = MagicMock()
        custom = {"mild": {"heterogeneity": 0.1}}
        runner = ScenarioRunner(model, custom_configs=custom)
        configs = runner.get_configs()
        assert "mild" in configs
        assert "realistic" not in configs

    def test_run_power_analysis_calls_func(self):
        model = MagicMock()
        runner = ScenarioRunner(model)

        mock_func = MagicMock(
            return_value={
                "model": {"target_tests": ["x1"], "target_power": 80.0, "sample_size": 100, "correction": None},
                "results": {"individual_powers": {"x1": 80.0}, "cumulative_probabilities": {"all": 80.0}},
            }
        )

        result = runner.run_power_analysis(
            sample_size=100,
            target_tests=["x1"],
            correction=None,
            run_find_power_func=mock_func,
            print_results=False,
        )

        # Called 3 times: optimistic + realistic + doomer
        assert mock_func.call_count == 3
        assert "scenarios" in result

    def test_run_power_analysis_print_false(self, capsys):
        model = MagicMock()
        runner = ScenarioRunner(model)

        mock_func = MagicMock(
            return_value={
                "model": {"target_tests": ["x1"], "target_power": 80.0, "sample_size": 100, "correction": None},
                "results": {"individual_powers": {"x1": 80.0}, "cumulative_probabilities": {"all": 80.0}},
            }
        )

        runner.run_power_analysis(
            sample_size=100,
            target_tests=["x1"],
            correction=None,
            run_find_power_func=mock_func,
            print_results=False,
        )

        captured = capsys.readouterr()
        assert "SCENARIO-BASED" not in captured.out

    def test_create_scenario_plots_early_return(self):
        model = MagicMock()
        runner = ScenarioRunner(model)
        # No results data → should return early without error
        runner._create_scenario_plots({"scenarios": {"optimistic": {}}})


class TestApplyPerSimulationPerturbations:
    """Test apply_per_simulation_perturbations function."""

    def test_none_config_passthrough(self):
        corr = np.eye(3)
        var_types = np.array([0, 0, 0], dtype=np.int64)
        p_corr, p_types = apply_per_simulation_perturbations(corr, var_types, None, 42)
        assert np.array_equal(p_corr, corr)
        assert np.array_equal(p_types, var_types)

    def test_correlation_perturbation(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        var_types = np.array([0, 0], dtype=np.int64)
        config = {
            "correlation_noise_sd": 0.3,
            "distribution_change_prob": 0.0,
            "new_distributions": [],
        }
        p_corr, p_types = apply_per_simulation_perturbations(corr, var_types, config, 42)

        # Should be symmetric
        assert np.allclose(p_corr, p_corr.T)
        # Diagonal should be 1
        assert np.allclose(np.diag(p_corr), 1.0)
        # Should be clipped to [-0.8, 0.8]
        off_diag = p_corr[0, 1]
        assert -0.8 <= off_diag <= 0.8

    def test_var_type_perturbation(self):
        corr = np.eye(3)
        var_types = np.array([0, 0, 0], dtype=np.int64)
        config = {
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 1.0,  # Always change
            "new_distributions": ["right_skewed"],
        }
        p_corr, p_types = apply_per_simulation_perturbations(corr, var_types, config, 42)

        # All normal (type 0) vars should be changed to right_skewed (type 2)
        assert np.all(p_types == 2)
