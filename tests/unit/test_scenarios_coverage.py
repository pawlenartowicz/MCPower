"""Tests for scenario analysis — plot creation, correlation matrix repair, LME perturbations."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcpower.core.scenarios import (
    ScenarioRunner,
    apply_lme_perturbations,
    apply_per_simulation_perturbations,
)


class TestCorrelationMatrixRepair:
    """Spectral clipping when noise creates negative eigenvalues."""

    def test_negative_eigenvalue_repaired(self):
        """After heavy noise, result should be positive semi-definite with unit diagonal."""
        # Create a 3x3 identity correlation matrix
        corr = np.eye(3)
        var_types = np.zeros(3, dtype=np.int64)  # all normal

        config = {
            "correlation_noise_sd": 2.0,  # Very heavy noise → guaranteed negative eigenvalues
            "distribution_change_prob": 0.0,
            "new_distributions": [],
        }

        perturbed_corr, _ = apply_per_simulation_perturbations(corr, var_types, config, sim_seed=42)

        # Eigenvalues should all be >= 0
        eigvals = np.linalg.eigvalsh(perturbed_corr)
        assert np.all(eigvals >= -1e-10)

        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(perturbed_corr), 1.0, atol=1e-10)

        # Should be symmetric
        np.testing.assert_allclose(perturbed_corr, perturbed_corr.T, atol=1e-10)

    def test_no_repair_needed_when_no_noise(self):
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        var_types = np.zeros(2, dtype=np.int64)

        config = {
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 0.0,
            "new_distributions": [],
        }

        perturbed_corr, _ = apply_per_simulation_perturbations(corr, var_types, config, sim_seed=42)
        np.testing.assert_array_equal(perturbed_corr, corr)


class TestDistributionPerturbation:
    """Variable type swaps in scenario mode."""

    def test_distribution_swap_occurs(self):
        var_types = np.zeros(10, dtype=np.int64)  # All normal
        config = {
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 1.0,  # Always swap
            "new_distributions": ["right_skewed"],
        }

        _, perturbed_types = apply_per_simulation_perturbations(
            np.eye(10), var_types, config, sim_seed=42,
        )
        # All should be swapped from 0 to 2 (right_skewed)
        assert np.all(perturbed_types == 2)

    def test_non_normal_not_swapped(self):
        """Binary (1) and uploaded (99) vars should not be swapped."""
        var_types = np.array([0, 1, 99], dtype=np.int64)
        config = {
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 1.0,
            "new_distributions": ["right_skewed"],
        }

        _, perturbed_types = apply_per_simulation_perturbations(
            np.eye(3), var_types, config, sim_seed=42,
        )
        assert perturbed_types[0] == 2  # normal → right_skewed
        assert perturbed_types[1] == 1  # binary unchanged
        assert perturbed_types[2] == 99  # uploaded unchanged

    def test_none_config_passthrough(self):
        corr = np.eye(2)
        var_types = np.zeros(2, dtype=np.int64)
        result_corr, result_types = apply_per_simulation_perturbations(
            corr, var_types, None, sim_seed=42,
        )
        np.testing.assert_array_equal(result_corr, corr)
        np.testing.assert_array_equal(result_types, var_types)


class TestLMEPerturbations:
    """LME perturbation computation."""

    def test_icc_noise_creates_multipliers(self):
        cluster_specs = {"school": {"n_clusters": 20, "cluster_size": 10, "icc": 0.2}}
        config = {
            "icc_noise_sd": 0.3,
            "random_effect_dist": "normal",
            "random_effect_df": 5,
        }

        result = apply_lme_perturbations(cluster_specs, config, sim_seed=42)
        assert result is not None
        assert "tau_squared_multipliers" in result
        assert "school" in result["tau_squared_multipliers"]
        # Multiplier should be exp(N(0, 0.3)) — positive, around 1
        mult = result["tau_squared_multipliers"]["school"]
        assert mult > 0

    def test_no_perturbation_returns_none(self):
        cluster_specs = {"school": {"n_clusters": 20, "cluster_size": 10, "icc": 0.2}}
        config = {
            "icc_noise_sd": 0.0,
            "random_effect_dist": "normal",
            "random_effect_df": 5,
        }
        result = apply_lme_perturbations(cluster_specs, config, sim_seed=42)
        assert result is None

    def test_empty_cluster_specs_returns_none(self):
        result = apply_lme_perturbations({}, {"icc_noise_sd": 0.5}, sim_seed=42)
        assert result is None

    def test_heavy_tailed_re_dist(self):
        cluster_specs = {"school": {"n_clusters": 20, "cluster_size": 10, "icc": 0.2}}
        config = {
            "icc_noise_sd": 0.0,
            "random_effect_dist": "heavy_tailed",
            "random_effect_df": 3,
        }
        result = apply_lme_perturbations(cluster_specs, config, sim_seed=42)
        assert result is not None
        assert result["random_effect_dist"] == "heavy_tailed"
        assert result["random_effect_df"] == 3


class TestScenarioRunnerPlots:
    """Test _create_scenario_plots path."""

    def test_plot_creation_with_mock(self):
        model = MagicMock()
        model.power = 80.0
        runner = ScenarioRunner(model)

        results = {
            "analysis_type": "sample_size",
            "scenarios": {
                "optimistic": {
                    "model": {
                        "target_tests": ["x1"],
                        "correction": None,
                    },
                    "results": {
                        "sample_sizes_tested": [50, 100],
                        "powers_by_test": {"x1": [50.0, 85.0]},
                        "first_achieved": {"x1": 100},
                    },
                },
            },
        }

        with patch("mcpower.core.scenarios._create_power_plot") as mock_plot:
            runner._create_scenario_plots(results)
            mock_plot.assert_called_once()

    def test_plot_with_correction(self):
        model = MagicMock()
        model.power = 80.0
        runner = ScenarioRunner(model)

        results = {
            "analysis_type": "sample_size",
            "scenarios": {
                "optimistic": {
                    "model": {
                        "target_tests": ["x1"],
                        "correction": "bonferroni",
                    },
                    "results": {
                        "sample_sizes_tested": [50, 100],
                        "powers_by_test": {"x1": [50.0, 85.0]},
                        "powers_by_test_corrected": {"x1": [40.0, 75.0]},
                        "first_achieved": {"x1": 100},
                        "first_achieved_corrected": {"x1": 150},
                    },
                },
            },
        }

        with patch("mcpower.core.scenarios._create_power_plot") as mock_plot:
            runner._create_scenario_plots(results)
            # Should be called for both uncorrected and corrected
            assert mock_plot.call_count == 2

    def test_no_plot_when_missing_sample_sizes(self):
        model = MagicMock()
        model.power = 80.0
        runner = ScenarioRunner(model)

        results = {
            "scenarios": {
                "optimistic": {
                    "results": {"powers_by_test": {"x1": [50.0]}},
                },
            },
        }

        with patch("mcpower.core.scenarios._create_power_plot") as mock_plot:
            runner._create_scenario_plots(results)
            mock_plot.assert_not_called()
