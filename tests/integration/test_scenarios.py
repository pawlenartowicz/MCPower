"""
Tests for scenario analysis module.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mcpower.core.scenarios import (
    DEFAULT_SCENARIO_CONFIG,
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


class TestSetScenarioConfigs:
    """Test set_scenario_configs() merge behavior and KeyError prevention."""

    # All keys that must exist in every scenario config
    ALL_KEYS = sorted(DEFAULT_SCENARIO_CONFIG["optimistic"].keys())

    def _make_model(self):
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.3, x2=0.2")
        return m

    # ── Merge semantics ──────────────────────────────────────────

    def test_custom_scenario_inherits_all_optimistic_keys(self):
        """New custom scenario with one key still has every required key."""
        m = self._make_model()
        m.set_scenario_configs({"extreme": {"heterogeneity": 0.6}})
        cfg = m._scenario_configs["extreme"]
        missing = set(self.ALL_KEYS) - set(cfg.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_custom_scenario_overrides_value(self):
        """Provided key overrides the optimistic default."""
        m = self._make_model()
        m.set_scenario_configs({"extreme": {"heterogeneity": 0.6}})
        assert m._scenario_configs["extreme"]["heterogeneity"] == 0.6

    def test_custom_scenario_non_overridden_keys_are_optimistic(self):
        """Non-overridden keys equal the optimistic baseline."""
        m = self._make_model()
        m.set_scenario_configs({"extreme": {"heterogeneity": 0.6}})
        opt = DEFAULT_SCENARIO_CONFIG["optimistic"]
        cfg = m._scenario_configs["extreme"]
        for key in self.ALL_KEYS:
            if key != "heterogeneity":
                assert cfg[key] == opt[key], f"Key {key}: {cfg[key]} != {opt[key]}"

    def test_existing_scenario_update_preserves_other_keys(self):
        """Updating one key on 'realistic' keeps the rest intact."""
        m = self._make_model()
        m.set_scenario_configs({"realistic": {"heterogeneity": 0.99}})
        cfg = m._scenario_configs["realistic"]
        assert cfg["heterogeneity"] == 0.99
        # Other keys should match original realistic defaults
        assert cfg["correlation_noise_sd"] == DEFAULT_SCENARIO_CONFIG["realistic"]["correlation_noise_sd"]

    def test_defaults_still_present_after_adding_custom(self):
        """Adding a custom scenario doesn't remove optimistic/realistic/doomer."""
        m = self._make_model()
        m.set_scenario_configs({"custom": {"heterogeneity": 0.1}})
        for name in ("optimistic", "realistic", "doomer", "custom"):
            assert name in m._scenario_configs

    def test_multiple_custom_scenarios(self):
        """Multiple custom scenarios each inherit independently."""
        m = self._make_model()
        m.set_scenario_configs({
            "mild": {"heterogeneity": 0.05},
            "severe": {"heterogeneity": 0.8, "heteroskedasticity": 0.5},
        })
        assert m._scenario_configs["mild"]["heterogeneity"] == 0.05
        assert m._scenario_configs["mild"]["heteroskedasticity"] == 0.0  # optimistic default
        assert m._scenario_configs["severe"]["heterogeneity"] == 0.8
        assert m._scenario_configs["severe"]["heteroskedasticity"] == 0.5

    def test_empty_custom_scenario_equals_optimistic(self):
        """An empty custom config is identical to the optimistic baseline."""
        m = self._make_model()
        m.set_scenario_configs({"empty": {}})
        opt = DEFAULT_SCENARIO_CONFIG["optimistic"]
        for key in self.ALL_KEYS:
            assert m._scenario_configs["empty"][key] == opt[key]

    # ── Type validation ──────────────────────────────────────────

    def test_non_dict_raises_type_error(self):
        m = self._make_model()
        with pytest.raises(TypeError):
            m.set_scenario_configs("not_a_dict")

    def test_returns_self_for_chaining(self):
        m = self._make_model()
        result = m.set_scenario_configs({"custom": {"heterogeneity": 0.1}})
        assert result is m

    # ── End-to-end: no KeyError during simulation ────────────────

    def test_custom_partial_config_runs_without_error(self):
        """Custom scenario with only one key runs find_power without KeyError."""
        m = self._make_model()
        m.set_scenario_configs({"partial": {"heterogeneity": 0.3}})
        result = m.find_power(
            50, scenarios=True, print_results=False, return_results=True
        )
        assert "partial" in result["scenarios"]
        power = result["scenarios"]["partial"]["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100

    def test_custom_residual_only_config_runs(self):
        """Custom scenario with only residual keys runs without error."""
        m = self._make_model()
        m.set_scenario_configs({
            "residual_test": {
                "residual_change_prob": 1.0,
                "residual_dists": ["heavy_tailed"],
                "residual_df": 5,
            }
        })
        result = m.find_power(
            50, scenarios=True, print_results=False, return_results=True
        )
        assert "residual_test" in result["scenarios"]

    def test_custom_lme_keys_on_ols_model_ignored(self):
        """LME-specific keys on an OLS model don't cause errors."""
        m = self._make_model()
        m.set_scenario_configs({
            "lme_on_ols": {
                "icc_noise_sd": 0.3,
                "random_effect_dist": "heavy_tailed",
                "random_effect_df": 3,
            }
        })
        result = m.find_power(
            50, scenarios=True, print_results=False, return_results=True
        )
        assert "lme_on_ols" in result["scenarios"]

    def test_overriding_all_three_defaults(self):
        """Overriding optimistic, realistic, and doomer all at once."""
        m = self._make_model()
        m.set_scenario_configs({
            "optimistic": {"heterogeneity": 0.01},
            "realistic": {"heterogeneity": 0.5},
            "doomer": {"heterogeneity": 0.9},
        })
        assert m._scenario_configs["optimistic"]["heterogeneity"] == 0.01
        assert m._scenario_configs["realistic"]["heterogeneity"] == 0.5
        assert m._scenario_configs["doomer"]["heterogeneity"] == 0.9
        # Other keys preserved from defaults
        assert m._scenario_configs["realistic"]["correlation_noise_sd"] == DEFAULT_SCENARIO_CONFIG["realistic"]["correlation_noise_sd"]
        assert m._scenario_configs["doomer"]["correlation_noise_sd"] == DEFAULT_SCENARIO_CONFIG["doomer"]["correlation_noise_sd"]


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


class TestScenarioConfigKeysE2E:
    """End-to-end tests for each individual config key and mixed combinations.

    Each test verifies that setting a single config key (or combination)
    via set_scenario_configs() runs find_power(scenarios=True) without
    error and produces valid power values.
    """

    N_SIMS = 50
    SAMPLE_SIZE = 80

    def _make_model(self):
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_effects("x1=0.3, x2=0.2")
        m.set_simulations(self.N_SIMS)
        return m

    def _run(self, model, config, scenario_name="test_scenario"):
        model.set_scenario_configs({scenario_name: config})
        result = model.find_power(
            self.SAMPLE_SIZE,
            scenarios=True,
            print_results=False,
            return_results=True,
        )
        power = result["scenarios"][scenario_name]["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100, f"Power out of range: {power}"
        return result

    # ── Individual general keys ───────────────────────────────────

    def test_heterogeneity_only(self):
        self._run(self._make_model(), {"heterogeneity": 0.3})

    def test_heteroskedasticity_only(self):
        self._run(self._make_model(), {"heteroskedasticity": 0.2})

    def test_correlation_noise_sd_only(self):
        m = self._make_model()
        m.set_correlations("(x1,x2)=0.4")
        self._run(m, {"correlation_noise_sd": 0.3})

    def test_distribution_change_prob_only(self):
        self._run(self._make_model(), {"distribution_change_prob": 0.5})

    def test_new_distributions_with_change_prob(self):
        self._run(self._make_model(), {
            "distribution_change_prob": 1.0,
            "new_distributions": ["uniform"],
        })

    # ── Individual residual keys ──────────────────────────────────

    def test_residual_change_prob_only(self):
        self._run(self._make_model(), {"residual_change_prob": 0.5})

    def test_residual_df_only(self):
        self._run(self._make_model(), {
            "residual_change_prob": 1.0,
            "residual_df": 3,
        })

    def test_residual_dists_only(self):
        self._run(self._make_model(), {
            "residual_change_prob": 1.0,
            "residual_dists": ["heavy_tailed"],
        })

    # ── Mixed general combinations ────────────────────────────────

    def test_heterogeneity_and_correlation_noise(self):
        m = self._make_model()
        m.set_correlations("(x1,x2)=0.3")
        self._run(m, {
            "heterogeneity": 0.25,
            "correlation_noise_sd": 0.3,
        })

    def test_distribution_change_and_heteroskedasticity(self):
        self._run(self._make_model(), {
            "distribution_change_prob": 0.5,
            "heteroskedasticity": 0.15,
        })

    def test_all_general_keys_together(self):
        m = self._make_model()
        m.set_correlations("(x1,x2)=0.3")
        self._run(m, {
            "heterogeneity": 0.2,
            "heteroskedasticity": 0.1,
            "correlation_noise_sd": 0.2,
            "distribution_change_prob": 0.3,
        })

    # ── Mixed general + residual ──────────────────────────────────

    def test_general_plus_residual_keys(self):
        self._run(self._make_model(), {
            "heterogeneity": 0.2,
            "residual_change_prob": 0.5,
            "residual_df": 5,
        })

    def test_all_ols_keys_together(self):
        m = self._make_model()
        m.set_correlations("(x1,x2)=0.3")
        self._run(m, {
            "heterogeneity": 0.3,
            "heteroskedasticity": 0.15,
            "correlation_noise_sd": 0.25,
            "distribution_change_prob": 0.4,
            "new_distributions": ["right_skewed", "uniform"],
            "residual_change_prob": 0.5,
            "residual_dists": ["heavy_tailed", "skewed"],
            "residual_df": 6,
        })

    # ── Boundary values ───────────────────────────────────────────

    def test_zero_perturbation_matches_optimistic(self):
        """A custom scenario with all zeros should match optimistic power."""
        m = self._make_model()
        m.set_seed(42)
        result = self._run(m, {
            "heterogeneity": 0.0,
            "heteroskedasticity": 0.0,
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 0.0,
            "residual_change_prob": 0.0,
        })
        opt_power = result["scenarios"]["optimistic"]["results"]["individual_powers"]["overall"]
        custom_power = result["scenarios"]["test_scenario"]["results"]["individual_powers"]["overall"]
        # Same seed, same zero config → should be close (not exact due to seed offsets)
        assert abs(opt_power - custom_power) < 15

    def test_max_perturbation_runs(self):
        """Extreme perturbation values should not crash."""
        self._run(self._make_model(), {
            "heterogeneity": 0.9,
            "heteroskedasticity": 0.5,
            "correlation_noise_sd": 0.8,
            "distribution_change_prob": 1.0,
            "residual_change_prob": 1.0,
            "residual_df": 2,
        })
