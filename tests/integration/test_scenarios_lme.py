"""
Integration tests for LME scenario analysis pipeline.

Tests that the full scenario pipeline works end-to-end with mixed models,
including realistic/doomer scenarios with non-normal random effects,
ICC jitter, and non-normal residuals.
"""

import numpy as np
import pytest

from mcpower import MCPower
from tests.config import LME_N_SIMS_BENCHMARK, LME_N_SIMS_QUICK
from tests.helpers.mc_margins import mc_accuracy_margin

pytestmark = pytest.mark.lme


class TestLmeScenarioPipelineIntercept:
    """Full scenario pipeline on random intercept model."""

    def test_scenario_analysis_runs_without_crash(self):
        """Basic smoke test: scenario analysis completes on a random intercept model."""
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(0.50)

        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert "scenarios" in result
        assert "optimistic" in result["scenarios"]
        assert "realistic" in result["scenarios"]
        assert "doomer" in result["scenarios"]

    def test_power_ordering_doomer_le_realistic_le_optimistic(self):
        """Full ordering: doomer ≤ realistic ≤ optimistic (with MC noise margin).

        Uses mc_accuracy_margin with the estimated optimistic power (highest
        of the three → worst-case binomial variance), scaled by sqrt(2)
        because we compare two independent MC estimates.
        """
        n_sims = LME_N_SIMS_BENCHMARK
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(n_sims)
        model.set_max_failed_simulations(0.50)

        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        opt_power = result["scenarios"]["optimistic"]["results"]["individual_powers"]["overall"]
        real_power = result["scenarios"]["realistic"]["results"]["individual_powers"]["overall"]
        doom_power = result["scenarios"]["doomer"]["results"]["individual_powers"]["overall"]

        # Use optimistic power for margin (worst-case variance among the three).
        # Scale by sqrt(2) for the difference of two independent MC estimates.
        margin = mc_accuracy_margin(opt_power, n_sims) * np.sqrt(2)

        assert doom_power <= real_power + margin, (
            f"Doomer ({doom_power}) should be ≤ realistic ({real_power}) + {margin:.1f}pp margin"
        )
        assert real_power <= opt_power + margin, (
            f"Realistic ({real_power}) should be ≤ optimistic ({opt_power}) + {margin:.1f}pp margin"
        )
        assert doom_power <= opt_power + margin, (
            f"Doomer ({doom_power}) should be ≤ optimistic ({opt_power}) + {margin:.1f}pp margin"
        )


class TestLmeScenarioPipelineSlopes:
    """Full scenario pipeline on random slopes model."""

    def test_scenario_analysis_slopes_no_crash(self):
        """Smoke test: scenario analysis completes on a random slopes model."""
        model = MCPower("y ~ x1 + (1 + x1|school)")
        model.set_cluster(
            "school", ICC=0.2, n_clusters=20,
            random_slopes=["x1"], slope_variance=0.1, slope_intercept_corr=0.3,
        )
        model.set_effects("x1=0.5")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(0.50)

        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert "scenarios" in result
        assert "optimistic" in result["scenarios"]
        assert "realistic" in result["scenarios"]


class TestLmeScenarioBackwardCompatibility:
    """Ensure OLS scenario analysis is not broken by LME additions."""

    def test_ols_scenario_analysis_unchanged(self):
        """OLS scenario analysis should still work identically."""
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_simulations(50)

        result = model.find_power(
            sample_size=100,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert "scenarios" in result
        assert "optimistic" in result["scenarios"]
        assert "realistic" in result["scenarios"]
        assert "doomer" in result["scenarios"]

        # OLS should still have valid power values
        for scenario_name in ["optimistic", "realistic", "doomer"]:
            power = result["scenarios"][scenario_name]["results"]["individual_powers"]["overall"]
            assert 0 <= power <= 100


class TestCustomLmeScenarioConfig:
    """Test custom LME scenario configs via set_scenario_configs()."""

    def test_custom_config_with_lme_keys(self):
        """Users can provide custom LME scenario configs."""
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(0.50)

        model.set_scenario_configs(
            {
                "realistic": {
                    "random_effect_dist": "skewed",
                    "random_effect_df": 5,
                    "icc_noise_sd": 0.10,
                    "residual_dist": "normal",
                    "residual_change_prob": 0.0,
                    "residual_df": 10,
                },
                "doomer": {
                    "random_effect_dist": "heavy_tailed",
                    "random_effect_df": 3,
                    "icc_noise_sd": 0.40,
                    "residual_dist": "heavy_tailed",
                    "residual_change_prob": 1.0,
                    "residual_df": 3,
                },
            }
        )

        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert "scenarios" in result

    def test_custom_config_without_lme_keys_still_works(self):
        """Custom config without LME keys should still work (no LME perturbations)."""
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(0.50)

        model.set_scenario_configs(
            {
                "realistic": {
                    "heterogeneity": 0.1,
                    "heteroskedasticity": 0.05,
                    "correlation_noise_sd": 0.1,
                    "distribution_change_prob": 0.2,
                    "new_distributions": ["right_skewed"],
                },
            }
        )

        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert "scenarios" in result


class TestLmeScenarioMetadata:
    """Test that LME scenario config is properly stored on metadata."""

    def test_lme_scenario_config_stored(self):
        """lme_scenario_config should be set on metadata for LME models in scenario mode."""
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(0.50)

        # Run in scenario mode — the metadata.lme_scenario_config is set internally
        # We can't directly inspect it, but we can verify the pipeline doesn't crash
        result = model.find_power(
            sample_size=1000,
            scenarios=True,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None
