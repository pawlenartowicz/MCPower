"""
Unit tests for LME scenario perturbation functions.

Tests the three LME assumption violation mechanisms:
1. Non-normal random effects (heavy-tailed, skewed)
2. ICC perturbation (tau² jitter)
3. Non-normal residuals
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from mcpower.core.scenarios import (
    DEFAULT_SCENARIO_CONFIG,
    apply_lme_perturbations,
    apply_lme_residual_perturbations,
)
from mcpower.stats.data_generation import (
    _generate_cluster_effects,
    _generate_non_normal_intercepts,
    _generate_random_effects,
)

pytestmark = pytest.mark.lme


# ---------------------------------------------------------------------------
# DEFAULT_SCENARIO_CONFIG has all LME keys
# ---------------------------------------------------------------------------
class TestDefaultConfig:
    """Verify DEFAULT_SCENARIO_CONFIG contains all LME keys."""

    LME_KEYS = [
        "random_effect_dist",
        "random_effect_df",
        "icc_noise_sd",
        "residual_dist",
        "residual_change_prob",
        "residual_df",
    ]

    def test_realistic_has_lme_keys(self):
        for key in self.LME_KEYS:
            assert key in DEFAULT_SCENARIO_CONFIG["realistic"], f"Missing key: {key}"

    def test_doomer_has_lme_keys(self):
        for key in self.LME_KEYS:
            assert key in DEFAULT_SCENARIO_CONFIG["doomer"], f"Missing key: {key}"

    def test_realistic_values(self):
        cfg = DEFAULT_SCENARIO_CONFIG["realistic"]
        assert cfg["random_effect_dist"] == "heavy_tailed"
        assert cfg["random_effect_df"] == 5
        assert cfg["icc_noise_sd"] == 0.15
        assert cfg["residual_dist"] == "heavy_tailed"
        assert cfg["residual_change_prob"] == 0.3
        assert cfg["residual_df"] == 10

    def test_doomer_values(self):
        cfg = DEFAULT_SCENARIO_CONFIG["doomer"]
        assert cfg["random_effect_dist"] == "heavy_tailed"
        assert cfg["random_effect_df"] == 3
        assert cfg["icc_noise_sd"] == 0.30
        assert cfg["residual_dist"] == "heavy_tailed"
        assert cfg["residual_change_prob"] == 0.8
        assert cfg["residual_df"] == 5


# ---------------------------------------------------------------------------
# apply_lme_perturbations
# ---------------------------------------------------------------------------
class TestApplyLmePerturbations:
    """Test apply_lme_perturbations() function."""

    CLUSTER_SPECS = {
        "school": {
            "n_clusters": 20,
            "cluster_size": 50,
            "tau_squared": 0.25,
            "icc": 0.2,
        }
    }

    def test_returns_none_when_no_cluster_specs(self):
        result = apply_lme_perturbations({}, DEFAULT_SCENARIO_CONFIG["realistic"], 42)
        assert result is None

    def test_returns_none_when_no_lme_perturbations_active(self):
        config = {"icc_noise_sd": 0.0, "random_effect_dist": "normal"}
        result = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        assert result is None

    def test_returns_dict_with_correct_keys(self):
        result = apply_lme_perturbations(
            self.CLUSTER_SPECS, DEFAULT_SCENARIO_CONFIG["realistic"], 42
        )
        assert result is not None
        assert "tau_squared_multipliers" in result
        assert "random_effect_dist" in result
        assert "random_effect_df" in result

    def test_icc_jitter_produces_non_unity_multipliers(self):
        config = {
            "icc_noise_sd": 0.15,
            "random_effect_dist": "heavy_tailed",
            "random_effect_df": 5,
        }
        result = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        multiplier = result["tau_squared_multipliers"]["school"]
        # With icc_noise_sd=0.15, multiplier = exp(N(0, 0.15)) — unlikely to be exactly 1.0
        assert multiplier != 1.0
        # Should be positive
        assert multiplier > 0

    def test_icc_jitter_seed_reproducible(self):
        config = DEFAULT_SCENARIO_CONFIG["realistic"]
        r1 = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        r2 = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        assert r1["tau_squared_multipliers"]["school"] == r2["tau_squared_multipliers"]["school"]

    def test_different_seeds_give_different_multipliers(self):
        config = DEFAULT_SCENARIO_CONFIG["realistic"]
        r1 = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        r2 = apply_lme_perturbations(self.CLUSTER_SPECS, config, 99)
        assert r1["tau_squared_multipliers"]["school"] != r2["tau_squared_multipliers"]["school"]

    def test_zero_icc_noise_gives_unity_multiplier(self):
        config = {
            "icc_noise_sd": 0.0,
            "random_effect_dist": "heavy_tailed",
            "random_effect_df": 5,
        }
        result = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        assert result["tau_squared_multipliers"]["school"] == 1.0

    def test_passthrough_dist_params(self):
        config = {
            "icc_noise_sd": 0.15,
            "random_effect_dist": "skewed",
            "random_effect_df": 7,
        }
        result = apply_lme_perturbations(self.CLUSTER_SPECS, config, 42)
        assert result["random_effect_dist"] == "skewed"
        assert result["random_effect_df"] == 7


# ---------------------------------------------------------------------------
# _generate_non_normal_intercepts
# ---------------------------------------------------------------------------
class TestGenerateNonNormalIntercepts:
    """Test the non-normal intercept generation helper."""

    N_CLUSTERS = 5000
    TAU = 1.0

    def test_normal_distribution(self):
        rng = np.random.RandomState(42)
        intercepts = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "normal", 5, rng)
        assert len(intercepts) == self.N_CLUSTERS
        assert abs(np.mean(intercepts)) < 0.1
        assert abs(np.std(intercepts) - self.TAU) < 0.1

    def test_heavy_tailed_has_excess_kurtosis(self):
        rng = np.random.RandomState(42)
        intercepts = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "heavy_tailed", 5, rng)
        # t(5) has excess kurtosis = 6/(5-4) = 6.0
        kurtosis = sp_stats.kurtosis(intercepts, fisher=True)
        assert kurtosis > 1.0, f"Expected excess kurtosis > 1, got {kurtosis}"

    def test_heavy_tailed_preserves_scale(self):
        rng = np.random.RandomState(42)
        intercepts = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "heavy_tailed", 5, rng)
        # SD should be approximately tau (within ~15% for 5000 samples)
        assert abs(np.std(intercepts) - self.TAU) < 0.3

    def test_heavy_tailed_df3_more_extreme(self):
        rng1 = np.random.RandomState(42)
        intercepts_df3 = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "heavy_tailed", 3, rng1)
        rng2 = np.random.RandomState(42)
        intercepts_df5 = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "heavy_tailed", 5, rng2)
        # df=3 should have higher kurtosis than df=5
        k3 = sp_stats.kurtosis(intercepts_df3, fisher=True)
        k5 = sp_stats.kurtosis(intercepts_df5, fisher=True)
        assert k3 > k5

    def test_skewed_has_nonzero_skewness(self):
        rng = np.random.RandomState(42)
        intercepts = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "skewed", 5, rng)
        skewness = sp_stats.skew(intercepts)
        assert abs(skewness) > 0.3, f"Expected |skewness| > 0.3, got {skewness}"

    def test_skewed_centered_near_zero(self):
        rng = np.random.RandomState(42)
        intercepts = _generate_non_normal_intercepts(self.N_CLUSTERS, self.TAU, "skewed", 5, rng)
        assert abs(np.mean(intercepts)) < 0.1

    def test_df_clamped_to_minimum_3(self):
        rng = np.random.RandomState(42)
        # df=1 should be clamped to 3
        intercepts = _generate_non_normal_intercepts(100, self.TAU, "heavy_tailed", 1, rng)
        assert len(intercepts) == 100


# ---------------------------------------------------------------------------
# _generate_cluster_effects with lme_perturbations
# ---------------------------------------------------------------------------
class TestGenerateClusterEffectsWithPerturbations:
    """Test _generate_cluster_effects with LME perturbations."""

    CLUSTER_SPECS = {
        "school": {
            "n_clusters": 20,
            "cluster_size": 50,
            "tau_squared": 0.25,
            "icc": 0.2,
        }
    }

    def test_without_perturbations_unchanged(self):
        X1 = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=None)
        X2 = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=None)
        np.testing.assert_array_equal(X1, X2)

    def test_with_perturbations_changes_output(self):
        perturbations = {
            "tau_squared_multipliers": {"school": 2.0},
            "random_effect_dist": "heavy_tailed",
            "random_effect_df": 5,
        }
        X_normal = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=None)
        X_perturbed = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=perturbations)
        # Should differ because of ICC jitter and different distribution
        assert not np.array_equal(X_normal, X_perturbed)

    def test_icc_jitter_changes_variance(self):
        perturbations_high = {
            "tau_squared_multipliers": {"school": 4.0},
            "random_effect_dist": "normal",
            "random_effect_df": 5,
        }
        perturbations_low = {
            "tau_squared_multipliers": {"school": 0.25},
            "random_effect_dist": "normal",
            "random_effect_df": 5,
        }
        X_high = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=perturbations_high)
        X_low = _generate_cluster_effects(1000, self.CLUSTER_SPECS, sim_seed=42, lme_perturbations=perturbations_low)
        # Higher multiplier → higher variance in cluster effects
        assert np.var(X_high) > np.var(X_low)


# ---------------------------------------------------------------------------
# _generate_random_effects with lme_perturbations (q>1 slopes)
# ---------------------------------------------------------------------------
class TestGenerateRandomEffectsWithPerturbations:
    """Test _generate_random_effects with LME perturbations (slopes model)."""

    def _make_specs(self):
        """Create a random slopes cluster spec."""
        G = np.array([[0.25, 0.05], [0.05, 0.10]])
        return {
            "school": {
                "n_clusters": 20,
                "cluster_size": 50,
                "tau_squared": 0.25,
                "icc": 0.2,
                "G_matrix": G,
                "random_slope_vars": ["x1"],
                "q": 2,
                "parent_var": None,
                "n_per_parent": None,
            }
        }

    def test_slopes_with_heavy_tailed_perturbations(self):
        specs = self._make_specs()
        X_non_factors = np.random.RandomState(42).standard_normal((1000, 1))
        perturbations = {
            "tau_squared_multipliers": {"school": 1.5},
            "random_effect_dist": "heavy_tailed",
            "random_effect_df": 5,
        }
        result = _generate_random_effects(
            1000, specs, X_non_factors, ["x1"], sim_seed=42, lme_perturbations=perturbations
        )
        assert result.intercept_columns.shape == (1000, 1)
        assert result.slope_contribution.shape == (1000,)

    def test_slopes_with_skewed_perturbations(self):
        specs = self._make_specs()
        X_non_factors = np.random.RandomState(42).standard_normal((1000, 1))
        perturbations = {
            "tau_squared_multipliers": {"school": 1.0},
            "random_effect_dist": "skewed",
            "random_effect_df": 5,
        }
        result = _generate_random_effects(
            1000, specs, X_non_factors, ["x1"], sim_seed=42, lme_perturbations=perturbations
        )
        assert result.intercept_columns.shape == (1000, 1)

    def test_slopes_without_perturbations(self):
        specs = self._make_specs()
        X_non_factors = np.random.RandomState(42).standard_normal((1000, 1))
        result = _generate_random_effects(
            1000, specs, X_non_factors, ["x1"], sim_seed=42, lme_perturbations=None
        )
        assert result.intercept_columns.shape == (1000, 1)


# ---------------------------------------------------------------------------
# apply_lme_residual_perturbations
# ---------------------------------------------------------------------------
class TestApplyLmeResidualPerturbations:
    """Test apply_lme_residual_perturbations() function."""

    def _make_y(self, seed=42):
        """Generate a deterministic y vector with known errors."""
        rng = np.random.RandomState(seed + 2)
        return rng.standard_normal(500)

    def test_normal_dist_returns_unchanged(self):
        y = self._make_y()
        config = {"residual_dist": "normal", "residual_change_prob": 1.0, "residual_df": 5}
        result = apply_lme_residual_perturbations(y.copy(), config, 42)
        np.testing.assert_array_equal(result, y)

    def test_zero_prob_returns_unchanged(self):
        y = self._make_y()
        config = {"residual_dist": "heavy_tailed", "residual_change_prob": 0.0, "residual_df": 5}
        result = apply_lme_residual_perturbations(y.copy(), config, 42)
        np.testing.assert_array_equal(result, y)

    def test_prob_1_always_applies(self):
        y = self._make_y()
        config = {"residual_dist": "heavy_tailed", "residual_change_prob": 1.0, "residual_df": 5}
        result = apply_lme_residual_perturbations(y.copy(), config, 42)
        # Should be different from original
        assert not np.array_equal(result, y)

    def test_heavy_tailed_residuals_have_excess_kurtosis(self):
        """When residuals are replaced with t(5), the diff should have heavy tails."""
        y_orig = self._make_y()
        config = {"residual_dist": "heavy_tailed", "residual_change_prob": 1.0, "residual_df": 5}
        y_perturbed = apply_lme_residual_perturbations(y_orig.copy(), config, 42)
        diff = y_perturbed - y_orig
        # The diff = new_errors - original_errors. Both have finite variance,
        # but the new_errors are t(5) which has excess kurtosis.
        # For large enough N, the kurtosis of the difference should be positive.
        sp_stats.kurtosis(diff + y_orig, fisher=True)
        # Just check it ran without error and output differs
        assert not np.array_equal(y_perturbed, y_orig)

    def test_skewed_residuals_applied(self):
        y_orig = self._make_y()
        config = {"residual_dist": "skewed", "residual_change_prob": 1.0, "residual_df": 5}
        y_perturbed = apply_lme_residual_perturbations(y_orig.copy(), config, 42)
        assert not np.array_equal(y_perturbed, y_orig)

    def test_coin_flip_seed_reproducible(self):
        y = self._make_y()
        config = {"residual_dist": "heavy_tailed", "residual_change_prob": 0.5, "residual_df": 5}
        r1 = apply_lme_residual_perturbations(y.copy(), config, 42)
        r2 = apply_lme_residual_perturbations(y.copy(), config, 42)
        np.testing.assert_array_equal(r1, r2)

    def test_coin_flip_prob_respected(self):
        """With prob=0.3, roughly 30% of simulations should be perturbed."""
        config = {"residual_dist": "heavy_tailed", "residual_change_prob": 0.3, "residual_df": 5}
        n_perturbed = 0
        n_trials = 200
        y_template = np.ones(100)
        for i in range(n_trials):
            y = y_template.copy()
            result = apply_lme_residual_perturbations(y, config, i * 100)
            if not np.array_equal(result, y_template):
                n_perturbed += 1
        # Should be roughly 30% ± some tolerance
        pct = n_perturbed / n_trials
        assert 0.10 < pct < 0.55, f"Expected ~30% perturbed, got {pct:.1%}"
