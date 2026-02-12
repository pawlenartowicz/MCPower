"""
Tests for Linear Mixed-Effects (LME) model support in MCPower.

Tests Phase 2.B implementation:
- Basic LME with random intercepts
- Convergence retry strategy
- LME vs OLS comparison (conservativeness)
- Full integration workflow
- Failed simulation threshold
"""

import numpy as np
import pytest

from mcpower import MCPower
from tests.config import (
    EFFECT_MEDIUM,
    ICC_HIGH,
    ICC_LOW,
    ICC_MODERATE,
    ICC_MODERATE_HIGH,
    LME_N_SIMS_QUICK,
    LME_N_SIMS_STANDARD,
    LME_THRESHOLD_STRICT,
    N_CLUSTERS_FEW,
    N_CLUSTERS_MANY,
    N_CLUSTERS_MODERATE,
)

pytestmark = [pytest.mark.lme, pytest.mark.slow]


class TestMixedModelsBasic:
    """Basic LME functionality tests."""

    def test_lme_simple_random_intercept(self):
        """Test basic LME with 2 clusters, 50 obs each (5 params → 10 obs/param)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=2)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=100, return_results=True)

        assert "results" in result
        assert "individual_powers" in result["results"]
        power = result["results"]["individual_powers"]["overall"]
        assert power >= 0
        assert power <= 100
        assert "n_simulations_failed" in result["results"]
        assert result["results"]["n_simulations_failed"] >= 0

    def test_lme_multiple_predictors(self):
        """Test LME with multiple fixed effects (6 params → 60 obs/cluster × 5 clusters)."""
        model = MCPower("y ~ x1 + x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.3")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=300, return_results=True)

        assert "results" in result
        assert "individual_powers" in result["results"]
        # Should return results for overall test
        assert "overall" in result["results"]["individual_powers"]

    def test_lme_with_interaction(self):
        """Test LME with interaction term (7 params → 70 obs/cluster × 15 clusters)."""
        model = MCPower("y ~ x1 + x2 + x1:x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model.set_effects("x1=0.4, x2=0.3, x1:x2=0.2")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=1050, return_results=True)

        assert "results" in result
        assert result["results"]["n_simulations_failed"] < 3  # Less than 20% failure


class TestMixedModelsConvergence:
    """Test convergence and retry strategy."""

    def test_lme_convergence_with_small_clusters(self):
        """Test convergence with challenging scenario (5 params → 50 obs/cluster × 30 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_LOW, n_clusters=N_CLUSTERS_MANY)
        model.set_effects("x=0.4")
        # Should succeed with retry strategy
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=1500, return_results=True)

        assert "results" in result
        # Allow up to 3% failures (default)
        assert result["results"]["n_simulations_failed"] <= 1  # 1/20 = 5%, but should be less

    def test_lme_convergence_with_high_icc(self):
        """Test convergence with high ICC (5 params → 50 obs/cluster × 5 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_HIGH, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=250, return_results=True)

        assert "results" in result
        # High ICC can make convergence harder
        assert result["results"]["n_simulations_failed"] <= 1


class TestMixedModelsVsOLS:
    """Compare LME and OLS to verify conservativeness."""

    def test_lme_more_conservative_than_ols(self):
        """LME should be more conservative (lower power) than OLS for clustered data."""
        # Setup: same data structure, different analysis methods
        np.random.seed(42)

        # OLS path (ignores clustering - anti-conservative)
        model_ols = MCPower("y ~ x")
        model_ols.set_effects(f"x={EFFECT_MEDIUM}")
        model_ols.set_seed(42)
        model_ols.set_simulations(LME_N_SIMS_STANDARD)
        result_ols = model_ols.find_power(sample_size=100, return_results=True)

        # LME path (proper clustering - conservative, 5 params → 50 obs/cluster × 5 clusters)
        model_lme = MCPower("y ~ x + (1|cluster)")
        model_lme.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model_lme.set_effects(f"x={EFFECT_MEDIUM}")
        model_lme.set_seed(42)
        model_lme.set_simulations(LME_N_SIMS_STANDARD)
        result_lme = model_lme.find_power(sample_size=250, return_results=True)

        # LME should have lower power (more conservative)
        # Note: This is a probabilistic test, may occasionally fail
        lme_power = result_lme["results"]["individual_powers"]["overall"]
        ols_power = result_ols["results"]["individual_powers"]["overall"]
        assert lme_power <= ols_power + 10  # Allow some variance


class TestMixedModelsIntegration:
    """Full workflow integration tests."""

    def test_mixed_model_full_workflow(self):
        """Test complete workflow from setup to power calculation."""
        model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
        model.set_cluster("school", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MANY)
        model.set_effects(f"treatment={EFFECT_MEDIUM}, motivation=0.3")
        model.set_seed(12137)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=1500, return_results=True)  # 1500/30 = 50 obs/cluster → 50/5 params = 10.0 ratio

        # Verify all expected fields
        assert "results" in result
        assert "individual_powers" in result["results"]
        assert "n_simulations_failed" in result["results"]  # Fixed: in results, not top level
        assert result["results"]["individual_powers"]["overall"] >= 0
        assert result["results"]["individual_powers"]["overall"] <= 100

        # Verify reasonable power for this design
        # With effect=0.5, n=600, 30 clusters, should have good power
        assert result["results"]["individual_powers"]["overall"] > 50

    def test_mixed_model_with_correction_methods(self):
        """Test LME with different multiple comparison corrections."""
        corrections = [None, "bonferroni", "benjamini-hochberg", "holm"]

        model = MCPower("y ~ x1 + x2 + x3 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.4, x3=0.3")
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_QUICK)  # Set simulations BEFORE loop

        for correction in corrections:
            result = model.find_power(
                sample_size=1200,  # 1200/20 = 60 obs/cluster → 60/6 params = 10.0 ratio
                correction=correction,
                return_results=True,  # Get full result dict
            )

            assert "results" in result
            assert result["results"]["individual_powers"]["overall"] >= 0

    def test_mixed_model_find_sample_size(self):
        """Test find_sample_size with mixed models (5 params → 50 obs/cluster × 15 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_power(80)
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_sample_size(
            from_size=750,  # 5 params → 50 obs/cluster × 15 clusters = 750 minimum
            to_size=1200,  # Upper bound for search
            by=150,
            return_results=True,  # Need to get results dict
        )

        assert "results" in result
        assert "first_achieved" in result["results"]
        assert "overall" in result["results"]["first_achieved"]
        assert result["results"]["first_achieved"]["overall"] >= 750


class TestFailedSimulationThreshold:
    """Test max_failed_simulations setting."""

    def test_set_max_failed_simulations(self):
        """Test setting max_failed_simulations."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")

        # Test valid values
        model.set_max_failed_simulations(LME_THRESHOLD_STRICT)
        assert model.max_failed_simulations == LME_THRESHOLD_STRICT

        model.set_max_failed_simulations(0.0)
        assert model.max_failed_simulations == 0.0

        model.set_max_failed_simulations(1.0)
        assert model.max_failed_simulations == 1.0

    def test_max_failed_simulations_validation(self):
        """Test validation of max_failed_simulations."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")

        # Should raise error for invalid values
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            model.set_max_failed_simulations(-0.1)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            model.set_max_failed_simulations(1.5)

    def test_failed_simulation_tracking(self):
        """Test that failed simulations are tracked and reported (5 params → 50 obs/cluster × 20 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=1000, return_results=True)

        # Check that n_simulations_failed is present and non-negative
        assert "results" in result
        assert "n_simulations_failed" in result["results"]
        assert result["results"]["n_simulations_failed"] >= 0
        assert result["results"]["n_simulations_failed"] < LME_N_SIMS_STANDARD


class TestMixedModelsEdgeCases:
    """Test edge cases and error handling."""

    def test_lme_single_cluster(self):
        """Test behavior with very few clusters (edge case, 5 params → 50 obs/cluster × 2 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=2)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        # Should work but may have convergence issues
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=100, return_results=True)
        assert "results" in result

    def test_lme_zero_icc(self):
        """Test LME with ICC=0 (should behave like OLS, 5 params → 50 obs/cluster × 5 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.0, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=250, return_results=True)

        assert "results" in result
        # With ICC=0, should be similar to OLS

    def test_lme_high_effect_size(self):
        """Test LME with large effect size (easier convergence, 5 params → 50 obs/cluster × 15 clusters)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model.set_effects("x=1.0")  # Large effect
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=750, return_results=True)

        assert "results" in result
        assert result["results"]["individual_powers"]["overall"] > 80  # Should have high power
        assert result["results"]["n_simulations_failed"] <= 1  # Should converge easily


class TestWarmStartOptimization:
    """Test warm start optimization behavior."""

    def test_warm_start_reset(self):
        """Test that warm start cache can be reset."""
        from mcpower.utils.mixed_models import reset_warm_start_cache

        # Run a model to populate cache (5 params → 50 obs/cluster × 5 clusters)
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.find_power(sample_size=250)

        # Reset cache
        reset_warm_start_cache()

        # Run another model (should work with cold start, 5 params → 50 obs/cluster × 15 clusters)
        model2 = MCPower("y ~ x + (1|cluster)")
        model2.set_cluster("cluster", ICC=ICC_MODERATE_HIGH, n_clusters=15)
        model2.set_effects("x=0.6")
        model2.set_simulations(LME_N_SIMS_QUICK)
        result = model2.find_power(sample_size=750, return_results=True)

        assert "results" in result


class TestMixedModelsCorrectionMethods:
    """Test FDR and Holm corrections produce different results than uncorrected."""

    def test_fdr_correction_reduces_power(self):
        """FDR (Benjamini-Hochberg) correction should produce lower or equal power."""
        model = MCPower("y ~ x1 + x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.3")
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(
            sample_size=1200,
            correction="benjamini-hochberg",
            return_results=True,
        )

        assert "results" in result
        powers = result["results"]["individual_powers"]
        powers_corr = result["results"].get("individual_powers_corrected", {})
        # Corrected power should be <= uncorrected (or close)
        for test in powers:
            if test in powers_corr:
                assert powers_corr[test] <= powers[test] + 5  # allow small MC noise

    def test_holm_correction_reduces_power(self):
        """Holm correction should produce lower or equal power."""
        model = MCPower("y ~ x1 + x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.3")
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(
            sample_size=1200,
            correction="holm",
            return_results=True,
        )

        assert "results" in result
        powers = result["results"]["individual_powers"]
        powers_corr = result["results"].get("individual_powers_corrected", {})
        for test in powers:
            if test in powers_corr:
                assert powers_corr[test] <= powers[test] + 5

    def test_bonferroni_most_conservative(self):
        """Bonferroni should be more conservative than FDR."""
        model = MCPower("y ~ x1 + x2 + x3 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.4, x3=0.3")
        model.set_seed(42)
        model.set_simulations(LME_N_SIMS_STANDARD)

        result_bonf = model.find_power(
            sample_size=1200,
            correction="bonferroni",
            return_results=True,
        )
        result_fdr = model.find_power(
            sample_size=1200,
            correction="benjamini-hochberg",
            return_results=True,
        )

        # Bonferroni corrected power <= FDR corrected power (+ noise tolerance)
        bonf_powers = result_bonf["results"].get("individual_powers_corrected", {})
        fdr_powers = result_fdr["results"].get("individual_powers_corrected", {})
        for test in bonf_powers:
            if test in fdr_powers:
                assert bonf_powers[test] <= fdr_powers[test] + 10  # allow MC noise


class TestLmeAnalysisWrapper:
    """Test _lme_analysis_wrapper routing."""

    def test_unknown_backend_raises(self):
        from mcpower.utils.mixed_models import _lme_analysis_wrapper

        with pytest.raises(ValueError, match="Unknown backend"):
            _lme_analysis_wrapper(
                np.zeros((10, 2)),
                np.zeros(10),
                np.array([0]),
                np.zeros(10, dtype=int),
                [],
                0,
                0.05,
                backend="nonexistent",
            )
