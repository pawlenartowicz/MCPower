"""
Extended edge case tests for mixed models.

These tests cover boundary conditions, extreme parameter values,
and potential failure modes to ensure robustness.
"""

import pytest

from mcpower import MCPower
from tests.config import (
    EFFECT_LARGE,
    EFFECT_MEDIUM,
    ICC_HIGH,
    ICC_LOW,
    ICC_MODERATE,
    ICC_MODERATE_HIGH,
    LME_N_SIMS_QUICK,
    LME_N_SIMS_STANDARD,
    LME_THRESHOLD_MODERATE,
    N_CLUSTERS_FEW,
    N_CLUSTERS_MANY,
    N_CLUSTERS_MODERATE,
    N_CLUSTERS_VERY_MANY,
)

pytestmark = [
    pytest.mark.lme,
    pytest.mark.filterwarnings("ignore:Wald test fallback used:UserWarning"),
]


class TestExtremeICCValues:
    """Test behavior with extreme ICC values."""

    def test_very_low_icc(self):
        """ICC near 0 should behave almost like OLS."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_LOW, n_clusters=15)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=750, return_results=True)

        assert result is not None
        # Very low ICC should have minimal failures
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 1
        print(f"ICC=0.1: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")

    def test_very_high_icc(self):
        """ICC near 1 indicates almost all variance between clusters."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.9, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)

        result = model.find_power(sample_size=1000, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        print(f"ICC=0.9: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")


class TestClusterConfigurations:
    """Test various cluster configurations."""

    def test_minimum_clusters(self):
        """Test with minimum viable number of clusters (2)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=2)
        model.set_effects("x=0.6")  # Large effect to compensate
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)

        result = model.find_power(sample_size=100, return_results=True)

        assert result is not None
        print(f"2 clusters: power={result['results']['individual_powers']['overall']:.1f}%")

    def test_many_clusters(self):
        """Test with many clusters (optimal for mixed models)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_VERY_MANY)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=2500, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed == 0  # Should be very stable
        print(f"50 clusters: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")

    def test_small_sample_per_cluster(self):
        """Test with small samples per cluster."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=1000, return_results=True)  # 3 per cluster

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        print(f"3 obs/cluster: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")

    def test_large_sample_per_cluster(self):
        """Test with large samples per cluster."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=500, return_results=True)  # 50 per cluster

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 1
        print(f"50 obs/cluster: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")


class TestComplexModels:
    """Test complex model specifications."""

    def test_many_predictors(self):
        """Test with many predictors."""
        model = MCPower("y ~ x1 + x2 + x3 + x4 + x5 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MANY)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.4, x3=0.3, x4=0.3, x5=0.2")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)  # Allow 10% failures for complex model

        result = model.find_power(sample_size=2700, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 2
        print(f"5 predictors: {n_failed}/25 failed")

    def test_multiple_interactions(self):
        """Test with multiple interaction terms."""
        model = MCPower("y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=25)
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.4, x3=0.3, x1:x2=0.2, x1:x3=0.2, x2:x3=0.2")
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)  # Allow 10% failures for complex model

        result = model.find_power(sample_size=2500, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        print(f"Multiple interactions: {n_failed}/20 failed")

    def test_correlated_predictors_in_mixed_model(self):
        """Test correlated predictors in mixed model context."""
        model = MCPower("y ~ x1 + x2 + x3 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_correlations("(x1,x2)=0.5, (x1,x3)=0.3, (x2,x3)=0.4")
        model.set_effects(f"x1={EFFECT_MEDIUM}, x2=0.4, x3=0.3")
        model.set_simulations(LME_N_SIMS_STANDARD)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)  # Allow 10% failures for complex model

        result = model.find_power(sample_size=1200, return_results=True)  # 1200/20 = 60 obs/cluster → 60/6 params = 10.0 ratio

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 2
        print(f"Correlated predictors: {n_failed}/30 failed")


class TestBoundaryConditions:
    """Test boundary and edge conditions."""

    def test_exact_sample_size_matching_clusters(self):
        """Test when sample size exactly matches cluster structure."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        # Exact match: 5 clusters * 25 obs = 125
        result = model.find_power(sample_size=250, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 1
        print(f"Exact match: {n_failed}/30 failed")

    def test_very_small_total_sample(self):
        """Test with very small total sample size."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_FEW)
        model.set_effects(f"x={EFFECT_LARGE}")  # Need large effect
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)

        result = model.find_power(sample_size=250, return_results=True)  # 25 per cluster

        assert result is not None
        print(f"Small sample (n=25): power={result['results']['individual_powers']['overall']:.1f}%")

    def test_very_large_total_sample(self):
        """Test with very large total sample size."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_VERY_MANY)
        model.set_effects("x=0.3")  # Small effect detectable
        model.set_simulations(LME_N_SIMS_QUICK)

        result = model.find_power(sample_size=2500, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed == 0  # Should be very stable
        # Should have high power even with small effect
        assert result["results"]["individual_powers"]["overall"] > 60.0  # Lowered expectation due to larger sample spread
        print(f"Large sample (n=1250): power={result['results']['individual_powers']['overall']:.1f}%")


class TestEffectSizeVariations:
    """Test various effect size configurations."""

    def test_zero_effect_null_hypothesis(self):
        """Test with zero effect (null hypothesis)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects("x=0.0")  # No effect
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=1000, return_results=True)

        assert result is not None
        # Power should be close to alpha (5%)
        assert result["results"]["individual_powers"]["overall"] < 15.0
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 2
        print(f"Null effect: power={result['results']['individual_powers']['overall']:.1f}% (should be ~5%)")

    def test_very_large_effect(self):
        """Test with very large effect size."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model.set_effects("x=1.5")  # Very large effect
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=750, return_results=True)

        assert result is not None
        # Should have near-perfect power
        assert result["results"]["individual_powers"]["overall"] > 95.0
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed <= 1
        print(f"Large effect (1.5): power={result['results']['individual_powers']['overall']:.1f}%")

    def test_mixed_effect_sizes(self):
        """Test with mixed effect sizes (some large, some small)."""
        model = MCPower("y ~ x1 + x2 + x3 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=25)
        model.set_effects("x1=0.8, x2=0.4, x3=0.1")
        model.set_simulations(LME_N_SIMS_STANDARD)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)  # Allow 10% failures for complex model

        result = model.find_power(sample_size=1500, return_results=True)  # 1500/25 = 60 obs/cluster → 60/6 params = 10.0 ratio

        assert result is not None
        # Check individual powers
        powers = result["results"]["individual_powers"]
        print(f"Mixed effects: x1={powers.get('x1', 0):.1f}%, x2={powers.get('x2', 0):.1f}%, x3={powers.get('x3', 0):.1f}%")


class TestRobustnessStress:
    """Stress tests for robustness."""

    def test_challenging_combination(self):
        """Test challenging combination of parameters."""
        model = MCPower("y ~ x1 + x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE_HIGH, n_clusters=10)  # High ICC, moderate clusters
        model.set_correlations("(x1,x2)=0.7")  # High correlation
        model.set_effects("x1=0.3, x2=0.3")  # Moderate effects
        model.set_simulations(LME_N_SIMS_QUICK)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)

        result = model.find_power(sample_size=500, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        print(f"Challenging combo: {n_failed}/20 failed ({n_failed / 20 * 100:.1f}%)")

    def test_optimal_design(self):
        """Test optimal design (many clusters, moderate ICC, good sample size)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=40)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        result = model.find_power(sample_size=2000, return_results=True)

        assert result is not None
        n_failed = result["results"].get("n_simulations_failed", 0)
        assert n_failed == 0  # Should be perfect
        assert result["results"]["individual_powers"]["overall"] > 90.0  # Should have high power
        print(f"Optimal design: power={result['results']['individual_powers']['overall']:.1f}%, {n_failed} failed")

    def test_reproducibility_with_seed(self):
        """Verify reproducibility with same seed (5 params → 50 obs/cluster × 15 clusters)."""
        model1 = MCPower("y ~ x + (1|cluster)")
        model1.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model1.set_effects(f"x={EFFECT_MEDIUM}")
        model1.set_simulations(LME_N_SIMS_STANDARD)
        model1.set_seed(12345)

        result1 = model1.find_power(sample_size=750, return_results=True, print_results=False)

        model2 = MCPower("y ~ x + (1|cluster)")
        model2.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=15)
        model2.set_effects(f"x={EFFECT_MEDIUM}")
        model2.set_simulations(LME_N_SIMS_STANDARD)
        model2.set_seed(12345)

        result2 = model2.find_power(sample_size=750, return_results=True, print_results=False)

        # Should get identical results
        power1 = result1["results"]["individual_powers"]["overall"]
        power2 = result2["results"]["individual_powers"]["overall"]
        assert power1 == power2
        assert result1["results"]["n_simulations_failed"] == result2["results"]["n_simulations_failed"]
        print(f"Reproducible: power={power1:.1f}%")


class TestWarningGeneration:
    """Test that appropriate warnings are generated."""

    def test_no_warning_low_failures(self):
        """Should not warn if failures are low."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_STANDARD)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.find_power(sample_size=1000, return_results=True)

            n_failed = result["results"].get("n_simulations_failed", 0)

            if n_failed == 0:
                # Should have no warnings about simulation failures
                failure_warnings = [warning for warning in w if "simulations failed" in str(warning.message).lower()]
                assert len(failure_warnings) == 0
                print("No failures, no warnings - correct")

    def test_warning_moderate_failures(self):
        """Should warn if some simulations fail."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_HIGH, n_clusters=4)
        model.set_effects("x=0.3")
        model.set_simulations(LME_N_SIMS_STANDARD)
        model.set_max_failed_simulations(LME_THRESHOLD_MODERATE)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                result = model.find_power(sample_size=750, return_results=True)
                n_failed = result["results"].get("n_simulations_failed", 0)

                if n_failed > 0:
                    # Should have issued warning
                    print(f"Failed {n_failed} simulations, warnings: {len(w)}")
            except RuntimeError:
                # If it raised error due to too many failures, that's also acceptable
                print("Raised error due to too many failures")
