"""
Validation tests for mixed models comparing MCPower with direct statsmodels/scipy usage.

These tests verify:
- ICC recovery from generated data
- Fixed effect estimation accuracy
- Type I error control (false positive rate)
- Power against theoretical calculations
"""

import numpy as np
import pytest

from mcpower import MCPower
from tests.config import (
    DEFAULT_ALPHA,
    EFFECT_LARGE,
    EFFECT_MEDIUM,
    ICC_LOW,
    ICC_MODERATE,
    ICC_MODERATE_HIGH,
    ICC_RECOVERY_TOLERANCE_HIGH,
    ICC_RECOVERY_TOLERANCE_LOW,
    LME_N_SIMS_RIGOROUS,
    LME_N_SIMS_VALIDATION,
    N_CLUSTERS_MANY,
    N_CLUSTERS_MODERATE,
    N_CLUSTERS_VERY_MANY,
    THEORETICAL_POWER_TOLERANCE,
    TYPE1_ERROR_RANGE,
)

pytestmark = pytest.mark.lme


class TestICCRecovery:
    """Verify that generated data has correct ICC."""

    def test_icc_recovery_low(self):
        """Verify ICC=0.1 is correctly recovered from generated data."""
        from statsmodels.regression.mixed_linear_model import MixedLM

        np.random.seed(42)

        # Generate data using MCPower's internal functions
        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 500
        n_clusters = 20
        icc_target = ICC_LOW

        # Generate cluster effects
        # Corrected τ² accounts for fixed-effect variance: β=0.5, Var(X)=1.0
        sigma_sq_fixed = 0.5**2 * 1.0
        sigma_sq_within = 1.0 + sigma_sq_fixed
        tau_sq = icc_target / (1.0 - icc_target) * sigma_sq_within
        cluster_specs = {"cluster": {"n_clusters": n_clusters, "cluster_size": sample_size // n_clusters, "tau_squared": tau_sq}}
        X_cluster = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=42)
        cluster_ids = np.repeat(np.arange(n_clusters), sample_size // n_clusters)

        # Generate fixed effects
        X_fixed = np.random.randn(sample_size, 1)

        # Generate y = fixed_effect * X + cluster_effects + noise
        y = 0.5 * X_fixed[:, 0] + X_cluster[:, 0] + np.random.randn(sample_size)

        # Fit with statsmodels directly
        X_with_intercept = np.column_stack([np.ones(sample_size), X_fixed])
        lme = MixedLM(endog=y, exog=X_with_intercept, groups=cluster_ids)
        result = lme.fit(reml=True, maxiter=200)

        # Recover marginal ICC (includes fixed-effect variance in denominator)
        cov_re = result.cov_re
        if hasattr(cov_re, "iloc"):
            tau_sq_est = cov_re.iloc[0, 0]
        else:
            tau_sq_est = cov_re[0, 0]
        sigma_sq_est = result.scale
        icc_recovered = tau_sq_est / (tau_sq_est + sigma_sq_est + sigma_sq_fixed)

        # Should be close to input ICC (within tolerance)
        assert abs(icc_recovered - icc_target) < ICC_RECOVERY_TOLERANCE_LOW
        print(f"Target ICC: {icc_target:.3f}, Recovered: {icc_recovered:.3f}")

    def test_icc_recovery_medium(self):
        """Verify ICC=0.3 is correctly recovered."""
        from statsmodels.regression.mixed_linear_model import MixedLM

        np.random.seed(123)

        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 1000
        n_clusters = 20
        icc_target = ICC_MODERATE_HIGH

        # Corrected τ² accounts for fixed-effect variance: β=0.5, Var(X)=1.0
        sigma_sq_fixed = 0.5**2 * 1.0
        sigma_sq_within = 1.0 + sigma_sq_fixed
        tau_sq = icc_target / (1.0 - icc_target) * sigma_sq_within
        cluster_specs = {"cluster": {"n_clusters": n_clusters, "cluster_size": sample_size // n_clusters, "tau_squared": tau_sq}}
        X_cluster = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=123)
        cluster_ids = np.repeat(np.arange(n_clusters), sample_size // n_clusters)

        X_fixed = np.random.randn(sample_size, 1)
        y = 0.5 * X_fixed[:, 0] + X_cluster[:, 0] + np.random.randn(sample_size)

        X_with_intercept = np.column_stack([np.ones(sample_size), X_fixed])
        lme = MixedLM(endog=y, exog=X_with_intercept, groups=cluster_ids)
        result = lme.fit(reml=True, maxiter=200)

        # Recover marginal ICC (includes fixed-effect variance in denominator)
        cov_re = result.cov_re
        if hasattr(cov_re, "iloc"):
            tau_sq_est = cov_re.iloc[0, 0]
        else:
            tau_sq_est = cov_re[0, 0]
        sigma_sq_est = result.scale
        icc_recovered = tau_sq_est / (tau_sq_est + sigma_sq_est + sigma_sq_fixed)

        assert abs(icc_recovered - icc_target) < ICC_RECOVERY_TOLERANCE_HIGH
        print(f"Target ICC: {icc_target:.3f}, Recovered: {icc_recovered:.3f}")


class TestFixedEffectRecovery:
    """Verify that fixed effect estimates are accurate."""

    def test_single_effect_recovery(self):
        """Verify β=0.5 is correctly estimated."""
        from statsmodels.regression.mixed_linear_model import MixedLM

        np.random.seed(42)

        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 500
        n_clusters = 25
        icc = 0.2
        true_effect = 0.5

        # Corrected τ² accounts for fixed-effect variance: β=0.5, Var(X)=1.0
        sigma_sq_fixed = true_effect**2 * 1.0
        sigma_sq_within = 1.0 + sigma_sq_fixed
        tau_sq = icc / (1.0 - icc) * sigma_sq_within
        cluster_specs = {"cluster": {"n_clusters": n_clusters, "cluster_size": sample_size // n_clusters, "tau_squared": tau_sq}}
        X_cluster = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=42)
        cluster_ids = np.repeat(np.arange(n_clusters), sample_size // n_clusters)

        X_fixed = np.random.randn(sample_size, 1)
        y = true_effect * X_fixed[:, 0] + X_cluster[:, 0] + np.random.randn(sample_size)

        X_with_intercept = np.column_stack([np.ones(sample_size), X_fixed])
        lme = MixedLM(endog=y, exog=X_with_intercept, groups=cluster_ids)
        result = lme.fit(reml=True, maxiter=200)

        estimated_effect = result.fe_params[1]  # Exclude intercept

        # Should be within reasonable range
        assert 0.35 < estimated_effect < 0.65
        print(f"True effect: {true_effect:.3f}, Estimated: {estimated_effect:.3f}")


class TestTypeIErrorControl:
    """Verify Type I error rate (false positive rate) under null hypothesis."""

    def test_null_hypothesis_alpha_control(self):
        """With effect_size=0, power should be approximately alpha (5%)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
        model.set_effects("x=0.0")  # NULL hypothesis
        model.set_simulations(LME_N_SIMS_VALIDATION)
        model.set_seed(42)

        result = model.find_power(sample_size=1000, return_results=True)

        power = result["results"]["individual_powers"]["overall"]

        # Should be close to alpha (within TYPE1_ERROR_RANGE)
        assert DEFAULT_ALPHA * 100 - TYPE1_ERROR_RANGE < power < DEFAULT_ALPHA * 100 + TYPE1_ERROR_RANGE
        print(f"Type I error rate: {power:.1f}% (expected ~5%)")


class TestPowerVsTheoretical:
    """Compare empirical power to theoretical power calculations."""

    def calculate_theoretical_power(self, effect_size, sample_size, n_clusters, icc, alpha=0.05):
        """
        Calculate theoretical power using design effect formula.

        Deff = 1 + (m-1)*ICC, where m = cluster_size
        effective_n = n / Deff
        """
        from scipy.stats import norm

        cluster_size = sample_size / n_clusters
        design_effect = 1 + (cluster_size - 1) * icc
        effective_n = sample_size / design_effect

        # Approximate power with normal distribution
        t_stat = effect_size * np.sqrt(effective_n)
        t_crit = norm.ppf(1 - alpha / 2)

        power = 1 - norm.cdf(t_crit - t_stat) + norm.cdf(-t_crit - t_stat)
        return power * 100

    def test_power_matches_theoretical_low_icc(self):
        """Power should match theoretical calculation (ICC=0.1, 5 params → 50 obs/cluster × 25 clusters)."""
        effect = EFFECT_MEDIUM
        n = 1250  # 5 params → 50 obs/cluster × 25 clusters
        n_clusters = 25
        icc = ICC_LOW

        theoretical = self.calculate_theoretical_power(effect, n, n_clusters, icc)

        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=icc, n_clusters=n_clusters)
        model.set_effects(f"x={effect}")
        model.set_simulations(LME_N_SIMS_RIGOROUS)
        model.set_seed(42)

        result = model.find_power(sample_size=n, return_results=True)
        empirical = result["results"]["individual_powers"]["overall"]

        # Should be within 20% of theoretical (Monte Carlo variation)
        assert abs(empirical - theoretical) < THEORETICAL_POWER_TOLERANCE
        print(f"Theoretical: {theoretical:.1f}%, Empirical: {empirical:.1f}%")

    def test_power_matches_theoretical_medium_icc(self):
        """Power should match theoretical calculation (ICC=0.3, 5 params → 50 obs/cluster × 30 clusters)."""
        effect = EFFECT_MEDIUM
        n = 1500  # 5 params → 50 obs/cluster × 30 clusters
        n_clusters = N_CLUSTERS_MANY
        icc = ICC_MODERATE_HIGH

        theoretical = self.calculate_theoretical_power(effect, n, n_clusters, icc)

        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=icc, n_clusters=n_clusters)
        model.set_effects(f"x={effect}")
        model.set_simulations(LME_N_SIMS_RIGOROUS)
        model.set_seed(123)

        result = model.find_power(sample_size=n, return_results=True)
        empirical = result["results"]["individual_powers"]["overall"]

        # Should be within 20% of theoretical
        assert abs(empirical - theoretical) < THEORETICAL_POWER_TOLERANCE
        print(f"Theoretical: {theoretical:.1f}%, Empirical: {empirical:.1f}%")


class TestConvergenceDiagnostics:
    """Verify diagnostic output in verbose mode."""

    def test_diagnostics_available(self):
        """Verify diagnostics are captured when using internal API."""
        from mcpower.stats.mixed_models import _lme_analysis_wrapper

        np.random.seed(42)

        # Generate simple test data
        sample_size = 500
        n_clusters = 20
        cluster_ids = np.repeat(np.arange(n_clusters), sample_size // n_clusters)

        X = np.random.randn(sample_size, 1)
        y = 0.5 * X[:, 0] + np.random.randn(sample_size)

        result = _lme_analysis_wrapper(
            X_expanded=X,
            y=y,
            target_indices=np.array([0]),
            cluster_ids=cluster_ids,

            correction_method=0,
            alpha=0.05,
            backend="statsmodels",
            verbose=True,
        )

        assert "diagnostics" in result
        diagnostics = result["diagnostics"]

        # Check expected fields
        assert "converged" in diagnostics
        assert "convergence_level" in diagnostics
        assert "test_method" in diagnostics
        assert "log_likelihood" in diagnostics
        assert "fixed_effects" in diagnostics

        print(f"Test method used: {diagnostics['test_method']}")
        print(f"Converged: {diagnostics['converged']}")
        print(f"Convergence level: {diagnostics['convergence_level']}")


class TestPositiveControls:
    """Positive controls that SHOULD have high power."""

    def test_large_effect_many_clusters_low_icc(self):
        """Ideal design: large effect, many clusters, low ICC."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_LOW, n_clusters=N_CLUSTERS_VERY_MANY)
        model.set_effects(f"x={EFFECT_LARGE}")
        model.set_simulations(LME_N_SIMS_RIGOROUS)
        model.set_seed(42)

        result = model.find_power(sample_size=2500, return_results=True)

        # This MUST achieve high power
        power = result["results"]["individual_powers"]["overall"]
        assert power > 85
        assert result["results"]["n_simulations_failed"] < 5

    def test_moderate_effect_adequate_sample(self):
        """Moderate effect with adequate sample size."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MANY)
        model.set_effects(f"x={EFFECT_MEDIUM}")
        model.set_simulations(LME_N_SIMS_RIGOROUS)
        model.set_seed(456)

        result = model.find_power(sample_size=1500, return_results=True)

        # Should achieve target power
        power = result["results"]["individual_powers"]["overall"]
        assert power > 80
        assert result["results"]["n_simulations_failed"] < 3
