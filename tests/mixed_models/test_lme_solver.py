"""Unit tests for the custom LME solver (mcpower.stats.lme_solver).

Tests the mathematical correctness of:
- Sufficient statistics computation
- Profiled deviance evaluation
- Parameter recovery from known DGP
- Boundary case (tau^2 = 0)
- Warm start correctness
- Critical value precomputation
"""

import numpy as np
import pytest

pytestmark = pytest.mark.lme


class TestSufficientStatistics:
    """Verify per-cluster cross-products are computed correctly."""

    def test_basic_shapes(self):
        """Sufficient stats have correct shapes."""
        from mcpower.stats.lme_solver import compute_sufficient_statistics

        np.random.seed(42)
        K, n_per = 5, 20
        N = K * n_per
        p = 3  # intercept + 2 predictors
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N, p - 1)])
        y = np.random.randn(N)

        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=1)

        assert stats.K == K
        assert stats.q == 1
        assert stats.p == p
        assert stats.N == N
        assert stats.cluster_sizes.shape == (K,)
        assert stats.ZtZ.shape == (K,)
        assert stats.ZtX.shape == (K, p)
        assert stats.Zty.shape == (K,)
        assert stats.XtX.shape == (K, p, p)
        assert stats.Xty.shape == (K, p)
        assert stats.yty.shape == (K,)

    def test_cluster_sizes(self):
        """Cluster sizes are correctly computed."""
        from mcpower.stats.lme_solver import compute_sufficient_statistics

        cluster_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        N = len(cluster_ids)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        y = np.random.randn(N)

        stats = compute_sufficient_statistics(X, y, cluster_ids, K=3, q=1)

        np.testing.assert_array_equal(stats.cluster_sizes, [3, 2, 4])

    def test_cross_products_match_direct(self):
        """Per-cluster cross-products match direct computation."""
        from mcpower.stats.lme_solver import compute_sufficient_statistics

        np.random.seed(123)
        K, n_per = 4, 15
        N = K * n_per
        p = 3
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N, p - 1)])
        y = np.random.randn(N)

        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=1)

        for j in range(K):
            mask = cluster_ids == j
            Xj = X[mask]
            yj = y[mask]
            nj = mask.sum()

            # ZtZ_j = n_j for q=1
            assert stats.ZtZ[j] == nj

            # ZtX_j = colsum(X_j)
            np.testing.assert_allclose(stats.ZtX[j], Xj.sum(axis=0), atol=1e-12)

            # Zty_j = sum(y_j)
            np.testing.assert_allclose(stats.Zty[j], yj.sum(), atol=1e-12)

            # XtX_j = X_j' @ X_j
            np.testing.assert_allclose(stats.XtX[j], Xj.T @ Xj, atol=1e-12)

            # Xty_j = X_j' @ y_j
            np.testing.assert_allclose(stats.Xty[j], Xj.T @ yj, atol=1e-12)

            # yty_j = y_j' @ y_j
            np.testing.assert_allclose(stats.yty[j], yj @ yj, atol=1e-12)


class TestProfiledDeviance:
    """Test the profiled deviance function behavior."""

    def _generate_data(self, K=20, n_per=50, beta=None, sigma2=1.0, tau2=0.5, seed=42):
        """Helper to generate LME data with known parameters."""
        np.random.seed(seed)
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        x = np.random.randn(N)
        X = np.column_stack([np.ones(N), x])
        if beta is None:
            beta = np.array([1.0, 0.5])
        b = np.random.normal(0, np.sqrt(tau2), K)
        y = X @ beta + b[cluster_ids] + np.random.normal(0, np.sqrt(sigma2), N)
        return X, y, cluster_ids, K

    def test_deviance_minimum_near_true_theta(self):
        """Profiled deviance has minimum near the true lambda^2."""
        from mcpower.stats.lme_solver import (
            _profiled_deviance_q1_core,
            compute_sufficient_statistics,
        )

        tau2_true = 0.5
        sigma2_true = 1.0
        lam_sq_true = tau2_true / sigma2_true  # 0.5

        X, y, cluster_ids, K = self._generate_data(
            K=30, n_per=100, tau2=tau2_true, sigma2=sigma2_true, seed=42
        )
        stats = compute_sufficient_statistics(X, y, cluster_ids, K)

        # Evaluate deviance at various lambda^2 values
        lam_sqs = np.logspace(-3, 2, 100)
        deviances = [
            _profiled_deviance_q1_core(
                ls, stats.K, stats.p, stats.N,
                stats.cluster_sizes, stats.ZtZ, stats.ZtX, stats.Zty,
                stats.XtX, stats.Xty, stats.yty, 1
            )
            for ls in lam_sqs
        ]

        # Minimum should be in a reasonable range around true value
        min_idx = np.argmin(deviances)
        optimal_lam_sq = lam_sqs[min_idx]
        # Within factor of 5 of true value (generous for finite sample)
        assert 0.1 <= optimal_lam_sq <= 2.5, (
            f"Optimal lam_sq={optimal_lam_sq:.3f}, expected near {lam_sq_true}"
        )

    def test_deviance_monotone_away_from_minimum(self):
        """Deviance increases away from minimum (unimodal)."""
        from mcpower.stats.lme_solver import (
            _profiled_deviance_q1_core,
            compute_sufficient_statistics,
            lme_fit,
        )

        X, y, cluster_ids, K = self._generate_data(K=20, n_per=50, tau2=0.3, seed=77)
        stats = compute_sufficient_statistics(X, y, cluster_ids, K)
        result = lme_fit(X, y, cluster_ids, K, reml=True)

        opt_lam_sq = result.theta[0] ** 2

        def dev(ls):
            return _profiled_deviance_q1_core(
                ls, stats.K, stats.p, stats.N,
                stats.cluster_sizes, stats.ZtZ, stats.ZtX, stats.Zty,
                stats.XtX, stats.Xty, stats.yty, 1
            )

        dev_opt = dev(opt_lam_sq)

        # Values far from optimum should have higher deviance
        if opt_lam_sq > 0.01:
            assert dev(opt_lam_sq * 0.01) > dev_opt
        assert dev(opt_lam_sq * 100 + 100) > dev_opt


class TestParameterRecovery:
    """Test that lme_fit recovers known parameters."""

    def test_recover_beta(self):
        """Fixed effects are recovered accurately."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 30, 100
        N = K * n_per
        beta_true = np.array([2.0, -0.8])
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, 0.5, K)
        y = X @ beta_true + b[cluster_ids] + np.random.randn(N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        # Intercept recovery is noisy with random effects, slope is tighter
        np.testing.assert_allclose(result.beta[1:], beta_true[1:], atol=0.1)
        assert abs(result.beta[0] - beta_true[0]) < 0.5  # Intercept has wider CI

    def test_recover_variance_components(self):
        """sigma^2 and tau^2 are recovered within reasonable tolerance."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 50, 100
        N = K * n_per
        sigma2_true = 1.0
        tau2_true = 0.5
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, np.sqrt(tau2_true), K)
        y = X @ np.array([1.0, 0.5]) + b[cluster_ids] + np.random.normal(0, np.sqrt(sigma2_true), N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        assert abs(result.sigma2 - sigma2_true) < 0.15, f"sigma2={result.sigma2:.3f}"
        assert abs(result.tau2 - tau2_true) < 0.25, f"tau2={result.tau2:.3f}"

    def test_recover_standard_errors(self):
        """Standard errors are reasonable (positive and finite)."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, 0.5, K)
        y = X @ np.array([1.0, 0.5]) + b[cluster_ids] + np.random.randn(N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        assert np.all(result.se_beta > 0), "SEs should be positive"
        assert np.all(np.isfinite(result.se_beta)), "SEs should be finite"
        # Intercept SE should be larger than slope SE (due to cluster effect)
        assert result.se_beta[0] > result.se_beta[1]


class TestBoundaryCase:
    """Test when tau^2 = 0 (no random effects)."""

    def test_tau2_zero_recovers_ols(self):
        """When tau^2 = 0, beta and SE should match OLS."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(N)  # No cluster effects

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        # tau2 should be near zero
        assert result.tau2 < 0.05, f"tau2={result.tau2:.4f} should be near 0"

        # Beta should be close to OLS estimate
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(result.beta, beta_ols, atol=0.01)

    def test_tau2_zero_standard_errors(self):
        """When tau^2 = 0, SEs should be close to OLS SEs."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        # OLS SEs
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta_ols
        mse_ols = np.sum(resid**2) / (N - 2)
        se_ols = np.sqrt(np.diag(mse_ols * np.linalg.inv(X.T @ X)))

        # LME SEs should be close to OLS SEs when tau2 ≈ 0
        np.testing.assert_allclose(result.se_beta, se_ols, rtol=0.1)


class TestWarmStart:
    """Test warm start produces same results as cold start."""

    def test_warm_start_same_result(self):
        """Warm-started fit should produce identical results to cold start."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, 0.5, K)
        y = X @ np.array([1.0, 0.5]) + b[cluster_ids] + np.random.randn(N)

        cold = lme_fit(X, y, cluster_ids, K, reml=True)
        warm = lme_fit(X, y, cluster_ids, K, reml=True, warm_theta=cold.theta[0])

        np.testing.assert_allclose(cold.beta, warm.beta, atol=1e-6)
        np.testing.assert_allclose(cold.sigma2, warm.sigma2, atol=1e-6)
        np.testing.assert_allclose(cold.tau2, warm.tau2, atol=1e-4)


class TestMLFitting:
    """Test ML (not REML) fitting for LR tests."""

    def test_ml_vs_reml(self):
        """ML and REML should give similar but not identical results."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, 0.5, K)
        y = X @ np.array([1.0, 0.5]) + b[cluster_ids] + np.random.randn(N)

        reml_result = lme_fit(X, y, cluster_ids, K, reml=True)
        ml_result = lme_fit(X, y, cluster_ids, K, reml=False)

        # Beta should be very close
        np.testing.assert_allclose(reml_result.beta, ml_result.beta, atol=0.01)

        # REML sigma2 is typically slightly larger than ML sigma2
        # (because REML divides by N-p, ML divides by N)
        assert reml_result.sigma2 >= ml_result.sigma2 * 0.99

    def test_null_model_fit(self):
        """Null model (intercept only) should fit successfully."""
        from mcpower.stats.lme_solver import lme_fit_ml_null

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        b = np.random.normal(0, 0.5, K)
        y = 1.0 + b[cluster_ids] + np.random.randn(N)

        result = lme_fit_ml_null(y, cluster_ids, K)

        assert result.converged
        assert len(result.beta) == 1  # Intercept only
        assert np.isfinite(result.log_likelihood)

    def test_lr_stat_positive(self):
        """LR statistic should be positive for a model with a real effect."""
        from mcpower.stats.lme_solver import lme_fit, lme_fit_ml_null

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        b = np.random.normal(0, 0.5, K)
        y = X @ np.array([1.0, 0.5]) + b[cluster_ids] + np.random.randn(N)

        full_ml = lme_fit(X, y, cluster_ids, K, reml=False)
        null_ml = lme_fit_ml_null(y, cluster_ids, K)

        lr_stat = 2 * (full_ml.log_likelihood - null_ml.log_likelihood)
        assert lr_stat > 0, f"LR stat should be positive, got {lr_stat}"


class TestCriticalValues:
    """Test LME critical value precomputation."""

    def test_basic_values(self):
        """Critical values should be positive and finite."""
        from mcpower.stats.lme_solver import compute_lme_critical_values

        chi2_crit, z_crit, corr_z = compute_lme_critical_values(
            alpha=0.05, n_fixed=2, n_targets=2, correction_method=0
        )

        assert chi2_crit > 0
        assert z_crit > 0
        assert np.isfinite(chi2_crit)
        assert np.isfinite(z_crit)
        assert len(corr_z) == 2
        np.testing.assert_allclose(corr_z, [z_crit, z_crit])

    def test_bonferroni_correction(self):
        """Bonferroni correction yields larger critical values."""
        from mcpower.stats.lme_solver import compute_lme_critical_values

        _, z_none, _ = compute_lme_critical_values(0.05, 2, 3, 0)
        _, z_bonf, corr_bonf = compute_lme_critical_values(0.05, 2, 3, 1)

        # Bonferroni critical z should be larger
        assert corr_bonf[0] > z_none

    def test_fdr_correction_decreasing(self):
        """FDR critical values should be decreasing (largest for rank 1)."""
        from mcpower.stats.lme_solver import compute_lme_critical_values

        _, _, corr_fdr = compute_lme_critical_values(0.05, 2, 5, 2)

        assert len(corr_fdr) == 5
        # FDR crits should be non-increasing
        for i in range(len(corr_fdr) - 1):
            assert corr_fdr[i] >= corr_fdr[i + 1] - 1e-10

    def test_holm_correction(self):
        """Holm correction yields non-increasing critical values."""
        from mcpower.stats.lme_solver import compute_lme_critical_values

        _, _, corr_holm = compute_lme_critical_values(0.05, 2, 4, 3)

        assert len(corr_holm) == 4
        for i in range(len(corr_holm) - 1):
            assert corr_holm[i] >= corr_holm[i + 1] - 1e-10


class TestMultipleFixedEffects:
    """Test with more than one predictor."""

    def test_two_predictors(self):
        """Solver handles two predictors correctly."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        x1 = np.random.randn(N)
        x2 = np.random.randn(N)
        X = np.column_stack([np.ones(N), x1, x2])
        beta_true = np.array([1.0, 0.5, -0.3])
        b = np.random.normal(0, 0.5, K)
        y = X @ beta_true + b[cluster_ids] + np.random.randn(N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        assert result.converged
        # Slope recovery should be good; intercept is noisier
        np.testing.assert_allclose(result.beta[1:], beta_true[1:], atol=0.15)
        assert abs(result.beta[0] - beta_true[0]) < 0.5
        assert result.cov_beta.shape == (3, 3)
        assert len(result.se_beta) == 3

    def test_many_predictors(self):
        """Solver handles 5+ predictors."""
        from mcpower.stats.lme_solver import lme_fit

        np.random.seed(42)
        K, n_per = 20, 100
        N = K * n_per
        p_pred = 5
        cluster_ids = np.repeat(np.arange(K), n_per)
        X_pred = np.random.randn(N, p_pred)
        X = np.column_stack([np.ones(N), X_pred])
        beta_true = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.4])
        b = np.random.normal(0, 0.3, K)
        y = X @ beta_true + b[cluster_ids] + np.random.randn(N)

        result = lme_fit(X, y, cluster_ids, K, reml=True)

        assert result.converged
        np.testing.assert_allclose(result.beta, beta_true, atol=0.2)
