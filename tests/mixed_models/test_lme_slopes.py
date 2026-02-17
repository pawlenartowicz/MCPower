"""Tests for LME solver with random slopes (q > 1).

Generates data with known parameters, fits with the custom solver,
and verifies parameter recovery and agreement with statsmodels.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.lme


def _generate_slope_data(
    n_clusters=30, cluster_size=50, beta=None, sigma2=1.0,
    tau_int=0.5, tau_slope=0.3, rho=0.2, seed=42,
):
    """Generate data from a random-slope model (1 + x | group).

    y_ij = beta0 + beta1*x_ij + b0_j + b1_j*x_ij + eps_ij
    [b0_j, b1_j] ~ MVN(0, G)
    G = [[tau_int^2, rho*tau_int*tau_slope], [rho*tau_int*tau_slope, tau_slope^2]]
    """
    if beta is None:
        beta = np.array([1.0, 0.5])  # intercept, slope

    rng = np.random.RandomState(seed)
    N = n_clusters * cluster_size

    G = np.array([
        [tau_int**2, rho * tau_int * tau_slope],
        [rho * tau_int * tau_slope, tau_slope**2],
    ])

    # Random effects
    b = rng.multivariate_normal([0, 0], G, size=n_clusters)

    # Fixed effects design
    x = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])

    # Cluster assignments
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

    # Z matrix: [1, x_ij] for each observation
    Z = np.column_stack([np.ones(N), x])

    # Generate y
    y = X @ beta
    for j in range(n_clusters):
        mask = cluster_ids == j
        y[mask] += b[j, 0] + b[j, 1] * x[mask]
    y += rng.normal(0, np.sqrt(sigma2), N)

    return X, y, x, Z, cluster_ids, n_clusters, G


class TestGeneralSufficientStatistics:
    """Test that general sufficient statistics are computed correctly."""

    def test_general_stats_shape(self):
        """Check shapes of general sufficient statistics."""
        from mcpower.stats.lme_solver import compute_sufficient_statistics

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=2, Z=Z)

        assert stats.ZtZ.shape == (K, 2, 2)
        assert stats.ZtX.shape == (K, 2, 2)
        assert stats.Zty.shape == (K, 2)
        assert stats.XtX.shape == (K, 2, 2)
        assert stats.cluster_sizes.sum() == len(y)

    def test_general_stats_match_direct(self):
        """Verify general sufficient stats match direct per-cluster computation."""
        from mcpower.stats.lme_solver import compute_sufficient_statistics

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data(
            n_clusters=5, cluster_size=20
        )
        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=2, Z=Z)

        # Verify for first cluster
        mask = cluster_ids == 0
        X0 = X[mask]
        y0 = y[mask]
        Z0 = Z[mask]

        np.testing.assert_allclose(stats.ZtZ[0], Z0.T @ Z0, atol=1e-10)
        np.testing.assert_allclose(stats.ZtX[0], Z0.T @ X0, atol=1e-10)
        np.testing.assert_allclose(stats.Zty[0], Z0.T @ y0, atol=1e-10)
        np.testing.assert_allclose(stats.XtX[0], X0.T @ X0, atol=1e-10)


class TestGeneralProfiledDeviance:
    """Test general profiled deviance computation."""

    def test_deviance_finite(self):
        """Profiled deviance should return finite values for reasonable theta."""
        from mcpower.stats.lme_solver import (
            _profiled_deviance_general,
            compute_sufficient_statistics,
        )

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=2, Z=Z)

        # theta = [lambda_11, lambda_21, lambda_22]
        theta = np.array([1.0, 0.0, 1.0])
        dev = _profiled_deviance_general(theta, stats, 1)
        assert np.isfinite(dev)

    def test_deviance_identity_theta(self):
        """Identity T should give a valid deviance."""
        from mcpower.stats.lme_solver import (
            _profiled_deviance_general,
            compute_sufficient_statistics,
        )

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        stats = compute_sufficient_statistics(X, y, cluster_ids, K, q=2, Z=Z)

        theta = np.array([1.0, 0.0, 1.0])
        dev = _profiled_deviance_general(theta, stats, 1)
        assert dev > 0
        assert dev < 1e20


class TestFitGeneral:
    """Test the full q>1 fitting pipeline."""

    def test_fit_recovers_beta(self):
        """Beta estimates should be close to true values with large data."""
        from mcpower.stats.lme_solver import lme_fit

        true_beta = np.array([1.0, 0.5])
        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data(
            n_clusters=50, cluster_size=100, beta=true_beta,
        )

        result = lme_fit(X, y, cluster_ids, K, q=2, Z=Z)
        assert result.converged
        np.testing.assert_allclose(result.beta, true_beta, atol=0.15)

    def test_fit_se_positive(self):
        """Standard errors should be positive."""
        from mcpower.stats.lme_solver import lme_fit

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        result = lme_fit(X, y, cluster_ids, K, q=2, Z=Z)
        assert result.converged
        assert np.all(result.se_beta > 0)

    def test_fit_G_matrix_psd(self):
        """Estimated G matrix should be positive semi-definite."""
        from mcpower.stats.lme_solver import lme_fit

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        result = lme_fit(X, y, cluster_ids, K, q=2, Z=Z)
        G = result.G
        eigenvals = np.linalg.eigvalsh(G)
        assert np.all(eigenvals >= -1e-10)

    def test_analysis_general_returns_array(self):
        """lme_analysis_general should return a results array."""
        from mcpower.stats.lme_solver import (
            compute_lme_critical_values,
            lme_analysis_general,
        )

        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data()
        # X without intercept for analysis function
        X_no_int = X[:, 1:]
        target_indices = np.array([0])

        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(
            0.05, 1, 1, 0
        )
        result = lme_analysis_general(
            X_no_int, y, cluster_ids, K, q=2, Z=Z,
            target_indices=target_indices,
            chi2_crit=chi2_crit, z_crit=z_crit,
            correction_z_crits=correction_z_crits,
            correction_method=0,
        )
        assert result is not None
        assert len(result) == 1 + 2 * 1 + 1  # f_sig + 2*n_targets + wald_flag


@pytest.mark.slow
class TestSlopesVsStatsmodels:
    """Cross-validate random slope solver against statsmodels."""

    def test_beta_matches_statsmodels(self):
        """Beta estimates should match statsmodels within tolerance."""
        pytest.importorskip("statsmodels")
        from statsmodels.regression.mixed_linear_model import MixedLM

        from mcpower.stats.lme_solver import lme_fit

        true_beta = np.array([1.0, 0.5])
        X, y, x, Z, cluster_ids, K, _ = _generate_slope_data(
            n_clusters=30, cluster_size=50, beta=true_beta, seed=123
        )

        # Custom solver
        custom_result = lme_fit(X, y, cluster_ids, K, q=2, Z=Z)

        # Statsmodels
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_model = MixedLM(y, X, cluster_ids, exog_re=Z)
            sm_result = sm_model.fit(reml=True, method="lbfgs", maxiter=500)

        np.testing.assert_allclose(
            custom_result.beta, sm_result.fe_params, atol=0.05
        )
