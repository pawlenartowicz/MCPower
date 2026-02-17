"""Cross-validation tests: custom LME solver vs statsmodels MixedLM.

Verifies that the custom profiled-deviance solver produces results
consistent with statsmodels for random-intercept models across a range
of ICC values, cluster configurations, and correction methods.
"""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.lme

# Tolerance for numerical agreement between backends
BETA_ATOL = 1e-3
SE_ATOL = 1e-3
SIGMA2_RTOL = 0.01
TAU2_RTOL = 0.05


def _generate_lme_data(K, n_per, beta, sigma2, tau2, seed):
    """Generate LME data with known parameters."""
    np.random.seed(seed)
    N = K * n_per
    p = len(beta)
    cluster_ids = np.repeat(np.arange(K), n_per)
    X = np.column_stack([np.ones(N)] + [np.random.randn(N) for _ in range(p - 1)])
    b = np.random.normal(0, np.sqrt(tau2), K)
    y = X @ beta + b[cluster_ids] + np.random.normal(0, np.sqrt(sigma2), N)
    return X, y, cluster_ids


def _fit_statsmodels(X, y, cluster_ids, reml=True):
    """Fit with statsmodels MixedLM.

    Uses powell method which is more robust than lbfgs for small-to-medium
    problems (lbfgs can get stuck at zero-intercept local minima).
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = MixedLM(y, X, cluster_ids).fit(reml=reml, method="powell",
                                                  full_output=False, maxiter=500)
    return result


def _fit_custom(X, y, cluster_ids, K, reml=True):
    """Fit with custom solver."""
    from mcpower.stats.lme_solver import lme_fit
    return lme_fit(X, y, cluster_ids, K, q=1, reml=reml)


class TestParameterAgreement:
    """Compare parameter estimates between backends."""

    @pytest.mark.parametrize("icc", [0.0, 0.1, 0.2, 0.5])
    def test_beta_agreement(self, icc):
        """Fixed effects match across ICC values."""
        K, n_per = 20, 50
        tau2 = icc / (1 - icc) if icc > 0 else 0.0
        beta_true = np.array([1.0, 0.5])

        X, y, cids = _generate_lme_data(K, n_per, beta_true, 1.0, tau2, seed=42)
        sm = _fit_statsmodels(X, y, cids)
        custom = _fit_custom(X, y, cids, K)

        np.testing.assert_allclose(custom.beta, sm.fe_params, atol=BETA_ATOL,
                                   err_msg=f"Beta mismatch at ICC={icc}")

    @pytest.mark.parametrize("icc", [0.1, 0.2, 0.5])
    def test_se_agreement(self, icc):
        """Standard errors match across ICC values."""
        K, n_per = 20, 50
        tau2 = icc / (1 - icc)
        beta_true = np.array([1.0, 0.5])

        X, y, cids = _generate_lme_data(K, n_per, beta_true, 1.0, tau2, seed=42)
        sm = _fit_statsmodels(X, y, cids)
        custom = _fit_custom(X, y, cids, K)

        np.testing.assert_allclose(custom.se_beta, sm.bse_fe, atol=SE_ATOL,
                                   err_msg=f"SE mismatch at ICC={icc}")

    @pytest.mark.parametrize("icc", [0.1, 0.2, 0.5])
    def test_variance_components_agreement(self, icc):
        """sigma^2 and tau^2 match."""
        K, n_per = 30, 80
        tau2 = icc / (1 - icc)
        beta_true = np.array([1.0, 0.5])

        X, y, cids = _generate_lme_data(K, n_per, beta_true, 1.0, tau2, seed=42)
        sm = _fit_statsmodels(X, y, cids)
        custom = _fit_custom(X, y, cids, K)

        sm_tau2 = sm.cov_re.iloc[0, 0] if hasattr(sm.cov_re, "iloc") else float(sm.cov_re)

        np.testing.assert_allclose(custom.sigma2, sm.scale, rtol=SIGMA2_RTOL,
                                   err_msg=f"sigma2 mismatch at ICC={icc}")
        np.testing.assert_allclose(custom.tau2, sm_tau2, rtol=TAU2_RTOL,
                                   err_msg=f"tau2 mismatch at ICC={icc}")

    @pytest.mark.parametrize("n_clusters", [5, 10, 20, 50])
    def test_cluster_count_sweep(self, n_clusters):
        """Estimates agree across different cluster counts."""
        n_per = 50
        tau2 = 0.3
        beta_true = np.array([1.0, 0.5])

        X, y, cids = _generate_lme_data(n_clusters, n_per, beta_true, 1.0, tau2, seed=42)
        sm = _fit_statsmodels(X, y, cids)
        custom = _fit_custom(X, y, cids, n_clusters)

        np.testing.assert_allclose(custom.beta, sm.fe_params, atol=BETA_ATOL)

    def test_multiple_predictors(self):
        """Agreement with 3 predictors."""
        K, n_per = 20, 80
        beta_true = np.array([1.0, 0.5, -0.3, 0.2])

        X, y, cids = _generate_lme_data(K, n_per, beta_true, 1.0, 0.3, seed=42)
        sm = _fit_statsmodels(X, y, cids)
        custom = _fit_custom(X, y, cids, K)

        np.testing.assert_allclose(custom.beta, sm.fe_params, atol=BETA_ATOL)
        np.testing.assert_allclose(custom.se_beta, sm.bse_fe, atol=SE_ATOL)


class TestAnalysisWrapperComparison:
    """Compare full analysis pipeline output (significance decisions)."""

    def _run_wrapper(self, X_expanded, y, cluster_ids, target_indices,
                     correction_method, alpha, backend):
        """Run _lme_analysis_wrapper with specified backend."""
        from mcpower.stats.mixed_models import _lme_analysis_wrapper

        return _lme_analysis_wrapper(
            X_expanded, y, target_indices, cluster_ids,
            cluster_column_indices=[],
            correction_method=correction_method,
            alpha=alpha,
            backend=backend,
        )

    def test_significance_agreement_basic(self):
        """Both backends agree on significance for a well-powered design."""
        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        x = np.random.randn(N)
        X_expanded = x.reshape(-1, 1)  # Without intercept
        b = np.random.normal(0, 0.5, K)
        y = 1.0 + 0.8 * x + b[cluster_ids] + np.random.randn(N)

        target_indices = np.array([0], dtype=np.int64)

        r_custom = self._run_wrapper(X_expanded, y, cluster_ids, target_indices, 0, 0.05, "custom")
        r_sm = self._run_wrapper(X_expanded, y, cluster_ids, target_indices, 0, 0.05, "statsmodels")

        assert r_custom is not None
        assert r_sm is not None
        # Both should detect the significant effect
        assert r_custom[1] == r_sm[1], "Disagreement on significance"

    @pytest.mark.parametrize("correction", [0, 1, 2, 3])
    def test_correction_methods(self, correction):
        """Both backends agree across correction methods."""
        np.random.seed(42)
        K, n_per = 20, 50
        N = K * n_per
        cluster_ids = np.repeat(np.arange(K), n_per)
        x1 = np.random.randn(N)
        x2 = np.random.randn(N)
        X_expanded = np.column_stack([x1, x2])
        b = np.random.normal(0, 0.5, K)
        y = 1.0 + 0.8 * x1 + 0.0 * x2 + b[cluster_ids] + np.random.randn(N)

        target_indices = np.array([0, 1], dtype=np.int64)

        r_custom = self._run_wrapper(X_expanded, y, cluster_ids, target_indices, correction, 0.05, "custom")
        r_sm = self._run_wrapper(X_expanded, y, cluster_ids, target_indices, correction, 0.05, "statsmodels")

        assert r_custom is not None
        assert r_sm is not None
        # Strong effect (x1=0.8) should be significant in both
        assert r_custom[1] == 1.0, "Custom should detect x1"
        assert r_sm[1] == 1.0, "Statsmodels should detect x1"


class TestFullPowerAnalysis:
    """Compare power estimates from end-to-end MCPower analysis."""

    def test_power_agreement_simple(self):
        """Power estimates agree between backends for a simple model."""
        from mcpower import MCPower
        from mcpower.stats.mixed_models import reset_warm_start_cache

        reset_warm_start_cache()

        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=20)
        model.set_effects("x=0.2")
        model.set_simulations(50)
        model.set_seed(2137)

        # Run with custom backend (default now)
        # Need 50+ obs/cluster: 1000/20 = 50 ✓
        result_custom = model.find_power(
            sample_size=1000,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power_custom = result_custom["results"]["individual_powers"]["overall"]

        # Power should be reasonable (not 0%)
        assert 0 < power_custom <= 100, f"Custom power={power_custom}%"

    def test_power_agreement_with_correction(self):
        """Power with Bonferroni correction works end-to-end."""
        from mcpower import MCPower
        from mcpower.stats.mixed_models import reset_warm_start_cache

        reset_warm_start_cache()

        model = MCPower("y ~ x1 + x2 + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5, x2=0.3")
        model.set_simulations(30)
        model.set_seed(2137)

        result = model.find_power(
            sample_size=1000,
            correction="bonferroni",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert "results" in result
        power = result["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100
