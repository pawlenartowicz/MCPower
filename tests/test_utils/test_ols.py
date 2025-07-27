"""
Tests for OLS analysis utilities.
"""

import pytest
import numpy as np
from mcpower.utils.ols import _ols_analysis, _generate_y


class TestOLSAnalysis:
    """Test OLS statistical analysis."""

    def test_basic_ols(self):
        """Test basic OLS analysis."""
        # Simple linear relationship: y = 0.5*x + noise
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        y = 0.5 * X[:, 0] + np.random.normal(0, 0.5, n)

        target_indices = np.array([0])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should return: [f_significant, target_significances, corrected_significances]
        assert len(results) == 3  # f_sig + 1 target + 1 corrected
        assert results[0] in [0.0, 1.0]  # F-test significance
        assert results[1] in [0.0, 1.0]  # Target significance
        assert results[2] in [0.0, 1.0]  # Corrected significance

    def test_multiple_predictors(self):
        """Test OLS with multiple predictors."""
        np.random.seed(42)
        n = 150
        X = np.random.normal(0, 1, (n, 3))
        y = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.5, n)

        target_indices = np.array([0, 1, 2])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        expected_length = 1 + 2 * len(target_indices)  # f_sig + uncorr + corr
        assert len(results) == expected_length

    def test_no_effect(self):
        """Test OLS with no real effects."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 2))
        y = np.random.normal(0, 1, n)  # Pure noise

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Most results should be non-significant
        assert len(results) == 5  # f + 2 uncorr + 2 corr

    def test_strong_correlation(self):
        """Test OLS with strong relationship."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        y = 2.0 * X[:, 0] + np.random.normal(0, 0.1, n)  # Strong but not perfect

        target_indices = np.array([0])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should be highly significant
        assert results[0] == 1.0  # F-test significant
        assert results[1] == 1.0  # Target significant


class TestCorrections:
    """Test multiple testing corrections."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = 0.5 * X[:, 0] + np.random.normal(
            0, 0.5, n
        )  # Only first predictor has effect

        target_indices = np.array([0, 1, 2])
        results = _ols_analysis(X, y, target_indices, correction_method=1, alpha=0.05)

        # Bonferroni should be more conservative
        uncorrected = results[1:4]
        corrected = results[4:7]

        # Corrected should have fewer significances
        assert np.sum(corrected) <= np.sum(uncorrected)

    def test_bh_correction(self):
        """Test Benjamini-Hochberg correction."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = 0.6 * X[:, 0] + 0.4 * X[:, 1] + np.random.normal(0, 0.5, n)

        target_indices = np.array([0, 1, 2])
        results = _ols_analysis(X, y, target_indices, correction_method=2, alpha=0.05)

        assert len(results) == 7  # f + 3 uncorr + 3 corr

    def test_holm_correction(self):
        """Test Holm correction."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 2))
        y = 0.8 * X[:, 0] + np.random.normal(0, 0.5, n)

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=3, alpha=0.05)

        assert len(results) == 5  # f + 2 uncorr + 2 corr

    def test_no_correction(self):
        """Test no correction (method=0)."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 2))
        y = 0.5 * X[:, 0] + np.random.normal(0, 0.5, n)

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # No correction: uncorrected == corrected
        uncorrected = results[1:3]
        corrected = results[3:5]
        np.testing.assert_array_equal(uncorrected, corrected)


class TestGenerateY:
    """Test dependent variable generation."""

    def test_basic_y_generation(self):
        """Test basic Y generation."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        effect_sizes = np.array([0.5, 0.3])

        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        assert len(y) == 100
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))

    def test_heterogeneity(self):
        """Test Y generation with heterogeneity."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        effect_sizes = np.array([0.5])

        y1 = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )
        y2 = _generate_y(
            X, effect_sizes, heterogeneity=0.3, heteroskedasticity=0.0, sim_seed=42
        )

        # With heterogeneity, relationship should be different
        assert not np.array_equal(y1, y2)

    def test_heteroskedasticity(self):
        """Test Y generation with heteroskedasticity."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        effect_sizes = np.array([0.5])

        y1 = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )
        y2 = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.3, sim_seed=42
        )

        # With heteroskedasticity, error structure should be different
        assert not np.array_equal(y1, y2)

    def test_reproducibility(self):
        """Test reproducible Y generation."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        effect_sizes = np.array([0.4, 0.3])

        y1 = _generate_y(
            X, effect_sizes, heterogeneity=0.1, heteroskedasticity=0.1, sim_seed=123
        )
        y2 = _generate_y(
            X, effect_sizes, heterogeneity=0.1, heteroskedasticity=0.1, sim_seed=123
        )

        np.testing.assert_array_equal(y1, y2)

    def test_zero_effects(self):
        """Test Y generation with zero effects."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        effect_sizes = np.array([0.0, 0.0])

        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        # Should be mostly noise
        assert len(y) == 100
        assert abs(np.mean(y)) < 0.3  # Should be close to 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_predictor(self):
        """Test with single predictor."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        y = 0.7 * X[:, 0] + np.random.normal(0, 0.5, 100)

        target_indices = np.array([0])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        assert len(results) == 3

    def test_small_sample_size(self):
        """Test with small sample size."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (10, 2))
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.5, 10)

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        assert len(results) == 5

    def test_large_effects(self):
        """Test with large effect sizes."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        effect_sizes = np.array([2.0])  # Large effect

        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        target_indices = np.array([0])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should be highly significant
        assert results[1] == 1.0

    def test_negative_effects(self):
        """Test with negative effect sizes."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        effect_sizes = np.array([-0.6])

        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        assert len(y) == 100
        # Check negative relationship
        correlation = np.corrcoef(X[:, 0], y)[0, 1]
        assert correlation < 0

    def test_collinear_predictors(self):
        """Test with highly collinear predictors."""
        np.random.seed(42)
        X1 = np.random.normal(0, 1, (100, 1))
        X2 = X1 + np.random.normal(0, 0.01, (100, 1))  # Nearly identical
        X = np.hstack([X1, X2])
        y = X1[:, 0] + np.random.normal(0, 0.5, 100)

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should handle collinearity gracefully
        assert len(results) == 5
        assert not np.any(np.isnan(results))

    def test_no_targets(self):
        """Test with empty target indices."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, 100)

        target_indices = np.array([], dtype=int)
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should only return F-test
        assert len(results) == 1


class TestStatisticalAccuracy:
    """Test statistical accuracy of OLS implementation."""

    def test_known_relationship(self):
        """Test OLS with known statistical relationship."""
        # Create data with known properties
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, (n, 1))
        beta = 0.5
        y = beta * X[:, 0] + np.random.normal(0, 1, n)

        target_indices = np.array([0])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        # With n=1000 and effect=0.5, should be significant
        assert results[1] == 1.0  # Should detect the effect

    def test_type_i_error_control(self):
        """Test Type I error control under null hypothesis."""
        # Run multiple tests under null
        np.random.seed(42)
        significances = []

        for i in range(100):
            X = np.random.normal(0, 1, (50, 1))
            y = np.random.normal(0, 1, 50)  # No relationship

            target_indices = np.array([0])
            results = _ols_analysis(
                X, y, target_indices, correction_method=0, alpha=0.05
            )
            significances.append(results[1])

        # Type I error rate should be around 5%
        error_rate = np.mean(significances)
        assert 0.01 < error_rate < 0.15  # Loose bounds due to randomness

    def test_different_alpha_levels(self):
        """Test different significance levels."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        y = 0.3 * X[:, 0] + np.random.normal(0, 1, 100)  # Weak effect

        target_indices = np.array([0])

        # Test stricter alpha
        results_strict = _ols_analysis(
            X, y, target_indices, correction_method=0, alpha=0.01
        )
        results_lenient = _ols_analysis(
            X, y, target_indices, correction_method=0, alpha=0.10
        )

        # Lenient should be more likely to find significance
        assert results_lenient[1] >= results_strict[1]


class TestInteractionWithDataGeneration:
    """Test OLS with data from data generation utilities."""

    def test_with_binary_predictors(self):
        """Test OLS with binary predictors from data generation."""
        from mcpower.utils.data_generation import _generate_X

        X = _generate_X(
            sample_size=200,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([1, 1]),  # binary
            var_params=np.array([0.5, 0.3]),
            seed=42,
        )

        effect_sizes = np.array([0.6, 0.4])
        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        assert len(results) == 5
        assert not np.any(np.isnan(results))

    def test_with_correlated_predictors(self):
        """Test OLS with correlated predictors."""
        from mcpower.utils.data_generation import _generate_X

        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        X = _generate_X(
            sample_size=150,
            n_vars=2,
            correlation_matrix=corr_matrix,
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5]),
            seed=42,
        )

        effect_sizes = np.array([0.5, 0.3])
        y = _generate_y(
            X, effect_sizes, heterogeneity=0.0, heteroskedasticity=0.0, sim_seed=42
        )

        target_indices = np.array([0, 1])
        results = _ols_analysis(X, y, target_indices, correction_method=0, alpha=0.05)

        assert len(results) == 5
