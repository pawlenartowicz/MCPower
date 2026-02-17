"""
Tests for OLS analysis utilities.
"""

import numpy as np


def _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05):
    """Test helper: compute critical values and call OLS core function."""
    from mcpower.stats.ols import _ols_jit, compute_critical_values

    n, p = X.shape
    dof = n - p - 1
    f_crit, t_crit, correction_t_crits = compute_critical_values(alpha, p, dof, len(target_indices), correction_method)
    return _ols_jit(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)


class TestOLSAnalysis:
    """Test OLS analysis via core function with precomputed critical values."""

    def test_basic_ols(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 2)
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5

        target_indices = np.array([0, 1], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05)

        # Result should contain: f_sig, uncorr_0, uncorr_1, corr_0, corr_1
        assert len(result) == 5

    def test_significant_effect(self):
        np.random.seed(42)

        n = 200
        X = np.random.randn(n, 1)
        y = 2.0 * X[:, 0] + np.random.randn(n) * 0.5  # Strong effect

        target_indices = np.array([0], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05)

        # Should detect significance
        assert result[0] == 1  # F-test significant
        assert result[1] == 1  # t-test significant

    def test_no_effect(self):
        np.random.seed(42)

        n = 50
        X = np.random.randn(n, 1)
        y = np.random.randn(n)  # No relationship

        target_indices = np.array([0], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05)

        # Usually should not detect significance (but not guaranteed)
        # Just check it runs without error
        assert len(result) == 3

    def test_multiple_predictors(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 3)
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(n) * 0.5

        target_indices = np.array([0, 1, 2], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05)

        # f_sig + 3 uncorrected + 3 corrected
        assert len(result) == 7

    def test_bonferroni_correction(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 2)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n) * 0.5

        target_indices = np.array([0, 1], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=1, alpha=0.05)

        # Bonferroni should be more conservative
        assert len(result) == 5


class TestGenerateY:
    """Test _generate_y_jit function."""

    def test_basic_generation(self):
        from mcpower.stats.ols import _generate_y_jit

        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 2)
        effects = np.array([0.5, 0.3])

        y = _generate_y_jit(X, effects, 0.0, 0.0, 42)

        assert y.shape == (n,)

    def test_reproducibility(self):
        from mcpower.stats.ols import _generate_y_jit

        n = 50
        X = np.random.randn(n, 2)
        effects = np.array([0.5, 0.3])

        y1 = _generate_y_jit(X, effects, 0.0, 0.0, 42)
        y2 = _generate_y_jit(X, effects, 0.0, 0.0, 42)

        assert np.allclose(y1, y2)

    def test_effect_sizes_matter(self):
        from mcpower.stats.ols import _generate_y_jit

        n = 1000
        np.random.seed(42)
        X = np.random.randn(n, 1)

        y_small = _generate_y_jit(X, np.array([0.1]), 0.0, 0.0, 42)
        y_large = _generate_y_jit(X, np.array([1.0]), 0.0, 0.0, 42)

        # Larger effect should produce higher correlation with X
        corr_small = np.corrcoef(X.flatten(), y_small)[0, 1]
        corr_large = np.corrcoef(X.flatten(), y_large)[0, 1]

        assert abs(corr_large) > abs(corr_small)

    def test_with_heterogeneity(self):
        from mcpower.stats.ols import _generate_y_jit

        n = 100
        np.random.seed(42)
        X = np.random.randn(n, 2)
        effects = np.array([0.5, 0.3])

        y = _generate_y_jit(X, effects, 0.1, 0.0, 42)

        # Should still produce valid output
        assert y.shape == (n,)
        assert not np.any(np.isnan(y))

    def test_with_heteroskedasticity(self):
        from mcpower.stats.ols import _generate_y_jit

        n = 100
        np.random.seed(42)
        X = np.random.randn(n, 2)
        effects = np.array([0.5, 0.3])

        y = _generate_y_jit(X, effects, 0.0, 0.3, 42)

        # Should still produce valid output
        assert y.shape == (n,)
        assert not np.any(np.isnan(y))


class TestOLSCorrections:
    """Test multiple comparison corrections."""

    def test_no_correction(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 3)
        y = 0.5 * X[:, 0] + np.random.randn(n)

        target_indices = np.array([0, 1, 2], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=0, alpha=0.05)

        # With no correction, uncorrected == corrected
        assert result[1] == result[4]
        assert result[2] == result[5]
        assert result[3] == result[6]

    def test_holm_correction(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 2)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n) * 0.5

        target_indices = np.array([0, 1], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=3, alpha=0.05)

        # Should run without error
        assert len(result) == 5

    def test_bh_correction(self):
        np.random.seed(42)

        n = 100
        X = np.random.randn(n, 2)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n) * 0.5

        target_indices = np.array([0, 1], dtype=np.int64)
        result = _ols_analysis_helper(X, y, target_indices, correction_method=2, alpha=0.05)

        # Should run without error
        assert len(result) == 5


class TestComputeCriticalValues:
    """Test compute_critical_values function."""

    def test_basic_critical_values(self):
        from mcpower.stats.ols import compute_critical_values

        f_crit, t_crit, correction_t_crits = compute_critical_values(alpha=0.05, dfn=2, dfd=97, n_targets=2, correction_method=0)

        assert np.isfinite(f_crit)
        assert np.isfinite(t_crit)
        assert f_crit > 0
        assert t_crit > 0
        assert len(correction_t_crits) == 2
        # No correction: all thresholds equal to t_crit
        np.testing.assert_allclose(correction_t_crits, t_crit)

    def test_bonferroni_stricter(self):
        from mcpower.stats.ols import compute_critical_values

        _, t_crit_none, crits_none = compute_critical_values(alpha=0.05, dfn=2, dfd=97, n_targets=3, correction_method=0)
        _, _, crits_bonf = compute_critical_values(alpha=0.05, dfn=2, dfd=97, n_targets=3, correction_method=1)

        # Bonferroni critical values should be higher (stricter)
        assert np.all(crits_bonf > crits_none)

    def test_zero_dfd(self):
        from mcpower.stats.ols import compute_critical_values

        f_crit, t_crit, correction_t_crits = compute_critical_values(alpha=0.05, dfn=2, dfd=0, n_targets=2, correction_method=0)

        assert f_crit == np.inf
        assert t_crit == np.inf

    def test_zero_targets(self):
        from mcpower.stats.ols import compute_critical_values

        f_crit, t_crit, correction_t_crits = compute_critical_values(alpha=0.05, dfn=2, dfd=97, n_targets=0, correction_method=0)

        assert np.isfinite(f_crit)
        assert np.isfinite(t_crit)
        assert len(correction_t_crits) == 0
