"""
Unit tests for post-hoc pairwise comparison functions.
"""

import numpy as np
import pytest


class TestComputeTukeyCriticalValue:
    """Test compute_tukey_critical_value function."""

    def test_basic_tukey_critical(self):
        from mcpower.stats.ols import compute_tukey_critical_value

        crit = compute_tukey_critical_value(alpha=0.05, n_levels=3, dfd=97)
        assert np.isfinite(crit)
        assert crit > 0

    def test_tukey_more_conservative_than_uncorrected(self):
        """Tukey critical value should be higher than uncorrected t critical."""
        from scipy.stats import t as t_dist

        from mcpower.stats.ols import compute_tukey_critical_value

        dfd = 97
        alpha = 0.05
        t_crit = t_dist.ppf(1 - alpha / 2, dfd)
        tukey_crit = compute_tukey_critical_value(alpha, n_levels=3, dfd=dfd)

        assert tukey_crit > t_crit

    def test_tukey_increases_with_levels(self):
        """More levels should require a higher critical value."""
        from mcpower.stats.ols import compute_tukey_critical_value

        crit_3 = compute_tukey_critical_value(0.05, n_levels=3, dfd=100)
        crit_5 = compute_tukey_critical_value(0.05, n_levels=5, dfd=100)

        assert crit_5 > crit_3

    def test_tukey_zero_dfd(self):
        from mcpower.stats.ols import compute_tukey_critical_value

        crit = compute_tukey_critical_value(0.05, n_levels=3, dfd=0)
        assert crit == np.inf

    def test_tukey_vs_scipy(self):
        """Verify Tukey critical values match scipy's studentized range."""
        from scipy.stats import studentized_range

        from mcpower.stats.ols import compute_tukey_critical_value

        alpha = 0.05
        n_levels = 4
        dfd = 50

        expected = studentized_range.ppf(1 - alpha, n_levels, dfd) / np.sqrt(2)
        actual = compute_tukey_critical_value(alpha, n_levels, dfd)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestComputePosthocContrasts:
    """Test compute_posthoc_contrasts function."""

    def _make_factor_data(self, n=200, seed=42):
        """Create data with a 3-level factor and continuous predictor."""
        np.random.seed(seed)
        # 3 levels, equal proportions
        group = np.repeat([0, 1, 2], n // 3 + 1)[:n]
        # Dummy code: group[2] and group[3] (internal levels 2,3)
        d2 = (group == 1).astype(float)
        d3 = (group == 2).astype(float)
        x1 = np.random.randn(n)
        X_expanded = np.column_stack([x1, d2, d3])

        # y = 0.2*x1 + 0.5*d2 + 0.8*d3 + noise
        y = 0.2 * x1 + 0.5 * d2 + 0.8 * d3 + np.random.randn(n)
        return X_expanded, y

    def test_reference_vs_nonref(self):
        """Post-hoc comparing ref (level 1) vs non-ref should be like a t-test on that dummy."""
        from mcpower.core.variables import PostHocSpec
        from mcpower.stats.ols import compute_posthoc_contrasts

        X, y = self._make_factor_data(n=200)

        spec = PostHocSpec(
            factor_name="group",
            level_a=1,
            level_b=2,
            col_idx_a=None,  # reference
            col_idx_b=1,     # group[2] is at effect index 1
            n_levels=3,
            label="group[1] vs group[2]",
        )

        from scipy.stats import t as t_dist
        n, p = X.shape
        dfd = n - p - 1
        t_crit = t_dist.ppf(1 - 0.05 / 2, dfd)

        uncorr, corr, _ = compute_posthoc_contrasts(
            X, y, [spec], method="t-test", t_crit=t_crit, tukey_crits={},
        )

        assert len(uncorr) == 1
        assert len(corr) == 1
        # With effect 0.5, n=200, should be significant
        assert uncorr[0] == True

    def test_nonref_vs_nonref(self):
        """Post-hoc comparing two non-reference levels."""
        from mcpower.core.variables import PostHocSpec
        from mcpower.stats.ols import compute_posthoc_contrasts

        X, y = self._make_factor_data(n=200)

        spec = PostHocSpec(
            factor_name="group",
            level_a=2,
            level_b=3,
            col_idx_a=1,  # group[2] is effect index 1
            col_idx_b=2,  # group[3] is effect index 2
            n_levels=3,
            label="group[2] vs group[3]",
        )

        from scipy.stats import t as t_dist
        n, p = X.shape
        dfd = n - p - 1
        t_crit = t_dist.ppf(1 - 0.05 / 2, dfd)

        uncorr, corr, _ = compute_posthoc_contrasts(
            X, y, [spec], method="t-test", t_crit=t_crit, tukey_crits={},
        )

        assert len(uncorr) == 1
        # Effect difference is 0.8 - 0.5 = 0.3, n=200, should be detectable
        # but not guaranteed every time

    def test_tukey_method(self):
        """Tukey method should return same values for uncorrected and corrected."""
        from mcpower.core.variables import PostHocSpec
        from mcpower.stats.ols import compute_posthoc_contrasts, compute_tukey_critical_value

        X, y = self._make_factor_data(n=200)
        n, p = X.shape
        dfd = n - p - 1

        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(1 - 0.05 / 2, dfd)
        tukey_crit = compute_tukey_critical_value(0.05, 3, dfd)

        spec = PostHocSpec(
            factor_name="group",
            level_a=1,
            level_b=2,
            col_idx_a=None,
            col_idx_b=1,
            n_levels=3,
            label="group[1] vs group[2]",
        )

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, [spec], method="tukey", t_crit=t_crit,
            tukey_crits={"group": tukey_crit},
        )

        # Tukey: corrected == uncorrected
        np.testing.assert_array_equal(uncorr, corr)
        assert override is None

    def test_empty_specs(self):
        """Empty posthoc_specs should return empty arrays."""
        from mcpower.stats.ols import compute_posthoc_contrasts

        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, [], method="t-test", t_crit=2.0, tukey_crits={},
        )

        assert len(uncorr) == 0
        assert len(corr) == 0
        assert override is None


class TestPostHocSpec:
    """Test PostHocSpec dataclass."""

    def test_creation(self):
        from mcpower.core.variables import PostHocSpec

        spec = PostHocSpec(
            factor_name="group",
            level_a=1,
            level_b=2,
            col_idx_a=None,
            col_idx_b=2,
            n_levels=3,
            label="group[1] vs group[2]",
        )

        assert spec.factor_name == "group"
        assert spec.level_a == 1
        assert spec.level_b == 2
        assert spec.col_idx_a is None
        assert spec.col_idx_b == 2
        assert spec.n_levels == 3
        assert spec.label == "group[1] vs group[2]"
