"""Tests for OLS post-hoc contrast corrections and edge cases."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from mcpower.stats.ols import compute_posthoc_contrasts


@dataclass
class _PostHocSpec:
    """Minimal PostHocSpec stub for tests."""
    factor_name: str
    col_idx_a: Optional[int]
    col_idx_b: Optional[int]
    label: str = ""
    level_a: str = ""
    level_b: str = ""
    n_levels: int = 3


def _make_ols_data(n=100, p=3, seed=42):
    """Generate simple OLS data: X, y, and target_indices."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta = np.array([0.5, 0.3, -0.2])[:p]
    y = X @ beta + rng.randn(n)
    return X, y


class TestDegenerateDesign:
    """When dof <= 0, posthoc should return zeros."""

    def test_dof_zero_returns_zeros(self):
        # n = p+1 → dof = 0
        n, p = 4, 3
        rng = np.random.RandomState(42)
        X = rng.randn(n, p)
        y = rng.randn(n)
        specs = [_PostHocSpec("grp", 0, 1)]

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {}, target_indices=np.array([0, 1, 2]),
        )
        assert uncorr.shape == (1,)
        assert not uncorr[0]
        assert not corr[0]
        assert override is None

    def test_singular_contrast_variance_stays_zero(self):
        """When both col_idx_a and col_idx_b are None, t_abs stays 0."""
        X, y = _make_ols_data()
        specs = [_PostHocSpec("grp", None, None)]

        uncorr, corr, _ = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {},
        )
        assert not uncorr[0]
        assert not corr[0]


class TestCombinedFDR:
    """FDR (correction_method=2) step-up across regular+posthoc t-stats."""

    def test_fdr_combined_ranking(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [
            _PostHocSpec("grp", 0, 1),
            _PostHocSpec("grp", 0, 2),
        ]
        target_indices = np.array([0, 1, 2])
        # Create combined crits of length n_regular + n_posthoc = 5
        # Use very lenient crits so everything passes
        combined_crits = np.full(5, 0.01)

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 0.01, {},
            target_indices=target_indices,
            correction_method=2,
            correction_t_crits_combined=combined_crits,
        )
        assert override is not None
        assert len(override) == 3  # n_regular
        assert len(corr) == 2  # n_posthoc

    def test_fdr_no_significant(self):
        """With very strict crits, nothing should be significant."""
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])
        # Very strict thresholds
        combined_crits = np.full(4, 100.0)

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 100.0, {},
            target_indices=target_indices,
            correction_method=2,
            correction_t_crits_combined=combined_crits,
        )
        assert not np.any(corr)
        assert override is not None
        assert not np.any(override)


class TestCombinedHolm:
    """Holm (correction_method=3) step-down with early termination."""

    def test_holm_combined_ranking(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])
        combined_crits = np.full(4, 0.01)  # Very lenient

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 0.01, {},
            target_indices=target_indices,
            correction_method=3,
            correction_t_crits_combined=combined_crits,
        )
        assert override is not None
        assert len(override) == 3

    def test_holm_early_termination(self):
        """If the most significant test doesn't pass, none should."""
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])
        combined_crits = np.full(4, 1000.0)  # Impossible threshold

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 1000.0, {},
            target_indices=target_indices,
            correction_method=3,
            correction_t_crits_combined=combined_crits,
        )
        assert not np.any(corr)


class TestFallbackPaths:
    """Fallback when correction_t_crits_combined is None or wrong length."""

    def test_combined_crits_none_fallback(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {},
            target_indices=target_indices,
            correction_method=2,
            correction_t_crits_combined=None,
        )
        # Fallback: corrected = uncorrected copy, no override
        np.testing.assert_array_equal(corr, uncorr)
        assert override is None

    def test_combined_crits_wrong_length_fallback(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])
        # Wrong length: should be 4 (3 regular + 1 posthoc)
        wrong_crits = np.full(2, 2.0)

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {},
            target_indices=target_indices,
            correction_method=2,
            correction_t_crits_combined=wrong_crits,
        )
        np.testing.assert_array_equal(corr, uncorr)
        assert override is None


class TestTukeyMethod:
    """Tukey post-hoc method path."""

    def test_tukey_uses_factor_crit(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1, n_levels=3)]
        tukey_crits = {"grp": 0.01}  # Very lenient

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "tukey", 2.0, tukey_crits,
        )
        # Tukey correction: uncorrected == corrected
        np.testing.assert_array_equal(uncorr, corr)
        assert override is None

    def test_tukey_missing_factor_uses_inf(self):
        """When factor not in tukey_crits, inf is used → not significant."""
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("missing_factor", 0, 1)]

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "tukey", 2.0, {},
        )
        assert not uncorr[0]
        assert not corr[0]


class TestBonferroniPosthoc:
    """Bonferroni correction for posthoc (correction_method=1)."""

    def test_bonferroni_uses_combined_first_crit(self):
        X, y = _make_ols_data(n=200, p=3, seed=10)
        specs = [_PostHocSpec("grp", 0, 1)]
        target_indices = np.array([0, 1, 2])
        combined_crits = np.full(4, 0.01)  # Very lenient

        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, specs, "t-test", 0.01, {},
            target_indices=target_indices,
            correction_method=1,
            correction_t_crits_combined=combined_crits,
        )
        assert override is None  # Bonferroni doesn't produce override


class TestEmptySpecs:
    """Empty posthoc specs return empty arrays."""

    def test_no_specs(self):
        X, y = _make_ols_data()
        uncorr, corr, override = compute_posthoc_contrasts(
            X, y, [], "t-test", 2.0, {},
        )
        assert len(uncorr) == 0
        assert len(corr) == 0
        assert override is None


class TestSingleColumnContrasts:
    """Contrasts where one side is the reference level (None)."""

    def test_col_idx_a_none(self):
        X, y = _make_ols_data(n=200)
        specs = [_PostHocSpec("grp", None, 1)]
        uncorr, corr, _ = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {},
        )
        assert uncorr.shape == (1,)

    def test_col_idx_b_none(self):
        X, y = _make_ols_data(n=200)
        specs = [_PostHocSpec("grp", 0, None)]
        uncorr, corr, _ = compute_posthoc_contrasts(
            X, y, specs, "t-test", 2.0, {},
        )
        assert uncorr.shape == (1,)
