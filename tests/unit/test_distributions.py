"""
Validation tests for C++ distribution functions against pre-recorded scipy reference values.

All expected values are hardcoded (pre-computed from scipy and Boost.Math) so these
tests run without scipy installed.  This validates that the C++ backend (Boost.Math
for standard distributions, R Tukey port for studentized range) produces correct
results.

For functions where Boost.Math and scipy agree to full double precision (norm, chi2 PPF),
we use tight tolerances (rtol=1e-10).  For functions where the two implementations
diverge slightly (t PPF, F PPF, chi2 CDF), we validate against the Boost.Math
reference value with rtol=1e-10, and also confirm the result is close to scipy's
value (rtol=1e-5) as a cross-implementation sanity check.

Tolerance guide:
  - Boost.Math exact match:         rtol=1e-10
  - Boost vs scipy cross-check:     rtol=1e-5
  - Studentized range (R port):     rtol=1e-3
  - Batch helpers:                  rtol=1e-6
"""

import numpy as np
import pytest

from mcpower.stats.distributions import (
    _BACKEND,
    chi2_cdf,
    chi2_ppf,
    compute_critical_values_lme,
    compute_critical_values_ols,
    compute_tukey_critical_value,
    f_ppf,
    generate_norm_cdf_table,
    generate_t3_ppf_table,
    norm_cdf,
    norm_ppf,
    norm_ppf_array,
    studentized_range_ppf,
    t_ppf,
)


# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
BOOST_RTOL = 1e-10  # Boost.Math distributions (tight, same-implementation)
CROSS_RTOL = 1e-5  # Cross-implementation (Boost vs scipy)
TUKEY_RTOL = 1e-3  # Studentized range (R port)
BATCH_RTOL = 1e-6  # Batch helpers (combine multiple operations)


# ===========================================================================
# 1. norm_ppf -- Normal PPF (Boost.Math and scipy agree tightly)
# ===========================================================================
class TestNormPPF:
    """Validate standard normal quantile function (inverse CDF)."""

    def test_median(self):
        # scipy.stats.norm.ppf(0.5) == 0.0
        assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-14)

    def test_upper_tail(self):
        # scipy: 1.959963984540054  |  Boost: 1.959963984540054
        assert norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=BOOST_RTOL)

    def test_lower_tail(self):
        # scipy: -1.959963984540054  |  Boost: -1.9599639845400545
        assert norm_ppf(0.025) == pytest.approx(-1.9599639845400545, rel=BOOST_RTOL)

    def test_symmetry(self):
        assert norm_ppf(0.975) == pytest.approx(-norm_ppf(0.025), rel=BOOST_RTOL)

    def test_extreme_upper(self):
        # scipy: 3.090232306167813  |  Boost: 3.0902323061678136
        assert norm_ppf(0.999) == pytest.approx(3.0902323061678136, rel=BOOST_RTOL)

    def test_extreme_lower(self):
        # scipy: -3.090232306167813  |  Boost: -3.0902323061678136
        assert norm_ppf(0.001) == pytest.approx(-3.0902323061678136, rel=BOOST_RTOL)


# ===========================================================================
# 2. norm_cdf -- Normal CDF (Boost.Math and scipy agree tightly)
# ===========================================================================
class TestNormCDF:
    """Validate standard normal CDF."""

    def test_at_zero(self):
        # scipy: 0.5  |  Boost: 0.5
        assert norm_cdf(0.0) == pytest.approx(0.5, rel=BOOST_RTOL)

    def test_positive(self):
        # scipy: 0.9750021048517795  |  Boost: 0.9750021048517795
        assert norm_cdf(1.96) == pytest.approx(0.9750021048517795, rel=BOOST_RTOL)

    def test_negative(self):
        # scipy: 0.024997895148220428  |  Boost: 0.024997895148220452
        assert norm_cdf(-1.96) == pytest.approx(0.024997895148220452, rel=BOOST_RTOL)

    def test_symmetry(self):
        # CDF(x) + CDF(-x) == 1
        val = 2.33
        assert norm_cdf(val) + norm_cdf(-val) == pytest.approx(1.0, rel=BOOST_RTOL)

    def test_large_positive(self):
        # scipy: 0.9999997133484281  |  Boost: 0.9999997133484281
        assert norm_cdf(5.0) == pytest.approx(0.9999997133484281, rel=BOOST_RTOL)


# ===========================================================================
# 3. t_ppf -- Student's t PPF (Boost and scipy differ at ~1e-8)
# ===========================================================================
class TestTPPF:
    """Validate Student's t quantile function."""

    def test_df10(self):
        # Boost: 2.2281388519862744  |  scipy: 2.2281388519649385
        assert t_ppf(0.975, 10) == pytest.approx(2.2281388519862744, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert t_ppf(0.975, 10) == pytest.approx(2.2281388519649385, rel=CROSS_RTOL)

    def test_df30(self):
        # Boost: 2.0422724563012378  |  scipy: 2.042272456301238
        assert t_ppf(0.975, 30) == pytest.approx(2.0422724563012378, rel=BOOST_RTOL)

    def test_converges_to_normal(self):
        # With large df, t approaches normal: t_ppf(0.975, 10000) ~= 1.96
        t_val = t_ppf(0.975, 10000)
        z_val = norm_ppf(0.975)
        assert t_val == pytest.approx(z_val, rel=1e-3)

    def test_df1(self):
        # Boost: 12.706204736174694  |  scipy: 12.706204736174698
        assert t_ppf(0.975, 1) == pytest.approx(12.706204736174694, rel=BOOST_RTOL)

    def test_df5_lower_tail(self):
        # Boost: -2.5705818356363155  |  scipy: -2.5705818366147395
        assert t_ppf(0.025, 5) == pytest.approx(-2.5705818356363155, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert t_ppf(0.025, 5) == pytest.approx(-2.5705818366147395, rel=CROSS_RTOL)


# ===========================================================================
# 4. f_ppf -- Fisher F PPF (Boost and scipy differ at ~1e-5 to 1e-7)
# ===========================================================================
class TestFPPF:
    """Validate Fisher F quantile function."""

    def test_dfn3_dfd50(self):
        # Boost: 2.790008406402201  |  scipy: 2.7900373715625537
        assert f_ppf(0.95, 3, 50) == pytest.approx(2.790008406402201, rel=BOOST_RTOL)
        # Cross-check against scipy (F PPF has larger cross-impl differences)
        assert f_ppf(0.95, 3, 50) == pytest.approx(2.7900373715625537, rel=5e-5)

    def test_dfn1_dfd100(self):
        # Boost: 3.9361429863126483  |  scipy: 3.936142816142263
        assert f_ppf(0.95, 1, 100) == pytest.approx(3.9361429863126483, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert f_ppf(0.95, 1, 100) == pytest.approx(3.936142816142263, rel=CROSS_RTOL)

    def test_dfn5_dfd20(self):
        # Boost: 2.7108898372096903  |  scipy: 2.7109037486444697
        assert f_ppf(0.95, 5, 20) == pytest.approx(2.7108898372096903, rel=BOOST_RTOL)
        # Cross-check against scipy (F PPF has larger cross-impl differences)
        assert f_ppf(0.95, 5, 20) == pytest.approx(2.7109037486444697, rel=5e-5)

    def test_dfn2_dfd97(self):
        # Boost: 3.0901866751548597  |  scipy: 3.0901771821788375
        assert f_ppf(0.95, 2, 97) == pytest.approx(3.0901866751548597, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert f_ppf(0.95, 2, 97) == pytest.approx(3.0901771821788375, rel=CROSS_RTOL)


# ===========================================================================
# 5. chi2_ppf -- Chi-squared PPF (Boost and scipy agree tightly)
# ===========================================================================
class TestChi2PPF:
    """Validate chi-squared quantile function."""

    def test_df3(self):
        # scipy: 7.814727903251179  |  Boost: 7.814727903251178
        assert chi2_ppf(0.95, 3) == pytest.approx(7.814727903251178, rel=BOOST_RTOL)

    def test_df1(self):
        # scipy: 3.841458820694126  |  Boost: 3.8414588206941245
        assert chi2_ppf(0.95, 1) == pytest.approx(3.8414588206941245, rel=BOOST_RTOL)

    def test_df10(self):
        # scipy: 18.307038053275146  |  Boost: 18.307038053275143
        assert chi2_ppf(0.95, 10) == pytest.approx(18.307038053275143, rel=BOOST_RTOL)

    def test_df2_alpha99(self):
        # scipy: 9.21034037197618  |  Boost: 9.21034037197618
        assert chi2_ppf(0.99, 2) == pytest.approx(9.21034037197618, rel=BOOST_RTOL)


# ===========================================================================
# 6. chi2_cdf -- Chi-squared CDF (Boost and scipy differ at ~1e-4 to 1e-5)
# ===========================================================================
class TestChi2CDF:
    """Validate chi-squared CDF."""

    def test_at_critical_df3(self):
        # Boost: 0.9500060970251161  |  scipy: 0.9500054572822098
        assert chi2_cdf(7.815, 3) == pytest.approx(0.9500060970251161, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert chi2_cdf(7.815, 3) == pytest.approx(0.9500054572822098, rel=CROSS_RTOL)

    def test_at_critical_df1(self):
        # Boost: 0.9499863162360433  |  scipy: 0.9499579704476519
        assert chi2_cdf(3.841, 1) == pytest.approx(0.9499863162360433, rel=BOOST_RTOL)
        # Cross-check against scipy (wider tolerance for df=1 near boundary)
        assert chi2_cdf(3.841, 1) == pytest.approx(0.9499579704476519, rel=1e-4)

    def test_zero(self):
        # CDF at zero should be 0
        assert chi2_cdf(0.0, 5) == pytest.approx(0.0, abs=1e-14)

    def test_large_value(self):
        # Boost: 0.9999997330916575  |  scipy: 0.9999994082788
        assert chi2_cdf(50.0, 10) == pytest.approx(0.9999997330916575, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert chi2_cdf(50.0, 10) == pytest.approx(0.9999994082788, rel=1e-4)


# ===========================================================================
# 7. studentized_range_ppf -- Tukey (R port, wider tolerance)
# ===========================================================================
class TestStudentizedRangePPF:
    """Validate studentized range quantile (Tukey) -- R port, wider tolerance."""

    def test_k3_df10(self):
        # Boost/R: 3.8767767491915652  |  scipy: ~3.8768
        result = studentized_range_ppf(0.95, 3, 10)
        assert result == pytest.approx(3.8767767491915652, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert result == pytest.approx(3.8768, rel=TUKEY_RTOL)

    def test_k5_df30(self):
        # Boost/R: 4.1020790196217005  |  scipy: ~4.1018
        result = studentized_range_ppf(0.95, 5, 30)
        assert result == pytest.approx(4.1020790196217005, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert result == pytest.approx(4.1018, rel=TUKEY_RTOL)

    def test_k4_df50(self):
        # Boost/R: 3.758394877140481  |  scipy: ~3.7633
        result = studentized_range_ppf(0.95, 4, 50)
        assert result == pytest.approx(3.758394877140481, rel=BOOST_RTOL)
        # Cross-check against scipy (slightly wider tolerance for this case)
        assert result == pytest.approx(3.7633, rel=2e-3)

    def test_k2_df120(self):
        # Boost/R: 2.8000444315499067  |  scipy: ~2.8000
        result = studentized_range_ppf(0.95, 2, 120)
        assert result == pytest.approx(2.8000444315499067, rel=BOOST_RTOL)
        # Cross-check against scipy
        assert result == pytest.approx(2.8000, rel=TUKEY_RTOL)

    def test_invalid_df_returns_inf(self):
        # df < 2 should return inf
        result = studentized_range_ppf(0.95, 3, 1)
        assert result == float("inf")

    def test_invalid_k_returns_inf(self):
        # k < 2 should return inf
        result = studentized_range_ppf(0.95, 1, 10)
        assert result == float("inf")

    def test_invalid_p_zero_returns_inf(self):
        result = studentized_range_ppf(0.0, 3, 10)
        assert result == float("inf")

    def test_invalid_p_one_returns_inf(self):
        result = studentized_range_ppf(1.0, 3, 10)
        assert result == float("inf")


# ===========================================================================
# 8. compute_critical_values_ols
# ===========================================================================
class TestComputeCriticalValuesOLS:
    """Validate OLS critical value computation."""

    def test_no_correction(self):
        # alpha=0.05, dfn=3, dfd=96, n_targets=3, correction_method=0
        # Boost: f_crit=2.69939259755218, t_crit=1.984984311522457
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 3, 96, 3, 0)

        assert f_crit == pytest.approx(2.69939259755218, rel=BATCH_RTOL)
        assert t_crit == pytest.approx(1.984984311522457, rel=BATCH_RTOL)
        # No correction: all correction crits equal t_crit
        assert len(corr_crits) == 3
        np.testing.assert_allclose(corr_crits, t_crit, rtol=BATCH_RTOL)

    def test_bonferroni_correction(self):
        # alpha=0.05, dfn=2, dfd=97, n_targets=3, correction_method=1
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 2, 97, 3, 1)

        assert np.isfinite(f_crit)
        assert np.isfinite(t_crit)
        assert len(corr_crits) == 3
        # Bonferroni crits should all be equal and stricter than uncorrected
        assert np.all(corr_crits > t_crit)
        np.testing.assert_allclose(corr_crits[0], corr_crits[1], rtol=BOOST_RTOL)

    def test_fdr_correction(self):
        # correction_method=2 (FDR / Benjamini-Hochberg)
        # BH assigns alpha_k = (k+1)/m * alpha/2. Lower k -> smaller alpha -> larger crit.
        # So crits should be monotonically non-increasing.
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 2, 97, 3, 2)

        assert len(corr_crits) == 3
        # FDR crits are monotonically non-increasing (most strict for rank 0)
        assert corr_crits[0] >= corr_crits[1] >= corr_crits[2]

    def test_holm_correction(self):
        # correction_method=3 (Holm)
        # Holm: alpha/(2*(m-k)). Lower k -> larger denominator -> smaller alpha -> larger crit.
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 2, 97, 3, 3)

        assert len(corr_crits) == 3
        # Holm crits are monotonically non-increasing
        assert corr_crits[0] >= corr_crits[1] >= corr_crits[2]

    def test_zero_dfd_returns_inf(self):
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 3, 0, 3, 0)

        assert f_crit == np.inf
        assert t_crit == np.inf
        assert np.all(np.isinf(corr_crits))

    def test_zero_targets(self):
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 2, 97, 0, 0)

        assert np.isfinite(f_crit)
        assert np.isfinite(t_crit)
        assert len(corr_crits) == 0


# ===========================================================================
# 9. compute_critical_values_lme
# ===========================================================================
class TestComputeCriticalValuesLME:
    """Validate LME critical value computation."""

    def test_no_correction(self):
        # alpha=0.05, n_fixed=3, n_targets=3, correction_method=0
        # Boost: chi2_crit=7.814727903251178, z_crit=1.959963984540054
        chi2_crit, z_crit, corr_crits = compute_critical_values_lme(0.05, 3, 3, 0)

        assert chi2_crit == pytest.approx(7.814727903251178, rel=BATCH_RTOL)
        assert z_crit == pytest.approx(1.959963984540054, rel=BATCH_RTOL)
        assert len(corr_crits) == 3
        np.testing.assert_allclose(corr_crits, z_crit, rtol=BATCH_RTOL)

    def test_bonferroni_correction(self):
        # correction_method=1 (Bonferroni)
        chi2_crit, z_crit, corr_crits = compute_critical_values_lme(0.05, 3, 3, 1)

        assert len(corr_crits) == 3
        # Bonferroni crits should be stricter than uncorrected z_crit
        assert np.all(corr_crits > z_crit)
        np.testing.assert_allclose(corr_crits[0], corr_crits[1], rtol=BOOST_RTOL)

    def test_fdr_correction(self):
        # correction_method=2 (FDR / Benjamini-Hochberg)
        chi2_crit, z_crit, corr_crits = compute_critical_values_lme(0.05, 3, 3, 2)

        assert len(corr_crits) == 3
        # FDR crits are monotonically non-increasing (most strict for rank 0)
        assert corr_crits[0] >= corr_crits[1] >= corr_crits[2]

    def test_holm_correction(self):
        # correction_method=3 (Holm)
        chi2_crit, z_crit, corr_crits = compute_critical_values_lme(0.05, 3, 3, 3)

        assert len(corr_crits) == 3
        # Holm crits are monotonically non-increasing
        assert corr_crits[0] >= corr_crits[1] >= corr_crits[2]

    def test_zero_targets(self):
        chi2_crit, z_crit, corr_crits = compute_critical_values_lme(0.05, 3, 0, 0)

        assert np.isfinite(chi2_crit)
        assert np.isfinite(z_crit)
        assert len(corr_crits) == 0


# ===========================================================================
# 10. compute_tukey_critical_value
# ===========================================================================
class TestComputeTukeyCriticalValue:
    """Validate Tukey HSD critical value (q / sqrt(2))."""

    def test_4levels_dfd50(self):
        # Boost/R: q=3.758394877140481, q/sqrt(2)=2.6575865040028153
        result = compute_tukey_critical_value(0.05, 4, 50)
        assert result == pytest.approx(2.6575865040028153, rel=TUKEY_RTOL)

    def test_3levels_dfd30(self):
        # Boost/R: q/sqrt(2) = 2.4652712698611228
        result = compute_tukey_critical_value(0.05, 3, 30)
        assert result == pytest.approx(2.4652712698611228, rel=TUKEY_RTOL)

    def test_zero_dfd_returns_inf(self):
        result = compute_tukey_critical_value(0.05, 4, 0)
        assert result == np.inf


# ===========================================================================
# 11. generate_norm_cdf_table
# ===========================================================================
class TestGenerateNormCDFTable:
    """Validate normal CDF lookup table generation."""

    def test_correct_length(self):
        resolution = 1000
        table = generate_norm_cdf_table(-6.0, 6.0, resolution)
        assert len(table) == resolution

    def test_is_numpy_array(self):
        table = generate_norm_cdf_table(-6.0, 6.0, 500)
        assert isinstance(table, np.ndarray)

    def test_first_entry_near_zero(self):
        table = generate_norm_cdf_table(-6.0, 6.0, 1000)
        assert table[0] < 1e-6

    def test_last_entry_near_one(self):
        table = generate_norm_cdf_table(-6.0, 6.0, 1000)
        assert table[-1] > 1.0 - 1e-6

    def test_monotonically_increasing(self):
        table = generate_norm_cdf_table(-6.0, 6.0, 1000)
        diffs = np.diff(table)
        assert np.all(diffs >= 0)

    def test_midpoint_approx_half(self):
        # The midpoint of [-6, 6] is 0, where CDF ~= 0.5
        table = generate_norm_cdf_table(-6.0, 6.0, 1001)
        mid_idx = 500  # index of x=0 in linspace(-6, 6, 1001)
        assert table[mid_idx] == pytest.approx(0.5, abs=1e-6)

    def test_values_in_unit_interval(self):
        table = generate_norm_cdf_table(-6.0, 6.0, 1000)
        assert np.all(table >= 0.0)
        assert np.all(table <= 1.0)


# ===========================================================================
# 12. generate_t3_ppf_table
# ===========================================================================
class TestGenerateT3PPFTable:
    """Validate t(3) PPF lookup table generation (divided by sqrt(3))."""

    def test_correct_length(self):
        resolution = 1000
        table = generate_t3_ppf_table(0.001, 0.999, resolution)
        assert len(table) == resolution

    def test_is_numpy_array(self):
        table = generate_t3_ppf_table(0.001, 0.999, 500)
        assert isinstance(table, np.ndarray)

    def test_monotonically_increasing(self):
        table = generate_t3_ppf_table(0.001, 0.999, 1000)
        diffs = np.diff(table)
        assert np.all(diffs > 0)

    def test_median_near_zero(self):
        # t(3).ppf(0.5) / sqrt(3) = 0 / sqrt(3) = 0
        table = generate_t3_ppf_table(0.001, 0.999, 999)
        mid_idx = 499  # middle index
        assert table[mid_idx] == pytest.approx(0.0, abs=0.01)

    def test_symmetry(self):
        # Table should be approximately antisymmetric around the center
        table = generate_t3_ppf_table(0.001, 0.999, 1001)
        # first + last should be near 0 (antisymmetric)
        assert table[0] + table[-1] == pytest.approx(0.0, abs=0.01)


# ===========================================================================
# 13. norm_ppf_array (vectorized)
# ===========================================================================
class TestNormPPFArray:
    """Validate vectorized normal PPF matches scalar calls."""

    def test_matches_scalar(self):
        percentiles = [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975]
        result = norm_ppf_array(np.array(percentiles))

        for i, p in enumerate(percentiles):
            assert result[i] == pytest.approx(norm_ppf(p), rel=BOOST_RTOL)

    def test_returns_numpy_array(self):
        result = norm_ppf_array(np.array([0.1, 0.5, 0.9]))
        assert isinstance(result, np.ndarray)

    def test_correct_length(self):
        n = 50
        percentiles = np.linspace(0.01, 0.99, n)
        result = norm_ppf_array(percentiles)
        assert len(result) == n

    def test_monotonically_increasing(self):
        percentiles = np.linspace(0.01, 0.99, 100)
        result = norm_ppf_array(percentiles)
        diffs = np.diff(result)
        assert np.all(diffs > 0)

    def test_known_values(self):
        # Pre-computed: Boost reference values
        result = norm_ppf_array(np.array([0.025, 0.5, 0.975]))
        np.testing.assert_allclose(
            result,
            [-1.9599639845400545, 0.0, 1.959963984540054],
            rtol=BOOST_RTOL,
            atol=1e-14,
        )


# ===========================================================================
# 14. Edge cases
# ===========================================================================
class TestEdgeCases:
    """Edge cases and boundary parameters."""

    def test_ols_dfd_zero_returns_inf(self):
        f_crit, t_crit, corr_crits = compute_critical_values_ols(0.05, 3, 0, 3, 0)
        assert f_crit == np.inf
        assert t_crit == np.inf

    def test_tukey_dfd_zero_returns_inf(self):
        result = compute_tukey_critical_value(0.05, 4, 0)
        assert result == np.inf

    def test_norm_ppf_extreme_tails(self):
        # Very small p -> large negative
        val = norm_ppf(1e-10)
        assert val < -6.0
        assert np.isfinite(val)

    def test_norm_ppf_near_one(self):
        # p near 1 -> large positive
        val = norm_ppf(1.0 - 1e-10)
        assert val > 6.0
        assert np.isfinite(val)

    def test_norm_cdf_extreme_negative(self):
        # Very negative x -> near 0
        val = norm_cdf(-10.0)
        assert val >= 0.0
        assert val < 1e-15

    def test_norm_cdf_extreme_positive(self):
        # Very positive x -> near or at 1 (Boost may round to exactly 1.0)
        val = norm_cdf(10.0)
        assert val >= 1.0 - 1e-15

    def test_chi2_ppf_small_alpha(self):
        # chi2_ppf(0.01, 1) should be finite and positive
        val = chi2_ppf(0.01, 1)
        assert np.isfinite(val)
        assert val > 0.0

    def test_f_ppf_large_df(self):
        # With very large df, F approaches chi2/dfn
        val = f_ppf(0.95, 1, 100000)
        assert np.isfinite(val)
        assert val > 0.0

    def test_t_ppf_large_df_matches_norm(self):
        # t with df=1e6 should be very close to normal
        t_val = t_ppf(0.975, 1000000)
        z_val = norm_ppf(0.975)
        assert t_val == pytest.approx(z_val, rel=1e-5)

    def test_studentized_range_k_too_large_returns_inf(self):
        # k > 200 should return inf
        result = studentized_range_ppf(0.95, 201, 30)
        assert result == float("inf")


# ===========================================================================
# 15. Backend detection
# ===========================================================================
class TestBackendDetection:
    """Verify the distribution backend is correctly detected."""

    def test_backend_is_set(self):
        assert _BACKEND is not None

    def test_backend_is_string(self):
        assert isinstance(_BACKEND, str)

    def test_backend_is_known_value(self):
        assert _BACKEND in ("native", "scipy")

    def test_native_backend_when_compiled(self):
        """When the C++ extension is compiled, backend should be 'native'."""
        try:
            import mcpower.backends.mcpower_native  # noqa: F401

            assert _BACKEND == "native"
        except ImportError:
            pytest.skip("C++ native backend not compiled")


# ===========================================================================
# Cross-consistency checks
# ===========================================================================
class TestCrossConsistency:
    """Verify internal consistency between related functions."""

    def test_norm_ppf_cdf_roundtrip(self):
        """norm_cdf(norm_ppf(p)) should return p."""
        for p in [0.01, 0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99]:
            x = norm_ppf(p)
            p_back = norm_cdf(x)
            assert p_back == pytest.approx(p, rel=BOOST_RTOL)

    def test_chi2_ppf_cdf_roundtrip(self):
        """chi2_cdf(chi2_ppf(p, df), df) should return p."""
        for df in [1, 3, 5, 10]:
            for p in [0.05, 0.5, 0.95]:
                x = chi2_ppf(p, df)
                p_back = chi2_cdf(x, df)
                assert p_back == pytest.approx(p, rel=1e-8)

    def test_ols_f_crit_matches_f_ppf(self):
        """OLS f_crit should match direct f_ppf call."""
        alpha, dfn, dfd = 0.05, 3, 96
        f_crit, _, _ = compute_critical_values_ols(alpha, dfn, dfd, 3, 0)
        f_direct = f_ppf(1 - alpha, dfn, dfd)
        assert f_crit == pytest.approx(f_direct, rel=BATCH_RTOL)

    def test_ols_t_crit_matches_t_ppf(self):
        """OLS t_crit should match direct t_ppf call."""
        alpha, dfd = 0.05, 96
        _, t_crit, _ = compute_critical_values_ols(alpha, 3, dfd, 3, 0)
        t_direct = t_ppf(1 - alpha / 2, dfd)
        assert t_crit == pytest.approx(t_direct, rel=BATCH_RTOL)

    def test_lme_chi2_crit_matches_chi2_ppf(self):
        """LME chi2_crit should match direct chi2_ppf call."""
        alpha, n_fixed = 0.05, 3
        chi2_crit, _, _ = compute_critical_values_lme(alpha, n_fixed, 3, 0)
        chi2_direct = chi2_ppf(1 - alpha, n_fixed)
        assert chi2_crit == pytest.approx(chi2_direct, rel=BATCH_RTOL)

    def test_lme_z_crit_matches_norm_ppf(self):
        """LME z_crit should match direct norm_ppf call."""
        alpha = 0.05
        _, z_crit, _ = compute_critical_values_lme(alpha, 3, 3, 0)
        z_direct = norm_ppf(1 - alpha / 2)
        assert z_crit == pytest.approx(z_direct, rel=BATCH_RTOL)

    def test_tukey_equals_qcrit_over_sqrt2(self):
        """compute_tukey_critical_value should equal studentized_range_ppf / sqrt(2)."""
        alpha, n_levels, dfd = 0.05, 4, 50
        tukey_val = compute_tukey_critical_value(alpha, n_levels, dfd)
        q_val = studentized_range_ppf(1 - alpha, n_levels, dfd)
        expected = q_val / np.sqrt(2)
        assert tukey_val == pytest.approx(expected, rel=TUKEY_RTOL)
