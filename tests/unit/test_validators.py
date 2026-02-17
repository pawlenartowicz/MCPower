"""
Tests for validation utilities.
"""

import numpy as np
import pytest


class TestValidatePower:
    """Test _validate_power function."""

    def test_valid_power(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power(80.0)
        assert result.is_valid

    def test_power_at_zero(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power(0)
        assert result.is_valid

    def test_power_at_hundred(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power(100)
        assert result.is_valid

    def test_power_negative(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power(-10)
        assert not result.is_valid

    def test_power_over_hundred(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power(150)
        assert not result.is_valid

    def test_power_wrong_type(self):
        from mcpower.utils.validators import _validate_power

        result = _validate_power("80")
        assert not result.is_valid


class TestValidateAlpha:
    """Test _validate_alpha function."""

    def test_valid_alpha(self):
        from mcpower.utils.validators import _validate_alpha

        result = _validate_alpha(0.05)
        assert result.is_valid

    def test_alpha_at_zero(self):
        from mcpower.utils.validators import _validate_alpha

        result = _validate_alpha(0)
        assert result.is_valid

    def test_alpha_at_max(self):
        from mcpower.utils.validators import _validate_alpha

        result = _validate_alpha(0.25)
        assert result.is_valid

    def test_alpha_negative(self):
        from mcpower.utils.validators import _validate_alpha

        result = _validate_alpha(-0.05)
        assert not result.is_valid

    def test_alpha_too_high(self):
        from mcpower.utils.validators import _validate_alpha

        result = _validate_alpha(0.5)
        assert not result.is_valid


class TestValidateSimulations:
    """Test _validate_simulations function."""

    def test_valid_simulations(self):
        from mcpower.utils.validators import _validate_simulations

        n_sims, result = _validate_simulations(1000)
        assert result.is_valid
        assert n_sims == 1000

    def test_low_simulations_warning(self):
        from mcpower.utils.validators import _validate_simulations

        n_sims, result = _validate_simulations(100)
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_float_rounding(self):
        from mcpower.utils.validators import _validate_simulations

        n_sims, result = _validate_simulations(1000.5)
        assert n_sims == 1000 or n_sims == 1001

    def test_zero_simulations(self):
        from mcpower.utils.validators import _validate_simulations

        n_sims, result = _validate_simulations(0)
        assert not result.is_valid


class TestValidateSampleSize:
    """Test _validate_sample_size function."""

    def test_valid_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(100)
        assert result.is_valid

    def test_minimum_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(20)
        assert result.is_valid

    def test_below_minimum_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(19)
        assert not result.is_valid
        assert "at least 20" in result.errors[0]

    def test_zero_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(0)
        assert not result.is_valid

    def test_negative_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(-10)
        assert not result.is_valid

    def test_very_large_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(200000)
        assert not result.is_valid

    def test_float_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size

        result = _validate_sample_size(100.5)
        assert not result.is_valid


class TestValidateSampleSizeForModel:
    """Test _validate_sample_size_for_model function."""

    def test_sufficient_sample_size(self):
        from mcpower.utils.validators import _validate_sample_size_for_model

        # 3 variables: need 15 + 3 = 18, have 100
        result = _validate_sample_size_for_model(100, 3)
        assert result.is_valid

    def test_exact_minimum(self):
        from mcpower.utils.validators import _validate_sample_size_for_model

        # 5 variables: need 15 + 5 = 20
        result = _validate_sample_size_for_model(20, 5)
        assert result.is_valid

    def test_below_minimum(self):
        from mcpower.utils.validators import _validate_sample_size_for_model

        # 10 variables: need 15 + 10 = 25, have 20
        result = _validate_sample_size_for_model(20, 10)
        assert not result.is_valid
        assert "25" in result.errors[0]
        assert "10 variables" in result.errors[0]

    def test_many_variables(self):
        from mcpower.utils.validators import _validate_sample_size_for_model

        # 20 variables (e.g. factor with many levels): need 15 + 20 = 35
        result = _validate_sample_size_for_model(34, 20)
        assert not result.is_valid
        result = _validate_sample_size_for_model(35, 20)
        assert result.is_valid


class TestValidateSampleSizeRange:
    """Test _validate_sample_size_range function."""

    def test_valid_range(self):
        from mcpower.utils.validators import _validate_sample_size_range

        result = _validate_sample_size_range(50, 200, 10)
        assert result.is_valid

    def test_from_greater_than_to(self):
        from mcpower.utils.validators import _validate_sample_size_range

        result = _validate_sample_size_range(200, 50, 10)
        assert not result.is_valid

    def test_step_too_large(self):
        from mcpower.utils.validators import _validate_sample_size_range

        result = _validate_sample_size_range(50, 100, 200)
        assert not result.is_valid

    def test_many_tests_warning(self):
        from mcpower.utils.validators import _validate_sample_size_range

        result = _validate_sample_size_range(10, 1000, 1)
        assert result.is_valid
        assert len(result.warnings) > 0


class TestValidateCorrelationMatrix:
    """Test _validate_correlation_matrix function."""

    def test_valid_matrix(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = _validate_correlation_matrix(matrix)
        assert result.is_valid

    def test_none_matrix(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        result = _validate_correlation_matrix(None)
        assert not result.is_valid

    def test_non_square_matrix(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4]])
        result = _validate_correlation_matrix(matrix)
        assert not result.is_valid

    def test_non_unit_diagonal(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.array([[0.9, 0.5], [0.5, 1.0]])
        result = _validate_correlation_matrix(matrix)
        assert not result.is_valid

    def test_asymmetric_matrix(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.array([[1.0, 0.5], [0.3, 1.0]])
        result = _validate_correlation_matrix(matrix)
        assert not result.is_valid

    def test_out_of_range_correlation(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.array([[1.0, 1.5], [1.5, 1.0]])
        result = _validate_correlation_matrix(matrix)
        assert not result.is_valid

    def test_not_positive_semidefinite(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        # This matrix has a negative eigenvalue
        matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        result = _validate_correlation_matrix(matrix)
        assert not result.is_valid

    def test_identity_matrix(self):
        from mcpower.utils.validators import _validate_correlation_matrix

        matrix = np.eye(5)
        result = _validate_correlation_matrix(matrix)
        assert result.is_valid


class TestValidateCorrectionMethod:
    """Test _validate_correction_method function."""

    def test_none_correction(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method(None)
        assert result.is_valid

    def test_bonferroni(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("bonferroni")
        assert result.is_valid

    def test_benjamini_hochberg(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("benjamini-hochberg")
        assert result.is_valid

    def test_bh_alias(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("BH")
        assert result.is_valid

    def test_fdr_alias(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("FDR")
        assert result.is_valid

    def test_holm(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("holm")
        assert result.is_valid

    def test_tukey(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("tukey")
        assert result.is_valid

    def test_invalid_method(self):
        from mcpower.utils.validators import _validate_correction_method

        result = _validate_correction_method("invalid")
        assert not result.is_valid


class TestValidateModelReady:
    """Test _validate_model_ready function."""

    def test_ready_model(self, configured_model):
        from mcpower.utils.validators import _validate_model_ready

        result = _validate_model_ready(configured_model)
        assert result.is_valid

    def test_no_effects(self, simple_model):
        from mcpower.utils.validators import _validate_model_ready

        result = _validate_model_ready(simple_model)
        assert not result.is_valid
        assert any("Effect sizes" in e for e in result.errors)


class TestValidationResult:
    """Test _ValidationResult class."""

    def test_raise_if_invalid(self):
        from mcpower.utils.validators import _ValidationResult

        result = _ValidationResult(False, ["Error 1"], [])
        with pytest.raises(ValueError, match="Error 1"):
            result.raise_if_invalid()

    def test_no_raise_if_valid(self):
        from mcpower.utils.validators import _ValidationResult

        result = _ValidationResult(True, [], [])
        result.raise_if_invalid()  # Should not raise


class TestValidateParallelSettings:
    """Test _validate_parallel_settings function."""

    def test_enable_true(self):
        from mcpower.utils.validators import _validate_parallel_settings

        (enable, cores), result = _validate_parallel_settings(True, None)
        assert result.is_valid
        assert enable is True
        assert cores >= 1

    def test_enable_mixedmodels(self):
        from mcpower.utils.validators import _validate_parallel_settings

        (enable, cores), result = _validate_parallel_settings("mixedmodels", None)
        assert result.is_valid
        assert enable == "mixedmodels"

    def test_enable_false(self):
        from mcpower.utils.validators import _validate_parallel_settings

        (enable, cores), result = _validate_parallel_settings(False, None)
        assert result.is_valid
        assert enable is False

    def test_custom_n_cores(self):
        from mcpower.utils.validators import _validate_parallel_settings

        (enable, cores), result = _validate_parallel_settings(True, 2)
        assert result.is_valid
        assert cores <= 2  # capped at min(n_cores, max_cores)

    def test_invalid_enable_value(self):
        from mcpower.utils.validators import _validate_parallel_settings

        _, result = _validate_parallel_settings("invalid", None)
        assert not result.is_valid
        assert "enable" in result.errors[0]

    def test_invalid_n_cores(self):
        from mcpower.utils.validators import _validate_parallel_settings

        _, result = _validate_parallel_settings(True, -1)
        assert not result.is_valid

    def test_n_cores_zero(self):
        from mcpower.utils.validators import _validate_parallel_settings

        _, result = _validate_parallel_settings(True, 0)
        assert not result.is_valid


class TestValidateTestFormula:
    """Test _validate_test_formula function."""

    def test_single_variable(self):
        from mcpower.utils.validators import _validate_test_formula

        result = _validate_test_formula("x1", ["x1", "x2"])
        assert result.is_valid

    def test_interaction_formula(self):
        from mcpower.utils.validators import _validate_test_formula

        result = _validate_test_formula("x1 + x2:x3", ["x1", "x2", "x3"])
        assert result.is_valid

    def test_missing_variable(self):
        from mcpower.utils.validators import _validate_test_formula

        result = _validate_test_formula("x1 + x99", ["x1", "x2"])
        assert not result.is_valid
        assert "x99" in result.errors[0]

    def test_empty_formula(self):
        from mcpower.utils.validators import _validate_test_formula

        result = _validate_test_formula("", ["x1"])
        assert not result.is_valid

    def test_non_string_formula(self):
        from mcpower.utils.validators import _validate_test_formula

        result = _validate_test_formula(123, ["x1"])
        assert not result.is_valid


class TestValidateFactorSpecification:
    """Test _validate_factor_specification function."""

    def test_valid_2_level_factor(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(2, [0.5, 0.5])
        assert result.is_valid

    def test_valid_3_level_factor(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(3, [0.5, 0.3, 0.2])
        assert result.is_valid

    def test_too_many_levels(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(21, [1 / 21] * 21)
        assert not result.is_valid
        assert "20 levels" in result.errors[0]

    def test_less_than_2_levels(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(1, [1.0])
        assert not result.is_valid

    def test_proportions_mismatch(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(3, [0.5, 0.5])
        assert not result.is_valid
        assert "must match" in result.errors[0]

    def test_negative_proportion(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(2, [1.5, -0.5])
        assert not result.is_valid

    def test_zero_proportion_warning(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(2, [1.0, 0.0])
        assert result.is_valid
        assert any("zero" in w for w in result.warnings)

    def test_proportions_not_summing_to_one(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(2, [0.3, 0.3])
        assert result.is_valid  # just a warning
        assert any("sum to" in w for w in result.warnings)

    def test_many_levels_warning(self):
        from mcpower.utils.validators import _validate_factor_specification

        result = _validate_factor_specification(15, [1 / 15] * 15)
        assert result.is_valid
        assert any("15 levels" in w for w in result.warnings)


class TestValidateClusterSampleSize:
    """Test _validate_cluster_sample_size function."""

    def test_sufficient_cluster_size(self):
        from mcpower.utils.validators import _validate_cluster_sample_size

        # 1000 / 10 = 100 obs per cluster >= 25
        result = _validate_cluster_sample_size(1000, 10, None)
        assert result.is_valid

    def test_insufficient_cluster_size(self):
        from mcpower.utils.validators import _validate_cluster_sample_size

        # 100 / 10 = 10 obs per cluster < 25
        result = _validate_cluster_sample_size(100, 10, None)
        assert not result.is_valid
        assert "25" in result.errors[0]

    def test_boundary_exact_25(self):
        from mcpower.utils.validators import _validate_cluster_sample_size

        # 250 / 10 = 25 — exactly at threshold
        result = _validate_cluster_sample_size(250, 10, None)
        assert result.is_valid

    def test_fixed_cluster_size(self):
        from mcpower.utils.validators import _validate_cluster_sample_size

        # cluster_size=30 provided directly, should be sufficient
        result = _validate_cluster_sample_size(300, 10, 30)
        assert result.is_valid

    def test_fixed_cluster_size_insufficient(self):
        from mcpower.utils.validators import _validate_cluster_sample_size

        result = _validate_cluster_sample_size(300, 10, 20)
        assert not result.is_valid


class TestValidateLmeModelComplexity:
    """Test _validate_lme_model_complexity function."""

    def test_sufficient_obs_per_param(self):
        from mcpower.utils.validators import _validate_lme_model_complexity

        # 2 fixed effects → params = 1 + 2 + 2 = 5; cluster_size = 1000/10 = 100; 100/5 = 20 > 10
        result = _validate_lme_model_complexity(1000, 10, 2)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_warning_zone(self):
        from mcpower.utils.validators import _validate_lme_model_complexity

        # 2 fixed effects → params = 5; cluster_size = 40; 40/5 = 8 → warning (7 < 8 < 10)
        result = _validate_lme_model_complexity(400, 10, 2)
        assert result.is_valid  # is_valid but has warnings
        assert len(result.warnings) > 0

    def test_error_zone(self):
        from mcpower.utils.validators import _validate_lme_model_complexity

        # 2 fixed effects → params = 5; cluster_size = 30; 30/5 = 6 < 7 → error
        result = _validate_lme_model_complexity(300, 10, 2)
        assert not result.is_valid

    def test_many_fixed_effects(self):
        from mcpower.utils.validators import _validate_lme_model_complexity

        # 10 fixed effects → params = 1 + 10 + 2 = 13; cluster_size = 200/10 = 20; 20/13 ≈ 1.5 < 7
        result = _validate_lme_model_complexity(200, 10, 10)
        assert not result.is_valid

    def test_with_explicit_cluster_size(self):
        from mcpower.utils.validators import _validate_lme_model_complexity

        # Explicit cluster_size=100, 2 fixed → 5 params, 100/5 = 20 > 10
        result = _validate_lme_model_complexity(500, 5, 2, cluster_size=100)
        assert result.is_valid
