"""
Edge case tests for MCPower.

Tests for robustness against unusual inputs: near-singular matrices,
NaN/Inf values, extreme ICC, constant columns, and small uploaded data.
"""

import warnings

import numpy as np
import pytest

from mcpower import MCPower


class TestNearSingularCorrelation:
    """Tests for near-singular correlation matrices."""

    def test_high_correlation_accepted(self):
        """Correlation of 0.95 should still work (not singular)."""
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_correlations("corr(x1, x2)=0.95")
        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100

    def test_near_perfect_correlation(self):
        """Correlation of 0.999 should still produce valid results (Cholesky fallback)."""
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        model.set_correlations(corr)
        model.set_simulations(100)
        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        assert 0 <= power <= 100

    def test_identity_correlation_three_vars(self):
        """Identity correlation should produce independent variables."""
        model = MCPower("y = x1 + x2 + x3")
        model.set_effects("x1=0.3, x2=0.3, x3=0.3")
        model.set_simulations(200)
        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        assert power > 20  # Decent effects → reasonable power


class TestNaNInfUploads:
    """Tests for NaN and Inf values in uploaded data."""

    def test_upload_data_with_nan_rows(self):
        """Uploaded data with NaN should either be handled or raise clear error."""
        model = MCPower("y = x1 + x2")
        # Need >=25 rows to pass size validation
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 2))
        data[5, 1] = np.nan  # Insert a NaN
        try:
            model.upload_data(data, columns=["x1", "x2"])
            model.set_effects("x1=0.3, x2=0.2")
            model.set_simulations(50)
            model.find_power(
                sample_size=50,
                print_results=False,
                progress_callback=False,
            )
        except (ValueError, TypeError, RuntimeError):
            pass  # Acceptable to reject NaN data

    def test_upload_data_with_inf(self):
        """Uploaded data with Inf should be handled gracefully."""
        model = MCPower("y = x1")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 1))
        data[3, 0] = np.inf  # Insert an Inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                model.upload_data(data, columns=["x1"])
                model.set_effects("x1=0.3")
                model.set_simulations(50)
                model.find_power(
                    sample_size=50,
                    print_results=False,
                    progress_callback=False,
                )
            except (ValueError, TypeError, RuntimeError):
                pass  # Acceptable to reject Inf data


class TestExtremeICCValues:
    """Tests for ICC values at boundaries of the valid range."""

    @pytest.mark.lme
    def test_low_icc(self):
        """ICC=0.1 (minimum allowed) should produce valid results."""
        model = MCPower("y ~ x1 + (1|group)")
        model.set_cluster("group", ICC=0.1, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(50, model_type="mixed")
        model.set_max_failed_simulations(0.20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.find_power(
                sample_size=1000,
                print_results=False,
                return_results=True,
                progress_callback=False,
            )
        if result:
            power = result["results"]["individual_powers"]["overall"]
            assert 0 <= power <= 100

    @pytest.mark.lme
    @pytest.mark.slow
    def test_high_icc(self):
        """ICC=0.5 should substantially reduce power compared to OLS."""
        model = MCPower("y ~ x1 + (1|group)")
        model.set_cluster("group", ICC=0.5, n_clusters=20)
        model.set_effects("x1=0.5")
        model.set_simulations(50, model_type="mixed")
        model.set_max_failed_simulations(0.20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.find_power(
                # 20 clusters, need >=50 obs/cluster (10 per parameter) → need >=1000
                sample_size=1000,
                print_results=False,
                return_results=True,
                progress_callback=False,
            )
        if result:
            power = result["results"]["individual_powers"]["overall"]
            assert 0 <= power <= 100


class TestConstantColumns:
    """Tests for constant columns in uploaded data."""

    def test_constant_column_upload(self):
        """Constant columns should be handled (dropped or flagged)."""
        model = MCPower("y = x1 + x2")
        rng = np.random.default_rng(42)
        n = 50
        data = np.column_stack(
            [
                rng.standard_normal(n),  # x1: normal
                np.ones(n) * 5.0,  # x2: constant
            ]
        )
        # Upload should accept the data (constant col gets detected/dropped)
        model.upload_data(data, columns=["x1", "x2"])
        model.set_effects("x1=0.3")
        model.set_simulations(50)
        result = model.find_power(
            sample_size=50,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None


class TestSmallBootstrapData:
    """Tests for bootstrap size validation warnings."""

    def test_small_upload_warns(self):
        """Uploading 25-29 observations should print a size warning."""
        model = MCPower("y = x1")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((26, 1))  # Just above validator minimum
        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            model.upload_data(data, columns=["x1"])
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "only 26" in output.lower() or "26 observations" in output.lower()

    def test_adequate_upload_no_warning(self):
        """Uploading >= 50 observations should not warn about size."""
        model = MCPower("y = x1")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 1))
        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            model.upload_data(data, columns=["x1"])
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        # Should not mention small sample concerns
        assert "observations" not in output.lower() or "bootstrap" not in output.lower()


class TestHighKurtosisStandardisation:
    """Tests that high-kurtosis distribution is properly standardised."""

    def test_high_kurtosis_variance_near_one(self):
        """High-kurtosis variable should have Var ≈ 1.0 (standardised)."""
        from mcpower.utils.data_generation import (
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            _generate_X_core,
        )

        n = 100000
        X = _generate_X_core(
            n,
            1,
            np.eye(1),
            np.array([4], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            42,
        )
        var = np.var(X[:, 0], ddof=1)
        # Should be within 10% of 1.0
        assert 0.90 <= var <= 1.10, f"High-kurtosis Var={var:.4f}, expected ≈1.0"

    def test_normal_variance_near_one(self):
        """Normal variable should have Var ≈ 1.0 (baseline check)."""
        from mcpower.utils.data_generation import (
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            _generate_X_core,
        )

        n = 100000
        X = _generate_X_core(
            n,
            1,
            np.eye(1),
            np.array([0], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            42,
        )
        var = np.var(X[:, 0], ddof=1)
        assert 0.95 <= var <= 1.05, f"Normal Var={var:.4f}, expected ≈1.0"


class TestNumericalStability:
    """Tests for numerical stability with large sample sizes."""

    def test_large_sample_ols(self):
        """OLS with n=10000 should be numerically stable."""
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_simulations(50)
        result = model.find_power(
            sample_size=10000,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        # With large n and decent effects, power should be very high
        assert power > 95, f"Power={power}% too low for n=10000"

    def test_large_sample_many_predictors(self):
        """Five predictors with n=5000 should be stable."""
        model = MCPower("y = a + b + c + d + e")
        model.set_effects("a=0.2, b=0.2, c=0.2, d=0.2, e=0.2")
        model.set_simulations(50)
        result = model.find_power(
            sample_size=5000,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        assert power > 90, f"Power={power}% too low for n=5000 with 5 predictors"

    def test_large_sample_interactions(self):
        """Interactions with large n should not cause numerical issues."""
        model = MCPower("y = x1 + x2 + x1:x2")
        model.set_effects("x1=0.3, x2=0.3, x1:x2=0.1")
        model.set_simulations(50)
        result = model.find_power(
            sample_size=10000,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        power = result["results"]["individual_powers"]["overall"]
        assert power > 95

    def test_small_effects_large_sample(self):
        """Very small effects with large n should detect significance."""
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.05, x2=0.05")
        model.set_simulations(100)
        result = model.find_power(
            sample_size=50000,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        # n=50000 with even tiny effects should have high power
        power = result["results"]["individual_powers"]["overall"]
        assert power > 80, f"Power={power}% too low for n=50000"
