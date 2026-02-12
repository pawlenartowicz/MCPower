"""
Comprehensive tests for upload_data() functionality.

Tests auto-detection, data_types override, and all three preserve_correlation modes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcpower import MCPower

# Load test data
TEST_DIR = Path(__file__).parent
CARS_CSV = TEST_DIR / "cars.csv"


@pytest.fixture
def cars_data():
    """Load cars.csv dataset."""
    df = pd.read_csv(CARS_CSV, index_col=0)
    return df


class TestAutoDetection:
    """Test auto-detection of variable types based on unique value counts."""

    def test_binary_auto_detection(self, cars_data):
        """Test that 2-value columns are auto-detected as binary."""
        model = MCPower("mpg = vs + am")
        model.upload_data(cars_data[["vs", "am"]])
        model.set_effects("vs=0.3, am=0.4")
        model.apply()

        # Check that vs and am were detected as uploaded_binary
        vs_pred = model._registry.get_predictor("vs")
        am_pred = model._registry.get_predictor("am")

        assert vs_pred.var_type == "uploaded_binary"
        assert am_pred.var_type == "uploaded_binary"

    def test_factor_auto_detection(self, cars_data):
        """Test that 3-6 value columns are auto-detected as factor."""
        model = MCPower("mpg = cyl + gear")
        model.upload_data(cars_data[["cyl", "gear"]])
        model.set_effects("cyl[2]=0.3, cyl[3]=0.4, gear[2]=0.2, gear[3]=0.3")
        model.apply()

        # Check that cyl and gear were detected as factor
        # After expansion, check the factor names
        assert "cyl" in model._registry.factor_names
        assert "gear" in model._registry.factor_names

        # Check dummy variables exist
        assert "cyl[2]" in model._registry.dummy_names
        assert "cyl[3]" in model._registry.dummy_names
        assert "gear[2]" in model._registry.dummy_names
        assert "gear[3]" in model._registry.dummy_names

    def test_continuous_auto_detection(self, cars_data):
        """Test that 7+ value columns are auto-detected as continuous."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]])
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # Check that hp and wt were detected as continuous (uploaded_data)
        hp_pred = model._registry.get_predictor("hp")
        wt_pred = model._registry.get_predictor("wt")

        assert hp_pred.var_type == "uploaded_data"
        assert wt_pred.var_type == "uploaded_data"

    def test_constant_column_dropped(self, cars_data):
        """Test that columns with 1 unique value are dropped with warning."""
        # Add a constant column
        data_with_constant = cars_data[["mpg", "hp"]].copy()
        data_with_constant["constant"] = 1.0

        model = MCPower("mpg = hp + constant")

        # Should raise error because 'constant' will be dropped
        with pytest.raises(ValueError, match="All uploaded columns were dropped"):
            model.upload_data(data_with_constant[["constant"]])
            model.apply()

    def test_mixed_types_auto_detection(self, cars_data):
        """Test auto-detection with mixed variable types."""
        model = MCPower("mpg = vs + cyl + hp")
        model.upload_data(cars_data[["vs", "cyl", "hp"]])
        model.set_effects("vs=0.3, cyl[2]=0.2, cyl[3]=0.4, hp=0.5")
        model.apply()

        vs_pred = model._registry.get_predictor("vs")
        hp_pred = model._registry.get_predictor("hp")

        assert vs_pred.var_type == "uploaded_binary"
        assert "cyl" in model._registry.factor_names  # Factor was expanded
        assert hp_pred.var_type == "uploaded_data"


class TestDataTypesOverride:
    """Test data_types parameter to override auto-detection."""

    def test_override_to_continuous(self, cars_data):
        """Test forcing a factor-like column to be continuous."""
        model = MCPower("mpg = cyl + hp")
        model.upload_data(cars_data[["cyl", "hp"]], data_types={"cyl": "continuous"})
        model.set_effects("cyl=0.4, hp=0.5")
        model.apply()

        cyl_pred = model._registry.get_predictor("cyl")
        # Should be uploaded_data (continuous) instead of factor
        assert cyl_pred.var_type == "uploaded_data"

    def test_override_to_factor(self, cars_data):
        """Test forcing a continuous column to be factor."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], data_types={"hp": "factor"})
        # hp will become a factor with many levels (22 levels)
        # Just apply data to check it's detected as factor
        model._apply_data()  # Apply just the data part
        # After expansion, hp should be in factor_names
        assert "hp" in model._registry.factor_names

    def test_override_to_binary(self, cars_data):
        """Test forcing a continuous column to be binary."""
        # Use median split for hp
        data = cars_data[["hp", "wt"]].copy()
        data["hp_binary"] = (data["hp"] > data["hp"].median()).astype(int)

        model_binary = MCPower("mpg = hp_binary + wt")
        model_binary.upload_data(data[["hp_binary", "wt"]], data_types={"hp_binary": "binary"})
        model_binary.set_effects("hp_binary=0.4, wt=0.3")
        model_binary.apply()

        hp_pred = model_binary._registry.get_predictor("hp_binary")
        assert hp_pred.var_type == "uploaded_binary"

    def test_invalid_data_type_raises_error(self, cars_data):
        """Test that invalid data type raises error."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="Invalid type 'invalid'"):
            model.upload_data(cars_data[["hp", "wt"]], data_types={"hp": "invalid"})

    def test_unknown_column_in_data_types_raises_error(self, cars_data):
        """Test that unknown column name in data_types raises error."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="Variable 'unknown' in data_types not found"):
            model.upload_data(cars_data[["hp", "wt"]], data_types={"unknown": "continuous"})


class TestPreserveCorrelationNo:
    """Test preserve_correlation='no' mode."""

    def test_no_correlation_from_data(self, cars_data):
        """Test that mode='no' does not compute correlations from data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="no")
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # Correlation matrix should be identity (or user-specified)
        corr = model.correlation_matrix
        # hp and wt should have 0 correlation (identity matrix off-diagonal)
        assert corr[0, 1] == 0.0
        assert corr[1, 0] == 0.0

    def test_binary_uses_standard_generation(self, cars_data):
        """Test that binary variables use standard generation in 'no' mode."""
        model = MCPower("mpg = vs + am")
        model.upload_data(cars_data[["vs", "am"]], preserve_correlation="no")
        model.set_effects("vs=0.3, am=0.4")
        model.apply()

        # Should detect proportions from data
        vs_pred = model._registry.get_predictor("vs")
        assert vs_pred.var_type == "binary"
        assert vs_pred.proportion is not None

    def test_continuous_uses_lookup_tables(self, cars_data):
        """Test that continuous variables use lookup tables in 'no' mode."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="no")
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # Should have lookup tables populated
        assert model.upload_normal_values.shape[0] > 0
        assert model.upload_data_values.shape[0] > 0


class TestPreserveCorrelationPartial:
    """Test preserve_correlation='partial' mode (default)."""

    def test_strict_is_default(self, cars_data):
        """Test that 'strict' is the default mode."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]])
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        assert model._preserve_correlation == "strict"

    def test_correlations_computed_from_data(self, cars_data):
        """Test that correlations are computed from uploaded data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="partial")
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # Correlation should match data correlation
        data_corr = np.corrcoef(cars_data[["hp", "wt"]].values, rowvar=False)
        model_corr = model.correlation_matrix

        # Should be approximately equal (hp-wt correlation)
        assert abs(model_corr[0, 1] - data_corr[0, 1]) < 0.01

    def test_user_can_override_correlations(self, cars_data):
        """Test that user-specified correlations can override data correlations."""
        model = MCPower("mpg = hp + wt + vs")

        # Upload data first
        model.upload_data(cars_data[["hp", "wt", "vs"]], preserve_correlation="partial")
        model.set_effects("hp=0.5, wt=0.3, vs=0.2")

        # Set custom correlation (will be applied in apply())
        # This tests that user correlations can override data correlations
        # For now, the implementation always uses data correlations
        # TODO: Implement user override priority
        model.apply()

        # Just verify it doesn't crash
        assert model.correlation_matrix is not None


class TestPreserveCorrelationStrict:
    """Test preserve_correlation='strict' mode."""

    def test_strict_mode_sets_metadata(self, cars_data):
        """Test that strict mode stores raw data for bootstrap."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        assert model._preserve_correlation == "strict"
        assert model._uploaded_raw_data is not None
        assert len(model._uploaded_var_metadata) == 2

    def test_strict_mode_warns_cross_correlations(self, cars_data, capsys):
        """Test that strict mode warns about cross-correlations."""
        model = MCPower("mpg = hp + wt + x1")  # x1 is created, hp/wt uploaded
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3, x1=0.4")
        model.apply()

        captured = capsys.readouterr()
        # Should warn about cross-correlations
        assert "Warning" in captured.out or "correlations" in captured.out.lower()

    def test_strict_mode_bootstrap_preserves_relationships(self, cars_data):
        """Test that strict mode uses bootstrap (can run simulation)."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # Should be able to run simulation without error
        result = model.find_power(sample_size=50, print_results=False, return_results=True)
        assert result is not None
        assert "results" in result  # Result structure has 'results' key

    def test_strict_mode_mixed_uploaded_and_created(self, cars_data):
        """Test strict mode with both uploaded and created (non-uploaded) variables."""
        model = MCPower("mpg = hp + wt + x1")  # x1 is created, hp/wt uploaded
        model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3, x1=0.4")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None
        assert "results" in result

    def test_strict_mode_mixed_find_power_target_all(self, cars_data):
        """Test find_power target_test='all' with mixed uploaded/created vars."""
        model = MCPower("mpg = hp + x1")
        model.upload_data(cars_data[["hp"]], preserve_correlation="strict")
        model.set_effects("hp=0.5, x1=0.3")
        result = model.find_power(
            sample_size=200,
            return_results=True,
            print_results=False,
            target_test="all",
        )
        assert result is not None

    def test_strict_mode_full_dataframe_mixed(self, cars_data):
        """Test strict mode with full DataFrame where only some vars match model."""
        model = MCPower("mpg = hp + x1")
        model.upload_data(cars_data)  # Full DataFrame, only hp matches
        model.set_effects("hp=0.5, x1=0.3")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None

    def test_strict_mode_with_binary(self, cars_data):
        """Test strict mode with binary variables."""
        model = MCPower("mpg = vs + am")
        model.upload_data(cars_data[["vs", "am"]], preserve_correlation="strict")
        model.set_effects("vs=0.3, am=0.4")
        model.apply()

        # Check metadata
        assert "vs" in model._uploaded_var_metadata
        assert "am" in model._uploaded_var_metadata
        assert model._uploaded_var_metadata["vs"]["type"] == "binary"
        assert model._uploaded_var_metadata["am"]["type"] == "binary"

    def test_strict_mode_with_factor(self, cars_data):
        """Test strict mode with factor variables."""
        model = MCPower("mpg = cyl + gear")
        model.upload_data(cars_data[["cyl", "gear"]], preserve_correlation="strict")
        model.set_effects("cyl[2]=0.3, cyl[3]=0.4, gear[2]=0.2, gear[3]=0.3")
        model.apply()

        # Check metadata
        assert "cyl" in model._uploaded_var_metadata
        assert "gear" in model._uploaded_var_metadata
        assert model._uploaded_var_metadata["cyl"]["type"] == "factor"
        assert model._uploaded_var_metadata["gear"]["type"] == "factor"


class TestWarnings:
    """Test warning messages."""

    def test_warning_for_unmatched_columns(self, cars_data, capsys):
        """Test warning when data columns don't match model variables."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt", "vs"]])  # vs not in model
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        captured = capsys.readouterr()
        assert "Ignoring unmatched columns" in captured.out
        assert "vs" in captured.out

    def test_warning_for_large_sample_size(self, cars_data, capsys):
        """Test warning when sample_size > 3x uploaded data size."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]])
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # 32 samples * 3 = 96, so 100 should trigger warning
        model.find_power(sample_size=100, print_results=False)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "more than 3x" in captured.out

    def test_warning_for_dropped_constant_columns(self, cars_data, capsys):
        """Test warning when constant columns are dropped."""
        data_with_constant = cars_data[["hp", "wt"]].copy()
        data_with_constant["constant"] = 1.0

        model = MCPower("mpg = hp + wt + constant")  # Include constant in model
        model.upload_data(data_with_constant)
        model.set_effects("hp=0.5, wt=0.3")

        # Clear output before apply
        capsys.readouterr()

        # This should raise an error because constant was dropped and no effect was set for it
        # But the auto-detection output should show it was dropped
        try:
            model.apply()
        except ValueError:
            pass  # Expected to fail because constant column missing

        # Check the output showed the constant being dropped
        captured = capsys.readouterr()
        assert "1 unique value" in captured.out
        assert "constant" in captured.out


class TestMixedTypeDataFrames:
    """Test uploading full DataFrames with mixed types (non-numeric columns)."""

    def test_full_dataframe_with_unmatched_columns(self, cars_data):
        """Test uploading full DataFrame where unmatched columns are non-numeric."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data)  # Full DataFrame, not pre-filtered
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        hp_pred = model._registry.get_predictor("hp")
        wt_pred = model._registry.get_predictor("wt")
        assert hp_pred.var_type == "uploaded_data"
        assert wt_pred.var_type == "uploaded_data"

    def test_dataframe_with_string_index_column(self):
        """Test DataFrame read without index_col (has 'Unnamed: 0' string column)."""
        df = pd.read_csv(CARS_CSV)  # No index_col=0 -> 'Unnamed: 0' is strings
        model = MCPower("mpg = hp + wt")
        model.upload_data(df)
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        hp_pred = model._registry.get_predictor("hp")
        assert hp_pred.var_type == "uploaded_data"

    def test_full_dataframe_find_power(self, cars_data):
        """Test find_power works end-to-end with full DataFrame upload."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data)
        model.set_effects("hp=0.5, wt=0.3")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None
        assert "results" in result

    def test_full_dataframe_with_mixed_var_types(self, cars_data):
        """Test full DataFrame with binary + factor + continuous auto-detection."""
        model = MCPower("mpg = vs + cyl + hp")
        model.upload_data(cars_data)  # Full DataFrame
        model.set_effects("vs=0.3, cyl[2]=0.2, cyl[3]=0.4, hp=0.5")
        model.apply()

        vs_pred = model._registry.get_predictor("vs")
        hp_pred = model._registry.get_predictor("hp")
        assert vs_pred.var_type == "uploaded_binary"
        assert "cyl" in model._registry.factor_names
        assert hp_pred.var_type == "uploaded_data"

    def test_full_dataframe_find_power_target_all(self, cars_data):
        """Test find_power with target_test='all' on full DataFrame upload."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data)
        model.set_effects("hp=0.5, wt=0.3")
        result = model.find_power(
            sample_size=200,
            return_results=True,
            print_results=False,
            target_test="all",
        )
        assert result is not None

    def test_non_numeric_matched_column_raises_error(self):
        """Test clear error when a matched column contains non-numeric data."""
        df = pd.DataFrame(
            {
                "x": ["a", "b", "c"] * 10,  # String data matching model variable
                "y": range(30),
            }
        )
        model = MCPower("y = x")
        model.upload_data(df)
        with pytest.raises(TypeError, match="non-numeric data"):
            model.apply()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_matching_columns_ignores_data(self, cars_data, capsys):
        """Test that zero matching columns prints warning and ignores data."""
        model = MCPower("mpg = x1 + x2")
        model.upload_data(cars_data[["hp", "wt"]])
        model.set_effects("x1=0.3, x2=0.4")
        model.apply()

        captured = capsys.readouterr()
        assert "uploaded data ignored" in captured.out.lower()
        assert model._pending_data is None

    def test_no_matching_columns_find_power(self, capsys):
        """Test find_power works when uploaded data has no matching columns."""
        df = pd.read_csv(CARS_CSV)
        model = MCPower("y ~ x")
        model.upload_data(df)
        model.set_effects("x=0.3")
        result = model.find_power(sample_size=200, return_results=True, print_results=False)
        assert result is not None

        captured = capsys.readouterr()
        assert "uploaded data ignored" in captured.out.lower()

    def test_empty_data_raises_error(self):
        """Test error with too few samples."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="at least 25 samples"):
            model.upload_data(np.empty((5, 2)), columns=["hp", "wt"])

    def test_invalid_preserve_correlation_mode_raises_error(self, cars_data):
        """Test error with invalid preserve_correlation mode."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="preserve_correlation must be one of"):
            model.upload_data(cars_data[["hp", "wt"]], preserve_correlation="invalid")

    def test_numpy_array_auto_columns(self, cars_data):
        """Test that numpy array without columns auto-generates column names."""
        model = MCPower("mpg = hp + wt")
        data_array = cars_data[["hp", "wt"]].values

        model.upload_data(data_array)
        assert model._pending_data["columns"] == ["column_1", "column_2"]

    def test_dict_format(self, cars_data):
        """Test uploading data as dict."""
        model = MCPower("mpg = hp + wt")
        data_dict = {
            "hp": cars_data["hp"].values,
            "wt": cars_data["wt"].values,
        }
        model.upload_data(data_dict)
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        assert model._applied is True

    def test_sample_size_warning_in_find_sample_size(self, cars_data, capsys):
        """Test warning in find_sample_size when max size > 3x data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data[["hp", "wt"]])
        model.set_effects("hp=0.5, wt=0.3")
        model.apply()

        # 32 * 3 = 96, so to_size=150 should trigger warning
        model.find_sample_size(from_size=30, to_size=150, by=20, print_results=False)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "more than 3x" in captured.out
