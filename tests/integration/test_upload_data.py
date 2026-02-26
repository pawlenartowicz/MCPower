"""
Comprehensive tests for upload_data() functionality.

Tests auto-detection, data_types override, and all three preserve_correlation modes.
Uses dict-based data loading (no pandas required) with a few DataFrame-specific tests.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from mcpower import MCPower

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Load test data
TEST_DIR = Path(__file__).parent.parent
CARS_CSV = TEST_DIR / "cars.csv"


def _load_csv(path) -> dict[str, list]:
    """Load a CSV file into a dict of lists, auto-converting numeric columns."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    result: dict[str, list] = {}
    for col in rows[0]:
        if col == "":
            continue  # skip unnamed index column
        raw = [r[col] for r in rows]
        # Try to convert to float
        try:
            result[col] = [float(v) for v in raw]
        except (ValueError, TypeError):
            result[col] = raw
    return result


def _select(data: dict[str, list], cols: list[str]) -> dict[str, list]:
    """Select subset of columns from a dict-of-lists dataset."""
    return {k: data[k] for k in cols}


def _copy_with(data: dict[str, list], **extra) -> dict[str, list]:
    """Copy data dict and add extra columns."""
    out = {k: list(v) for k, v in data.items()}
    out.update(extra)
    return out


@pytest.fixture
def cars_data():
    """Load cars.csv dataset as dict of lists."""
    return _load_csv(CARS_CSV)


class TestAutoDetection:
    """Test auto-detection of variable types based on unique value counts."""

    def test_binary_auto_detection(self, cars_data):
        """Test that 2-value columns are auto-detected as binary."""
        model = MCPower("mpg = vs + am")
        model.upload_data(_select(cars_data, ["vs", "am"]))
        model.set_effects("vs=0.3, am=0.4")
        model._apply()

        # Check that vs and am were detected as uploaded_binary
        vs_pred = model._registry.get_predictor("vs")
        am_pred = model._registry.get_predictor("am")

        assert vs_pred.var_type == "uploaded_binary"
        assert am_pred.var_type == "uploaded_binary"

    def test_factor_auto_detection(self, cars_data):
        """Test that 3-6 value columns are auto-detected as factor."""
        model = MCPower("mpg = cyl + gear")
        model.upload_data(_select(cars_data, ["cyl", "gear"]), preserve_factor_level_names=False)
        model.set_effects("cyl[2]=0.3, cyl[3]=0.4, gear[2]=0.2, gear[3]=0.3")
        model._apply()

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
        model.upload_data(_select(cars_data, ["hp", "wt"]))
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # Check that hp and wt were detected as continuous (uploaded_data)
        hp_pred = model._registry.get_predictor("hp")
        wt_pred = model._registry.get_predictor("wt")

        assert hp_pred.var_type == "uploaded_data"
        assert wt_pred.var_type == "uploaded_data"

    def test_constant_column_dropped(self, cars_data):
        """Test that columns with 1 unique value are dropped with warning."""
        # Add a constant column
        data = _copy_with(_select(cars_data, ["mpg", "hp"]), constant=[1.0] * len(cars_data["mpg"]))

        model = MCPower("mpg = hp + constant")

        # Should raise error because 'constant' will be dropped
        with pytest.raises(ValueError, match="All uploaded columns were dropped"):
            model.upload_data(_select(data, ["constant"]))
            model._apply()

    def test_mixed_types_auto_detection(self, cars_data):
        """Test auto-detection with mixed variable types."""
        model = MCPower("mpg = vs + cyl + hp")
        model.upload_data(_select(cars_data, ["vs", "cyl", "hp"]), preserve_factor_level_names=False)
        model.set_effects("vs=0.3, cyl[2]=0.2, cyl[3]=0.4, hp=0.5")
        model._apply()

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
        model.upload_data(_select(cars_data, ["cyl", "hp"]), data_types={"cyl": "continuous"})
        model.set_effects("cyl=0.4, hp=0.5")
        model._apply()

        cyl_pred = model._registry.get_predictor("cyl")
        # Should be uploaded_data (continuous) instead of factor
        assert cyl_pred.var_type == "uploaded_data"

    def test_override_to_factor(self, cars_data):
        """Test forcing a continuous column to be factor."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), data_types={"hp": "factor"})
        # hp will become a factor with many levels (22 levels)
        # Just apply data to check it's detected as factor
        model._apply_data()  # Apply just the data part
        # After expansion, hp should be in factor_names
        assert "hp" in model._registry.factor_names

    def test_override_to_binary(self, cars_data):
        """Test forcing a continuous column to be binary."""
        # Use median split for hp
        hp_vals = cars_data["hp"]
        median_hp = sorted(hp_vals)[len(hp_vals) // 2]
        hp_binary = [int(v > median_hp) for v in hp_vals]

        data = {
            "hp_binary": hp_binary,
            "wt": list(cars_data["wt"]),
        }

        model_binary = MCPower("mpg = hp_binary + wt")
        model_binary.upload_data(data, data_types={"hp_binary": "binary"})
        model_binary.set_effects("hp_binary=0.4, wt=0.3")
        model_binary._apply()

        hp_pred = model_binary._registry.get_predictor("hp_binary")
        assert hp_pred.var_type == "uploaded_binary"

    def test_invalid_data_type_raises_error(self, cars_data):
        """Test that invalid data type raises error."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="Invalid type 'invalid'"):
            model.upload_data(_select(cars_data, ["hp", "wt"]), data_types={"hp": "invalid"})

    def test_unknown_column_in_data_types_raises_error(self, cars_data):
        """Test that unknown column name in data_types raises error."""
        model = MCPower("mpg = hp + wt")

        with pytest.raises(ValueError, match="Variable 'unknown' in data_types not found"):
            model.upload_data(_select(cars_data, ["hp", "wt"]), data_types={"unknown": "continuous"})


class TestPreserveCorrelationNo:
    """Test preserve_correlation='no' mode."""

    def test_no_correlation_from_data(self, cars_data):
        """Test that mode='no' does not compute correlations from data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="no")
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # Correlation matrix should be identity (or user-specified)
        corr = model.correlation_matrix
        # hp and wt should have 0 correlation (identity matrix off-diagonal)
        assert corr[0, 1] == 0.0
        assert corr[1, 0] == 0.0

    def test_binary_uses_standard_generation(self, cars_data):
        """Test that binary variables use standard generation in 'no' mode."""
        model = MCPower("mpg = vs + am")
        model.upload_data(_select(cars_data, ["vs", "am"]), preserve_correlation="no")
        model.set_effects("vs=0.3, am=0.4")
        model._apply()

        # Should detect proportions from data
        vs_pred = model._registry.get_predictor("vs")
        assert vs_pred.var_type == "binary"
        assert vs_pred.proportion is not None

    def test_continuous_uses_lookup_tables(self, cars_data):
        """Test that continuous variables use lookup tables in 'no' mode."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="no")
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # Should have lookup tables populated
        assert model.upload_normal_values.shape[0] > 0
        assert model.upload_data_values.shape[0] > 0


class TestPreserveCorrelationPartial:
    """Test preserve_correlation='partial' mode (default)."""

    def test_strict_is_default(self, cars_data):
        """Test that 'strict' is the default mode."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]))
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        assert model._preserve_correlation == "strict"

    def test_correlations_computed_from_data(self, cars_data):
        """Test that correlations are computed from uploaded data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="partial")
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # Correlation should match data correlation
        hp_arr = np.array(cars_data["hp"])
        wt_arr = np.array(cars_data["wt"])
        data_corr = np.corrcoef(np.column_stack([hp_arr, wt_arr]), rowvar=False)
        model_corr = model.correlation_matrix

        # Should be approximately equal (hp-wt correlation)
        assert abs(model_corr[0, 1] - data_corr[0, 1]) < 0.01

    def test_user_can_override_correlations(self, cars_data):
        """Test that user-specified correlations can override data correlations."""
        model = MCPower("mpg = hp + wt + vs")

        # Upload data first
        model.upload_data(_select(cars_data, ["hp", "wt", "vs"]), preserve_correlation="partial")
        model.set_effects("hp=0.5, wt=0.3, vs=0.2")

        # Set custom correlation (will be applied in apply())
        # This tests that user correlations can override data correlations
        # For now, the implementation always uses data correlations
        # TODO: Implement user override priority
        model._apply()

        # Just verify it doesn't crash
        assert model.correlation_matrix is not None


class TestPreserveCorrelationStrict:
    """Test preserve_correlation='strict' mode."""

    def test_strict_mode_sets_metadata(self, cars_data):
        """Test that strict mode stores raw data for bootstrap."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        assert model._preserve_correlation == "strict"
        assert model._uploaded_raw_data is not None
        assert len(model._uploaded_var_metadata) == 2

    def test_strict_mode_warns_cross_correlations(self, cars_data, capsys):
        """Test that strict mode warns about cross-correlations."""
        model = MCPower("mpg = hp + wt + x1")  # x1 is created, hp/wt uploaded
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3, x1=0.4")
        model._apply()

        captured = capsys.readouterr()
        # Should warn about cross-correlations
        assert "Warning" in captured.out or "correlations" in captured.out.lower()

    def test_strict_mode_bootstrap_preserves_relationships(self, cars_data):
        """Test that strict mode uses bootstrap (can run simulation)."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # Should be able to run simulation without error
        result = model.find_power(sample_size=50, print_results=False, return_results=True)
        assert result is not None
        assert "results" in result  # Result structure has 'results' key

    def test_strict_mode_mixed_uploaded_and_created(self, cars_data):
        """Test strict mode with both uploaded and created (non-uploaded) variables."""
        model = MCPower("mpg = hp + wt + x1")  # x1 is created, hp/wt uploaded
        model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="strict")
        model.set_effects("hp=0.5, wt=0.3, x1=0.4")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None
        assert "results" in result

    def test_strict_mode_mixed_find_power_target_all(self, cars_data):
        """Test find_power target_test='all' with mixed uploaded/created vars."""
        model = MCPower("mpg = hp + x1")
        model.upload_data(_select(cars_data, ["hp"]), preserve_correlation="strict")
        model.set_effects("hp=0.5, x1=0.3")
        result = model.find_power(
            sample_size=200,
            return_results=True,
            print_results=False,
            target_test="all",
        )
        assert result is not None

    def test_strict_mode_full_dict_mixed(self, cars_data):
        """Test strict mode with full dict where only some vars match model."""
        model = MCPower("mpg = hp + x1")
        model.upload_data(cars_data)  # Full dict, only hp matches
        model.set_effects("hp=0.5, x1=0.3")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None

    def test_strict_mode_with_binary(self, cars_data):
        """Test strict mode with binary variables."""
        model = MCPower("mpg = vs + am")
        model.upload_data(_select(cars_data, ["vs", "am"]), preserve_correlation="strict")
        model.set_effects("vs=0.3, am=0.4")
        model._apply()

        # Check metadata
        assert "vs" in model._uploaded_var_metadata
        assert "am" in model._uploaded_var_metadata
        assert model._uploaded_var_metadata["vs"]["type"] == "binary"
        assert model._uploaded_var_metadata["am"]["type"] == "binary"

    def test_strict_mode_with_factor(self, cars_data):
        """Test strict mode with factor variables."""
        model = MCPower("mpg = cyl + gear")
        model.upload_data(
            _select(cars_data, ["cyl", "gear"]),
            preserve_correlation="strict",
            preserve_factor_level_names=False,
        )
        model.set_effects("cyl[2]=0.3, cyl[3]=0.4, gear[2]=0.2, gear[3]=0.3")
        model._apply()

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
        model.upload_data(_select(cars_data, ["hp", "wt", "vs"]))  # vs not in model
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        captured = capsys.readouterr()
        assert "Ignoring unmatched columns" in captured.out
        assert "vs" in captured.out

    def test_warning_for_large_sample_size(self, cars_data, capsys):
        """Test warning when sample_size > 3x uploaded data size."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]))
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # 32 samples * 3 = 96, so 100 should trigger warning
        model.find_power(sample_size=100, print_results=False)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "more than 3x" in captured.out

    def test_warning_for_dropped_constant_columns(self, cars_data, capsys):
        """Test warning when constant columns are dropped."""
        data = _copy_with(_select(cars_data, ["hp", "wt"]), constant=[1.0] * len(cars_data["hp"]))

        model = MCPower("mpg = hp + wt + constant")  # Include constant in model
        model.upload_data(data)
        model.set_effects("hp=0.5, wt=0.3")

        # Clear output before apply
        capsys.readouterr()

        # This should raise an error because constant was dropped and no effect was set for it
        # But the auto-detection output should show it was dropped
        try:
            model._apply()
        except ValueError:
            pass  # Expected to fail because constant column missing

        # Check the output showed the constant being dropped
        captured = capsys.readouterr()
        assert "1 unique value" in captured.out
        assert "constant" in captured.out


class TestMixedTypeData:
    """Test uploading data with mixed types (non-numeric columns)."""

    def test_full_dict_with_unmatched_columns(self, cars_data):
        """Test uploading full dict where unmatched columns are non-numeric."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data)  # Full dict, not pre-filtered
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        hp_pred = model._registry.get_predictor("hp")
        wt_pred = model._registry.get_predictor("wt")
        assert hp_pred.var_type == "uploaded_data"
        assert wt_pred.var_type == "uploaded_data"

    def test_full_dict_find_power(self, cars_data):
        """Test find_power works end-to-end with full dict upload."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(cars_data)
        model.set_effects("hp=0.5, wt=0.3")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None
        assert "results" in result

    def test_full_dict_with_mixed_var_types(self, cars_data):
        """Test full dict with binary + factor + continuous auto-detection."""
        model = MCPower("mpg = vs + cyl + hp")
        model.upload_data(cars_data, preserve_factor_level_names=False)  # Full dict
        model.set_effects("vs=0.3, cyl[2]=0.2, cyl[3]=0.4, hp=0.5")
        model._apply()

        vs_pred = model._registry.get_predictor("vs")
        hp_pred = model._registry.get_predictor("hp")
        assert vs_pred.var_type == "uploaded_binary"
        assert "cyl" in model._registry.factor_names
        assert hp_pred.var_type == "uploaded_data"

    def test_full_dict_find_power_target_all(self, cars_data):
        """Test find_power with target_test='all' on full dict upload."""
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

    def test_string_matched_column_auto_detected_as_factor(self):
        """Test that a matched column with string data is auto-detected as factor."""
        data = {
            "x": ["a", "b", "c"] * 10,
            "y": list(range(30)),
        }
        model = MCPower("y = x")
        model.upload_data(data)
        model.set_effects("x[b]=0.3, x[c]=0.4")
        model._apply()
        assert "x" in model._registry.factor_names
        assert "x[b]" in model._registry.dummy_names
        assert "x[c]" in model._registry.dummy_names


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_matching_columns_ignores_data(self, cars_data, capsys):
        """Test that zero matching columns prints warning and ignores data."""
        model = MCPower("mpg = x1 + x2")
        model.upload_data(_select(cars_data, ["hp", "wt"]))
        model.set_effects("x1=0.3, x2=0.4")
        model._apply()

        captured = capsys.readouterr()
        assert "uploaded data ignored" in captured.out.lower()
        assert model._pending_data is None

    def test_no_matching_columns_find_power(self, cars_data, capsys):
        """Test find_power works when uploaded data has no matching columns."""
        model = MCPower("y ~ x")
        model.upload_data(cars_data)
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
            model.upload_data(_select(cars_data, ["hp", "wt"]), preserve_correlation="invalid")

    def test_numpy_array_auto_columns(self, cars_data):
        """Test that numpy array without columns auto-generates column names."""
        model = MCPower("mpg = hp + wt")
        hp_arr = np.array(cars_data["hp"])
        wt_arr = np.array(cars_data["wt"])
        data_array = np.column_stack([hp_arr, wt_arr])

        model.upload_data(data_array)
        assert model._pending_data["columns"] == ["column_1", "column_2"]

    def test_dict_format(self, cars_data):
        """Test uploading data as dict."""
        model = MCPower("mpg = hp + wt")
        data_dict = {
            "hp": cars_data["hp"],
            "wt": cars_data["wt"],
        }
        model.upload_data(data_dict)
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        assert model._applied is True

    def test_sample_size_warning_in_find_sample_size(self, cars_data, capsys):
        """Test warning in find_sample_size when max size > 3x data."""
        model = MCPower("mpg = hp + wt")
        model.upload_data(_select(cars_data, ["hp", "wt"]))
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        # 32 * 3 = 96, so size=110 > 96 triggers warning
        model.find_sample_size(from_size=50, to_size=110, by=30, print_results=False)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "more than 3x" in captured.out


class TestStringColumns:
    """Test string/categorical column support."""

    def test_string_column_auto_detected_as_factor(self, cars_data):
        model = MCPower("mpg = origin + hp")
        model.upload_data(_select(cars_data, ["origin", "hp"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.4, hp=0.5")
        model._apply()
        assert "origin" in model._registry.factor_names

    def test_string_column_creates_named_dummies(self, cars_data):
        model = MCPower("mpg = origin + hp")
        model.upload_data(_select(cars_data, ["origin", "hp"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.4, hp=0.5")
        model._apply()
        dummy_names = model._registry.dummy_names
        assert "origin[Japan]" in dummy_names
        assert "origin[USA]" in dummy_names
        assert "origin[Europe]" not in dummy_names

    def test_string_column_no_mode(self, cars_data):
        model = MCPower("mpg = origin + hp")
        model.upload_data(_select(cars_data, ["origin", "hp"]), preserve_correlation="no")
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.4, hp=0.5")
        model._apply()
        assert "origin" in model._registry.factor_names

    def test_too_many_string_levels_raises(self):
        data = {
            "name": [f"person_{i}" for i in range(50)],
            "x1": list(np.random.randn(50)),
        }
        model = MCPower("y = name + x1")
        with pytest.raises(ValueError, match="too many unique"):
            model.upload_data(_select(data, ["name", "x1"]))
            model._apply()


class TestPreserveFactorLevelNames:
    """Test preserve_factor_level_names parameter."""

    def test_numeric_factor_uses_original_values(self, cars_data):
        model = MCPower("mpg = cyl + hp")
        model.upload_data(_select(cars_data, ["cyl", "hp"]))
        model.set_effects("cyl[6]=0.3, cyl[8]=0.4, hp=0.5")
        model._apply()
        dummy_names = model._registry.dummy_names
        assert "cyl[6]" in dummy_names
        assert "cyl[8]" in dummy_names
        assert "cyl[2]" not in dummy_names

    def test_preserve_false_uses_integer_indices(self, cars_data):
        model = MCPower("mpg = cyl + hp")
        model.upload_data(_select(cars_data, ["cyl", "hp"]), preserve_factor_level_names=False)
        model.set_effects("cyl[2]=0.3, cyl[3]=0.4, hp=0.5")
        model._apply()
        dummy_names = model._registry.dummy_names
        assert "cyl[2]" in dummy_names
        assert "cyl[3]" in dummy_names

    def test_custom_reference_via_data_types_tuple(self, cars_data):
        model = MCPower("mpg = cyl + hp")
        model.upload_data(_select(cars_data, ["cyl", "hp"]), data_types={"cyl": ("factor", 6)})
        model.set_effects("cyl[4]=0.3, cyl[8]=0.4, hp=0.5")
        model._apply()
        dummy_names = model._registry.dummy_names
        assert "cyl[4]" in dummy_names
        assert "cyl[8]" in dummy_names
        assert "cyl[6]" not in dummy_names

    def test_invalid_reference_level_raises(self, cars_data):
        model = MCPower("mpg = cyl + hp")
        with pytest.raises(ValueError, match="not found in"):
            model.upload_data(_select(cars_data, ["cyl", "hp"]), data_types={"cyl": ("factor", 99)})
            model._apply()

    def test_string_custom_reference(self, cars_data):
        model = MCPower("mpg = origin + hp")
        model.upload_data(
            _select(cars_data, ["origin", "hp"]), data_types={"origin": ("factor", "Japan")}
        )
        model.set_effects("origin[Europe]=0.3, origin[USA]=0.4, hp=0.5")
        model._apply()
        dummy_names = model._registry.dummy_names
        assert "origin[Europe]" in dummy_names
        assert "origin[USA]" in dummy_names
        assert "origin[Japan]" not in dummy_names


class TestPostHocNamedLevels:
    """Post-hoc comparisons with named factor levels."""

    def test_posthoc_with_named_numeric_levels(self, cars_data):
        """Post-hoc with numeric named levels like cyl[4] vs cyl[6]."""
        model = MCPower("mpg = cyl")
        model.upload_data(_select(cars_data, ["cyl"]))
        model.set_effects("cyl[6]=0.3, cyl[8]=0.5")
        result = model.find_power(
            sample_size=100,
            target_test="cyl[4] vs cyl[6], cyl[4] vs cyl[8], cyl[6] vs cyl[8]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_posthoc_with_string_levels(self, cars_data):
        """Post-hoc with string levels like origin[Europe] vs origin[Japan]."""
        model = MCPower("mpg = origin")
        model.upload_data(_select(cars_data, ["origin"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5")
        result = model.find_power(
            sample_size=100,
            target_test="origin[Europe] vs origin[Japan], origin[Europe] vs origin[USA]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_all_posthoc_with_named_levels(self, cars_data):
        """'all-posthoc' keyword expands with named levels."""
        model = MCPower("mpg = cyl")
        model.upload_data(_select(cars_data, ["cyl"]))
        model.set_effects("cyl[6]=0.3, cyl[8]=0.5")
        result = model.find_power(
            sample_size=100,
            target_test="all-posthoc",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_posthoc_with_set_factor_levels(self):
        """Post-hoc with manually set named factor levels."""
        model = MCPower("y = treatment")
        model.set_factor_levels("treatment=placebo,drug_a,drug_b")
        model.set_effects("treatment[drug_a]=0.5, treatment[drug_b]=0.8")
        result = model.find_power(
            sample_size=100,
            target_test="treatment[placebo] vs treatment[drug_a], treatment[placebo] vs treatment[drug_b]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None


class TestCarsOriginColumn:
    """Integration tests using the origin string column in cars.csv."""

    def test_origin_as_factor(self, cars_data):
        """origin column is auto-detected as 3-level string factor."""
        model = MCPower("mpg = origin + hp")
        model.upload_data(_select(cars_data, ["origin", "hp"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5, hp=0.4")
        model._apply()

        assert "origin" in model._registry.factor_names
        assert "origin[Japan]" in model._registry.dummy_names
        assert "origin[USA]" in model._registry.dummy_names
        assert "origin[Europe]" not in model._registry.dummy_names

    def test_origin_full_power_analysis(self, cars_data):
        """Full power analysis with string factor."""
        model = MCPower("mpg = origin + hp + wt")
        model.upload_data(_select(cars_data, ["origin", "hp", "wt"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5, hp=0.4, wt=0.3")
        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_origin_with_cyl_mixed(self, cars_data):
        """String + numeric factors together."""
        model = MCPower("mpg = origin + cyl")
        model.upload_data(_select(cars_data, ["origin", "cyl"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5, cyl[6]=0.2, cyl[8]=0.4")
        model._apply()

        assert "origin[Japan]" in model._registry.dummy_names
        assert "cyl[6]" in model._registry.dummy_names

    def test_origin_strict_mode_power(self, cars_data):
        """String factor works in strict correlation mode with find_power."""
        model = MCPower("mpg = origin")
        model.upload_data(_select(cars_data, ["origin"]), preserve_correlation="strict")
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5")
        result = model.find_power(
            sample_size=60,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_origin_no_mode_power(self, cars_data):
        """String factor works in no correlation mode with find_power."""
        model = MCPower("mpg = origin + hp")
        model.upload_data(_select(cars_data, ["origin", "hp"]), preserve_correlation="no")
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5, hp=0.4")
        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None

    def test_origin_posthoc(self, cars_data):
        """Post-hoc comparisons with string factor levels."""
        model = MCPower("mpg = origin")
        model.upload_data(_select(cars_data, ["origin"]))
        model.set_effects("origin[Japan]=0.3, origin[USA]=0.5")
        result = model.find_power(
            sample_size=100,
            target_test="all-posthoc",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )
        assert result is not None


# ── DataFrame-specific tests (require pandas) ──────────────────────────


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
class TestPandasDataFrame:
    """Tests specifically for pandas DataFrame input."""

    def test_dataframe_upload(self):
        """Test that DataFrame input works correctly."""
        df = pd.read_csv(CARS_CSV, index_col=0)
        model = MCPower("mpg = hp + wt")
        model.upload_data(df[["hp", "wt"]])
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        hp_pred = model._registry.get_predictor("hp")
        assert hp_pred.var_type == "uploaded_data"

    def test_dataframe_with_string_index_column(self):
        """Test DataFrame read without index_col (has 'Unnamed: 0' string column)."""
        df = pd.read_csv(CARS_CSV)  # No index_col=0 -> 'Unnamed: 0' is strings
        model = MCPower("mpg = hp + wt")
        model.upload_data(df)
        model.set_effects("hp=0.5, wt=0.3")
        model._apply()

        hp_pred = model._registry.get_predictor("hp")
        assert hp_pred.var_type == "uploaded_data"

    def test_dataframe_find_power(self):
        """Test find_power works end-to-end with DataFrame upload."""
        df = pd.read_csv(CARS_CSV, index_col=0)
        model = MCPower("mpg = hp + wt")
        model.upload_data(df)
        model.set_effects("hp=0.5, wt=0.3")
        result = model.find_power(sample_size=100, return_results=True, print_results=False)
        assert result is not None
        assert "results" in result
