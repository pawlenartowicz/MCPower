"""
Tests for result formatting utilities.
"""

import pytest

from mcpower.utils.formatters import (
    _format_results,
    _get_significance_code,
    _ResultFormatter,
    _TableFormatter,
)


# ---------------------------------------------------------------------------
# TableFormatter
# ---------------------------------------------------------------------------
class TestTableFormatter:
    """Test _TableFormatter utility methods."""

    def setup_method(self):
        self.tf = _TableFormatter()

    def test_create_table_basic(self):
        headers = ["Name", "Value"]
        rows = [["alpha", "0.05"], ["beta", "0.20"]]
        table = self.tf._create_table(headers, rows)
        lines = table.split("\n")
        assert len(lines) == 4  # header + separator + 2 rows
        assert "Name" in lines[0]
        assert "-" in lines[1]
        assert "alpha" in lines[2]

    def test_create_table_custom_col_widths(self):
        headers = ["A", "B"]
        rows = [["x", "y"]]
        table = self.tf._create_table(headers, rows, col_widths=[10, 10])
        header_line = table.split("\n")[0]
        # Each column padded to 10 chars + 1 space separator
        assert len(header_line) == 21

    def test_create_table_auto_col_widths(self):
        headers = ["H"]
        rows = [["longvalue"]]
        table = self.tf._create_table(headers, rows)
        # Auto width should accommodate the longest cell
        assert "longvalue" in table

    def test_format_value_float_small(self):
        result = self.tf._format_value(0.00001)
        assert result == "0.000010"

    def test_format_value_float_normal(self):
        result = self.tf._format_value(3.14159)
        assert result == "3.1416"

    def test_format_value_float_with_spec(self):
        result = self.tf._format_value(3.14159, ".2f")
        assert result == "3.14"

    def test_format_value_non_float(self):
        assert self.tf._format_value("hello") == "hello"
        assert self.tf._format_value(42) == "42"


# ---------------------------------------------------------------------------
# Significance codes
# ---------------------------------------------------------------------------
class TestGetSignificanceCode:
    """Test _get_significance_code function."""

    def test_highly_significant(self):
        assert _get_significance_code(0.0001) == "***"

    def test_very_significant(self):
        assert _get_significance_code(0.005) == "**"

    def test_significant(self):
        assert _get_significance_code(0.03) == "*"

    def test_not_significant(self):
        assert _get_significance_code(0.10) == ""

    def test_boundary_001(self):
        assert _get_significance_code(0.001) == "**"

    def test_boundary_01(self):
        # 0.01 is NOT < 0.01, so it falls to the < 0.05 branch → "*"
        assert _get_significance_code(0.01) == "*"

    def test_boundary_05(self):
        assert _get_significance_code(0.05) == ""


# ---------------------------------------------------------------------------
# Power formatting
# ---------------------------------------------------------------------------
class TestFormatPower:
    """Test power analysis formatting."""

    def setup_method(self):
        self.formatter = _ResultFormatter()

    def _make_power_data(self, *, correction=None, powers=None, powers_corr=None):
        if powers is None:
            powers = {"x1": 85.0, "x2": 72.0}
        data = {
            "model": {
                "sample_size": 100,
                "target_tests": list(powers.keys()),
                "target_power": 80.0,
                "correction": correction,
            },
            "results": {
                "individual_powers": powers,
                "cumulative_probabilities": {"at_least_1": 95.0, "all": 62.0},
            },
        }
        if powers_corr:
            data["results"]["individual_powers_corrected"] = powers_corr
            data["results"]["cumulative_probabilities_corrected"] = {"at_least_1": 90.0, "all": 55.0}
        return data

    def test_short_power_basic(self):
        data = self._make_power_data()
        out = self.formatter._format_short_power(data)
        assert "Power Analysis Results" in out
        assert "N=100" in out
        assert "85.0" in out

    def test_short_power_achieved_count(self):
        data = self._make_power_data(powers={"x1": 85.0, "x2": 72.0})
        out = self.formatter._format_short_power(data)
        assert "1/2 tests achieved" in out

    def test_short_power_with_correction(self):
        data = self._make_power_data(
            correction="bonferroni",
            powers={"x1": 90.0},
            powers_corr={"x1": 80.0},
        )
        out = self.formatter._format_short_power(data)
        assert "bonferroni" in out.lower()

    def test_long_power_no_correction(self):
        data = self._make_power_data()
        out = self.formatter._format_long_power(data)
        assert "Individual Test Powers" in out
        assert "Cumulative Probabilities" in out

    def test_long_power_with_correction(self):
        data = self._make_power_data(
            correction="holm",
            powers={"x1": 90.0},
            powers_corr={"x1": 82.0},
        )
        out = self.formatter._format_long_power(data)
        assert "Corrected" in out

    def test_format_results_dispatches_power(self):
        data = self._make_power_data()
        out = _format_results("power", data, "short")
        assert "Power Analysis" in out


# ---------------------------------------------------------------------------
# Sample-size formatting
# ---------------------------------------------------------------------------
class TestFormatSampleSize:
    """Test sample size formatting."""

    def setup_method(self):
        self.formatter = _ResultFormatter()

    def _make_ss_data(self, *, correction=None, achieved=None, achieved_corr=None):
        if achieved is None:
            achieved = {"x1": 80, "x2": 120}
        data = {
            "model": {
                "target_tests": list(achieved.keys()),
                "correction": correction,
                "sample_size_range": {"to_size": 200},
            },
            "results": {
                "first_achieved": achieved,
                "sample_sizes_tested": [50, 80, 100, 120, 150, 200],
                "powers_by_test": {k: [30.0, 50.0, 65.0, 80.0, 90.0, 95.0] for k in achieved},
            },
        }
        if achieved_corr:
            data["results"]["first_achieved_corrected"] = achieved_corr
            data["results"]["powers_by_test_corrected"] = {k: [20.0, 40.0, 55.0, 70.0, 82.0, 90.0] for k in achieved}
        return data

    def test_short_sample_size_basic(self):
        data = self._make_ss_data()
        out = self.formatter._format_short_sample_size(data)
        assert "Sample Size Requirements" in out
        assert "80" in out
        assert "120" in out

    def test_short_sample_size_not_achieved(self):
        data = self._make_ss_data(achieved={"x1": 0})
        out = self.formatter._format_short_sample_size(data)
        assert ">200" in out

    def test_short_sample_size_with_correction(self):
        data = self._make_ss_data(
            correction="bonferroni",
            achieved={"x1": 80},
            achieved_corr={"x1": 120},
        )
        out = self.formatter._format_short_sample_size(data)
        assert "Uncorrected N" in out
        assert "Corrected N" in out

    def test_long_sample_size_includes_probability_table(self):
        data = self._make_ss_data()
        # Add model fields needed by long format
        data["model"]["target_power"] = 80.0
        out = self.formatter._format_long_sample_size(data)
        assert "Cumulative Significance Probability" in out

    def test_format_results_dispatches_sample_size(self):
        data = self._make_ss_data()
        out = _format_results("sample_size", data, "short")
        assert "Sample Size" in out


# ---------------------------------------------------------------------------
# Scenario power formatting
# ---------------------------------------------------------------------------
class TestFormatScenarioPower:
    """Test scenario power formatting."""

    def setup_method(self):
        self.formatter = _ResultFormatter()

    def _make_scenario_power_data(self, *, correction=None):
        target_tests = ["x1", "x2"]

        def scenario_template(powers, powers_corr=None):
            return {
                "model": {
                    "sample_size": 100,
                    "target_tests": target_tests,
                    "target_power": 80.0,
                    "correction": correction,
                },
                "results": {
                    "individual_powers": dict(zip(target_tests, powers, strict=True)),
                    "individual_powers_corrected": (dict(zip(target_tests, powers_corr, strict=True)) if powers_corr else {}),
                    "cumulative_probabilities": {"at_least_1": 95.0, "all": 70.0},
                },
            }

        data = {
            "scenarios": {
                "optimistic": scenario_template([95.0, 90.0]),
                "realistic": scenario_template([80.0, 70.0]),
                "doomer": scenario_template([50.0, 40.0]),
            }
        }
        return data

    def test_scenario_power_short(self):
        data = self._make_scenario_power_data()
        out = self.formatter._format_scenario_power(data, "short")
        assert "SCENARIO SUMMARY" in out
        assert "Optimistic" in out
        assert "Realistic" in out
        assert "Doomer" in out

    def test_scenario_power_short_uncorrected_table(self):
        data = self._make_scenario_power_data()
        out = self.formatter._format_scenario_power_short(data["scenarios"], ["x1", "x2"], None)
        assert "Uncorrected Power" in out
        assert "95.0" in out

    def test_scenario_power_no_data(self):
        out = self.formatter._format_scenario_power({"scenarios": {}}, "short")
        assert "No scenario data" in out

    def test_scenario_power_long_robustness(self):
        data = self._make_scenario_power_data()
        out = self.formatter._format_scenario_power(data, "long")
        assert "ROBUSTNESS ANALYSIS" in out


# ---------------------------------------------------------------------------
# Scenario sample-size formatting
# ---------------------------------------------------------------------------
class TestFormatScenarioSampleSize:
    """Test scenario sample size formatting."""

    def setup_method(self):
        self.formatter = _ResultFormatter()

    def _make_scenario_ss_data(self, *, correction=None):
        target_tests = ["x1"]

        def scenario(n_required, n_corr=None):
            s = {
                "model": {
                    "target_tests": target_tests,
                    "target_power": 80.0,
                    "correction": correction,
                    "sample_size_range": {"to_size": 200},
                },
                "results": {
                    "first_achieved": {"x1": n_required},
                    "sample_sizes_tested": [50, 100, 150, 200],
                    "powers_by_test": {"x1": [30.0, 60.0, 80.0, 95.0]},
                },
            }
            if n_corr is not None:
                s["results"]["first_achieved_corrected"] = {"x1": n_corr}
            return s

        return {
            "scenarios": {
                "optimistic": scenario(80),
                "realistic": scenario(120),
                "doomer": scenario(0),
            }
        }

    def test_scenario_ss_short_uncorrected(self):
        data = self._make_scenario_ss_data()
        out = self.formatter._format_scenario_sample_size(data, "short")
        assert "SCENARIO SUMMARY" in out
        assert "80" in out
        assert ">200" in out  # doomer not achieved

    def test_scenario_ss_short_with_correction(self):
        data = self._make_scenario_ss_data(correction="bonferroni")
        # Add corrected results
        for s in data["scenarios"].values():
            s["results"]["first_achieved_corrected"] = {"x1": 150}
        out = self.formatter._format_scenario_sample_size(data, "short")
        assert "(U)" in out or "Uncorrected" in out or "Opt(U)" in out

    def test_scenario_ss_long_recommendations(self):
        data = self._make_scenario_ss_data()
        out = self.formatter._format_scenario_sample_size(data, "long")
        assert "RECOMMENDATIONS" in out


# ---------------------------------------------------------------------------
# Regression formatting
# ---------------------------------------------------------------------------
class TestFormatRegression:
    """Test regression output formatting."""

    def setup_method(self):
        self.formatter = _ResultFormatter()

    def _make_regression_data(self, *, correction=None, inf_se=False):
        effect_names = ["x1", "x2"]
        se_x1 = float("inf") if inf_se else 0.123
        data = {
            "effect_names": effect_names,
            "dep_var": "y",
            "correction": correction,
            "results": {
                "overall_significant": True,
                "statistics": {
                    "r_squared": 0.35,
                    "f_statistic": 12.5,
                    "f_p_value": 0.0001,
                    "coefficients": {"x1": 0.5, "x2": 0.3},
                    "standard_errors": {"x1": se_x1, "x2": 0.098},
                    "t_statistics": {"x1": 0.0 if inf_se else 4.06, "x2": 3.06},
                    "p_values": {"x1": 1.0 if inf_se else 0.0001, "x2": 0.003},
                },
            },
        }
        if correction:
            data["results"]["x1_significant_corrected"] = True
            data["results"]["x2_significant_corrected"] = False
        return data

    def test_regression_basic(self):
        data = self._make_regression_data()
        out = self.formatter._format_regression(data, "short")
        assert "REGRESSION RESULTS" in out
        assert "R-squared" in out
        assert "Significance codes" in out

    def test_regression_with_correction(self):
        data = self._make_regression_data(correction="bonferroni")
        out = self.formatter._format_regression(data, "short")
        assert "Corrected" in out
        assert "bonferroni" in out.lower()

    def test_regression_inf_std_error(self):
        data = self._make_regression_data(inf_se=True)
        out = self.formatter._format_regression(data, "short")
        assert "inf" in out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class TestFormatResultsDispatcher:
    """Test the top-level _format_results dispatcher."""

    def test_unknown_result_type_raises(self):
        with pytest.raises(ValueError, match="Unknown result type"):
            _format_results("nonexistent", {})
