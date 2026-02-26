"""Tests for formatter edge cases — scenario sample-size long format, cumulative recs, NaN filtering."""

import math

import pytest

from mcpower.utils.formatters import _ResultFormatter, _is_nan


_fmt = _ResultFormatter()


def _make_scenario_sample_size_data(
    target_tests=("x1", "x2"),
    correction=None,
    sample_sizes=(50, 100, 150),
    optimistic_achieved=None,
    realistic_achieved=None,
    doomer_achieved=None,
):
    """Build a scenario sample_size result dict for formatting tests."""
    if optimistic_achieved is None:
        optimistic_achieved = {"x1": 50, "x2": 100}
    if realistic_achieved is None:
        realistic_achieved = {"x1": 100, "x2": 150}
    if doomer_achieved is None:
        doomer_achieved = {"x1": 0, "x2": 0}  # Not achieved

    def _make_scenario(achieved):
        achieved_corr = {t: -1 for t in target_tests} if not correction else achieved
        return {
            "model": {
                "target_tests": list(target_tests),
                "correction": correction,
                "sample_size_range": {"from_size": sample_sizes[0], "to_size": sample_sizes[-1]},
                "target_power": 80.0,
            },
            "results": {
                "first_achieved": achieved,
                "first_achieved_corrected": achieved_corr,
                "sample_sizes_tested": list(sample_sizes),
                "powers_by_test": {
                    t: [30.0 + 25.0 * i for i in range(len(sample_sizes))]
                    for t in target_tests
                },
                "powers_by_test_corrected": (
                    {t: [25.0 + 25.0 * i for i in range(len(sample_sizes))] for t in target_tests}
                    if correction
                    else None
                ),
            },
        }

    return {
        "analysis_type": "sample_size",
        "scenarios": {
            "optimistic": _make_scenario(optimistic_achieved),
            "realistic": _make_scenario(realistic_achieved),
            "doomer": _make_scenario(doomer_achieved),
        },
        "comparison": {},
    }


class TestScenarioSampleSizeLongFormat:
    """Test _format_scenario_sample_size with summary='long'."""

    def test_recommendations_present(self):
        data = _make_scenario_sample_size_data()
        output = _fmt.format("scenario_sample_size", data, "long")
        assert "RECOMMENDATIONS" in output

    def test_unachievable_tests_warning(self):
        data = _make_scenario_sample_size_data(
            doomer_achieved={"x1": 0, "x2": 0},
        )
        output = _fmt.format("scenario_sample_size", data, "long")
        assert "Warning" in output or "may not achieve" in output

    def test_realistic_recommendation_shown(self):
        data = _make_scenario_sample_size_data(
            realistic_achieved={"x1": 100, "x2": 150},
        )
        output = _fmt.format("scenario_sample_size", data, "long")
        assert "150" in output  # max N for realistic

    def test_short_format_produces_table(self):
        data = _make_scenario_sample_size_data()
        output = _fmt.format("scenario_sample_size", data, "short")
        assert "SCENARIO SUMMARY" in output

    def test_with_correction(self):
        data = _make_scenario_sample_size_data(correction="bonferroni")
        output = _fmt.format("scenario_sample_size", data, "short")
        assert "Opt(U)" in output or "Uncorrected" in output.lower() or "(U)" in output


class TestCumulativeRecommendations:
    """Test _format_cumulative_recommendations paths."""

    def test_non_scenario_target_met(self):
        data = {
            "model": {
                "target_tests": ["x1", "x2"],
                "target_power": 80.0,
            },
            "results": {
                "sample_sizes_tested": [50, 100, 150],
                "powers_by_test": {
                    "x1": [60.0, 85.0, 95.0],
                    "x2": [70.0, 90.0, 98.0],
                },
            },
        }
        lines = _fmt._format_cumulative_recommendations(data, is_scenario=False)
        joined = "\n".join(lines)
        assert "N=" in joined  # Found a sample size

    def test_non_scenario_target_not_met(self):
        data = {
            "model": {
                "target_tests": ["x1", "x2"],
                "target_power": 80.0,
            },
            "results": {
                "sample_sizes_tested": [50, 100],
                "powers_by_test": {
                    "x1": [10.0, 20.0],
                    "x2": [15.0, 25.0],
                },
            },
        }
        lines = _fmt._format_cumulative_recommendations(data, is_scenario=False)
        joined = "\n".join(lines)
        assert ">100" in joined  # Exceeded max tested

    def test_scenario_recommendations(self):
        data = _make_scenario_sample_size_data(
            sample_sizes=(50, 100, 150, 200),
            optimistic_achieved={"x1": 100, "x2": 150},
        )
        # Override powers so all > 80%
        for scenario in data["scenarios"].values():
            scenario["results"]["powers_by_test"] = {
                "x1": [50.0, 85.0, 92.0, 98.0],
                "x2": [40.0, 75.0, 88.0, 95.0],
            }
        lines = _fmt._format_cumulative_recommendations(data, is_scenario=True)
        assert len(lines) > 0

    def test_empty_scenarios(self):
        data = {"scenarios": {}}
        lines = _fmt._format_cumulative_recommendations(data, is_scenario=True)
        assert lines == []

    def test_no_results_key(self):
        data = {}
        lines = _fmt._format_cumulative_recommendations(data, is_scenario=False)
        assert lines == []


class TestNaNPowerFiltering:
    """NaN power values in cumulative table should be filtered out."""

    def test_nan_power_filtered_in_cumulative_sample_size_table(self):
        lines = []
        _fmt._add_cumulative_sample_size_table(
            lines,
            sample_sizes=[50, 100],
            target_tests=["x1", "x2_nan"],
            powers_by_test={
                "x1": [50.0, 80.0],
                "x2_nan": [float("nan"), float("nan")],
            },
        )
        # Should still produce output for x1 (x2_nan filtered out)
        output = "\n".join(lines)
        assert "N=50" in output or "50" in output

    def test_all_nan_produces_no_table(self):
        lines = []
        _fmt._add_cumulative_sample_size_table(
            lines,
            sample_sizes=[50],
            target_tests=["x1"],
            powers_by_test={"x1": [float("nan")]},
        )
        # All NaN → no valid tests → no table
        assert len(lines) == 0


class TestIsNan:
    """Test _is_nan utility."""

    def test_nan_float(self):
        assert _is_nan(float("nan"))

    def test_regular_float(self):
        assert not _is_nan(42.0)

    def test_non_float(self):
        assert not _is_nan("nan")
        assert not _is_nan(None)
        assert not _is_nan(42)


class TestExtractScenarioMeta:
    """Test _extract_scenario_meta."""

    def test_no_model_returns_none(self):
        target_tests, correction = _fmt._extract_scenario_meta({"opt": {"results": {}}})
        assert target_tests is None

    def test_extracts_from_first_scenario(self):
        scenarios = {
            "optimistic": {
                "model": {"target_tests": ["a", "b"], "correction": "holm"},
            }
        }
        target_tests, correction = _fmt._extract_scenario_meta(scenarios)
        assert target_tests == ["a", "b"]
        assert correction == "holm"


class TestFormatUnknownType:
    """Unknown result type should raise."""

    def test_unknown_result_type(self):
        with pytest.raises(ValueError, match="Unknown result type"):
            _fmt.format("nonexistent", {})
