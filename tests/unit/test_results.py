"""Unit tests for mcpower.core.results — ResultsProcessor and builder functions."""

import numpy as np
import pytest

from mcpower.core.results import ResultsProcessor, build_power_result, build_sample_size_result


class TestCalculatePowers:
    """Tests for ResultsProcessor.calculate_powers."""

    def test_basic_two_tests(self):
        """Power calculation with two tests (overall + one predictor)."""
        proc = ResultsProcessor(target_power=80.0)
        # 10 simulations, 2 columns: [overall, x1]
        # overall: 8/10 sig, x1: 6/10 sig
        results = [np.array([True, True])] * 6 + [
            np.array([True, False]),
            np.array([True, False]),
            np.array([False, False]),
            np.array([False, False]),
        ]
        corrected = results  # same for this test

        out = proc.calculate_powers(results, corrected, ["overall", "x1"])

        assert out["individual_powers"]["overall"] == pytest.approx(80.0)
        assert out["individual_powers"]["x1"] == pytest.approx(60.0)
        assert out["n_simulations_used"] == 10

    def test_all_significant(self):
        proc = ResultsProcessor()
        results = [np.array([True, True])] * 5
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        assert out["individual_powers"]["overall"] == pytest.approx(100.0)
        assert out["individual_powers"]["x1"] == pytest.approx(100.0)

    def test_none_significant(self):
        proc = ResultsProcessor()
        results = [np.array([False, False])] * 5
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        assert out["individual_powers"]["overall"] == pytest.approx(0.0)
        assert out["individual_powers"]["x1"] == pytest.approx(0.0)

    def test_combined_probabilities(self):
        proc = ResultsProcessor()
        # 4 sims, 2 tests: exactly 0, 1, 2 significant
        results = [
            np.array([False, False]),  # 0 sig
            np.array([True, False]),   # 1 sig
            np.array([False, True]),   # 1 sig
            np.array([True, True]),    # 2 sig
        ]
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        combined = out["combined_probabilities"]
        assert combined["exactly_0_significant"] == pytest.approx(25.0)
        assert combined["exactly_1_significant"] == pytest.approx(50.0)
        assert combined["exactly_2_significant"] == pytest.approx(25.0)

    def test_cumulative_probabilities(self):
        proc = ResultsProcessor()
        results = [
            np.array([False, False]),
            np.array([True, True]),
            np.array([True, True]),
            np.array([True, True]),
        ]
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        cumulative = out["cumulative_probabilities"]
        assert cumulative["at_least_0_significant"] == pytest.approx(100.0)
        assert cumulative["at_least_2_significant"] == pytest.approx(75.0)


class TestBuildPowerResult:
    """Tests for build_power_result."""

    def test_basic_structure(self):
        power_results = {
            "individual_powers": {"overall": 80.0},
            "n_simulations_used": 1000,
        }
        result = build_power_result(
            model_type="OLS",
            target_tests=["overall"],
            formula_to_test=None,
            equation="y = x1",
            sample_size=100,
            alpha=0.05,
            n_simulations=1000,
            correction=None,
            target_power=80.0,
            parallel=False,
            power_results=power_results,
        )
        assert result["model"]["model_type"] == "OLS"
        assert result["model"]["sample_size"] == 100
        assert result["model"]["alpha"] == 0.05
        assert result["results"] is power_results


class TestBuildSampleSizeResult:
    """Tests for build_sample_size_result."""

    def test_basic_structure(self):
        analysis_results = {"sample_sizes_tested": [50, 100]}
        result = build_sample_size_result(
            model_type="OLS",
            target_tests=["overall"],
            formula_to_test=None,
            equation="y = x1",
            sample_sizes=[50, 100],
            alpha=0.05,
            n_simulations=1000,
            correction=None,
            target_power=80.0,
            parallel=False,
            analysis_results=analysis_results,
        )
        assert result["model"]["sample_size_range"]["from_size"] == 50
        assert result["model"]["sample_size_range"]["to_size"] == 100
        assert result["model"]["sample_size_range"]["by"] == 50
        assert result["results"] is analysis_results

    def test_single_sample_size(self):
        result = build_sample_size_result(
            model_type="OLS",
            target_tests=["overall"],
            formula_to_test=None,
            equation="y = x1",
            sample_sizes=[100],
            alpha=0.05,
            n_simulations=1000,
            correction=None,
            target_power=80.0,
            parallel=False,
            analysis_results={},
        )
        assert result["model"]["sample_size_range"]["by"] == 1
