"""Unit tests for mcpower.core.results — ResultsProcessor and builder functions."""

import numpy as np
import pytest

from mcpower.core.results import ResultsProcessor, _wilson_ci, build_power_result, build_sample_size_result


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

    def test_ci_fields_present(self):
        """calculate_powers returns individual_powers_ci and individual_powers_corrected_ci."""
        proc = ResultsProcessor(target_power=80.0)
        results = [np.array([True, True])] * 6 + [
            np.array([True, False]),
            np.array([True, False]),
            np.array([False, False]),
            np.array([False, False]),
        ]
        out = proc.calculate_powers(results, results, ["overall", "x1"])

        assert "individual_powers_ci" in out
        assert "individual_powers_corrected_ci" in out
        # overall: 8/10 sig → 80%
        ci_overall = out["individual_powers_ci"]["overall"]
        assert len(ci_overall) == 2
        assert ci_overall[0] < 80.0
        assert ci_overall[1] > 80.0
        # x1: 6/10 sig → 60%
        ci_x1 = out["individual_powers_ci"]["x1"]
        assert ci_x1[0] < 60.0
        assert ci_x1[1] > 60.0

    def test_ci_boundary_all_significant(self):
        """CI upper bound is 100 when all simulations are significant."""
        proc = ResultsProcessor()
        results = [np.array([True, True])] * 100
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        ci = out["individual_powers_ci"]["overall"]
        assert ci[1] == pytest.approx(100.0)
        assert ci[0] < 100.0

    def test_ci_boundary_none_significant(self):
        """CI lower bound is 0 when no simulations are significant."""
        proc = ResultsProcessor()
        results = [np.array([False, False])] * 100
        out = proc.calculate_powers(results, results, ["overall", "x1"])
        ci = out["individual_powers_ci"]["overall"]
        assert ci[0] == pytest.approx(0.0)
        assert ci[1] > 0.0


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


class TestProcessSampleSizeResults:
    """Tests for ResultsProcessor.process_sample_size_results."""

    def _make_power_result(self, powers, powers_corrected, powers_ci, powers_corrected_ci):
        """Build a minimal power_result dict for one sample size."""
        return {
            "results": {
                "individual_powers": powers,
                "individual_powers_ci": powers_ci,
                "individual_powers_corrected": powers_corrected,
                "individual_powers_corrected_ci": powers_corrected_ci,
            }
        }

    def test_ci_fields_present(self):
        """process_sample_size_results returns powers_by_test_ci."""
        proc = ResultsProcessor(target_power=80.0)
        results = [
            (50, self._make_power_result(
                {"overall": 40.0, "x1": 30.0},
                {"overall": 35.0, "x1": 25.0},
                {"overall": [30.5, 50.1], "x1": [21.5, 39.8]},
                {"overall": [26.1, 44.8], "x1": [17.1, 34.5]},
            )),
            (100, self._make_power_result(
                {"overall": 85.0, "x1": 70.0},
                {"overall": 80.0, "x1": 65.0},
                {"overall": [79.3, 89.6], "x1": [63.0, 76.3]},
                {"overall": [73.8, 85.3], "x1": [57.9, 71.5]},
            )),
        ]
        out = proc.process_sample_size_results(results, ["overall", "x1"], correction=None)

        assert "powers_by_test_ci" in out
        assert out["powers_by_test_ci"]["overall"] == [
            [30.5, 50.1],
            [79.3, 89.6],
        ]
        assert out["powers_by_test_ci"]["x1"] == [
            [21.5, 39.8],
            [63.0, 76.3],
        ]
        # No correction → corrected CI is None
        assert out["powers_by_test_corrected_ci"] is None

    def test_ci_with_correction(self):
        """Corrected CI fields present when correction is specified."""
        proc = ResultsProcessor(target_power=80.0)
        results = [
            (100, self._make_power_result(
                {"overall": 85.0},
                {"overall": 80.0},
                {"overall": [79.3, 89.6]},
                {"overall": [73.8, 85.3]},
            )),
        ]
        out = proc.process_sample_size_results(results, ["overall"], correction="bonferroni")

        assert out["powers_by_test_corrected_ci"] is not None
        assert out["powers_by_test_corrected_ci"]["overall"] == [[73.8, 85.3]]


class TestWilsonCI:
    """Tests for _wilson_ci — Wilson score 95% confidence interval."""

    def test_midpoint_proportion(self):
        """p=0.5, n=100 gives a symmetric ~10pp-wide CI."""
        lo, hi = _wilson_ci(50, 100)
        assert lo == pytest.approx(40.2, abs=0.2)
        assert hi == pytest.approx(59.8, abs=0.2)

    def test_typical_power(self):
        """p=0.8, n=1600 gives a narrow CI around 80%."""
        lo, hi = _wilson_ci(1280, 1600)
        assert lo == pytest.approx(77.9, abs=0.2)
        assert hi == pytest.approx(82.0, abs=0.2)

    def test_all_significant(self):
        """p=1.0: upper is 100, lower is below 100 (non-degenerate)."""
        lo, hi = _wilson_ci(1600, 1600)
        assert hi == pytest.approx(100.0)
        assert lo < 100.0
        assert lo > 99.0  # should be ~99.76 for n=1600

    def test_none_significant(self):
        """p=0.0: lower is 0, upper is above 0 (non-degenerate)."""
        lo, hi = _wilson_ci(0, 1600)
        assert lo == pytest.approx(0.0)
        assert hi > 0.0
        assert hi < 1.0  # should be ~0.24 for n=1600

    def test_small_n(self):
        """Small n gives wider CI."""
        lo_small, hi_small = _wilson_ci(8, 10)
        lo_large, hi_large = _wilson_ci(800, 1000)
        assert (hi_small - lo_small) > (hi_large - lo_large)

    def test_returns_percentages(self):
        """Output is in percentage units (0-100), not proportions (0-1)."""
        lo, hi = _wilson_ci(50, 100)
        assert lo > 1.0  # would be ~0.40 if proportions
        assert hi < 100.0
