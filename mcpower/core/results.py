"""
Results processing for MCPower framework.

This module handles power calculation and result formatting from
simulation outputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ResultsProcessor:
    """Converts raw simulation output into power estimates and statistics.

    Computes individual power per test, combined significance
    probabilities (exactly-*k* and at-least-*k*), and aggregates
    sample-size sweep results to find the first sample size that
    achieves the target power.
    """

    def __init__(self, target_power: float = 80.0):
        """Initialise the results processor.

        Args:
            target_power: Target power as a percentage (0–100).
        """
        self.target_power = target_power

    def calculate_powers(
        self,
        all_results: List[np.ndarray],
        all_results_corrected: List[np.ndarray],
        target_tests: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate power estimates from simulation results.

        Args:
            all_results: List of uncorrected significance arrays per simulation
            all_results_corrected: List of corrected significance arrays per simulation
            target_tests: List of test names

        Returns:
            Dictionary with power estimates and probabilities
        """
        n_sims = len(all_results)
        n_tests = len(target_tests)

        # Convert to numpy arrays
        results_array = np.array(all_results, dtype=bool)
        results_corrected_array = np.array(all_results_corrected, dtype=bool)

        # Individual powers
        individual_powers = {}
        individual_powers_corrected = {}
        non_overall_tests = [t for t in target_tests if t != "overall"]

        for test in target_tests:
            if test == "overall":
                # F-test is always at column 0
                individual_powers[test] = np.mean(results_array[:, 0]) * 100
                individual_powers_corrected[test] = np.mean(results_corrected_array[:, 0]) * 100
            else:
                # Find position among non-'overall' tests and add 1 for F-test offset
                pos = non_overall_tests.index(test)
                col_idx = pos + 1  # +1 because column 0 is F-test
                individual_powers[test] = np.mean(results_array[:, col_idx]) * 100
                individual_powers_corrected[test] = np.mean(results_corrected_array[:, col_idx]) * 100

        # Combined probabilities
        combined = {}
        combined_corrected = {}
        cumulative = {}
        cumulative_corrected = {}

        # Count how many tests were significant in each simulation
        sig_counts = np.sum(results_array, axis=1)
        sig_counts_corrected = np.sum(results_corrected_array, axis=1)

        for k in range(n_tests + 1):
            # Exactly k significant
            exactly_k = np.sum(sig_counts == k) / n_sims * 100
            exactly_k_corrected = np.sum(sig_counts_corrected == k) / n_sims * 100

            combined[f"exactly_{k}_significant"] = exactly_k
            combined_corrected[f"exactly_{k}_significant"] = exactly_k_corrected

            # At least k significant
            at_least_k = np.sum(sig_counts >= k) / n_sims * 100
            at_least_k_corrected = np.sum(sig_counts_corrected >= k) / n_sims * 100

            cumulative[f"at_least_{k}_significant"] = at_least_k
            cumulative_corrected[f"at_least_{k}_significant"] = at_least_k_corrected

        return {
            "individual_powers": individual_powers,
            "individual_powers_corrected": individual_powers_corrected,
            "combined_probabilities": combined,
            "combined_probabilities_corrected": combined_corrected,
            "cumulative_probabilities": cumulative,
            "cumulative_probabilities_corrected": cumulative_corrected,
            "n_simulations_used": n_sims,
        }

    def process_sample_size_results(
        self,
        results: List[Tuple[int, Dict]],
        target_tests: List[str],
        correction: Optional[str],
    ) -> Dict[str, Any]:
        """
        Process power results from sample size analysis.

        Args:
            results: List of (sample_size, power_result) tuples
            target_tests: List of test names
            correction: Correction method name

        Returns:
            Dictionary with sample size analysis results
        """
        powers_by_test: Dict[str, List[float]] = {test: [] for test in target_tests}
        powers_by_test_corrected: Dict[str, List[float]] = {test: [] for test in target_tests}
        first_achieved = dict.fromkeys(target_tests, -1)
        first_achieved_corrected = dict.fromkeys(target_tests, -1)

        for sample_size, power_result in results:
            if power_result is None:
                continue

            for test in target_tests:
                power = power_result["results"]["individual_powers"][test]
                power_corrected = power_result["results"]["individual_powers_corrected"][test]

                powers_by_test[test].append(power)
                powers_by_test_corrected[test].append(power_corrected)

                if power >= self.target_power and first_achieved[test] == -1:
                    first_achieved[test] = sample_size

                if power_corrected >= self.target_power and first_achieved_corrected[test] == -1:
                    first_achieved_corrected[test] = sample_size

        return {
            "sample_sizes_tested": [r[0] for r in results],
            "powers_by_test": powers_by_test,
            "powers_by_test_corrected": (powers_by_test_corrected if correction else None),
            "first_achieved": first_achieved,
            "first_achieved_corrected": (first_achieved_corrected if correction else None),
        }


def build_power_result(
    model_type: str,
    target_tests: List[str],
    formula_to_test: Optional[Dict],
    equation: str,
    sample_size: int,
    alpha: float,
    n_simulations: int,
    correction: Optional[str],
    target_power: float,
    parallel: Union[bool, str],
    power_results: Dict,
) -> Dict[str, Any]:
    """
    Build complete power analysis result dictionary.

    Args:
        model_type: Type of statistical model
        target_tests: List of tests performed
        formula_to_test: Optional filtered effects
        equation: Original model equation
        sample_size: Sample size tested
        alpha: Significance level
        n_simulations: Number of simulations
        correction: Correction method
        target_power: Target power level
        parallel: Whether parallel processing was used
        power_results: Results from power calculation

    Returns:
        Complete result dictionary
    """
    return {
        "model": {
            "model_type": model_type,
            "target_tests": target_tests,
            "test_formula": (formula_to_test if formula_to_test is not None else target_tests),
            "data_formula": equation,
            "sample_size": sample_size,
            "alpha": alpha,
            "n_simulations": power_results.get("n_simulations_used", n_simulations),
            "correction": correction,
            "target_power": target_power,
            "parallel": parallel,
        },
        "results": power_results,
    }


def build_sample_size_result(
    model_type: str,
    target_tests: List[str],
    formula_to_test: Optional[Dict],
    equation: str,
    sample_sizes: List[int],
    alpha: float,
    n_simulations: int,
    correction: Optional[str],
    target_power: float,
    parallel: Union[bool, str],
    analysis_results: Dict,
) -> Dict[str, Any]:
    """
    Build complete sample size analysis result dictionary.

    Args:
        model_type: Type of statistical model
        target_tests: List of tests performed
        formula_to_test: Optional filtered effects
        equation: Original model equation
        sample_sizes: Sample sizes tested
        alpha: Significance level
        n_simulations: Number of simulations
        correction: Correction method
        target_power: Target power level
        parallel: Whether parallel processing was used
        analysis_results: Results from sample size analysis

    Returns:
        Complete result dictionary
    """
    return {
        "model": {
            "model_type": model_type,
            "target_tests": target_tests,
            "target_power": target_power,
            "data_formula": equation,
            "test_formula": (formula_to_test if formula_to_test is not None else target_tests),
            "alpha": alpha,
            "n_simulations": n_simulations,
            "correction": correction,
            "parallel": parallel,
            "sample_size_range": {
                "from_size": sample_sizes[0],
                "to_size": sample_sizes[-1],
                "by": sample_sizes[1] - sample_sizes[0] if len(sample_sizes) > 1 else 1,
            },
        },
        "results": analysis_results,
    }
