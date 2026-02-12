"""
Scenario analysis for MCPower framework.

This module provides scenario-based robustness analysis, allowing users
to test power under different assumption violations (optimistic, realistic, doomer).
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils.formatters import _format_results
from ..utils.visualization import _create_power_plot

# Default scenario configurations.
# "realistic" introduces moderate assumption violations; "doomer" introduces
# severe violations. Each simulation iteration draws random perturbations
# from these parameters (correlation noise, distribution swaps, etc.).
DEFAULT_SCENARIO_CONFIG = {
    "realistic": {
        "heterogeneity": 0.2,
        "heteroskedasticity": 0.1,
        "correlation_noise_sd": 0.2,
        "distribution_change_prob": 0.3,
        "new_distributions": ["right_skewed", "left_skewed", "uniform"],
    },
    "doomer": {
        "heterogeneity": 0.4,
        "heteroskedasticity": 0.2,
        "correlation_noise_sd": 0.4,
        "distribution_change_prob": 0.6,
        "new_distributions": ["right_skewed", "left_skewed", "uniform"],
    },
}


class ScenarioRunner:
    """Executes scenario-based robustness analysis.

    Runs the power (or sample-size) analysis under multiple conditions:

    - **Optimistic** — the user's original settings (no perturbations).
    - **Realistic** — moderate assumption violations.
    - **Doomer** — severe assumption violations.

    Custom scenarios can be supplied via ``set_scenario_configs`` on the
    ``MCPower`` model.
    """

    def __init__(self, model, custom_configs: Optional[Dict] = None):
        """Initialise the scenario runner.

        Args:
            model: ``MCPower`` instance (used for target power and config).
            custom_configs: Scenario configuration dict. Defaults to
                ``DEFAULT_SCENARIO_CONFIG``.
        """
        self.model = model
        self.configs = custom_configs if custom_configs is not None else DEFAULT_SCENARIO_CONFIG

    def get_configs(self) -> Dict:
        """Get current scenario configurations."""
        return self.configs

    def run_power_analysis(
        self,
        sample_size: int,
        target_tests: List[str],
        correction: Optional[str],
        run_find_power_func: Callable,
        summary: str = "short",
        print_results: bool = True,
        progress=None,
    ) -> Dict[str, Any]:
        """
        Run scenario-based power analysis.

        Args:
            sample_size: Sample size to test
            target_tests: List of effects to test
            correction: Multiple comparison correction
            run_find_power_func: Function to run power analysis
            summary: Output detail level
            print_results: Whether to print results
            progress: Optional ProgressReporter to start after header

        Returns:
            Dictionary with scenario comparison results
        """
        results = {}

        if print_results:
            print(f"\n{'=' * 80}")
            print("SCENARIO-BASED MONTE CARLO POWER ANALYSIS RESULTS")
            print(f"{'=' * 80}")

        if progress is not None:
            progress.start()

        # Optimistic (user's original settings)
        results["optimistic"] = run_find_power_func(
            sample_size=sample_size,
            target_tests=target_tests,
            correction=correction,
            scenario_config=None,
        )

        # Realistic & Doomer scenarios
        for scenario_name, config in self.configs.items():
            results[scenario_name] = run_find_power_func(
                sample_size=sample_size,
                target_tests=target_tests,
                correction=correction,
                scenario_config=config,
            )

        # Format results
        formatted_results = {
            "analysis_type": "power",
            "scenarios": results,
            "comparison": {},
        }

        if print_results:
            print(_format_results("scenario_power", formatted_results, summary))

        return formatted_results

    def run_sample_size_analysis(
        self,
        sample_sizes: List[int],
        target_tests: List[str],
        correction: Optional[str],
        run_sample_size_func: Callable,
        summary: str = "short",
        print_results: bool = True,
        progress=None,
    ) -> Dict[str, Any]:
        """
        Run scenario-based sample size analysis.

        Args:
            sample_sizes: List of sample sizes to test
            target_tests: List of effects to test
            correction: Multiple comparison correction
            run_sample_size_func: Function to run sample size analysis
            summary: Output detail level
            print_results: Whether to print results
            progress: Optional ProgressReporter to start after header

        Returns:
            Dictionary with scenario comparison results
        """
        results = {}

        if print_results:
            print(f"\n{'=' * 80}")
            print("SCENARIO-BASED MONTE CARLO POWER ANALYSIS RESULTS")
            print(f"{'=' * 80}")

        if progress is not None:
            progress.start()

        # Optimistic
        results["optimistic"] = run_sample_size_func(
            sample_sizes=sample_sizes,
            target_tests=target_tests,
            correction=correction,
            scenario_config=None,
        )

        # Other scenarios
        for scenario_name, config in self.configs.items():
            results[scenario_name] = run_sample_size_func(
                sample_sizes=sample_sizes,
                target_tests=target_tests,
                correction=correction,
                scenario_config=config,
            )

        formatted_results = {
            "analysis_type": "sample_size",
            "scenarios": results,
            "comparison": {},
        }

        if print_results:
            print(_format_results("scenario_sample_size", formatted_results, summary))

            if summary == "long":
                self._create_scenario_plots(formatted_results)

        return formatted_results

    def _create_scenario_plots(self, results: Dict) -> None:
        """Create visualizations for scenario analysis."""
        scenarios = results["scenarios"]
        scenario_names = ["optimistic", "realistic", "doomer"]
        scenario_labels = ["Optimistic", "Realistic", "Doomer"]

        first_scenario = scenarios.get("optimistic", {})
        if "results" not in first_scenario or "sample_sizes_tested" not in first_scenario["results"]:
            return

        sample_sizes = first_scenario["results"]["sample_sizes_tested"]
        target_tests = first_scenario["model"]["target_tests"]
        correction = first_scenario["model"].get("correction")
        target_power = self.model.power

        # Uncorrected plots
        for i, scenario in enumerate(scenario_names):
            if scenario in scenarios:
                powers_by_test = scenarios[scenario]["results"]["powers_by_test"]

                _create_power_plot(
                    sample_sizes=sample_sizes,
                    powers_by_test=powers_by_test,
                    first_achieved=scenarios[scenario]["results"]["first_achieved"],
                    target_tests=target_tests,
                    target_power=target_power,
                    title=f"{scenario_labels[i]} Scenario - Uncorrected Power",
                )

        # Corrected plots
        if correction:
            for i, scenario in enumerate(scenario_names):
                if scenario in scenarios and scenarios[scenario]["results"].get("powers_by_test_corrected"):
                    powers_by_test_corr = scenarios[scenario]["results"]["powers_by_test_corrected"]

                    _create_power_plot(
                        sample_sizes=sample_sizes,
                        powers_by_test=powers_by_test_corr,
                        first_achieved=scenarios[scenario]["results"]["first_achieved_corrected"],
                        target_tests=target_tests,
                        target_power=target_power,
                        title=f"{scenario_labels[i]} Scenario - Corrected Power ({correction})",
                    )


def apply_per_simulation_perturbations(
    correlation_matrix: np.ndarray,
    var_types: np.ndarray,
    scenario_config: Dict,
    sim_seed: Optional[int],
) -> tuple:
    """Apply random perturbations to correlations and distributions.

    Called once per simulation iteration in scenario mode. Adds symmetric
    Gaussian noise to the correlation matrix (clipped to [-0.8, 0.8]) and
    randomly swaps normal distributions for skewed/uniform alternatives.

    Args:
        correlation_matrix: Original correlation matrix (may be ``None``).
        var_types: Integer-coded distribution types.
        scenario_config: Scenario parameters (``correlation_noise_sd``,
            ``distribution_change_prob``, ``new_distributions``).
        sim_seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(perturbed_correlation_matrix, perturbed_var_types)``.
    """
    if scenario_config is None:
        return correlation_matrix, var_types

    np.random.seed(sim_seed)

    # Perturb correlation matrix
    perturbed_corr = correlation_matrix
    if correlation_matrix is not None and scenario_config["correlation_noise_sd"] > 0:
        perturbed_corr = correlation_matrix.copy()
        noise = np.random.normal(0, scenario_config["correlation_noise_sd"], correlation_matrix.shape)
        noise = (noise + noise.T) / 2  # Keep symmetric
        perturbed_corr += noise
        perturbed_corr = np.clip(perturbed_corr, -0.8, 0.8)
        np.fill_diagonal(perturbed_corr, 1.0)

    # Perturb variable types
    perturbed_var_types = var_types.copy()
    if scenario_config["distribution_change_prob"] > 0:
        type_mapping = {"right_skewed": 2, "left_skewed": 3, "uniform": 5}
        new_type_codes = [type_mapping[distribution] for distribution in scenario_config["new_distributions"]]

        for i in range(len(var_types)):
            if var_types[i] == 0 and np.random.random() < scenario_config["distribution_change_prob"]:
                perturbed_var_types[i] = np.random.choice(new_type_codes)

    return perturbed_corr, perturbed_var_types
