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
        # LME-specific keys (only consumed when cluster_specs present)
        "random_effect_dist": "heavy_tailed",
        "random_effect_df": 5,
        "icc_noise_sd": 0.15,
        "residual_dist": "heavy_tailed",
        "residual_change_prob": 0.3,
        "residual_df": 10,
    },
    "doomer": {
        "heterogeneity": 0.4,
        "heteroskedasticity": 0.2,
        "correlation_noise_sd": 0.4,
        "distribution_change_prob": 0.6,
        "new_distributions": ["right_skewed", "left_skewed", "uniform"],
        # LME-specific keys (only consumed when cluster_specs present)
        "random_effect_dist": "heavy_tailed",
        "random_effect_df": 3,
        "icc_noise_sd": 0.30,
        "residual_dist": "heavy_tailed",
        "residual_change_prob": 0.8,
        "residual_df": 5,
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


def apply_lme_perturbations(
    cluster_specs: Dict,
    scenario_config: Dict,
    sim_seed: Optional[int],
) -> Optional[Dict]:
    """Compute per-simulation LME perturbations.

    Returns a dict consumed by ``_generate_cluster_effects`` /
    ``_generate_random_effects``, or ``None`` when no LME keys are present.

    The returned dict contains:

    - ``tau_squared_multipliers``: ``{grouping_var: float}`` — multiplicative
      jitter on τ² via ``exp(N(0, icc_noise_sd))``.
    - ``random_effect_dist`` / ``random_effect_df``: passthrough from config.

    Args:
        cluster_specs: Dict of cluster specifications.
        scenario_config: Scenario parameters with LME keys.
        sim_seed: Random seed for reproducibility.

    Returns:
        Perturbation dict or ``None`` if no LME keys are active.
    """
    if not cluster_specs:
        return None

    icc_noise_sd = scenario_config.get("icc_noise_sd", 0.0)
    re_dist = scenario_config.get("random_effect_dist", "normal")
    re_df = scenario_config.get("random_effect_df", 5)

    # If all LME perturbations are effectively off, return None
    if icc_noise_sd == 0.0 and re_dist == "normal":
        return None

    rng = np.random.RandomState(sim_seed + 5000 if sim_seed is not None else None)

    # ICC jitter: multiplicative noise on tau_squared per grouping variable
    tau_squared_multipliers: Dict[str, float] = {}
    if icc_noise_sd > 0:
        for gv in cluster_specs:
            tau_squared_multipliers[gv] = float(np.exp(rng.normal(0, icc_noise_sd)))
    else:
        for gv in cluster_specs:
            tau_squared_multipliers[gv] = 1.0

    return {
        "tau_squared_multipliers": tau_squared_multipliers,
        "random_effect_dist": re_dist,
        "random_effect_df": re_df,
    }


def apply_lme_residual_perturbations(
    y: np.ndarray,
    scenario_config: Dict,
    sim_seed: Optional[int],
) -> np.ndarray:
    """Replace normal residuals with non-normal if coin flip succeeds.

    For each simulation, independently flips a coin (probability
    ``residual_change_prob``) to decide whether residuals are replaced.
    If activated, reproduces the original N(0,1) errors via the known
    seed, generates replacements from t(df) or shifted χ², and applies
    the correction ``y += (new_error - original_error)``.

    Args:
        y: Dependent variable array (modified in-place).
        scenario_config: Scenario parameters with residual keys.
        sim_seed: Random seed for reproducibility.

    Returns:
        The (possibly modified) dependent variable array.
    """
    residual_dist = scenario_config.get("residual_dist", "normal")
    residual_change_prob = scenario_config.get("residual_change_prob", 0.0)
    residual_df = scenario_config.get("residual_df", 10)

    if residual_dist == "normal" or residual_change_prob <= 0.0:
        return y

    rng = np.random.RandomState(sim_seed + 6000 if sim_seed is not None else None)

    # Coin flip: should this simulation have non-normal residuals?
    if rng.random() > residual_change_prob:
        return y

    n = len(y)

    # Reproduce the original N(0,1) errors using the same seed as generate_y
    # generate_y uses sim_seed + 2 for error generation
    original_rng = np.random.RandomState(sim_seed + 2 if sim_seed is not None else None)
    original_errors = original_rng.standard_normal(n)

    # Generate replacement errors
    replacement_rng = np.random.RandomState(sim_seed + 6001 if sim_seed is not None else None)

    if residual_dist == "heavy_tailed":
        # t(df) scaled to have variance 1
        df = max(residual_df, 3)
        raw = replacement_rng.standard_t(df, size=n)
        # t(df) has variance df/(df-2), scale to unit variance
        scale = 1.0 / np.sqrt(df / (df - 2))
        new_errors = raw * scale
    elif residual_dist == "skewed":
        # Shifted chi-squared: mean=0, variance=1
        df = max(residual_df, 3)
        raw = replacement_rng.chisquare(df, size=n)
        new_errors = (raw - df) / np.sqrt(2 * df)
    else:
        return y

    # Apply correction: swap out original errors for new ones
    y = y + (new_errors - original_errors)
    return y


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

    rng = np.random.RandomState(sim_seed)

    # Perturb correlation matrix
    perturbed_corr = correlation_matrix
    if correlation_matrix is not None and scenario_config["correlation_noise_sd"] > 0:
        perturbed_corr = correlation_matrix.copy()
        noise = rng.normal(0, scenario_config["correlation_noise_sd"], correlation_matrix.shape)
        noise = (noise + noise.T) / 2  # Keep symmetric
        perturbed_corr += noise
        perturbed_corr = np.clip(perturbed_corr, -0.8, 0.8)
        np.fill_diagonal(perturbed_corr, 1.0)

        # Ensure positive semi-definiteness via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(perturbed_corr)
        if np.any(eigvals < 0):
            eigvals = np.maximum(eigvals, 0.0)
            perturbed_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Re-normalize to unit diagonal
            d = np.sqrt(np.diag(perturbed_corr))
            perturbed_corr = perturbed_corr / np.outer(d, d)
            np.fill_diagonal(perturbed_corr, 1.0)

    # Perturb variable types
    perturbed_var_types = var_types.copy()
    if scenario_config["distribution_change_prob"] > 0:
        type_mapping = {"right_skewed": 2, "left_skewed": 3, "uniform": 5}
        new_type_codes = [type_mapping[distribution] for distribution in scenario_config["new_distributions"]]

        for i in range(len(var_types)):
            if var_types[i] == 0 and rng.random() < scenario_config["distribution_change_prob"]:
                perturbed_var_types[i] = rng.choice(new_type_codes)

    return perturbed_corr, perturbed_var_types
