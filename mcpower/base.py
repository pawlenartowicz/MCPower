"""
Base framework for Monte Carlo power analysis across statistical models.

This module provides the abstract base class MCPowerBase that defines the common
interface and functionality for all statistical model implementations.
Concrete models (LinearRegression, LogisticRegression, etc.) inherit from this
base to implement model-specific statistical tests while sharing common features
like data generation, scenario analysis, and result formatting.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .utils.parsers import (
    _parser,
    _parse_equation,
    _parse_independent_variables,
)
from .utils.validators import (
    _validate_power,
    _validate_alpha,
    _validate_simulations,
    _validate_parallel_settings,
    _validate_correlation_matrix,
    _validate_sample_size_range,
    _validate_correction_method,
    _validate_model_ready,
    _validate_test_formula,
    _validate_variable_type_timing,
)
from .utils.formatters import _format_results
from .utils.visualization import _create_power_plot
from .utils.data_generation import _generate_X, _generate_factors

DUMMY_MATRIX = np.zeros((2, 2), dtype=np.float64)


class MCPowerBase(ABC):
    """
    Base class for Monte Carlo power analysis across different statistical models.

    Provides common functionality for data generation, scenario analysis,
    and power/sample size calculations. Subclasses implement specific
    statistical tests.
    """

    def __init__(self, equation: str):
        """
        Initialize Monte Carlo Power Analysis with R-style equation.

        Args:
            equation: R-style formula string (e.g., 'y = x1 + x2 + x1:x2' or 'y ~ x1 * x2')

        Raises:
            ValueError: If equation is empty or contains no predictor variables

        Example:
            >>> model = LinearRegression("satisfaction = treatment + age + treatment:age")
        """

        # Model specification
        self.seed = 2137
        self.power = 80.0
        self.alpha = 0.05
        self.n_simulations = 1600
        self.heterogeneity = 0.0
        self.heteroskedasticity = 0.0

        # Parallel processing settings
        self.parallel = False
        self.n_cores = 1

        # State flags
        self.correlations_set = False
        self.effects_set = False
        self.variable_types_set = False
        self.data_uploaded = False

        # Model equation and variable definitions
        self.equation = equation.strip()
        self.variables = {}
        self.effects = {}
        self.correlations = {}
        self.correlation_matrix = None
        self.predictor_vars_order = (
            []
        )  # This will be the actual data columns after factor expansion
        self.factor_variables = (
            {}
        )  # {factor_name: {n_levels, proportions, reference_level}}
        self.factor_dummy_mapping = {}  # {dummy_var_name: {factor_name, level}}

        # Scenario configurations
        self.default_scenario_config = {
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
        self.custom_scenario_configs = None

        # Uploaded data
        self.uploaded_data = {}
        self.upload_normal_values = np.zeros((2, 2), dtype=np.float64)  # Dummy matrix
        self.upload_data_values = np.zeros((2, 2), dtype=np.float64)  # Dummy matrix

        # Parse equation
        dep_var, formula_part = _parse_equation(self.equation)
        self.variables["variable_0"] = {"name": dep_var}

        # Parse variables and effects
        variables, effects = _parse_independent_variables(formula_part)
        self.variables.update(variables)
        self.effects.update(effects)

        # Validate equation
        if not formula_part.strip():
            raise ValueError("Equation cannot be empty. Expected format: 'y = x1 + x2'")

        if not self.effects:
            raise ValueError("No predictor variables found in equation")

        # Set initial predictor order (before factor expansion)
        self.predictor_vars_order = [
            info["name"] for key, info in self.variables.items() if key != "variable_0"
        ]

        # Print summary
        if self.predictor_vars_order:
            print(
                f"Variables: {dep_var} (dependent), {', '.join(self.predictor_vars_order)} (predictors)"
            )
            print(f"Found {len(self.predictor_vars_order)} predictor variables")
            if len(self.predictor_vars_order) == 1:
                print("Single predictor - no correlation matrix needed")

    # =====================================
    # Configuration
    # =====================================

    def set_parallel(self, enable=True, n_cores=None):
        """
        Enable or disable parallel processing for simulations.

        Args:
            enable: Whether to enable parallel processing (default: True)
            n_cores: Number of cores to use (default: CPU count - 1)

        Returns:
            self: For method chaining

        Note:
            Requires joblib package. Falls back to sequential processing if unavailable.
        """

        if enable:
            try:
                from joblib import Parallel, delayed
                import multiprocessing as mp
            except:
                print("Warning: joblib not available. Install with: pip install joblib")
                print("Warning: Continuing with sequential processing.")
                self.parallel = False
                return self

            settings, result = _validate_parallel_settings(enable, n_cores)
            result.raise_if_invalid()

            self.parallel, self.n_cores = settings

            return self
        else:
            self.parallel, self.n_cores = False, 1
            return self

    def set_seed(self, seed=None):
        """
        Set default random seed for reproducible results.

        Args:
            seed: Random seed (integer) or None for random behavior

        Returns:
            self: For method chaining

        Raises:
            TypeError: If seed is not integer or None
            ValueError: If seed is negative
        """
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")
            if seed > 3000000000:
                raise ValueError(
                    "seed must be lower than 3 000 000 000, do not worry, there's plenty of room at the bottom"
                )

        self.seed = seed

        if seed is not None:
            print(f"Seed set to: {seed}")
        else:
            print("Random seeding enabled")

        return self

    def set_power(self, power: float):
        """
        Set target power level for analysis.

        Args:
            power: Target power as percentage (0-100)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If power is not between 0 and 100
        """

        result = _validate_power(power)
        result.raise_if_invalid()
        self.power = float(power)
        return self

    def set_alpha(self, alpha: float):
        """
        Set significance level (Type I error rate).

        Args:
            alpha: Significance level (0-0.25, typically 0.05)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If alpha is not between 0 and 0.25
        """

        result = _validate_alpha(alpha)
        result.raise_if_invalid()
        self.alpha = float(alpha)
        return self

    def set_simulations(self, n_simulations):
        """
        Set number of Monte Carlo simulations.

        Args:
            n_simulations: Number of simulations to run (minimum 1, recommended ≥1000)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If n_simulations is less than 1

        Note:
            Higher values increase precision but require more computation time.
        """

        n_sims, result = _validate_simulations(n_simulations)
        for warning in result.warnings:
            print(f"Warning: {warning}")
        result.raise_if_invalid()
        self.n_simulations = n_sims
        return self

    def set_effects(self, effects_string: str):
        """
        Set effect sizes for predictors using string assignments.
        Factor variables must use bracket notation like 'treatment[2]=0.5'.

        Args:
            effects_string: Comma-separated assignments (e.g., 'x1=0.5, treatment[2]=0.3, x1:treatment[2]=0.2')

        Returns:
            self: For method chaining

        Example:
            >>> model.set_effects("treatment[2]=0.6, age=0.2, treatment[2]:age=0.3")
        """
        if not isinstance(effects_string, str):
            raise TypeError(
                f"effects_string must be a string, got {type(effects_string).__name__}"
            )

        if not effects_string.strip():
            raise ValueError("effects_string cannot be empty")

        try:
            from .utils.parsers import _parser

            assignments = _parser._split_assignments(effects_string)

            valid_items = []
            errors = []

            for assignment in assignments:
                if "=" not in assignment:
                    errors.append(
                        f"Invalid format '{assignment}'. Expected: 'name=value'"
                    )
                    continue

                name, value_str = assignment.split("=", 1)
                name, value_str = name.strip(), value_str.strip()

                # Parse value
                try:
                    value = float(value_str)
                except ValueError:
                    errors.append(
                        f"Invalid value '{value_str}' for '{name}'. Must be a number."
                    )
                    continue

                # Check if effect exists
                if any(info["name"] == name for info in self.effects.values()):
                    valid_items.append({"name": name, "value": value})
                else:
                    available = [info["name"] for info in self.effects.values()]
                    errors.append(
                        f"Effect '{name}' not found. Available: {', '.join(available)}"
                    )

            if errors:
                raise ValueError(
                    "Validation failed:\n" + "\n".join(f"• {err}" for err in errors)
                )

            if not valid_items:
                available_effects = [info["name"] for info in self.effects.values()]
                raise ValueError(
                    f"No valid effect assignments found. Available: {', '.join(available_effects)}"
                )

            # Apply effects
            successful = []
            for item in valid_items:
                effect_name = item["name"]
                effect_value = item["value"]

                for effect_key, effect_info in self.effects.items():
                    if effect_info["name"] == effect_name:
                        effect_info["effect_size"] = effect_value
                        successful.append(f"{effect_name}={effect_value}")
                        break

            if successful:
                print(f"Successfully set effects: {', '.join(successful)}")

        except Exception as e:
            available_effects = [
                effect_info["name"] for effect_info in self.effects.values()
            ]
            raise ValueError(
                f"Error setting effects: {str(e)}\nAvailable: {', '.join(available_effects)}"
            ) from e

        self.effects_set = True
        return self

    def set_variable_type(self, variable_types_string: str):
        """
        Set distribution types for predictor variables with factor support.
        Must be called before set_correlations() or set_effects().
        """
        if variable_types_string != "":
            timing_result = _validate_variable_type_timing(
                self.correlations_set, self.effects_set
            )
            if not timing_result.is_valid:
                raise ValueError(
                    "Timing error:\n"
                    + "\n".join(f"• {err}" for err in timing_result.errors)
                )

        if not isinstance(variable_types_string, str):
            raise TypeError(f"variable_types_string must be a string")

        # Initialize defaults
        if not self.variable_types_set:
            for var_key, var_info in self.variables.items():
                var_info["type"] = "normal"
            self.variable_types_set = True

        if not variable_types_string.strip():
            print(
                "Warning: No variable types specified. All variables remain 'normal'."
            )
            return self

        # Use original names before factor expansion
        available_vars = self._original_predictor_names

        try:
            parsed_vars, errors = _parser._parse(
                variable_types_string, "variable_type", available_vars
            )

            if errors:
                error_msg = "Error setting variable types:\n" + "\n".join(
                    f"• {err}" for err in errors
                )
                raise ValueError(error_msg)

            # Apply parsed types and handle factors
            successful = []
            factors_to_expand = []

            for var_name, var_data in parsed_vars.items():
                for key, info in self.variables.items():
                    if key != "variable_0" and info["name"] == var_name:
                        info.update(var_data)

                        if var_data["type"] == "factor":
                            # Store factor info
                            self.factor_variables[var_name] = {
                                "n_levels": var_data["n_levels"],
                                "proportions": var_data["proportions"],
                                "reference_level": 1,  # Always level 1 is reference
                            }
                            factors_to_expand.append((var_name, var_data))
                            successful.append(
                                f"{var_name}=(factor,{var_data['n_levels']} levels)"
                            )
                        else:
                            if "proportion" in var_data:
                                successful.append(
                                    f"{var_name}=({var_data['type']},{var_data['proportion']})"
                                )
                            else:
                                successful.append(f"{var_name}={var_data['type']}")
                        break

            # Expand factors into dummy variables
            if factors_to_expand:
                self._expand_factors(factors_to_expand)

            if successful:
                print(f"Successfully set variable types: {', '.join(successful)}")

        except Exception as e:
            raise ValueError(
                f"Error setting variable types: {str(e)}\nAvailable variables: {', '.join(available_vars)}"
            ) from e

        return self

    def set_correlations(self, correlations_input):
        """
        Set correlations between predictor variables. They'll act like rank correlations for different distributions.

        Args:
            correlations_input: Either correlation string or numpy correlation matrix
                String format: 'corr(x1,x2)=0.3, corr(x1,x3)=-0.2'
                Matrix format: Square correlation matrix as numpy array

        Returns:
            self: For method chaining

        Raises:
            TypeError: If input is not string or numpy array
            ValueError: If correlation matrix is invalid or correlations are out of range

        Note:
            Correlation matrix must be positive semi-definite with 1s on diagonal.
            Factor variables cannot be used in correlations.
        """

        if not isinstance(correlations_input, (str, np.ndarray)):
            raise TypeError("correlations_input must be string or numpy array")

        if not self.variable_types_set:
            self.set_variable_type("")

        if len(self.predictor_vars_order) < 2:
            raise ValueError(f"Need at least 2 variables for correlations")

        # Validate factor usage in correlation strings
        if isinstance(correlations_input, str) and correlations_input.strip():
            if self.factor_variables:
                import re

                var_pattern = r"corr\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*\)|^\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*\)"
                matches = re.findall(var_pattern, correlations_input)

                factor_vars_found = []
                for match in matches:
                    vars_in_match = [v.strip() for v in match if v.strip()]
                    for var in vars_in_match:
                        if var in self.factor_variables:
                            factor_vars_found.append(var)

                if factor_vars_found:
                    continuous_vars = self._non_factor_vars
                    raise ValueError(
                        f"Factor validation failed:\n"
                        f"• Cannot use factor variables in correlations: {', '.join(set(factor_vars_found))}. "
                        f"Factors are categorical and expand into dummy variables.\n"
                        f"• Use continuous variables: {', '.join(continuous_vars)}"
                    )

        # Initialize correlation matrix
        if self.correlation_matrix is None:
            n_vars = len(self.predictor_vars_order)
            self.correlation_matrix = np.eye(n_vars)

        try:
            if isinstance(correlations_input, str):
                if not correlations_input.strip():
                    print("Warning: Empty correlation string provided.")
                    return self

                correlations, errors = _parser._parse(
                    correlations_input, "correlation", self.predictor_vars_order
                )

                if errors:
                    error_msg = "Error setting correlations:\n" + "\n".join(
                        f"• {err}" for err in errors
                    )
                    raise ValueError(error_msg)

                # Update correlation matrix
                for (var1, var2), correlation in correlations.items():
                    idx1 = self.predictor_vars_order.index(var1)
                    idx2 = self.predictor_vars_order.index(var2)
                    self.correlation_matrix[idx1, idx2] = correlation
                    self.correlation_matrix[idx2, idx1] = correlation

                # Validate matrix
                result = _validate_correlation_matrix(self.correlation_matrix)
                if not result.is_valid:
                    raise ValueError(
                        "Invalid correlation matrix:\n"
                        + "\n".join(f"• {err}" for err in result.errors)
                    )

                if correlations:
                    correlation_strs = [
                        f"{var1}_{var2}={corr}"
                        for (var1, var2), corr in correlations.items()
                    ]
                    print(f"Set correlations: {', '.join(correlation_strs)}")

            elif isinstance(correlations_input, np.ndarray):
                num_vars = len(self.predictor_vars_order)
                if correlations_input.shape != (num_vars, num_vars):
                    raise ValueError(
                        f"Matrix shape {correlations_input.shape} doesn't match {num_vars} variables"
                    )

                if self.factor_variables:
                    factor_names = list(self.factor_variables.keys())
                    print(
                        f"Warning: Factor variables detected ({', '.join(factor_names)}). "
                        f"Ensure correlation matrix only covers continuous variables."
                    )

                result = _validate_correlation_matrix(correlations_input)
                if not result.is_valid:
                    raise ValueError(
                        "Invalid correlation matrix:\n"
                        + "\n".join(f"• {err}" for err in result.errors)
                    )

                self.correlation_matrix = correlations_input.copy()
                print(f"Set correlation matrix")

        except Exception as e:
            raise ValueError(
                f"Error setting correlations: {str(e)}\nAvailable variables: {', '.join(self.predictor_vars_order)}"
            ) from e

        self.correlations_set = True
        return self

    def set_heterogeneity(self, heterogeneity):
        """
        Set heterogeneity in effect sizes across observations.

        Args:
            heterogeneity: Standard deviation of effect size variation (≥0) = true effect * heterogeneity
                0 = constant effects, >0 = variable effects across observations

        Returns:
            self: For method chaining

        Raises:
            TypeError: If heterogeneity is not numeric
            ValueError: If heterogeneity is negative
        """

        if not isinstance(heterogeneity, (int, float)):
            raise TypeError("heterogeneity must be a number")

        if heterogeneity < 0:
            raise ValueError("heterogeneity must be non-negative")

        self.heterogeneity = float(heterogeneity)

        if heterogeneity > 0:
            print(
                f"Heterogeneity enabled: effect sizes will vary with SD = {heterogeneity}"
            )
        else:
            print("Heterogeneity disabled: effect sizes will be constant")

        return self

    def set_heteroskedasticity(self, heteroskedasticity_correlation):
        """
        Set heteroskedasticity in error terms.

        Args:
            heteroskedasticity_correlation: Correlation between linear predictor and error variance (-1 to 1)
                0 = homoscedastic errors, ≠0 = heteroscedastic errors

        Returns:
            self: For method chaining

        Raises:
            TypeError: If heteroskedasticity_correlation is not numeric
            ValueError: If correlation is not between -1 and 1
        """

        if not isinstance(heteroskedasticity_correlation, (int, float)):
            raise TypeError("heteroskedasticity_correlation must be a number")

        if not -1 <= heteroskedasticity_correlation <= 1:
            raise ValueError("heteroskedasticity_correlation must be between -1 and 1")

        self.heteroskedasticity = float(heteroskedasticity_correlation)

        if (
            abs(heteroskedasticity_correlation) > 1e-8
        ):  # tolerance for floating point noise
            print(
                f"heteroskedasticity enabled: error-predictor correlation = {heteroskedasticity_correlation}"
            )
        else:
            print("Homoscedasticity: no correlation between errors and predictors")

        return self

    def set_scenario_configs(self, configs_dict):
        """
        Set custom scenario configurations.

        Args:
            configs_dict: Dict with 'realistic' and/or 'doomer' keys containing config dicts

        Returns:
            self: For method chaining

        Raises:
            TypeError: If configs_dict is not a dictionary
            ValueError: If configs contain invalid parameter names or values

        Example:
            >>> model.set_scenario_configs({
            ...     'realistic': {'heterogeneity': 0.3, 'heteroskedasticity': 0.15},
            ...     'doomer': {'heterogeneity': 0.5, 'correlation_noise_sd': 0.5}
            ... })
        """
        if not isinstance(configs_dict, dict):
            raise TypeError("configs_dict must be a dictionary")

        # Merge custom configs with defaults
        merged = self.default_scenario_config.copy()
        for scenario, config in configs_dict.items():
            if scenario in merged:
                merged[scenario].update(config)
            else:
                merged[scenario] = config

        self.custom_scenario_configs = merged
        print(f"Custom scenario configs set for: {', '.join(configs_dict.keys())}")
        return self

    def upload_own_data(self, dataframe, preserve_correlation=True):
        """
        Upload empirical data to use realistic variable distributions.

        Args:
            dataframe: pandas DataFrame with variable columns matching model
            preserve_correlation: Whether to extract and use empirical correlations (default: True)

        Returns:
            self: For method chaining

        Raises:
            TypeError: If dataframe is not pandas DataFrame
            ValueError: If dataframe is empty or has no matching variables

        Note:
            Dataset should be complete, with no missing values
            Values shuld be numeric
            Variables in dataframe are automatically standardized (mean=0, SD=1).
            Not uploaded variables are generated synthetically.
        """

        # Lazy load for rarely used method
        import pandas as pd
        from .utils.data_generation import create_uploaded_lookup_tables

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be pandas DataFrame")

        if dataframe.empty:
            raise ValueError("dataframe cannot be empty")

        # Find matching variables
        all_vars = [info["name"] for info in self.variables.values()]
        available_cols = list(dataframe.columns)

        matched_vars = [var for var in all_vars if var in available_cols]
        synthetic_vars = [var for var in all_vars if var not in available_cols]

        if not matched_vars:
            raise ValueError(f"No matching variables found")

        print(f"Uploaded: {', '.join(matched_vars)}")
        print(f"Synthetic: {', '.join(synthetic_vars)}")

        # Store normalized data and create data matrix for lookup tables
        self.uploaded_data = {}
        uploaded_predictors = [
            var for var in self.predictor_vars_order if var in matched_vars
        ]

        if uploaded_predictors:
            # Create data matrix for uploaded predictors only (in predictor order)
            data_matrix_list = []
            for var in self.predictor_vars_order:
                if var in matched_vars:
                    data = dataframe[var].values.astype(float)
                    normalized = (data - np.mean(data)) / np.std(data)
                    self.uploaded_data[var] = normalized
                    data_matrix_list.append(normalized)

            if data_matrix_list:
                # Create lookup tables for uploaded data
                data_matrix = np.column_stack(data_matrix_list)
                self.upload_normal_values, self.upload_data_values = (
                    create_uploaded_lookup_tables(data_matrix)
                )
                print(
                    f"Created lookup tables for {len(data_matrix_list)} uploaded variables"
                )
            else:
                # No uploaded predictors, keep dummy matrices
                self.upload_normal_values = DUMMY_MATRIX
                self.upload_data_values = DUMMY_MATRIX
        else:
            # No uploaded predictors, keep dummy matrices
            self.upload_normal_values = DUMMY_MATRIX
            self.upload_data_values = DUMMY_MATRIX

        # Set variable types - mark uploaded variables as uploaded_data
        for _, var_info in self.variables.items():
            if var_info["name"] in matched_vars:
                var_info["type"] = "uploaded_data"

        # Initialize synthetic variables if not already done
        if not self.variable_types_set:
            for _, var_info in self.variables.items():
                if var_info["name"] in synthetic_vars:
                    var_info["type"] = "normal"
            self.variable_types_set = True

        # Measure correlations and update correlation matrix if requested
        if preserve_correlation and len(uploaded_predictors) > 1:
            data_matrix = np.column_stack(
                [self.uploaded_data[var] for var in uploaded_predictors]
            )
            corr_matrix = np.corrcoef(data_matrix.T)

            if self.correlation_matrix is None:
                n_vars = len(self.predictor_vars_order)
                self.correlation_matrix = np.eye(n_vars)

            # Update correlation matrix for uploaded variables
            correlations_found = 0
            for i, var1 in enumerate(uploaded_predictors):
                for j, var2 in enumerate(uploaded_predictors):
                    if i != j:
                        idx1 = self.predictor_vars_order.index(var1)
                        idx2 = self.predictor_vars_order.index(var2)
                        self.correlation_matrix[idx1, idx2] = corr_matrix[i, j]
                        if i < j:
                            correlations_found += 1

            print(f"Updated correlation matrix with {correlations_found} correlations")

        self.data_uploaded = True
        print(f"Processed {len(dataframe)} observations")

        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(equation: {self.equation})"

    # =====================================
    # Main analysis methods
    # =====================================

    def find_power(
        self,
        sample_size,
        target_test="overall",
        correction=None,
        print_results=True,
        scenarios=False,
        summary="short",
        return_results=False,
        test_formula=None,
    ):
        """
        Calculate statistical power for given sample size via Monte Carlo simulation.

        Args:
            sample_size: Sample size to test
            target_test: Which effect(s) to test ('overall', effect name, or 'all')
            correction: Multiple comparison correction ('Bonferroni', 'BH', 'Holm', or None)
            print_results: Whether to print formatted results (default: True)
            scenarios: Whether to run robustness analysis across assumption violations (default: False)
            summary: Output detail level ('short' or 'long')
            return_results: Whether to return results dictionary (default: False)
            test_formula: Optional formula subset to test (e.g., 'x1 + x1:x2')

        Returns:
            dict or None: Results dictionary if return_results=True, otherwise None

        Raises:
            ValueError: If model is not properly configured (missing effect sizes, etc.)

        Example:
            >>> model.find_power(sample_size=100, target_test='treatment', scenarios=True)
        """

        # Validate and parse once at the start
        self._validate_analysis_inputs(correction)
        target_tests = self._parse_target_tests(target_test)

        # Parse and validate test_formula if provided
        formula_to_test = self._parse_and_validate_test_formula(test_formula)

        if formula_to_test is not None:
            available_effect_names = [info["name"] for info in formula_to_test.values()]
            target_tests = [
                test
                for test in target_tests
                if test == "overall" or test in available_effect_names
            ]

        # Pass it
        if scenarios:
            result = self._run_scenario_analysis(
                "power",
                sample_size=sample_size,
                target_tests=target_tests,
                formula_to_test=formula_to_test,
                correction=correction,
                summary=summary,
                print_results=print_results,
            )
        else:
            result = self._run_find_power(
                sample_size, target_tests, formula_to_test, correction
            )

        if print_results and not scenarios:
            print(f"\n{'='*80}")
            print("MONTE CARLO POWER ANALYSIS RESULTS")
            print(f"{'='*80}")
            if correction:
                print(f"Multiple comparison correction: {correction}")
            if test_formula:
                print(f"Testing subset: {test_formula}")
            print(_format_results("power", result, summary))

        return result if return_results else None

    def find_sample_size(
        self,
        target_test="overall",
        from_size=30,
        to_size=200,
        by=5,
        correction=None,
        print_results=True,
        scenarios=False,
        summary="short",
        return_results=False,
        test_formula=None,
    ):
        """
        Find minimum sample size needed to achieve target power via Monte Carlo simulation.

        Args:
            target_test: Which effect(s) to test ('overall', effect name, or 'all')
            from_size: Minimum sample size to test (default: 30)
            to_size: Maximum sample size to test (default: 200)
            by: Step size between sample sizes (default: 5)
            correction: Multiple comparison correction ('Bonferroni', 'BH', 'Holm', or None)
            print_results: Whether to print formatted results (default: True)
            scenarios: Whether to run robustness analysis across assumption violations (default: False)
            summary: Output detail level ('short' or 'long')
            return_results: Whether to return results dictionary (default: False)
            test_formula: Optional formula subset to test (e.g., 'x1 + x1:x2')

        Returns:
            dict or None: Results dictionary if return_results=True, otherwise None

        Raises:
            ValueError: If model is not properly configured or sample size range is invalid

        Example:
            >>> model.find_sample_size(target_test='treatment', scenarios=True, summary='long')
        """

        # Validate and parse once at the start
        self._validate_analysis_inputs(correction)
        validation_result = _validate_sample_size_range(from_size, to_size, by)
        for warning in validation_result.warnings:
            print(f"Warning: {warning}")
        validation_result.raise_if_invalid()

        target_tests = self._parse_target_tests(target_test)

        # Parse and validate test_formula if provided
        formula_to_test = self._parse_and_validate_test_formula(test_formula)

        if formula_to_test is not None:
            available_effect_names = [info["name"] for info in formula_to_test.values()]
            target_tests = [
                test
                for test in target_tests
                if test == "overall" or test in available_effect_names
            ]

        sample_sizes = list(range(from_size, to_size + 1, by))

        # Pass it
        if scenarios:
            result = self._run_scenario_analysis(
                "sample_size",
                target_tests=target_tests,
                formula_to_test=formula_to_test,
                sample_sizes=sample_sizes,
                correction=correction,
                summary=summary,
                print_results=print_results,
            )
        else:
            # Run analysis with test_formula
            result = self._run_sample_size_analysis(
                sample_sizes, target_tests, formula_to_test, correction
            )

            if print_results:
                print(f"\n{'='*80}")
                print("SAMPLE SIZE ANALYSIS RESULTS")
                print(f"{'='*80}")
                if correction:
                    print(f"Multiple comparison correction: {correction}")
                if test_formula:
                    print(f"Testing subset: {test_formula}")
                print(_format_results("sample_size", result, summary))

                if summary == "long":
                    self._create_sample_size_plots(result)

        return result if return_results else None

    # =====================================
    # Abstract methods for subclasses
    # =====================================

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type name for display."""
        pass

    @abstractmethod
    def _run_statistical_analysis(
        self,
        X_expanded: np.ndarray,
        y: np.ndarray,
        target_indices: np.ndarray,
        alpha: float,
        correction_method: int,
    ) -> np.ndarray:
        """
        Run model-specific statistical analysis on simulated data.

        Must return array: [f_test_significant, individual_tests..., corrected_tests...]
        where each element is 0.0 (not significant) or 1.0 (significant).
        """
        pass

    @abstractmethod
    def _generate_dependent_variable(
        self,
        X_expanded: np.ndarray,
        effect_sizes_expanded: np.ndarray,
        heterogeneity: float = 0.0,
        heteroskedasticity: float = 0.0,
        sim_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate dependent variable for Monte Carlo simulation.

        Args:
            X_expanded: Design matrix with interaction terms (n_samples, n_effects)
            effect_sizes_expanded: Effect sizes for each column in X_expanded
            heterogeneity: SD of effect size variation across observations (0=constant effects)
            heteroskedasticity: Correlation between predictor and error variance (-1 to 1)
            sim_seed: Random seed for reproducible generation

        Returns:
            Generated dependent variable array (n_samples,)

        """
        pass

    # =====================================
    # Helpers
    # =====================================

    # --- Parsing and preparing  ---

    def _validate_analysis_inputs(self, correction):
        """Validate inputs before analysis."""

        result = _validate_correction_method(correction)
        result.raise_if_invalid()

        model_result = _validate_model_ready(self)
        model_result.raise_if_invalid()

    def _parse_target_tests(self, target_test):
        """Parse target test specification into list."""

        if isinstance(target_test, str):
            if target_test.strip().lower() == "all":
                target_test = ["overall"] + [
                    effect_info["name"] for effect_info in self.effects.values()
                ]
            elif "," in target_test:
                target_test = [test.strip() for test in target_test.split(",")]
            else:
                target_test = [target_test.strip()]

        # Handle dependent variable name
        dep_var_name = self.variables["variable_0"]["name"]
        target_test = [
            "overall" if test in {dep_var_name, "y"} else test for test in target_test
        ]

        # Validate
        valid_effects = [effect_info["name"] for effect_info in self.effects.values()]
        invalid_tests = [
            test
            for test in target_test
            if test != "overall" and test not in valid_effects
        ]

        if invalid_tests:
            valid_options = ["overall"] + valid_effects
            raise ValueError(
                f"Invalid target test(s): {', '.join(invalid_tests)}. "
                f"Available options: {', '.join(valid_options)}"
            )

        return target_test

    def _parse_and_validate_test_formula(self, test_formula):
        """Parse and validate test_formula, return filtered effects dict."""

        if test_formula is None:
            return None

        # Simple validation first
        available_vars = self.predictor_vars_order
        result = _validate_test_formula(test_formula, available_vars)
        result.raise_if_invalid()

        # Parse to create filtered effects
        try:
            _, test_effects = _parse_independent_variables(test_formula.strip())

            if not test_effects:
                raise ValueError(
                    f"No valid effects found in test_formula: '{test_formula}'"
                )

            # Create filtered effects dict with proper column indices
            filtered_effects = {}
            effect_counter = 1

            for effect_key, effect_info in test_effects.items():
                # Try to find in original effects first
                original_key = None
                for orig_key, orig_info in self.effects.items():
                    if orig_info["name"] == effect_info["name"]:
                        original_key = orig_key
                        break

                if original_key:
                    # Use original effect (preserves effect_size if set)
                    filtered_effects[f"effect_{effect_counter}"] = self.effects[
                        original_key
                    ].copy()
                else:
                    # New effect (interaction not in original model)
                    new_effect = effect_info.copy()
                    # Set column indices for new interactions
                    if effect_info["type"] == "interaction":
                        var_names = effect_info["var_names"]
                        new_effect["column_indices"] = [
                            available_vars.index(var) for var in var_names
                        ]
                    elif effect_info["type"] == "main":
                        new_effect["column_index"] = available_vars.index(
                            effect_info["name"]
                        )

                    # Default effect size for new effects
                    new_effect["effect_size"] = 0.0
                    filtered_effects[f"effect_{effect_counter}"] = new_effect
                    print(
                        f"Warning: New effect '{effect_info['name']}' added with effect_size=0.0"
                    )

                effect_counter += 1

            return filtered_effects

        except Exception as e:
            raise ValueError(f"Error parsing test_formula '{test_formula}': {str(e)}")

    def _parse_bracket_syntax(self, effect_name):
        """Parse bracket syntax like 'treatment[2]' with validation."""
        import re

        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$", effect_name)
        if match:
            factor_name = match.group(1)
            level = int(match.group(2))

            if factor_name in self.factor_variables:
                max_levels = self.factor_variables[factor_name]["n_levels"]
                if level == 1:
                    raise ValueError(
                        f"Level 1 is the reference level for factor '{factor_name}' and cannot have an effect"
                    )
                elif level < 2 or level > max_levels:
                    raise ValueError(
                        f"Level {level} invalid for factor '{factor_name}' (has {max_levels} levels, valid: 2-{max_levels})"
                    )
                return factor_name, level
            else:
                raise ValueError(f"'{factor_name}' is not a factor variable")

        return None, None

    def _create_X_extended(self, X, effects_to_use=None):
        """Create extended design matrix with interactions."""
        if effects_to_use is None:
            effects_to_use = self.effects

        columns = []

        for effect_info in effects_to_use.values():
            if effect_info["type"] == "main":
                # All main effects (continuous + factor dummies) use column index
                col_idx = effect_info["column_index"]
                columns.append(X[:, col_idx])
            else:  # interaction
                indices = effect_info["column_indices"]
                col = X[:, indices[0]].copy()
                for idx in indices[1:]:
                    col *= X[:, idx]
                columns.append(col)

        return np.column_stack(columns) if columns else np.empty((X.shape[0], 0))

    def _prepare_metadata(self, target_tests, correction=None, formula_to_test=None):
        """Pre-compute arrays and metadata for simulations with factor support."""

        effects_to_use = (
            formula_to_test if formula_to_test is not None else self.effects
        )

        effect_order = [info["name"] for info in effects_to_use.values()]
        target_indices = np.array(
            [effect_order.index(test) for test in target_tests if test != "overall"],
            dtype=np.int64,
        )

        # Get non-factor variables only for data generation
        non_factor_vars = self._non_factor_vars
        n_non_factor_vars = len(non_factor_vars)

        # Correlation matrix - only for non-factor variables
        if n_non_factor_vars == 0:
            correlation_matrix = np.eye(1)  # Dummy
        elif self.correlation_matrix is None:
            correlation_matrix = np.eye(n_non_factor_vars)
        else:
            # Extract submatrix for non-factor variables
            non_factor_indices = [
                self.predictor_vars_order.index(var)
                for var in non_factor_vars
                if var in self.predictor_vars_order
            ]
            if non_factor_indices:
                correlation_matrix = self.correlation_matrix[
                    np.ix_(non_factor_indices, non_factor_indices)
                ]
            else:
                correlation_matrix = np.eye(max(1, n_non_factor_vars))

        # var_types, var_params - for non-factor variables only
        var_types = np.zeros(max(1, n_non_factor_vars), dtype=np.int64)
        var_params = np.zeros(max(1, n_non_factor_vars), dtype=np.float64)

        type_mapping = {
            "normal": 0,
            "binary": 1,
            "right_skewed": 2,
            "left_skewed": 3,
            "high_kurtosis": 4,
            "uniform": 5,
            "uploaded_data": 99,
        }

        for i, var_name in enumerate(non_factor_vars):
            for var_key, info in self.variables.items():
                if var_key != "variable_0" and info["name"] == var_name:
                    var_type = info.get("type", "normal")
                    var_types[i] = type_mapping.get(var_type, 0)
                    var_params[i] = info.get("proportion", 0.5)
                    break

        # Factor specifications
        factor_specs = []
        for factor_name, factor_info in self.factor_variables.items():
            factor_specs.append(
                {
                    "n_levels": factor_info["n_levels"],
                    "proportions": factor_info["proportions"],
                }
            )

        effect_sizes_expanded = np.array(
            [info.get("effect_size", 0.0) for info in effects_to_use.values()]
        )

        # Correction method
        correction_method = 0
        if correction:
            method = correction.lower().replace("-", "_").replace(" ", "_")
            if method == "bonferroni":
                correction_method = 1
            elif method in ["benjamini_hochberg", "bh", "fdr"]:
                correction_method = 2
            elif method == "holm":
                correction_method = 3

        return (
            target_indices,
            n_non_factor_vars,
            correlation_matrix,
            var_types,
            var_params,
            factor_specs,
            self.upload_normal_values,
            self.upload_data_values,
            effect_sizes_expanded,
            correction_method,
            effects_to_use,
        )

    # --- Factor helpers  ---

    def _expand_factors(self, factors_to_expand):
        """Expand factor variables into dummy variables and update predictor order."""

        original_effects = self.effects.copy()
        new_variables = {}
        new_effects = {}

        # Keep all non-factor variables
        for var_key, var_info in self.variables.items():
            if var_key == "variable_0" or var_info.get("type") != "factor":
                new_variables[var_key] = var_info.copy()

        # Keep ALL non-factor effects (both main and interactions)
        for effect_key, effect_info in self.effects.items():
            effect_name = effect_info["name"]
            if effect_name not in self.factor_variables and not any(
                factor in effect_info.get("var_names", [effect_name])
                for factor in self.factor_variables
            ):
                new_effects[effect_key] = effect_info.copy()

        var_counter = len([k for k in self.variables.keys() if k != "variable_0"]) + 1
        effect_counter = len(self.effects) + 1

        # Create dummy variables and effects
        for factor_name, factor_data in factors_to_expand:
            n_levels = factor_data["n_levels"]

            for level in range(2, n_levels + 1):
                dummy_name = f"{factor_name}[{level}]"

                new_var_key = f"variable_{var_counter}"
                new_variables[new_var_key] = {
                    "name": dummy_name,
                    "type": "factor_dummy",
                    "factor_source": factor_name,
                    "factor_level": level,
                }

                new_effect_key = f"effect_{effect_counter}"
                new_effects[new_effect_key] = {
                    "name": dummy_name,
                    "type": "main",
                    "column_index": var_counter - 1,
                    "factor_source": factor_name,
                    "factor_level": level,
                }

                self.factor_dummy_mapping[dummy_name] = {
                    "factor_name": factor_name,
                    "level": level,
                }

                var_counter += 1
                effect_counter += 1

        # Handle interactions involving factors
        for effect_key, effect_info in original_effects.items():
            if effect_info["type"] == "interaction":
                var_names = effect_info["var_names"]
                factor_vars_in_interaction = [
                    v for v in var_names if v in self.factor_variables
                ]

                if factor_vars_in_interaction:
                    # Create new interactions for each combination of factor levels
                    for factor_var in factor_vars_in_interaction:
                        if factor_var in self.factor_variables:
                            n_levels = self.factor_variables[factor_var]["n_levels"]
                            for level in range(2, n_levels + 1):
                                dummy_name = f"{factor_var}[{level}]"

                                # Replace factor name with dummy name in interaction
                                new_var_names = [
                                    dummy_name if v == factor_var else v
                                    for v in var_names
                                ]
                                new_interaction_name = ":".join(new_var_names)

                                new_effect_key = f"effect_{effect_counter}"
                                new_effects[new_effect_key] = {
                                    "name": new_interaction_name,
                                    "type": "interaction",
                                    "var_names": new_var_names,
                                    "column_indices": [],  # Will be updated later
                                }
                                effect_counter += 1

        self.variables = new_variables
        self.effects = new_effects

        # Update predictor order - this is now the ONLY predictor order list
        regular_predictors = []
        factor_dummies = []

        for var_key, var_info in self.variables.items():
            if var_key != "variable_0":
                if var_info.get("factor_source"):  # This is a dummy variable
                    factor_dummies.append(var_info["name"])
                elif (
                    var_info.get("type") != "factor"
                ):  # This is a regular variable (not the original factor)
                    regular_predictors.append(var_info["name"])

        # Update the single predictor order (this will be used for data generation)
        self.predictor_vars_order = regular_predictors + factor_dummies

        # Update column indices for all effects
        for effect_key, effect_info in self.effects.items():
            if effect_info["type"] == "main":
                var_name = effect_info["name"]
                if var_name in self.predictor_vars_order:
                    effect_info["column_index"] = self.predictor_vars_order.index(
                        var_name
                    )
            elif effect_info["type"] == "interaction":
                var_names = effect_info["var_names"]
                effect_info["column_indices"] = [
                    self.predictor_vars_order.index(var)
                    for var in var_names
                    if var in self.predictor_vars_order
                ]

        print(
            f"Expanded {len(factors_to_expand)} factors into {len(factor_dummies)} dummy variables"
        )

    def _is_dummy_variable(self, var_name):
        """Check if variable is a dummy variable for a factor."""
        return var_name in self.factor_dummy_mapping

    # --- Main process  ---

    def _run_find_power(
        self,
        sample_size,
        target_tests=None,
        formula_to_test=None,
        correction=None,
        scenario_config=None,
    ):
        """Execute power analysis with scenario support."""

        if scenario_config is None:
            # Run simulations parsed values
            power_results = self._run_power_simulations_fixed(
                sample_size, formula_to_test, target_tests, correction
            )
        else:
            # Run simulations with scenario config
            power_results = self._run_power_simulations_scenario(
                sample_size,
                formula_to_test,
                target_tests,
                correction,
                scenario_config=scenario_config,
            )

        return {
            "model": {
                "model_type": self.model_type,
                "target_tests": target_tests,
                "test_formula": (
                    formula_to_test if formula_to_test is not None else target_tests
                ),
                "data_formula": self.equation,
                "sample_size": sample_size,
                "alpha": self.alpha,
                "n_simulations": power_results.get(
                    "n_simulations_used", self.n_simulations
                ),
                "correction": correction,
                "target_power": self.power,
                "parallel": self.parallel,
            },
            "results": power_results,
        }

    def _run_power_simulations_fixed(
        self, sample_size, formula_to_test, target_tests, correction
    ):
        """Run simulations with constant parameters."""

        # Pre-compute metadata
        (
            target_indices,
            n_non_factor_vars,  # changed from n_vars
            correlation_matrix,
            var_types,
            var_params,
            factor_specs,  # new
            upload_normal_values,
            upload_data_values,
            effect_sizes_expanded,
            correction_method,
            effects_to_use,
        ) = self._prepare_metadata(target_tests, correction, formula_to_test)

        heterogeneity, heteroskedasticity = self.heterogeneity, self.heteroskedasticity
        alpha = self.alpha

        all_results = []
        all_results_corrected = []

        seed = self.seed
        for sim_id in range(self.n_simulations):
            sim_seed = seed + 3 * sim_id if seed is not None else None
            result = self._single_simulation(
                sim_id,
                target_indices,
                sample_size,
                n_non_factor_vars,
                correlation_matrix,
                var_types,
                var_params,
                factor_specs,
                upload_normal_values,
                upload_data_values,
                effect_sizes_expanded,
                heterogeneity,
                heteroskedasticity,
                alpha,
                correction_method,
                effects_to_use,
                sim_seed=sim_seed,
            )
            if result is not None:
                sim_significant, sim_significant_corrected = result
                all_results.append(sim_significant)
                all_results_corrected.append(sim_significant_corrected)

        if not all_results:
            return {}

        return self._calculate_powers(all_results, all_results_corrected, target_tests)

    def _run_power_simulations_scenario(
        self, sample_size, formula_to_test, target_tests, correction, scenario_config
    ):
        """Run simulations with per-simulation perturbations."""

        # Pre-compute metadata
        (
            target_indices,
            n_non_factor_vars,
            correlation_matrix,
            var_types,
            var_params,
            factor_specs,
            upload_normal_values,
            upload_data_values,
            effect_sizes_expanded,
            correction_method,
            effects_to_use,
        ) = self._prepare_metadata(target_tests, correction, formula_to_test)

        heterogeneity, heteroskedasticity = (
            scenario_config["heterogeneity"],
            scenario_config["heteroskedasticity"],
        )
        alpha = self.alpha

        all_results = []
        all_results_corrected = []

        seed = self.seed
        for sim_id in range(self.n_simulations):
            sim_seed = seed + 3 * sim_id if seed is not None else None

            # Apply per-simulation perturbations (only affects non-factor variables)
            perturbed_corr, perturbed_types = self._apply_per_simulation_perturbations(
                correlation_matrix, var_types, scenario_config, sim_seed
            )

            result = self._single_simulation(
                sim_id,
                target_indices,
                sample_size,
                n_non_factor_vars,
                perturbed_corr,
                perturbed_types,
                var_params,
                factor_specs,
                upload_normal_values,
                upload_data_values,
                effect_sizes_expanded,
                heterogeneity,
                heteroskedasticity,
                alpha,
                correction_method,
                effects_to_use,
                sim_seed=sim_seed,
            )

            if result is not None:
                sim_significant, sim_significant_corrected = result
                all_results.append(sim_significant)
                all_results_corrected.append(sim_significant_corrected)

        if not all_results:
            return {}

        return self._calculate_powers(all_results, all_results_corrected, target_tests)

    def _single_simulation(
        self,
        sim_id,
        target_indices,
        sample_size,
        n_non_factor_vars,
        correlation_matrix,
        var_types,
        var_params,
        factor_specs,
        upload_normal_values,
        upload_data_values,
        effect_sizes_expanded,
        heterogeneity,
        heteroskedasticity,
        alpha,
        correction_method,
        effects_to_use=None,
        sim_seed=None,
    ):
        """Execute single Monte Carlo simulation with factor support."""

        try:
            # Generate non-factor variables
            if n_non_factor_vars > 0:
                X_non_factors = _generate_X(
                    sample_size,
                    n_non_factor_vars,
                    correlation_matrix,
                    var_types,
                    var_params,
                    upload_normal_values,
                    upload_data_values,
                    sim_seed,
                )
            else:
                X_non_factors = np.empty((sample_size, 0), dtype=float)

            # Generate factor variables (as dummy variables)
            X_factors = _generate_factors(sample_size, factor_specs, sim_seed)

            # Merge: non-factors first, then factors (matches predictor_vars_order)
            if X_factors.shape[1] > 0:
                X = np.column_stack([X_non_factors, X_factors])
            else:
                X = X_non_factors

            # Use filtered effects for X_expanded
            X_expanded = self._create_X_extended(X, effects_to_use)

            # Generate Y
            y = self._generate_dependent_variable(
                X_expanded=X_expanded,
                effect_sizes_expanded=effect_sizes_expanded,
                heterogeneity=heterogeneity,
                heteroskedasticity=heteroskedasticity,
                sim_seed=sim_seed,
            )

            # Run statistical analysis
            results = self._run_statistical_analysis(
                X_expanded, y, target_indices, alpha, correction_method
            )

            # Extract results: [f_sig, uncorr..., corr...]
            n_targets = len(target_indices)
            f_significant = bool(results[0])
            uncorrected = results[1 : 1 + n_targets].astype(bool)
            corrected = results[1 + n_targets : 1 + 2 * n_targets].astype(bool)

            # Add F-test to beginning
            sim_significant = np.concatenate([[f_significant], uncorrected])
            sim_significant_corrected = np.concatenate([[f_significant], corrected])

            return sim_significant, sim_significant_corrected

        except Exception as e:
            print(f"SIMULATION {sim_id} FAILED: {str(e)}")
            return None

    def _run_sample_size_analysis(
        self,
        sample_sizes,
        target_tests,
        formula_to_test,
        correction,
        scenario_config=None,
    ):
        """Run sample size analysis with passed parameters."""

        if self.parallel and len(sample_sizes) > 1:
            results = self._run_sample_size_analysis_parallel(
                sample_sizes, target_tests, formula_to_test, correction, scenario_config
            )
        else:
            results = self._run_sample_size_analysis_seq(
                sample_sizes, target_tests, formula_to_test, correction, scenario_config
            )

        return {
            "model": {
                "model_type": self.model_type,
                "target_tests": target_tests,
                "target_power": self.power,
                "data_formula": self.equation,
                "test_formula": (
                    formula_to_test if formula_to_test is not None else target_tests
                ),
                "alpha": self.alpha,
                "n_simulations": self.n_simulations,
                "correction": correction,
                "parallel": self.parallel,
                "sample_size_range": {
                    "from_size": sample_sizes[0],
                    "to_size": sample_sizes[-1],
                    "by": (
                        sample_sizes[1] - sample_sizes[0]
                        if len(sample_sizes) > 1
                        else 1
                    ),
                },
            },
            "results": results,
        }

    def _run_sample_size_analysis_parallel(
        self, sample_sizes, target_tests, formula_to_test, correction, scenario_config
    ):
        """Run sample size analysis in parallel."""
        from joblib import Parallel, delayed

        n_jobs = min(len(sample_sizes), self.n_cores)

        def analyze_single_size(sample_size):
            return sample_size, self._run_find_power(
                sample_size=sample_size,
                target_tests=target_tests,
                formula_to_test=formula_to_test,
                correction=correction,
                scenario_config=scenario_config,
            )

        results = Parallel(n_jobs=n_jobs, verbose=0)(  # type: ignore
            delayed(analyze_single_size)(size) for size in reversed(sample_sizes)  # type: ignore
        )

        results.sort(key=lambda x: x[0])  # type: ignore

        return self._process_power_results(results, target_tests, correction)

    def _run_sample_size_analysis_seq(
        self, sample_sizes, target_tests, formula_to_test, correction, scenario_config
    ):
        """Run sample size analysis sequentially."""

        results = []

        for sample_size in sample_sizes:
            power_result = self._run_find_power(
                sample_size=sample_size,
                target_tests=target_tests,
                formula_to_test=formula_to_test,
                correction=correction,
                scenario_config=scenario_config,
            )
            results.append((sample_size, power_result))

        return self._process_power_results(results, target_tests, correction)

    def _process_power_results(self, results, target_tests, correction):
        """Process power results from sample size analysis."""

        powers_by_test = {test: [] for test in target_tests}
        powers_by_test_corrected = {test: [] for test in target_tests}
        first_achieved = {test: -1 for test in target_tests}
        first_achieved_corrected = {test: -1 for test in target_tests}

        for sample_size, power_result in results:
            if power_result is None:
                continue

            for test in target_tests:
                power = power_result["results"]["individual_powers"][test]
                power_corrected = power_result["results"][
                    "individual_powers_corrected"
                ][test]

                powers_by_test[test].append(power)
                powers_by_test_corrected[test].append(power_corrected)

                if power >= self.power and first_achieved[test] == -1:
                    first_achieved[test] = sample_size

                if (
                    power_corrected >= self.power
                    and first_achieved_corrected[test] == -1
                ):
                    first_achieved_corrected[test] = sample_size

        return {
            "sample_sizes_tested": list(
                dict(results).keys()
            ),  # Extract sample sizes in order
            "powers_by_test": powers_by_test,
            "powers_by_test_corrected": (
                powers_by_test_corrected if correction else None
            ),
            "first_achieved": first_achieved,
            "first_achieved_corrected": (
                first_achieved_corrected if correction else None
            ),
        }

    def _calculate_powers(self, all_results, all_results_corrected, target_tests):
        """Calculate exact probabilities from simulation results."""

        n_sims = len(all_results)
        n_tests = len(target_tests)

        # Convert to numpy arrays for easier computation
        results_array = np.array(all_results, dtype=bool)
        results_corrected_array = np.array(all_results_corrected, dtype=bool)

        # Individual powers - FIXED: properly map test names to columns
        individual_powers = {}
        individual_powers_corrected = {}

        for test in target_tests:
            if test == "overall":
                # F-test is always at column 0
                individual_powers[test] = np.mean(results_array[:, 0]) * 100
                individual_powers_corrected[test] = (
                    np.mean(results_corrected_array[:, 0]) * 100
                )
            else:
                # Find position among non-'overall' tests and add 1 for F-test offset
                non_overall_tests = [t for t in target_tests if t != "overall"]
                pos = non_overall_tests.index(test)
                col_idx = pos + 1  # +1 because column 0 is F-test
                individual_powers[test] = np.mean(results_array[:, col_idx]) * 100
                individual_powers_corrected[test] = (
                    np.mean(results_corrected_array[:, col_idx]) * 100
                )

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

    # --- Scenarios  ---

    def get_scenario_configs(self):
        """Return realistic/doomer scenario parameters."""
        return (
            self.default_scenario_config
            if self.custom_scenario_configs is None
            else self.custom_scenario_configs
        )

    def _run_scenario_analysis(self, analysis_type, **kwargs):
        """Execute optimistic/realistic/doomer scenario comparison."""
        configs = self.get_scenario_configs()
        results = {}
        summary = kwargs.pop("summary", "short")
        print_results = kwargs.pop("print_results", True)

        if print_results:
            print(f"\n{'='*80}")
            print("SCENARIO-BASED MONTE CARLO POWER ANALYSIS RESULTS")
            print(f"{'='*80}")
            if self.custom_scenario_configs is None:
                print("Default configuration")
            else:
                print("Custom configuration")
            # Optimistic (user's original settings)
            print("Running OPTIMISTIC scenario (original settings)...")
        if analysis_type == "power":
            results["optimistic"] = self._run_find_power(**kwargs)
        else:  # sample_size
            results["optimistic"] = self._run_sample_size_analysis(**kwargs)

        # Realistic & Doomer scenarios
        for scenario_name, config in configs.items():
            if print_results:
                print(f"\nRunning {scenario_name.upper()} scenario...")

            # Run analysis with scenario-specific parameters
            kwargs_copy = kwargs.copy()
            kwargs_copy["scenario_config"] = config

            if analysis_type == "power":
                results[scenario_name] = self._run_find_power(**kwargs_copy)
            else:  # sample_size
                results[scenario_name] = self._run_sample_size_analysis(**kwargs_copy)

        # Format and return results
        formatted_results = {
            "analysis_type": analysis_type,
            "scenarios": results,
            "comparison": {},
        }

        # Print results
        if print_results:
            if analysis_type == "power":
                print(_format_results("scenario_power", formatted_results, summary))
            else:
                print(
                    _format_results("scenario_sample_size", formatted_results, summary)
                )

            # Create plots for long summary
            if summary == "long":
                self._create_scenario_plots(formatted_results, analysis_type)

        return formatted_results

    def _apply_per_simulation_perturbations(
        self, correlation_matrix, var_types, scenario_config, sim_seed
    ):
        """Apply random perturbations for scenario analysis."""

        if scenario_config is None:
            return correlation_matrix, var_types

        np.random.seed(sim_seed)

        # Perturb correlation matrix
        perturbed_corr = correlation_matrix
        if (
            correlation_matrix is not None
            and scenario_config["correlation_noise_sd"] > 0
        ):
            perturbed_corr = correlation_matrix.copy()
            noise = np.random.normal(
                0, scenario_config["correlation_noise_sd"], correlation_matrix.shape
            )
            noise = (noise + noise.T) / 2  # Keep symmetric
            perturbed_corr += noise
            perturbed_corr = np.clip(perturbed_corr, -0.8, 0.8)
            np.fill_diagonal(perturbed_corr, 1.0)

        # Perturb variable types
        perturbed_var_types = var_types.copy()
        if scenario_config["distribution_change_prob"] > 0:
            type_mapping = {"right_skewed": 2, "left_skewed": 3, "uniform": 5}
            new_type_codes = [
                type_mapping[distribution]
                for distribution in scenario_config["new_distributions"]
            ]

            for i in range(len(var_types)):
                if (
                    var_types[i] == 0  # normal distribution
                    and np.random.random() < scenario_config["distribution_change_prob"]
                ):
                    perturbed_var_types[i] = np.random.choice(new_type_codes)

        return perturbed_corr, perturbed_var_types

    def _create_sample_size_plots(self, results):
        """Create plots for sample size analysis."""

        if results.get("model", {}).get("correction"):
            # Two plots for corrected analysis
            _create_power_plot(
                sample_sizes=results["results"]["sample_sizes_tested"],
                powers_by_test=results["results"]["powers_by_test"],
                first_achieved=results["results"]["first_achieved"],
                target_tests=results["model"]["target_tests"],
                target_power=self.power,
                title="Uncorrected Power",
            )

            _create_power_plot(
                sample_sizes=results["results"]["sample_sizes_tested"],
                powers_by_test=results["results"]["powers_by_test_corrected"],
                first_achieved=results["results"]["first_achieved_corrected"],
                target_tests=results["model"]["target_tests"],
                target_power=self.power,
                title=f"Corrected Power ({results['model']['correction'].title()})",
            )
        else:
            # Single plot
            _create_power_plot(
                sample_sizes=results["results"]["sample_sizes_tested"],
                powers_by_test=results["results"]["powers_by_test"],
                first_achieved=results["results"]["first_achieved"],
                target_tests=results["model"]["target_tests"],
                target_power=self.power,
                title="Power Analysis: Sample Size Requirements",
            )

    def _create_scenario_plots(self, results: Dict, analysis_type: str):
        """Create visualizations for scenario analysis."""

        if analysis_type != "sample_size":
            return

        scenarios = results["scenarios"]
        scenario_names = ["optimistic", "realistic", "doomer"]
        scenario_labels = ["Optimistic", "Realistic", "Doomer"]

        first_scenario = scenarios.get("optimistic", {})
        if (
            "results" not in first_scenario
            or "sample_sizes_tested" not in first_scenario["results"]
        ):
            return

        sample_sizes = first_scenario["results"]["sample_sizes_tested"]
        target_tests = first_scenario["model"]["target_tests"]
        correction = first_scenario["model"].get("correction")

        # UNCORRECTED PLOTS - one plot per scenario
        for i, scenario in enumerate(scenario_names):
            if scenario in scenarios:
                powers_by_test = scenarios[scenario]["results"]["powers_by_test"]

                _create_power_plot(
                    sample_sizes=sample_sizes,
                    powers_by_test=powers_by_test,
                    first_achieved=scenarios[scenario]["results"]["first_achieved"],
                    target_tests=target_tests,
                    target_power=self.power,
                    title=f"{scenario_labels[i]} Scenario - Uncorrected Power",
                )

        # CORRECTED PLOTS - one plot per scenario
        if correction:
            for i, scenario in enumerate(scenario_names):
                if scenario in scenarios and scenarios[scenario]["results"].get(
                    "powers_by_test_corrected"
                ):
                    powers_by_test_corr = scenarios[scenario]["results"][
                        "powers_by_test_corrected"
                    ]

                    _create_power_plot(
                        sample_sizes=sample_sizes,
                        powers_by_test=powers_by_test_corr,
                        first_achieved=scenarios[scenario]["results"][
                            "first_achieved_corrected"
                        ],
                        target_tests=target_tests,
                        target_power=self.power,
                        title=f"{scenario_labels[i]} Scenario - Corrected Power ({correction})",
                    )

    # --- Helpers   ---

    @property
    def _original_predictor_names(self):
        """Get original predictor names before factor expansion."""
        return [
            info["name"]
            for key, info in self.variables.items()
            if key != "variable_0" and info.get("type") != "factor_dummy"
        ]

    @property
    def _non_factor_vars(self):
        """Get non-factor variables from current predictor order."""
        return [
            var
            for var in self.predictor_vars_order
            if var not in self.factor_variables and not self._is_dummy_variable(var)
        ]
