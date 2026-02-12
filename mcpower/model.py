"""
MCPower - Monte Carlo Power Analysis.

This module provides the main MCPower class for conducting power analysis
using Monte Carlo simulations.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .backends import get_backend
from .core import (
    DEFAULT_SCENARIO_CONFIG,
    ResultsProcessor,
    ScenarioRunner,
    SimulationRunner,
    VariableRegistry,
    apply_per_simulation_perturbations,
    build_power_result,
    build_sample_size_result,
    prepare_metadata,
)
from .utils.formatters import _format_results
from .utils.ols import compute_critical_values
from .utils.upload_data_utils import normalize_upload_input
from .utils.validators import (
    _validate_alpha,
    _validate_correction_method,
    _validate_correlation_matrix,
    _validate_model_ready,
    _validate_parallel_settings,
    _validate_power,
    _validate_sample_size,
    _validate_sample_size_for_model,
    _validate_sample_size_range,
    _validate_simulations,
    _validate_upload_data,
)
from .utils.visualization import _create_power_plot


class MCPower:
    """Monte Carlo Power Analysis.

    Conducts simulation-based power analysis for linear regression and
    linear mixed-effects models. Supports continuous, binary, and factor
    predictors, interactions, correlated predictors, non-normal distributions,
    and empirical data upload.

    All configuration methods (``set_*``) use deferred application: settings
    are stored as pending and processed in the correct order when ``apply()``
    is called (or automatically before ``find_power``/``find_sample_size``).
    Most ``set_*`` methods return ``self`` for method chaining.

    Attributes:
        seed: Random seed for reproducibility (default: 2137).
        power: Target power level in percent (default: 80.0).
        alpha: Significance level (default: 0.05).
        n_simulations: Number of Monte Carlo simulations for OLS (default: 1600).
        n_simulations_mixed_model: Simulations for mixed models (default: 400).
        parallel: Parallel processing mode (default: ``"mixedmodels"``).
        n_cores: Number of CPU cores for parallel execution.
        max_failed_simulations: Maximum acceptable failure rate (default: 0.03).
        heterogeneity: Standard deviation for random effect-size perturbation.
        heteroskedasticity: Correlation between predictor and error variance.

    Example:
        >>> model = MCPower("y = x1 + x2 + x1:x2")
        >>> model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
        >>> model.find_power(sample_size=100)

        >>> # Mixed model with clustered data
        >>> model = MCPower("y ~ treatment + motivation + (1|school)")
        >>> model.set_cluster("school", ICC=0.2, n_clusters=20)
        >>> model.set_effects("treatment=0.5, motivation=0.3")
        >>> model.find_power(sample_size=1000)
    """

    def __init__(self, data_generation_formula: str):
        """Initialize Monte Carlo Power Analysis.

        Parses the formula, creates the variable registry, and sets all
        configuration attributes to their defaults. If the formula contains
        random effects (e.g. ``(1|school)``), the model is automatically
        configured for mixed-model generation and a warning is issued.

        Args:
            data_generation_formula: R-style formula specifying the model.
                Supports ``=`` or ``~`` separators, ``+`` for additive terms,
                ``:`` for interactions, and ``(1|group)`` for random intercepts.
                Examples: ``"y = x1 + x2 + x1:x2"``,
                ``"score ~ treatment + age + (1|school)"``.
        """
        # Formula types
        self._generation_method = "linear_regression"
        self._test_method = "linear_regression"

        # Core configuration (applied immediately)
        self.seed: Optional[int] = 2137
        self.power = 80.0
        self.alpha = 0.05
        self.n_simulations = 1600
        self.n_simulations_mixed_model = 400

        # Parallel processing
        import multiprocessing as mp

        self.parallel: Union[bool, str] = "mixedmodels"
        self.n_cores = max(1, (mp.cpu_count() or 1) // 2)

        # Simulation failure tolerance
        self.max_failed_simulations = 0.03  # 3% default

        # Variable registry
        self._registry = VariableRegistry(data_generation_formula)

        # Scenario configurations
        self._scenario_configs: Optional[Dict[str, Dict[str, Any]]] = None

        # Pending inputs (deferred until apply())
        self._pending_variable_types: Optional[str] = None
        self._pending_effects: Optional[str] = None
        self._pending_correlations: Optional[Union[str, np.ndarray]] = None
        self._pending_heterogeneity: Optional[float] = None
        self._pending_heteroskedasticity: Optional[float] = None
        self._pending_data: Optional[Dict[str, Any]] = None
        self._pending_clusters: Dict[str, Dict] = {}  # {grouping_var: {n_clusters, cluster_size, icc}}

        # Detect mixed model formula
        if self._registry._random_effects_parsed:
            self._generation_method = "mixed_model"
            warnings.warn(
                "Mixed-effects models are experimental and still under active development. "
                "Results may be unreliable. Use at your own risk.",
                UserWarning,
                stacklevel=2,
            )

        # Applied state
        self._applied = False
        self.heterogeneity = 0.0
        self.heteroskedasticity = 0.0

        # Data storage
        self.upload_normal_values: Optional[np.ndarray] = None
        self.upload_data_values: Optional[np.ndarray] = None
        self._uploaded_raw_data: Optional[np.ndarray] = None
        self._uploaded_data_n = 0  # Sample count for warning
        self._preserve_correlation = "strict"  # Default mode
        self._uploaded_var_metadata: Dict[str, Any] = {}

        # Phase 2 Optimization: Effect plan cache for _create_X_extended
        self._effect_plan_cache: Optional[List[Tuple[str, Any]]] = None

        # Print summary
        predictor_names = self._registry.predictor_names
        if predictor_names:
            print(f"Variables: {self._registry.dependent} (dependent), {', '.join(predictor_names)} (predictors)")
            print(f"Found {len(predictor_names)} predictor variables")
            if len(predictor_names) == 1:
                print("Single predictor - no correlation matrix needed")

    # =========================================================================
    # Formula properties
    # =========================================================================

    @property
    def model_type(self) -> str:
        """Human-readable string showing the generation and test methods."""
        return f"Generation method: {self._generation_method}, Test method: {self._test_method}"

    # =========================================================================
    # Model properties
    # =========================================================================

    @property
    def equation(self) -> str:
        """Original equation string."""
        return self._registry.equation

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Correlation matrix."""
        return self._registry.get_correlation_matrix()

    @correlation_matrix.setter
    def correlation_matrix(self, value: Optional[np.ndarray]):
        if value is not None:
            self._registry.set_correlation_matrix(value)

    @property
    def predictor_vars_order(self) -> List[str]:
        """Predictor variable order."""
        return self._registry.non_factor_names + self._registry.dummy_names

    # =========================================================================
    # Configuration methods
    # =========================================================================

    def set_parallel(self, enable: Union[bool, str] = True, n_cores: Optional[int] = None):
        """Enable or disable parallel processing.

        Requires ``joblib`` to be installed. Falls back to sequential
        processing with a warning if ``joblib`` is unavailable.

        Args:
            enable: Parallel mode:
                - ``True``: parallel for all analyses.
                - ``False``: sequential processing.
                - ``"mixedmodels"``: parallel only for mixed-model analyses
                  (default).
            n_cores: Number of CPU cores to use. Defaults to
                ``cpu_count // 2``.

        Returns:
            self: For method chaining.
        """
        if enable is False:
            self.parallel, self.n_cores = False, 1
            return self

        # enable is True or "mixedmodels"
        try:
            import joblib  # noqa: F401 — availability check only
        except ImportError:
            print("Warning: joblib not available. Install with: pip install joblib")
            print("Warning: Continuing with sequential processing.")
            self.parallel = False
            return self

        settings, result = _validate_parallel_settings(enable, n_cores)
        result.raise_if_invalid()
        self.parallel, self.n_cores = settings
        return self

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility.

        Args:
            seed: Non-negative integer up to 3,000,000,000.
                Pass ``None`` to enable fully random seeding.

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *seed* is not an integer or ``None``.
            ValueError: If *seed* is negative or exceeds the maximum.
        """
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")
            if seed > 3000000000:
                raise ValueError("seed must be lower than 3,000,000,000")

        self.seed = seed
        if seed is not None:
            print(f"Seed set to: {seed}")
        else:
            print("Random seeding enabled")
        return self

    def set_power(self, power: float):
        """Set the target statistical power level.

        Used by ``find_sample_size`` to determine when power is sufficient.

        Args:
            power: Target power as a percentage (0–100). Default is 80.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If *power* is outside the valid range.
        """
        result = _validate_power(power)
        result.raise_if_invalid()
        self.power = float(power)
        return self

    def set_alpha(self, alpha: float):
        """Set the significance level for hypothesis testing.

        Args:
            alpha: Type-I error rate (0–0.25). Default is 0.05.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If *alpha* is outside the valid range.
        """
        result = _validate_alpha(alpha)
        result.raise_if_invalid()
        self.alpha = float(alpha)
        return self

    def set_simulations(self, n_simulations: int, model_type: Optional[str] = None):
        """Set the number of Monte Carlo simulations.

        More simulations yield more precise power estimates at the cost of
        longer runtime. The default is 1600 for OLS and 400 for mixed models.

        Args:
            n_simulations: Number of simulations (positive integer).
            model_type: Which simulation count to update:
                - ``None`` (default): sets both OLS and mixed-model counts.
                - ``"linear"``: sets only the OLS count.
                - ``"mixed"``: sets only the mixed-model count.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If *n_simulations* is not a positive integer or
                *model_type* is unrecognised.
        """
        n_sims, result = _validate_simulations(n_simulations)
        for warning in result.warnings:
            print(f"Warning: {warning}")
        result.raise_if_invalid()

        if model_type is None:
            self.n_simulations = n_sims
            self.n_simulations_mixed_model = n_sims
        elif model_type == "linear":
            self.n_simulations = n_sims
        elif model_type == "mixed":
            self.n_simulations_mixed_model = n_sims
        else:
            raise ValueError(f"model_type must be None, 'linear', or 'mixed', got '{model_type}'")
        return self

    @property
    def _effective_n_simulations(self) -> int:
        """Return the simulation count appropriate for the current test method."""
        if self._test_method == "mixed_model":
            return self.n_simulations_mixed_model
        return self.n_simulations

    def set_max_failed_simulations(self, percentage: float):
        """Set the maximum acceptable proportion of failed simulations.

        When a simulation iteration fails (e.g. due to convergence issues in
        mixed models), it is discarded. If the failure rate exceeds this
        threshold, the analysis raises an error rather than returning
        unreliable results.

        Args:
            percentage: Maximum failure rate as a proportion (0–1).
                Default is 0.03 (3%). For mixed models with small samples
                or high ICC, consider raising to 0.10.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If *percentage* is not between 0 and 1.
        """
        if not 0 <= percentage <= 1:
            raise ValueError("percentage must be between 0 and 1")
        self.max_failed_simulations = percentage
        return self

    def set_effects(self, effects_string: str):
        """Set standardised effect sizes for predictors.

        Effect sizes are expressed as standardised regression coefficients
        (beta weights). Each assignment maps an effect name to its size.
        Interaction effects use ``:`` notation. For factor variables, specify
        effects for each dummy level with bracket notation.

        This setting is deferred until ``apply()`` is called.

        Args:
            effects_string: Comma-separated ``name=value`` pairs.
                Examples: ``"x1=0.5, x2=0.3, x1:x2=0.2"``,
                ``"treatment=0.4, cyl[2]=0.2, cyl[3]=0.5"``.

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *effects_string* is not a string.
            ValueError: If *effects_string* is empty or contains invalid
                assignments (checked at apply time).
        """
        if not isinstance(effects_string, str):
            raise TypeError("effects_string must be a string")
        if not effects_string.strip():
            raise ValueError("effects_string cannot be empty")

        self._pending_effects = effects_string
        self._applied = False
        return self

    def set_correlations(self, correlations_input):
        """Set correlations between predictor variables.

        Correlations are only defined for non-factor (continuous/binary)
        predictors. Factor dummies are generated independently.

        This setting is deferred until ``apply()`` is called.

        Args:
            correlations_input: Either a comma-separated string of pair-wise
                assignments (e.g. ``"x1:x2=0.3, x1:x3=-0.1"``) or a full
                NumPy correlation matrix whose dimensions match the number
                of non-factor predictors.

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *correlations_input* is not a string or ndarray.
            ValueError: If the matrix is not positive semi-definite or has
                wrong dimensions (checked at apply time).
        """
        if not isinstance(correlations_input, (str, np.ndarray)):
            raise TypeError("correlations_input must be string or numpy array")

        self._pending_correlations = correlations_input
        self._applied = False
        return self

    def set_variable_type(self, variable_types_string: str):
        """Set distribution types for predictor variables.

        Variables default to ``"normal"`` (standard Gaussian). Use this
        method to specify alternative distributions.

        This setting is deferred until ``apply()`` is called.

        Args:
            variable_types_string: Comma-separated ``name=type`` assignments.
                Supported types:

                - ``"normal"`` — standard normal (default).
                - ``"binary"`` or ``"binary(p)"`` — Bernoulli with proportion *p*
                  (default 0.5).
                - ``"skewed"`` — heavy-tailed (t-distribution, df=3).
                - ``"factor(k)"`` — categorical with *k* levels (creates *k-1*
                  dummy variables).
                - ``"factor(k, p1, p2, ...)"`` — factor with custom level
                  proportions.

                Example: ``"x1=binary, x2=skewed, x3=factor(3)"``.

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *variable_types_string* is not a string.
            ValueError: If types are unrecognised or proportions invalid
                (checked at apply time).
        """
        if not isinstance(variable_types_string, str):
            raise TypeError("variable_types_string must be a string")

        self._pending_variable_types = variable_types_string
        self._applied = False
        return self

    def set_heterogeneity(self, heterogeneity: float):
        """Set heterogeneity (random variation) in effect sizes.

        When non-zero, each simulation draws a per-simulation effect-size
        multiplier from a normal distribution with mean 1 and the given
        standard deviation. This models uncertainty about the true effect
        size — for example, ``heterogeneity=0.1`` means effect sizes vary
        by roughly +/- 10% across simulations.

        This setting is deferred until ``apply()`` is called.

        Args:
            heterogeneity: Standard deviation of the random effect-size
                multiplier. Must be non-negative. Default is 0 (no variation).

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *heterogeneity* is not numeric.
        """
        if not isinstance(heterogeneity, (int, float)):
            raise TypeError("heterogeneity must be a number")

        self._pending_heterogeneity = float(heterogeneity)
        self._applied = False
        return self

    def set_heteroskedasticity(self, heteroskedasticity_correlation: float):
        """Set heteroskedasticity (non-constant error variance).

        Introduces a correlation between the first predictor's values and
        the error standard deviation, producing variance that increases (or
        decreases) with the predictor. This violates the homoskedasticity
        assumption and typically reduces power.

        This setting is deferred until ``apply()`` is called.

        Args:
            heteroskedasticity_correlation: Correlation between the first
                predictor and the error standard deviation, in the range
                [-1, 1]. Default is 0 (homoskedastic errors).

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If the value is not numeric.
        """
        if not isinstance(heteroskedasticity_correlation, (int, float)):
            raise TypeError("heteroskedasticity_correlation must be a number")

        self._pending_heteroskedasticity = float(heteroskedasticity_correlation)
        self._applied = False
        return self

    def set_cluster(
        self,
        grouping_var: str,
        ICC: float,
        n_clusters: Optional[int] = None,
        cluster_size: Optional[int] = None,
    ):
        """Configure a cluster/grouping variable for a random intercept.

        Sets up the clustering structure for a linear mixed-effects model.
        The grouping variable must correspond to a ``(1|group)`` term in the
        formula. Specify either *n_clusters* or *cluster_size* — the other
        is derived from the sample size at analysis time.

        This setting is deferred until ``apply()`` is called.

        Args:
            grouping_var: Name of the grouping variable (must match a
                ``(1|group)`` term in the formula).
            ICC: Intraclass correlation coefficient (0 <= ICC < 1).
                Determines the proportion of total variance attributable to
                between-cluster differences.
            n_clusters: Number of clusters. Mutually exclusive with
                *cluster_size*.
            cluster_size: Number of observations per cluster. Mutually
                exclusive with *n_clusters*.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If *grouping_var* is not in the formula, both or
                neither of *n_clusters*/*cluster_size* are given, or *ICC*
                is out of range.

        Example:
            >>> model = MCPower("y ~ x1 + x2 + (1|school)")
            >>> model.set_cluster("school", ICC=0.2, n_clusters=20)
        """
        from .utils.validators import _validate_cluster_config

        # Validate cluster configuration
        parsed_grouping_vars = [re["grouping_var"] for re in self._registry._random_effects_parsed]
        result = _validate_cluster_config(grouping_var, ICC, n_clusters, cluster_size, parsed_grouping_vars)
        result.raise_if_invalid()

        # Store pending configuration
        self._pending_clusters[grouping_var] = {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "icc": ICC,
        }
        self._applied = False
        return self

    def upload_data(
        self,
        data,
        columns: Optional[List[str]] = None,
        preserve_correlation: str = "strict",
        data_types: Optional[Dict[str, str]] = None,
    ):
        """
        Upload empirical data to preserve distribution shapes (deferred until apply()).

        The uploaded data's distribution will be used for generating simulated
        data. Auto-detects variable types based on unique value counts:
        - 1 unique value: dropped (constant)
        - 2 unique values: binary
        - 3-6 unique values: factor
        - 7+ unique values: continuous

        Args:
            data: Empirical data as:
                - pandas DataFrame - column names used automatically
                - dict of {var_name: array} - keys are variable names
                - numpy array (n_samples, n_vars) or (n_samples,) for single variable
                - list (1D or 2D)
                When columns is not provided for array/list input, variables are
                auto-named column_1, column_2, etc.
            columns: Variable names for numpy/list columns (optional; auto-generated if omitted)
            preserve_correlation: How to handle correlations between uploaded variables:
                - 'no': No correlation preservation (default generation)
                - 'partial': Compute correlations from data, merge with user correlations
                - 'strict': Bootstrap whole rows to preserve exact relationships (default)
            data_types: Override auto-detection for specific variables
                (e.g., {"hp": "continuous", "cyl": "factor"})
                Valid types: "binary", "factor", "continuous"

        Example:
            >>> # Using pandas DataFrame with default partial correlation preservation
            >>> import pandas as pd
            >>> df = pd.read_csv("my_data.csv")
            >>> model.upload_data(df[["x1", "x2"]])

            >>> # Using numpy array with strict correlation preservation
            >>> model.upload_data(
            ...     np.array([[1,2], [3,4], [5,6]]),
            ...     columns=["x1", "x2"],
            ...     preserve_correlation="strict"
            ... )

            >>> # Override auto-detection
            >>> model.upload_data(df, data_types={"cyl": "factor", "hp": "continuous"})
        """
        # Validate preserve_correlation
        valid_modes = ["no", "partial", "strict"]
        if preserve_correlation not in valid_modes:
            raise ValueError(f"preserve_correlation must be one of {valid_modes}, got '{preserve_correlation}'")

        # Convert to standard format: numpy array + column names
        data, columns = normalize_upload_input(data, columns)

        # Validate data
        result = _validate_upload_data(data)
        result.raise_if_invalid()

        # Validate data_types if provided
        if data_types is not None:
            valid_types = ["binary", "factor", "continuous"]
            for var, dtype in data_types.items():
                if dtype not in valid_types:
                    raise ValueError(f"Invalid type '{dtype}' for variable '{var}'. Must be one of {valid_types}")
                if var not in columns:
                    raise ValueError(f"Variable '{var}' in data_types not found in data columns: {columns}")

        # Warn about very small uploaded datasets (>=25 passes validation,
        # but <30 is still marginal for bootstrap resampling)
        if data.shape[0] < 30:
            print(
                f"Warning: Uploaded data has only {data.shape[0]} observations. "
                f"Bootstrap resampling from fewer than 30 observations may produce "
                f"unreliable distributional estimates."
            )

        self._pending_data = {
            "data": data,
            "columns": columns,
            "preserve_correlation": preserve_correlation,
            "data_types": data_types or {},
            "uploaded_n": data.shape[0],
        }
        self._applied = False

    def set_scenario_configs(self, configs_dict: Dict):
        """Set custom scenario configurations for robustness analysis.

        Scenario analysis (enabled via ``scenarios=True`` in ``find_power``
        or ``find_sample_size``) runs the simulation under multiple
        assumption-violation profiles. The defaults define ``"realistic"``
        and ``"doomer"`` scenarios; use this method to override or add
        custom scenarios.

        Provided configs are merged with the defaults: existing scenario
        keys are updated, new keys are added.

        Args:
            configs_dict: Mapping of scenario names to configuration dicts.
                Each configuration may include keys such as
                ``"heterogeneity"``, ``"heteroskedasticity"``,
                ``"effect_size_jitter"``, and ``"distribution_jitter"``.

        Returns:
            self: For method chaining.

        Raises:
            TypeError: If *configs_dict* is not a dictionary.
        """
        if not isinstance(configs_dict, dict):
            raise TypeError("configs_dict must be a dictionary")

        merged = {k: dict(v) for k, v in DEFAULT_SCENARIO_CONFIG.items()}
        for scenario, config in configs_dict.items():
            if scenario in merged:
                merged[scenario].update(config)
            else:
                merged[scenario] = config

        self._scenario_configs = merged
        print(f"Custom scenario configs set: {', '.join(configs_dict.keys())}")
        return self

    # =========================================================================
    # Apply method (processes all pending settings)
    # =========================================================================

    def apply(self):
        """
        Apply all pending settings to the model.

        Processes settings in the correct order:
        1. Variable types (expands factors)
        2. Cluster configurations (random effects)
        3. Uploaded data (sets variable types to uploaded_data)
        4. Effects
        5. Correlations
        6. Heterogeneity/heteroskedasticity

        This method is called automatically before find_power() or find_sample_size()
        if any settings have changed since the last apply().

        Returns:
            self for method chaining
        """
        from .utils.parsers import _parser

        # 1. Apply variable types first (may expand factors)
        self._apply_variable_types(_parser)

        # 2. Apply cluster configurations (needs expanded factor names)
        self._apply_clusters()

        # 3. Apply uploaded data (overrides variable types for matched columns)
        self._apply_data()

        # 4. Apply effects (needs expanded factor names + cluster effects)
        self._apply_effects(_parser)

        # 5. Apply correlations
        self._apply_correlations(_parser)

        # 6. Apply heterogeneity/heteroskedasticity
        self._apply_heterogeneity()

        # 6. Validate model is ready
        model_result = _validate_model_ready(self)
        model_result.raise_if_invalid()

        # Invalidate effect plan cache when settings change (Phase 2 optimization)
        self._effect_plan_cache = None

        self._applied = True
        print("Model settings applied successfully")
        return self

    def _apply_variable_types(self, _parser):
        """Apply pending variable types and expand factor variables into dummies."""
        # Set defaults for all predictors
        for name in self._registry.predictor_names:
            pred = self._registry.get_predictor(name)
            if pred and pred.var_type is None:
                pred.var_type = "normal"

        if self._pending_variable_types is None or not self._pending_variable_types.strip():
            return

        available_vars = [name for name in self._registry.predictor_names if not self._registry.is_dummy_variable(name)]
        parsed_vars, errors = _parser._parse(self._pending_variable_types, "variable_type", available_vars)

        if errors:
            raise ValueError("Error setting variable types:\n" + "\n".join(f"- {e}" for e in errors))

        successful = []
        for var_name, var_data in parsed_vars.items():
            var_type = var_data["type"]

            if var_type == "factor":
                self._registry.set_variable_type(
                    var_name,
                    var_type,
                    n_levels=var_data["n_levels"],
                    proportions=var_data["proportions"],
                )
                successful.append(f"{var_name}=(factor,{var_data['n_levels']} levels)")
            else:
                self._registry.set_variable_type(var_name, var_type)
                if "proportion" in var_data:
                    successful.append(f"{var_name}=({var_type},{var_data['proportion']})")
                else:
                    successful.append(f"{var_name}={var_type}")

        # Expand factors
        if self._registry.factor_names:
            self._registry.expand_factors()
            print(f"Expanded {len(self._registry.factor_names)} factors")

        if successful:
            print(f"Variable types: {', '.join(successful)}")

    def _apply_clusters(self):
        """Register pending cluster specifications in the variable registry."""
        if not self._pending_clusters:
            return

        # Validate that all parsed random effects have cluster config
        parsed_grouping_vars = [re["grouping_var"] for re in self._registry._random_effects_parsed]

        for gv in parsed_grouping_vars:
            if gv not in self._pending_clusters:
                raise ValueError(
                    f"Random effect (1|{gv}) found in formula but set_cluster('{gv}', ...) "
                    f"not called. You must configure all random effects."
                )

        for gv in self._pending_clusters:
            if gv not in parsed_grouping_vars:
                raise ValueError(f"set_cluster('{gv}', ...) called but (1|{gv}) not found in formula.")

        # Register each cluster
        for grouping_var, config in self._pending_clusters.items():
            # Warn if ICC=0
            if config["icc"] == 0:
                print(f"Warning: ICC=0 for '{grouping_var}'. No random variation between clusters.")

            self._registry.register_cluster(
                grouping_var=grouping_var,
                n_clusters=config["n_clusters"],
                cluster_size=config["cluster_size"],
                icc=config["icc"],
            )

        print(f"Cluster variables configured: {', '.join(self._pending_clusters.keys())}")

    def _apply_data(self):
        """Apply uploaded empirical data, auto-detecting types and routing to the appropriate mode."""
        if self._pending_data is None:
            return

        data = self._pending_data["data"]
        columns = self._pending_data["columns"]
        preserve_correlation = self._pending_data["preserve_correlation"]
        data_types_override = self._pending_data["data_types"]

        # Store sample count for warnings
        self._uploaded_data_n = data.shape[0]
        self._preserve_correlation = preserve_correlation

        # Get all predictor variables (both factor and non-factor)
        all_predictor_vars = self._registry.predictor_names
        matched_columns = []
        matched_indices = []
        unmatched = []

        for i, col in enumerate(columns):
            if col in all_predictor_vars:
                matched_columns.append(col)
                matched_indices.append(i)
            else:
                unmatched.append(col)

        if not matched_columns:
            print("Warning: No data columns match model variables — uploaded data ignored.")
            self._pending_data = None
            return

        if unmatched:
            print(f"Warning: Ignoring unmatched columns: {', '.join(unmatched)}")

        # Extract matched data
        matched_data = data[:, matched_indices]

        # Convert to float64 if object dtype (common with mixed-type DataFrames)
        if matched_data.dtype == object:
            try:
                matched_data = matched_data.astype(np.float64)
            except (ValueError, TypeError):
                # Per-column conversion for mixed numeric/string matched columns
                for ci in range(matched_data.shape[1]):
                    try:
                        matched_data[:, ci] = matched_data[:, ci].astype(np.float64)
                    except (ValueError, TypeError):
                        raise TypeError(
                            f"Column '{matched_columns[ci]}' contains non-numeric data "
                            f"that cannot be converted to float. Please ensure uploaded "
                            f"data columns contain numeric values."
                        ) from None
                matched_data = matched_data.astype(np.float64)

        # Auto-detect variable types based on unique values
        print("\n=== Auto-detecting variable types ===")
        auto_detected_types = {}
        dropped_columns = []

        for i, col in enumerate(matched_columns):
            col_data = matched_data[:, i]
            unique_vals = np.unique(col_data[~np.isnan(col_data)])
            n_unique = len(unique_vals)

            # Check for override
            if col in data_types_override:
                detected_type = data_types_override[col]
                print(f"{col}: {n_unique} unique values -> {detected_type} (overridden)")
            elif n_unique == 1:
                print(f"{col}: {n_unique} unique value -> DROPPED (constant)")
                dropped_columns.append(col)
                continue
            elif n_unique == 2:
                detected_type = "binary"
                print(f"{col}: {n_unique} unique values -> binary")
            elif 3 <= n_unique <= 6:
                detected_type = "factor"
                print(f"{col}: {n_unique} unique values -> factor")
            else:  # n_unique >= 7
                detected_type = "continuous"
                print(f"{col}: {n_unique} unique values -> continuous")

            auto_detected_types[col] = {
                "type": detected_type,
                "unique_count": n_unique,
                "unique_values": unique_vals,
                "data_index": i,
            }

        # Remove dropped columns
        if dropped_columns:
            matched_columns = [c for c in matched_columns if c not in dropped_columns]
            matched_data = matched_data[:, [auto_detected_types[c]["data_index"] for c in matched_columns]]
            # Reindex
            for i, col in enumerate(matched_columns):
                auto_detected_types[col]["data_index"] = i

        if not matched_columns:
            raise ValueError("All uploaded columns were dropped (constant values)")

        # Process based on mode
        if preserve_correlation == "strict":
            self._apply_data_strict_mode(matched_data, matched_columns, auto_detected_types)
        else:  # 'no' or 'partial'
            self._apply_data_normal_mode(matched_data, matched_columns, auto_detected_types, preserve_correlation)

        print(f"\nUploaded data: {', '.join(matched_columns)} ({data.shape[0]} samples)")
        print(f"Correlation preservation mode: {preserve_correlation}")

    def _apply_data_normal_mode(self, data, columns, type_info, mode):
        """Apply uploaded data in 'no' or 'partial' mode using quantile-matched lookup tables."""
        from .utils.data_generation import create_uploaded_lookup_tables

        # Process each variable based on detected type
        for col in columns:
            info = type_info[col]
            col_data = data[:, info["data_index"]]
            detected_type = info["type"]

            if detected_type == "binary":
                # Convert to standard binary (type 1)
                # Detect proportion (which value is "1")
                unique_vals = info["unique_values"]
                # Use the more frequent value as 0, less frequent as 1
                counts = [np.sum(col_data == val) for val in unique_vals]
                if counts[0] < counts[1]:
                    # First value is less frequent -> it's "1"
                    proportion = counts[0] / len(col_data)
                else:
                    # Second value is less frequent -> it's "1"
                    proportion = counts[1] / len(col_data)

                self._registry.set_variable_type(col, "binary", proportion=proportion)

            elif detected_type == "factor":
                # Convert to existing factor system
                unique_vals = info["unique_values"]
                n_levels = len(unique_vals)

                # Calculate proportions for each level
                proportions = []
                for val in unique_vals:
                    prop = np.sum(col_data == val) / len(col_data)
                    proportions.append(prop)

                self._registry.set_variable_type(col, "factor", n_levels=n_levels, proportions=proportions)

            else:  # continuous
                # Normalize: mean=0, sd=1
                normalized = (col_data - np.mean(col_data)) / np.std(col_data, ddof=1)

                # Create lookup tables (type 99)
                normal_vals, uploaded_vals = create_uploaded_lookup_tables(normalized.reshape(-1, 1))

                # Store in lookup tables
                non_factor_vars = self._registry.non_factor_names
                if col in non_factor_vars:
                    var_idx = non_factor_vars.index(col)
                    n_vars = len(non_factor_vars)

                    # Resize if needed
                    if self.upload_normal_values is None or self.upload_normal_values.shape != (n_vars, len(col_data)):
                        self.upload_normal_values = np.zeros((n_vars, len(col_data)), dtype=np.float64)
                        self.upload_data_values = np.zeros((n_vars, len(col_data)), dtype=np.float64)

                    assert self.upload_normal_values is not None
                    assert self.upload_data_values is not None
                    self.upload_normal_values[var_idx] = normal_vals[0]
                    self.upload_data_values[var_idx] = uploaded_vals[0]

                    # Set variable type to uploaded_data
                    pred = self._registry.get_predictor(col)
                    if pred:
                        pred.var_type = "uploaded_data"

        # Expand factors if any were detected
        if self._registry.factor_names:
            self._registry.expand_factors()

        # Mode 'partial': compute correlation matrix from continuous uploaded data
        if mode == "partial":
            self._compute_data_correlations(data, columns, type_info)
        elif mode == "no":
            # Ensure correlation matrix is initialized to identity
            non_factor_vars = self._registry.non_factor_names
            if len(non_factor_vars) > 0:
                existing_corr = self._registry.get_correlation_matrix()
                if existing_corr is None:
                    n_vars = len(non_factor_vars)
                    self._registry.set_correlation_matrix(np.eye(n_vars))

    def _compute_data_correlations(self, data, columns, type_info):
        """Compute and merge a correlation matrix from uploaded continuous variables."""
        # Get continuous uploaded variables
        continuous_cols = [col for col in columns if type_info[col]["type"] == "continuous"]

        if len(continuous_cols) < 2:
            return  # No correlations to compute

        # Extract continuous data
        continuous_indices = [type_info[col]["data_index"] for col in continuous_cols]
        continuous_data = data[:, continuous_indices]

        # Compute correlation matrix
        data_corr = np.corrcoef(continuous_data, rowvar=False)

        # Merge with existing correlation matrix
        non_factor_vars = self._registry.non_factor_names
        existing_corr = self._registry.get_correlation_matrix()

        if existing_corr is None:
            # Create identity matrix
            n_vars = len(non_factor_vars)
            existing_corr = np.eye(n_vars)

        # Fill in correlations between uploaded continuous variables
        for i, col1 in enumerate(continuous_cols):
            if col1 not in non_factor_vars:
                continue
            idx1 = non_factor_vars.index(col1)

            for j, col2 in enumerate(continuous_cols):
                if col2 not in non_factor_vars:
                    continue
                idx2 = non_factor_vars.index(col2)

                # Only update if not user-specified (user override)
                # For now, always update from data
                existing_corr[idx1, idx2] = data_corr[i, j]
                existing_corr[idx2, idx1] = data_corr[j, i]

        self._registry.set_correlation_matrix(existing_corr)
        print(f"Computed correlations from {len(continuous_cols)} continuous variables")

    def _apply_data_strict_mode(self, data, columns, type_info):
        """Apply uploaded data in 'strict' mode, storing normalised rows for bootstrapping."""
        # Store raw data for bootstrap
        # Normalize continuous variables
        normalized_data = np.copy(data)

        binary_cols = []
        factor_cols = []
        continuous_cols = []

        for col in columns:
            info = type_info[col]
            idx = info["data_index"]
            detected_type = info["type"]

            if detected_type == "binary":
                binary_cols.append(idx)
                # Store mapping info
                unique_vals = info["unique_values"]
                self._uploaded_var_metadata[col] = {
                    "type": "binary",
                    "unique_values": unique_vals,
                    "data_index": idx,
                }
                # Mark as uploaded_binary (type 98)
                pred = self._registry.get_predictor(col)
                if pred:
                    pred.var_type = "uploaded_binary"

            elif detected_type == "factor":
                factor_cols.append(idx)
                unique_vals = info["unique_values"]
                n_levels = len(unique_vals)

                self._uploaded_var_metadata[col] = {
                    "type": "factor",
                    "unique_values": unique_vals,
                    "n_levels": n_levels,
                    "data_index": idx,
                }
                # Register as factor but mark for bootstrap
                self._registry.set_variable_type(col, "factor", n_levels=n_levels, proportions=[1 / n_levels] * n_levels)
                # Mark as uploaded_factor
                pred = self._registry.get_predictor(col)
                if pred:
                    pred.var_type = "uploaded_factor"

            else:  # continuous
                continuous_cols.append(idx)
                # Normalize
                col_data = data[:, idx]
                normalized_data[:, idx] = (col_data - np.mean(col_data)) / np.std(col_data, ddof=1)

                self._uploaded_var_metadata[col] = {
                    "type": "continuous",
                    "data_index": idx,
                }
                # Mark as uploaded continuous (but this goes through non-factor pipeline)
                pred = self._registry.get_predictor(col)
                if pred:
                    pred.var_type = "uploaded_data"  # Will be handled by bootstrap

        # Expand factors
        if self._registry.factor_names:
            self._registry.expand_factors()

        # Store normalized data for bootstrap
        self._uploaded_raw_data = normalized_data

        # Warn about cross-correlations
        non_factor_vars = self._registry.non_factor_names
        uploaded_non_factor = [c for c in columns if c in non_factor_vars]
        created_vars = [v for v in non_factor_vars if v not in uploaded_non_factor]

        if created_vars and uploaded_non_factor:
            print(
                f"Warning: Cross-correlations between uploaded ({', '.join(uploaded_non_factor)}) "
                f"and created ({', '.join(created_vars)}) variables will be set to zero."
            )
            # Zero out cross-correlations in correlation matrix
            existing_corr = self._registry.get_correlation_matrix()
            if existing_corr is None:
                n_vars = len(non_factor_vars)
                existing_corr = np.eye(n_vars)

            for up_var in uploaded_non_factor:
                up_idx = non_factor_vars.index(up_var)
                for cr_var in created_vars:
                    cr_idx = non_factor_vars.index(cr_var)
                    existing_corr[up_idx, cr_idx] = 0.0
                    existing_corr[cr_idx, up_idx] = 0.0

            self._registry.set_correlation_matrix(existing_corr)

    def _apply_effects(self, _parser):
        """Parse and apply pending effect-size assignments to the registry."""
        if self._pending_effects is None:
            return

        assignments = _parser._split_assignments(self._pending_effects)
        effect_names = self._registry.effect_names

        successful = []
        errors = []

        for assignment in assignments:
            if "=" not in assignment:
                errors.append(f"Invalid format '{assignment}'. Expected: 'name=value'")
                continue

            name, value_str = assignment.split("=", 1)
            name, value_str = name.strip(), value_str.strip()

            try:
                value = float(value_str)
            except ValueError:
                errors.append(f"Invalid value '{value_str}' for '{name}'")
                continue

            if name in effect_names:
                self._registry.set_effect_size(name, value)
                successful.append(f"{name}={value}")
            else:
                errors.append(f"Effect '{name}' not found. Available: {', '.join(effect_names)}")

        if errors:
            raise ValueError("Effect validation failed:\n" + "\n".join(f"- {e}" for e in errors))

        if successful:
            print(f"Effects: {', '.join(successful)}")

    def _apply_correlations(self, _parser):
        """Parse and apply pending correlation settings to the registry."""
        if self._pending_correlations is None:
            return

        # Correlations only apply to non-factor variables (not dummies)
        non_factor_vars = self._registry.non_factor_names
        if len(non_factor_vars) < 2:
            raise ValueError("Need at least 2 non-factor variables for correlations")

        n_vars = len(non_factor_vars)

        if self.correlation_matrix is None:
            self._registry.set_correlation_matrix(np.eye(n_vars))

        correlations_input = self._pending_correlations

        if isinstance(correlations_input, str):
            if not correlations_input.strip():
                return

            correlations, errors = _parser._parse(correlations_input, "correlation", non_factor_vars)

            if errors:
                raise ValueError("Error setting correlations:\n" + "\n".join(f"- {e}" for e in errors))

            for (var1, var2), value in correlations.items():
                self._registry.set_correlation(var1, var2, value)

            result = _validate_correlation_matrix(self.correlation_matrix)
            if not result.is_valid:
                raise ValueError("Invalid correlation matrix:\n" + "\n".join(f"- {e}" for e in result.errors))

            print(f"Correlations: {len(correlations)} set")

        elif isinstance(correlations_input, np.ndarray):
            if correlations_input.shape != (n_vars, n_vars):
                raise ValueError(f"Matrix shape {correlations_input.shape} doesn't match {n_vars} non-factor variables")

            result = _validate_correlation_matrix(correlations_input)
            if not result.is_valid:
                raise ValueError("Invalid correlation matrix:\n" + "\n".join(f"- {e}" for e in result.errors))

            self._registry.set_correlation_matrix(correlations_input)
            print("Correlation matrix set")

    def _apply_heterogeneity(self):
        """Validate and apply pending heterogeneity and heteroskedasticity settings."""
        if self._pending_heterogeneity is not None:
            if self._pending_heterogeneity < 0:
                raise ValueError("heterogeneity must be non-negative")
            self.heterogeneity = self._pending_heterogeneity
            if self.heterogeneity > 0:
                print(f"Heterogeneity: SD = {self.heterogeneity}")

        if self._pending_heteroskedasticity is not None:
            if not -1 <= self._pending_heteroskedasticity <= 1:
                raise ValueError("heteroskedasticity_correlation must be between -1 and 1")
            self.heteroskedasticity = self._pending_heteroskedasticity
            if abs(self.heteroskedasticity) > 1e-8:
                print(f"Heteroskedasticity: correlation = {self.heteroskedasticity}")

    # =========================================================================
    # Analysis methods
    # =========================================================================

    def find_power(
        self,
        sample_size: int,
        target_test: str = "overall",
        correction: Optional[str] = None,
        print_results: bool = True,
        scenarios: bool = False,
        summary: str = "short",
        return_results: bool = False,
        test_formula: str = "",
        progress_callback=None,
        cancel_check=None,
    ):
        """
        Calculate statistical power for given sample size.

        Args:
            sample_size: Number of observations per simulation
            target_test: Effect(s) to test - "overall", specific name, or "all"
            correction: Multiple comparison correction (None, "bonferroni", "benjamini-hochberg", "holm")
            print_results: Whether to print results
            scenarios: Run scenario analysis
            summary: Output detail level ("short" or "long")
            return_results: Return results dict
            test_formula: Formula for statistical testing (default: use data generation formula).
                If the formula contains random effects like (1|school), analysis switches to
                mixed model testing (not yet implemented).
            progress_callback: Progress reporting control:
                - ``None`` (default): auto-use ``PrintReporter`` when
                  *print_results* is ``True``.
                - ``False``: explicitly disable progress.
                - callable ``(current, total)``: custom callback.
            cancel_check: Optional callable returning ``True`` to abort.

        Returns:
            dict or None: If *return_results* is ``True``, returns a
            results dictionary with keys ``"model"`` (metadata) and
            ``"results"`` (power estimates). Returns ``None`` otherwise.
        """
        # Auto-apply if settings have changed
        if not self._applied:
            self.apply()

        # Validate sample size (basic: >= 20, type check)
        _validate_sample_size(sample_size).raise_if_invalid()

        # Validate sample size against model complexity (>= 15 + n_variables)
        n_variables = len(self._registry.effect_names)
        _validate_sample_size_for_model(sample_size, n_variables).raise_if_invalid()

        # Validate and adjust cluster sample sizes
        self._validate_cluster_sample_size(sample_size)

        # Validate model complexity for mixed models
        if self._registry._random_effects_parsed and self._registry._cluster_specs:
            from .utils.validators import _validate_lme_model_complexity

            # Get first cluster spec (currently only support single clustering variable)
            cluster_spec = list(self._registry._cluster_specs.values())[0]
            n_fixed_effects = len(self._registry.effect_names)

            assert cluster_spec.n_clusters is not None
            complexity_result = _validate_lme_model_complexity(
                sample_size=sample_size,
                n_clusters=cluster_spec.n_clusters,
                n_fixed_effects=n_fixed_effects,
                cluster_size=cluster_spec.cluster_size,
            )

            # Print warnings (don't fail on warnings, only errors)
            for warning in complexity_result.warnings:
                print(f"Warning: {warning}")

            # Fail on errors (ratio < 7)
            complexity_result.raise_if_invalid()

        # Warn if sample size is much larger than uploaded data
        if self._uploaded_data_n > 0 and sample_size > 3 * self._uploaded_data_n:
            print(
                f"\nWarning: Requested sample size ({sample_size}) is more than 3x "
                f"the uploaded data size ({self._uploaded_data_n}). "
                f"This may lead to unrealistic extrapolation from the empirical distribution."
            )

        self._validate_analysis_inputs(correction)
        resolved_test_formula = self._resolve_test_formula(test_formula)
        target_tests = self._parse_target_tests(target_test)

        # Resolve progress callback
        from .progress import PrintReporter, ProgressReporter, compute_total_simulations

        if progress_callback is None:
            effective_cb = PrintReporter() if print_results else None
        elif progress_callback is False:
            effective_cb = None
        else:
            effective_cb = progress_callback

        reporter = None
        if effective_cb is not None:
            n_scenarios = (len(self._scenario_configs or DEFAULT_SCENARIO_CONFIG) + 1) if scenarios else 1
            total = compute_total_simulations(self._effective_n_simulations, 1, n_scenarios)
            reporter = ProgressReporter(total, effective_cb)

        if scenarios:
            result = self._run_scenario_analysis(
                "power",
                sample_size=sample_size,
                target_tests=target_tests,
                correction=correction,
                summary=summary,
                print_results=print_results,
                test_formula=resolved_test_formula,
                progress=reporter,
                cancel_check=cancel_check,
            )
        else:
            if reporter is not None:
                reporter.start()
            result = self._run_find_power(
                sample_size,
                target_tests,
                correction,
                test_formula=resolved_test_formula,
                progress=reporter,
                cancel_check=cancel_check,
            )

        if reporter is not None:
            reporter.finish()

        if not scenarios and print_results:
            print(f"\n{'=' * 80}")
            print("MONTE CARLO POWER ANALYSIS RESULTS")
            print(f"{'=' * 80}")
            if correction:
                print(f"Multiple comparison correction: {correction}")
            print(_format_results("power", result, summary))

        return result if return_results else None

    def find_sample_size(
        self,
        target_test: str = "overall",
        from_size: int = 30,
        to_size: int = 200,
        by: int = 5,
        correction: Optional[str] = None,
        print_results: bool = True,
        scenarios: bool = False,
        summary: str = "short",
        return_results: bool = False,
        test_formula: str = "",
        progress_callback=None,
        cancel_check=None,
    ):
        """
        Find minimum sample size needed for target power.

        Args:
            target_test: Effect(s) to test - "overall", specific name, or "all"
            from_size: Minimum sample size to test
            to_size: Maximum sample size to test
            by: Step size between sample sizes
            correction: Multiple comparison correction
            print_results: Whether to print results
            scenarios: Run scenario analysis
            summary: Output detail level
            return_results: Return results dict
            test_formula: Formula for statistical testing (default: use data generation formula).
                If the formula contains random effects like (1|school), analysis switches to
                mixed model testing (not yet implemented).
            progress_callback: Progress reporting control:
                - ``None`` (default): auto-use ``PrintReporter`` when
                  *print_results* is ``True``.
                - ``False``: explicitly disable progress.
                - callable ``(current, total)``: custom callback.
            cancel_check: Optional callable returning ``True`` to abort.

        Returns:
            dict or None: If *return_results* is ``True``, returns a
            results dictionary with keys ``"model"`` (metadata) and
            ``"results"`` (per-sample-size power estimates, first-achieved
            sizes). Returns ``None`` otherwise.
        """
        # Auto-apply if settings have changed
        if not self._applied:
            self.apply()

        # Validate from_size meets minimum requirements
        _validate_sample_size(from_size).raise_if_invalid()
        n_variables = len(self._registry.effect_names)
        _validate_sample_size_for_model(from_size, n_variables).raise_if_invalid()

        # Warn if max sample size is much larger than uploaded data
        if self._uploaded_data_n > 0 and to_size > 3 * self._uploaded_data_n:
            print(
                f"\nWarning: Maximum sample size ({to_size}) is more than 3x "
                f"the uploaded data size ({self._uploaded_data_n}). "
                f"This may lead to oversimplistic extrapolation from the empirical distribution."
            )

        self._validate_analysis_inputs(correction)
        resolved_test_formula = self._resolve_test_formula(test_formula)
        validation_result = _validate_sample_size_range(from_size, to_size, by)
        for warning in validation_result.warnings:
            print(f"Warning: {warning}")
        validation_result.raise_if_invalid()

        target_tests = self._parse_target_tests(target_test)
        sample_sizes = list(range(from_size, to_size + 1, by))

        # Resolve progress callback
        from .progress import PrintReporter, ProgressReporter, compute_total_simulations

        if progress_callback is None:
            effective_cb = PrintReporter() if print_results else None
        elif progress_callback is False:
            effective_cb = None
        else:
            effective_cb = progress_callback

        reporter = None
        if effective_cb is not None:
            n_scenarios = (len(self._scenario_configs or DEFAULT_SCENARIO_CONFIG) + 1) if scenarios else 1
            total = compute_total_simulations(self._effective_n_simulations, len(sample_sizes), n_scenarios)
            reporter = ProgressReporter(total, effective_cb)

        if scenarios:
            result = self._run_scenario_analysis(
                "sample_size",
                target_tests=target_tests,
                sample_sizes=sample_sizes,
                correction=correction,
                summary=summary,
                print_results=print_results,
                test_formula=resolved_test_formula,
                progress=reporter,
                cancel_check=cancel_check,
            )
        else:
            if reporter is not None:
                reporter.start()
            result = self._run_sample_size_analysis(
                sample_sizes,
                target_tests,
                correction,
                test_formula=resolved_test_formula,
                progress=reporter,
                cancel_check=cancel_check,
            )

        if reporter is not None:
            reporter.finish()

        if not scenarios and print_results:
            print(f"\n{'=' * 80}")
            print("SAMPLE SIZE ANALYSIS RESULTS")
            print(f"{'=' * 80}")
            if correction:
                print(f"Multiple comparison correction: {correction}")
            print(_format_results("sample_size", result, summary))

            if summary == "long":
                self._create_sample_size_plots(result)

        return result if return_results else None

    # =========================================================================
    # Statistical methods (linear regression)
    # =========================================================================

    def _generate_dependent_variable(
        self,
        X_expanded: np.ndarray,
        effect_sizes_expanded: np.ndarray,
        heterogeneity: float = 0.0,
        heteroskedasticity: float = 0.0,
        sim_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate the dependent variable as y = X @ beta + error via the active backend."""
        return get_backend().generate_y(
            X_expanded,
            effect_sizes_expanded,
            heterogeneity,
            heteroskedasticity,
            sim_seed if sim_seed is not None else -1,
        )

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _validate_analysis_inputs(self, correction):
        """Validate the multiple-comparison correction method before analysis."""
        result = _validate_correction_method(correction)
        result.raise_if_invalid()

    def _validate_cluster_sample_size(self, sample_size: int):
        """Derive missing cluster dimensions from sample_size and validate minimums."""
        if not self._registry.cluster_names:
            return  # No clusters, nothing to do

        from .utils.validators import _validate_cluster_sample_size

        for gv, spec in self._registry._cluster_specs.items():
            # Derive the missing dimension from sample_size
            if spec.n_clusters is not None:
                spec.cluster_size = sample_size // spec.n_clusters
            else:
                assert spec.cluster_size is not None
                spec.n_clusters = sample_size // spec.cluster_size

            assert spec.n_clusters is not None and spec.cluster_size is not None
            actual_n = spec.n_clusters * spec.cluster_size
            if actual_n != sample_size:
                print(
                    f"Warning: sample_size {sample_size} not evenly divisible by "
                    f"n_clusters={spec.n_clusters} for '{gv}'. Using {actual_n} "
                    f"(cluster_size={spec.cluster_size})"
                )

            _validate_cluster_sample_size(sample_size, spec.n_clusters, spec.cluster_size).raise_if_invalid()

    def _parse_target_tests(self, target_test: Union[str, List[str]]) -> List[str]:
        """Parse a target_test argument into a list of effect names to test."""
        if isinstance(target_test, str):
            if target_test.strip().lower() == "all":
                # Exclude cluster effects - they are random effects, not fixed effects to test
                cluster_effects = self._registry.cluster_effect_names
                fixed_effects = [e for e in self._registry.effect_names if e not in cluster_effects]
                target_test = ["overall"] + fixed_effects
            elif "," in target_test:
                target_test = [t.strip() for t in target_test.split(",")]
            else:
                target_test = [target_test.strip()]

        dep_var_name = self._registry.dependent
        target_test = ["overall" if t in {dep_var_name, "y"} else t for t in target_test]

        # Validate: cluster effects cannot be tested (they're random effects)
        cluster_effects = self._registry.cluster_effect_names
        cluster_in_targets = [t for t in target_test if t in cluster_effects]
        if cluster_in_targets:
            raise ValueError(
                f"Cannot test cluster effects {cluster_in_targets} - they are random effects. "
                f"Mixed models test fixed effects only. Use target_test='all' or specify fixed effects."
            )

        valid_effects = self._registry.effect_names
        invalid = [t for t in target_test if t != "overall" and t not in valid_effects]

        if invalid:
            valid_options = ["overall"] + [e for e in valid_effects if e not in cluster_effects]
            raise ValueError(f"Invalid target test(s): {', '.join(invalid)}. Available: {', '.join(valid_options)}")

        return target_test

    def _create_X_extended(self, X):
        """Build the extended design matrix by adding interaction columns.

        Uses a cached effect plan to avoid re-inspecting the registry on
        every simulation iteration. Cluster random-effect columns are
        excluded (they are handled by the LME groups parameter).
        """
        # Phase 2 Optimization: Build effect plan on first call, then reuse
        if self._effect_plan_cache is None:
            self._effect_plan_cache = []
            cluster_effect_names = self._registry.cluster_effect_names

            for effect_name, effect in self._registry._effects.items():
                # Skip cluster effects - they're handled separately via groups parameter in LME
                if effect_name in cluster_effect_names:
                    continue

                if effect.effect_type == "main":
                    self._effect_plan_cache.append(("main", effect.column_index))
                else:
                    self._effect_plan_cache.append(("interaction", effect.column_indices))

        # Execute the cached plan
        columns = []
        for effect_type, effect_data in self._effect_plan_cache:
            if effect_type == "main":
                columns.append(X[:, effect_data])
            else:  # interaction
                indices = effect_data
                col = X[:, indices[0]].copy()
                for idx in indices[1:]:
                    col *= X[:, idx]
                columns.append(col)

        return np.column_stack(columns) if columns else np.empty((X.shape[0], 0))

    def _prepare_metadata(self, target_tests, correction=None):
        """Pre-compute all static simulation metadata from the current model state."""
        return prepare_metadata(self, target_tests, correction)

    def _resolve_test_formula(self, test_formula: str) -> str:
        """Resolve test formula and update _test_method accordingly.

        Returns the resolved formula string.
        """
        if not test_formula:
            resolved = self._registry.equation
        else:
            resolved = test_formula

        from .utils.parsers import _parse_equation

        _, _, random_effects = _parse_equation(resolved)

        if random_effects:
            self._test_method = "mixed_model"
            warnings.warn(
                "Mixed-effects models are experimental and still under active development. "
                "Results may be unreliable. Use at your own risk.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._test_method = "linear_regression"

        return resolved

    def _run_find_power(
        self,
        sample_size,
        target_tests,
        correction,
        scenario_config=None,
        test_formula=None,
        progress=None,
        cancel_check=None,
    ):
        """Run the Monte Carlo simulation loop and return a power result dict."""
        # Validate and adjust cluster sample sizes
        self._validate_cluster_sample_size(sample_size)

        # Route based on test method (routing logic handled in simulation.py)
        metadata = self._prepare_metadata(target_tests, correction)

        if scenario_config:
            metadata.heterogeneity = scenario_config["heterogeneity"]
            metadata.heteroskedasticity = scenario_config["heteroskedasticity"]

        runner = SimulationRunner(
            n_simulations=self._effective_n_simulations,
            seed=self.seed,
            alpha=self.alpha,
            parallel=self.parallel,
            n_cores=self.n_cores,
            max_failed_simulations=self.max_failed_simulations,
        )

        # Compute critical values once before the simulation loop
        p = len(metadata.effect_sizes)
        dof = sample_size - p - 1
        n_targets = len(metadata.target_indices)

        f_crit, t_crit, correction_t_crits = compute_critical_values(self.alpha, p, dof, n_targets, metadata.correction_method)

        # Create analyze function with precomputed critical values
        def analyze_func(X, y, indices, alpha, correction):
            return get_backend().ols_analysis(X, y, indices, f_crit, t_crit, correction_t_crits, correction)

        sim_results = runner.run_power_simulations(
            sample_size=sample_size,
            metadata=metadata,
            generate_y_func=self._generate_dependent_variable,
            analyze_func=analyze_func,
            create_X_extended_func=self._create_X_extended,
            scenario_config=scenario_config,
            apply_perturbations_func=(apply_per_simulation_perturbations if scenario_config else None),
            progress=progress,
            cancel_check=cancel_check,
        )

        if not sim_results:
            return {}

        processor = ResultsProcessor(target_power=self.power)
        power_results = processor.calculate_powers(
            sim_results["all_results"],
            sim_results["all_results_corrected"],
            target_tests,
        )

        # Add n_simulations_failed to power_results
        if "n_simulations_failed" in sim_results:
            power_results["n_simulations_failed"] = sim_results["n_simulations_failed"]

        return build_power_result(
            model_type=self.model_type,
            target_tests=target_tests,
            formula_to_test=test_formula,
            equation=self.equation,
            sample_size=sample_size,
            alpha=self.alpha,
            n_simulations=self._effective_n_simulations,
            correction=correction,
            target_power=self.power,
            parallel=self.parallel,
            power_results=power_results,
        )

    def _is_parallel_effective(self) -> bool:
        """Resolve current parallel setting to a boolean.

        Returns True if parallel processing should be used for this model.
        """
        if self.parallel is True:
            return True
        if self.parallel == "mixedmodels":
            return bool(self._registry._cluster_specs)
        return False

    def _run_sample_size_analysis(
        self,
        sample_sizes,
        target_tests,
        correction,
        scenario_config=None,
        test_formula=None,
        progress=None,
        cancel_check=None,
    ):
        """Iterate over sample sizes, running power analysis for each."""
        from .progress import SimulationCancelled

        if self._is_parallel_effective():
            from joblib import Parallel, delayed

            try:
                power_results = Parallel(
                    n_jobs=self.n_cores,
                    backend="loky",
                    verbose=0,
                    return_as="generator",
                )(delayed(self._run_find_power)(ss, target_tests, correction, scenario_config, test_formula) for ss in sample_sizes)
                results = []
                for ss, result in zip(sample_sizes, power_results, strict=False):
                    if cancel_check is not None and cancel_check():
                        raise SimulationCancelled("Simulation cancelled by user")
                    results.append((ss, result))
                    if progress is not None:
                        progress.advance(self._effective_n_simulations)
            except Exception as e:
                if isinstance(e, SimulationCancelled):
                    raise
                print(f"Warning: Parallel execution failed ({e}). Falling back to sequential.")
                results = []
                for ss in sample_sizes:
                    if cancel_check is not None and cancel_check():
                        raise SimulationCancelled("Simulation cancelled by user") from None
                    result = self._run_find_power(
                        ss,
                        target_tests,
                        correction,
                        scenario_config,
                        test_formula,
                        progress=progress,
                        cancel_check=cancel_check,
                    )
                    results.append((ss, result))
        else:
            results = []
            for sample_size in sample_sizes:
                if cancel_check is not None and cancel_check():
                    raise SimulationCancelled("Simulation cancelled by user")
                power_result = self._run_find_power(
                    sample_size,
                    target_tests,
                    correction,
                    scenario_config,
                    test_formula,
                    progress=progress,
                    cancel_check=cancel_check,
                )
                results.append((sample_size, power_result))

        processor = ResultsProcessor(target_power=self.power)
        analysis_results = processor.process_sample_size_results(results, target_tests, correction)

        return build_sample_size_result(
            model_type=self.model_type,
            target_tests=target_tests,
            formula_to_test=test_formula,
            equation=self.equation,
            sample_sizes=sample_sizes,
            alpha=self.alpha,
            n_simulations=self._effective_n_simulations,
            correction=correction,
            target_power=self.power,
            parallel=self.parallel,
            analysis_results=analysis_results,
        )

    def _run_scenario_analysis(self, analysis_type, **kwargs):
        """Delegate to ScenarioRunner for multi-scenario power or sample-size analysis."""
        from functools import partial

        configs = self._scenario_configs or DEFAULT_SCENARIO_CONFIG
        scenario_runner = ScenarioRunner(self, configs)
        test_formula = kwargs.get("test_formula")
        progress = kwargs.get("progress")
        cancel_check = kwargs.get("cancel_check")

        if analysis_type == "power":
            run_power_func = partial(
                self._run_find_power,
                test_formula=test_formula,
                progress=progress,
                cancel_check=cancel_check,
            )
            return scenario_runner.run_power_analysis(
                sample_size=kwargs["sample_size"],
                target_tests=kwargs["target_tests"],
                correction=kwargs.get("correction"),
                run_find_power_func=run_power_func,
                summary=kwargs.get("summary", "short"),
                print_results=kwargs.get("print_results", True),
                progress=progress,
            )
        else:
            run_ss_func = partial(
                self._run_sample_size_analysis,
                test_formula=test_formula,
                progress=progress,
                cancel_check=cancel_check,
            )
            return scenario_runner.run_sample_size_analysis(
                sample_sizes=kwargs["sample_sizes"],
                target_tests=kwargs["target_tests"],
                correction=kwargs.get("correction"),
                run_sample_size_func=run_ss_func,
                summary=kwargs.get("summary", "short"),
                print_results=kwargs.get("print_results", True),
                progress=progress,
            )

    def _create_sample_size_plots(self, results):
        """Create power-vs-sample-size plots (uncorrected and/or corrected)."""
        if results.get("model", {}).get("correction"):
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
            _create_power_plot(
                sample_sizes=results["results"]["sample_sizes_tested"],
                powers_by_test=results["results"]["powers_by_test"],
                first_achieved=results["results"]["first_achieved"],
                target_tests=results["model"]["target_tests"],
                target_power=self.power,
                title="Power Analysis",
            )

    def __repr__(self):
        return f"MCPower(equation='{self.equation}')"
