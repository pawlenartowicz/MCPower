"""
Validation utilities for Monte Carlo Power Analysis.

This module provides validation functions for model inputs, parameters,
and mathematical constraints.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

__all__ = []


@dataclass
class _ValidationResult:
    """Outcome of a validation check, carrying errors and warnings.

    Attributes:
        is_valid: ``True`` if no errors were found.
        errors: List of error messages (empty when valid).
        warnings: List of non-fatal warning messages.
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self):
        """Raise ``ValueError`` if the validation failed."""
        if not self.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(f"• {err}" for err in self.errors)
            raise ValueError(error_msg)


class _Validator:
    """Static helpers for type and range checks used by all validators."""

    @staticmethod
    def _check_type(value: Any, expected_types: tuple, name: str) -> Optional[str]:
        """Check if value has expected type."""
        if not isinstance(value, expected_types):
            actual_type = type(value).__name__
            expected = expected_types[0].__name__ if len(expected_types) == 1 else f"one of {[t.__name__ for t in expected_types]}"
            return f"{name} must be {expected}, got {actual_type}"
        return None

    @staticmethod
    def _check_range(
        value: Union[int, float],
        min_val: Optional[float],
        max_val: Optional[float],
        name: str,
    ) -> Optional[str]:
        """Check if value is within range."""
        if min_val is not None and value < min_val:
            return f"{name} must be >= {min_val}, got {value}"
        if max_val is not None and value > max_val:
            return f"{name} must be <= {max_val}, got {value}"
        return None


_validator = _Validator()


def _validate_numeric_parameter(
    value: Any,
    name: str,
    expected_types: tuple = (int, float),
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_rounding: bool = False,
) -> _ValidationResult:
    """Generic validation for numeric parameters."""
    errors: List[str] = []
    warnings: List[str] = []

    # Type check
    type_error = _validator._check_type(value, expected_types, name)
    if type_error:
        errors.append(type_error)
        return _ValidationResult(False, errors, warnings)

    # Range check
    range_error = _validator._check_range(value, min_val, max_val, name)
    if range_error:
        errors.append(range_error)

    # Rounding warning for floats when int expected
    if allow_rounding and isinstance(value, float) and (int, float) in expected_types:
        rounded = int(round(value))
        if value != rounded:
            warnings.append(f"{name} rounded from {value} to {rounded}")

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_power(power: Any) -> _ValidationResult:
    """Validate power parameter (0-100%)."""
    return _validate_numeric_parameter(power, "Power", min_val=0, max_val=100)


def _validate_alpha(alpha: Any) -> _ValidationResult:
    """Validate alpha level parameter (0-0.25)."""
    return _validate_numeric_parameter(alpha, "Alpha", min_val=0, max_val=0.25)


def _validate_simulations(n_simulations: Any) -> Tuple[int, _ValidationResult]:
    """Validate and process number of simulations."""
    result = _validate_numeric_parameter(n_simulations, "Number of simulations", min_val=1, allow_rounding=True)

    if result.is_valid:
        rounded = int(round(n_simulations))
        if rounded < 1000:
            result.warnings.append(f"Low simulation count ({rounded}). Consider using at least 1000 for reliable results.")
        return rounded, result

    return 0, result


def _validate_sample_size(sample_size: Any) -> _ValidationResult:
    """Validate sample size parameter.

    Requires an integer >= 20 and <= 100,000.
    """
    errors = []

    # Must be integer
    if not isinstance(sample_size, int):
        errors.append(f"sample_size must be an integer, got {type(sample_size).__name__}")
        return _ValidationResult(False, errors, [])

    # Range check
    if sample_size < 20:
        errors.append(f"sample_size must be at least 20, got {sample_size}")
    elif sample_size > 100000:
        errors.append(
            f"sample_size too large ({sample_size:,}). Maximum recommended: 100,000. We cannot guarantee stability for such small p-values."
        )

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_sample_size_for_model(sample_size: int, n_variables: int) -> _ValidationResult:
    """Validate that sample size is sufficient for the model complexity.

    Requires sample_size >= 15 + n_variables, where n_variables is the number
    of columns in the design matrix (continuous predictors, factor dummy
    variables, and interaction terms).

    Args:
        sample_size: Total number of observations.
        n_variables: Number of design-matrix columns (excluding intercept).

    Returns:
        _ValidationResult with errors if sample size is insufficient.
    """
    errors = []
    min_required = 15 + n_variables

    if sample_size < min_required:
        errors.append(
            f"sample_size ({sample_size}) is too small for a model with {n_variables} "
            f"variables. Minimum required: {min_required} (15 + {n_variables} variables)."
        )

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_sample_size_range(from_size: Any, to_size: Any, by: Any) -> _ValidationResult:
    """Validate sample size range parameters."""
    errors: List[str] = []
    warnings: List[str] = []

    # Type checks
    for param, name in [(from_size, "from_size"), (to_size, "to_size"), (by, "by")]:
        if not isinstance(param, int) or param <= 0:
            errors.append(f"{name} must be a positive integer, got {param}")

    if errors:
        return _ValidationResult(False, errors, warnings)

    # Logic checks
    if from_size >= to_size:
        errors.append(f"from_size ({from_size}) must be less than to_size ({to_size})")

    if by > (to_size - from_size):
        errors.append(f"Step size 'by' ({by}) is larger than range ({to_size - from_size}). This will only test one sample size.")

    # Warning for many tests
    n_tests = len(range(from_size, to_size + 1, by))
    if n_tests > 100:
        warnings.append(f"Large number of sample sizes to test ({n_tests}). This may take significant time.")

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_correlation_matrix(
    corr_matrix: Optional[np.ndarray],
) -> _ValidationResult:
    """Validate correlation matrix meets mathematical requirements."""
    errors = []

    if corr_matrix is None:
        errors.append("Correlation matrix is None")
        return _ValidationResult(False, errors, [])

    # Shape check
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        errors.append("Correlation matrix must be square")
        return _ValidationResult(False, errors, [])

    # Diagonal check
    if not np.allclose(np.diag(corr_matrix), 1.0):
        errors.append("Diagonal elements of correlation matrix must be 1")

    # Symmetry check
    if not np.allclose(corr_matrix, corr_matrix.T):
        errors.append("Correlation matrix must be symmetric")

    # Range check
    if np.any(np.abs(corr_matrix) > 1):
        errors.append("All correlations must be between -1 and 1")

    # Positive semi-definite check
    try:
        eigenvals = np.linalg.eigvals(corr_matrix)
        if np.any(eigenvals < -1e-8):  # Tolerance for floating point noise
            errors.append("Correlation matrix must be positive semi-definite. ")
    except np.linalg.LinAlgError:
        errors.append("Cannot compute eigenvalues of correlation matrix")

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_correction_method(correction: Optional[str]) -> _ValidationResult:
    """Validate correction method name."""
    if correction is None:
        return _ValidationResult(True, [], [])

    method = correction.lower().replace("-", "_").replace(" ", "_")
    valid_methods = ["bonferroni", "benjamini_hochberg", "bh", "fdr", "holm", "tukey"]

    if method not in valid_methods:
        return _ValidationResult(
            False,
            [
                f"Unknown correction method: {correction}. "
                "Valid options: 'Bonferroni', 'Benjamini-Hochberg' (or 'BH', 'FDR'), 'Holm', 'Tukey'"
            ],
            [],
        )

    return _ValidationResult(True, [], [])


def _validate_parallel_settings(enable: Any, n_cores: Optional[int]) -> Tuple[Tuple[Any, int], _ValidationResult]:
    """Validate parallel processing settings.

    Args:
        enable: True, False, or "mixedmodels"
        n_cores: Number of CPU cores (positive int or None for auto)

    Returns:
        ((enable, n_cores), ValidationResult)
    """
    import multiprocessing as mp

    errors = []

    # Validate enable
    valid_values = (True, False, "mixedmodels")
    if enable not in valid_values:
        errors.append(f"enable must be True, False, or 'mixedmodels', got {enable!r}")
        return (False, 1), _ValidationResult(False, errors, [])

    # Validate n_cores
    max_cores = mp.cpu_count()
    validated_n_cores = max(1, max_cores // 2)

    if n_cores is not None:
        if not isinstance(n_cores, int) or n_cores <= 0:
            errors.append(f"n_cores must be a positive integer, got {n_cores}")
        else:
            validated_n_cores = min(n_cores, max_cores)

    return (enable, validated_n_cores), _ValidationResult(len(errors) == 0, errors, [])


def _validate_model_ready(model) -> _ValidationResult:
    """
    Validate that model is ready for analysis.

    Args:
        model: Model instance to validate

    Returns:
        _ValidationResult with any errors or warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check effect sizes - check if pending effects were set
    has_effects = hasattr(model, "_pending_effects") and model._pending_effects is not None
    if not has_effects:
        if hasattr(model, "_registry"):
            available = model._registry.effect_names
            errors.append(
                f"Effect sizes must be set using set_effects() before running analysis. Available effects: {', '.join(available)}"
            )
        else:
            errors.append("Effect sizes must be set before running analysis")

    # Check other required attributes
    required_attrs = ["power", "alpha", "n_simulations"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            errors.append(f"Model missing required attribute: {attr}")

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_test_formula(test_formula: str, available_variables: List[str]) -> _ValidationResult:
    """
    Simple validation for test_formula - just check if variables exist.

    Args:
        test_formula: Formula string to test (e.g., "x1 + x2:x3")
        available_variables: List of base variable names

    Returns:
        _ValidationResult with any errors
    """
    import re

    errors = []

    if not isinstance(test_formula, str):
        errors.append("test_formula must be a string")
        return _ValidationResult(False, errors, [])

    if not test_formula.strip():
        errors.append("test_formula cannot be empty")
        return _ValidationResult(False, errors, [])

    try:
        # Extract all variable names from formula
        # Matches: word characters (letters, digits, underscore)
        variables_in_formula = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", test_formula))

        if not variables_in_formula:
            errors.append(f"No variables found in test_formula: '{test_formula}'")
            return _ValidationResult(False, errors, [])

        # Check if all variables exist
        missing_vars = variables_in_formula - set(available_variables)
        if missing_vars:
            errors.append(
                f"Variables not found in original model: {', '.join(sorted(missing_vars))}. Available: {', '.join(available_variables)}"
            )

        return _ValidationResult(len(errors) == 0, errors, [])

    except Exception as e:
        errors.append(f"Error parsing test_formula: {str(e)}")
        return _ValidationResult(False, errors, [])


def _validate_factor_specification(n_levels: int, proportions: List[float]) -> _ValidationResult:
    """Validate factor variable specification."""
    errors = []
    warnings = []

    # Validate n_levels
    if not isinstance(n_levels, int):
        errors.append("n_levels must be an integer")
    elif n_levels < 2:
        errors.append("Factor must have at least 2 levels")
    elif n_levels > 20:
        errors.append("Factor cannot have more than 20 levels (computational limits)")

    # Validate proportions
    if not isinstance(proportions, (list, tuple)):
        errors.append("proportions must be a list or tuple")
    elif len(proportions) != n_levels:
        errors.append(f"Number of proportions ({len(proportions)}) must match n_levels ({n_levels})")
    else:
        # Check individual proportions
        for i, prop in enumerate(proportions):
            if not isinstance(prop, (int, float)):
                errors.append(f"Proportion {i + 1} must be numeric")
            elif prop < 0:
                errors.append(f"Proportion {i + 1} cannot be negative")
            elif prop == 0:
                warnings.append(f"Proportion {i + 1} is zero - level will never appear")

        # Check if they sum to approximately 1
        if not errors:  # Only if no errors with individual proportions
            total = sum(proportions)
            if abs(total - 1.0) > 1e-6:
                warnings.append(f"Proportions sum to {total:.4f}, not 1.0 (will be normalized)")

    # Warn about many levels
    if n_levels > 10:
        warnings.append(f"Factor has {n_levels} levels. This creates {n_levels - 1} dummy variables, which may require large sample sizes")

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_upload_data(data: np.ndarray) -> _ValidationResult:
    """Validate uploaded data array after normalization.

    Checks:
        - Array is 2-dimensional.
        - At least 25 samples (rows) for reliable quantile matching.
    """
    errors = []

    if data.ndim != 2:
        errors.append("data must be 2-dimensional (samples x variables)")
        return _ValidationResult(False, errors, [])

    if data.shape[0] < 25:
        errors.append(f"Need at least 25 samples for reliable quantile matching, got {data.shape[0]}")

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_cluster_config(
    grouping_var: str,
    icc: float,
    n_clusters: Optional[int],
    cluster_size: Optional[int],
    parsed_grouping_vars: List[str],
    nested_child: bool = False,
) -> _ValidationResult:
    """Validate cluster configuration parameters."""
    errors: List[str] = []
    warnings: List[str] = []

    # Grouping var must be in formula
    if grouping_var not in parsed_grouping_vars:
        errors.append(
            f"Grouping variable '{grouping_var}' not found in formula random effects. Expected one of: {', '.join(parsed_grouping_vars)}"
        )

    # ICC range - strict validation for numerical stability
    if not isinstance(icc, (int, float)):
        errors.append("ICC must be a number")
    elif icc < 0 or icc >= 1:
        errors.append(f"ICC must be between 0 and 1 (exclusive on upper end), got {icc}")
    elif icc != 0 and (icc < 0.1 or icc > 0.9):
        errors.append(
            f"ICC must be 0 (no clustering) or between 0.1 and 0.9 for numerical stability. "
            f"Got {icc}. Extreme ICC values (< 0.1 or > 0.9) cause convergence issues in mixed models."
        )

    # For nested child terms, n_clusters/cluster_size are derived from parent
    if not nested_child:
        # Mutual exclusivity
        if n_clusters is not None and cluster_size is not None:
            errors.append("Specify either n_clusters OR cluster_size, not both")
        elif n_clusters is None and cluster_size is None:
            errors.append("Must specify either n_clusters or cluster_size")

        # n_clusters validation
        if n_clusters is not None:
            if not isinstance(n_clusters, int) or n_clusters < 2:
                errors.append(f"n_clusters must be an integer >= 2, got {n_clusters}")

        # cluster_size validation
        if cluster_size is not None:
            if not isinstance(cluster_size, int) or cluster_size < 15:
                errors.append(
                    f"cluster_size must be an integer >= 15 for reliable mixed model estimation. "
                    f"Got {cluster_size}. Small cluster sizes cause convergence issues."
                )

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_cluster_sample_size(
    sample_size: int,
    n_clusters: int,
    cluster_size: Optional[int],
) -> _ValidationResult:
    """
    Validate that cluster configuration with sample_size provides sufficient observations.

    Args:
        sample_size: Total number of observations
        n_clusters: Number of clusters
        cluster_size: Size per cluster (if fixed), or None (computed from sample_size)

    Returns:
        ValidationResult with errors if observations per cluster < 15
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Compute effective cluster size
    if cluster_size is not None:
        effective_cluster_size = cluster_size
    else:
        effective_cluster_size = sample_size // n_clusters

    # Check minimum observations per cluster
    if effective_cluster_size < 25:
        errors.append(
            f"Insufficient observations per cluster: {effective_cluster_size} (sample_size={sample_size}, "
            f"n_clusters={n_clusters}). Need at least 25 observations per cluster for reliable "
            f"mixed model estimation. Either increase sample_size to {n_clusters * 25} or reduce n_clusters to {sample_size // 25}."
        )

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_lme_model_complexity(
    sample_size: int,
    n_clusters: int,
    n_fixed_effects: int,
    cluster_size: Optional[int] = None,
) -> _ValidationResult:
    """
    Validate cluster configuration provides sufficient data for model complexity.

    Linear Mixed-Effects models require adequate observations per cluster relative
    to parameters being estimated. Conservative guideline: 10 observations per parameter.

    Parameters estimated:
    - Fixed effects (including intercept)
    - Random effect variance (1 per random intercept)
    - Residual variance

    Args:
        sample_size: Total number of observations
        n_clusters: Number of clusters
        n_fixed_effects: Number of fixed effects (excluding intercept)
        cluster_size: Size per cluster (if fixed), computed if None

    Returns:
        ValidationResult with warnings or errors
    """
    errors = []
    warnings = []

    # Compute effective cluster size
    if cluster_size is not None:
        effective_cluster_size = cluster_size
    else:
        effective_cluster_size = sample_size // n_clusters

    # Estimate total parameters: intercept + fixed effects + random variance + residual variance
    n_total_params = 1 + n_fixed_effects + 2

    # Compute observations per parameter ratio
    obs_per_param = effective_cluster_size / n_total_params

    # Conservative threshold: 10 observations per parameter
    # Warning threshold: 7 observations per parameter
    if obs_per_param < 7:
        errors.append(
            f"Insufficient observations per parameter for mixed model: {obs_per_param:.1f} "
            f"(cluster_size={effective_cluster_size}, parameters={n_total_params}). "
            f"Need at least {10 * n_total_params} observations per cluster (10 per parameter). "
            f"Either increase sample_size to {n_clusters * 10 * n_total_params} or reduce model complexity."
        )
    elif obs_per_param < 10:
        warnings.append(
            f"Low observations per parameter ratio: {obs_per_param:.1f} "
            f"(cluster_size={effective_cluster_size}, parameters={n_total_params}). "
            f"Recommended: at least {10 * n_total_params} observations per cluster for reliable estimation. "
            f"This may cause convergence issues in mixed models."
        )

    return _ValidationResult(len(errors) == 0, errors, warnings)
