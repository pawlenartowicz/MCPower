"""
Validation utilities for Monte Carlo Power Analysis.

This module provides validation functions for model inputs, parameters,
and mathematical constraints.
"""

import numpy as np
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

__all__ = []


@dataclass
class _ValidationResult:
    """Result of validation with errors and warnings."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self):
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(
                f"• {err}" for err in self.errors
            )
            raise ValueError(error_msg)


class _Validator:
    """Base validator with common functionality."""

    @staticmethod
    def _check_type(value: Any, expected_types: tuple, name: str) -> Optional[str]:
        """Check if value has expected type."""
        if not isinstance(value, expected_types):
            actual_type = type(value).__name__
            expected = (
                expected_types[0].__name__
                if len(expected_types) == 1
                else f"one of {[t.__name__ for t in expected_types]}"
            )
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
    errors = []
    warnings = []

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
    result = _validate_numeric_parameter(
        n_simulations, "Number of simulations", min_val=1, allow_rounding=True
    )

    if result.is_valid:
        rounded = int(round(n_simulations))
        if rounded < 1000:
            result.warnings.append(
                f"Low simulation count ({rounded}). Consider using at least 1000 for reliable results."
            )
        return rounded, result

    return 0, result


def _validate_sample_size(sample_size: Any) -> _ValidationResult:
    """Validate sample size parameter."""
    errors = []

    # Must be integer
    if not isinstance(sample_size, int):
        errors.append(
            f"sample_size must be an integer, got {type(sample_size).__name__}"
        )
        return _ValidationResult(False, errors, [])

    # Range check
    if sample_size <= 0:
        errors.append(f"sample_size must be positive, got {sample_size}")
    elif sample_size > 100000:
        errors.append(
            f"sample_size too large ({sample_size:,}). Maximum recommended: 100,000. We cannot guarantee stability for such small p-values."
        )

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_sample_size_range(
    from_size: Any, to_size: Any, by: Any
) -> _ValidationResult:
    """Validate sample size range parameters."""
    errors = []
    warnings = []

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
        errors.append(
            f"Step size 'by' ({by}) is larger than range ({to_size - from_size}). "
            "This will only test one sample size."
        )

    # Warning for many tests
    n_tests = len(range(from_size, to_size + 1, by))
    if n_tests > 100:
        warnings.append(
            f"Large number of sample sizes to test ({n_tests}). This may take significant time."
        )

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
            errors.append(f"Correlation matrix must be positive semi-definite. ")
    except np.linalg.LinAlgError:
        errors.append("Cannot compute eigenvalues of correlation matrix")

    return _ValidationResult(len(errors) == 0, errors, [])


def _validate_correction_method(correction: Optional[str]) -> _ValidationResult:
    """Validate correction method name."""
    if correction is None:
        return _ValidationResult(True, [], [])

    method = correction.lower().replace("-", "_").replace(" ", "_")
    valid_methods = ["bonferroni", "benjamini_hochberg", "bh", "fdr", "holm"]

    if method not in valid_methods:
        return _ValidationResult(
            False,
            [
                f"Unknown correction method: {correction}. "
                f"Valid options: 'Bonferroni', 'Benjamini-Hochberg' (or 'BH', 'FDR'), 'Holm'"
            ],
            [],
        )

    return _ValidationResult(True, [], [])


def _validate_parallel_settings(
    enable: Any, n_cores: Optional[int]
) -> Tuple[Tuple[bool, int], _ValidationResult]:
    """Validate parallel processing settings."""
    import multiprocessing as mp

    errors = []

    # Validate enable
    if not isinstance(enable, bool):
        errors.append(f"enable must be True or False, got {type(enable).__name__}")
        return (False, 1), _ValidationResult(False, errors, [])

    # Validate n_cores
    max_cores = mp.cpu_count()
    validated_n_cores = max(1, max_cores - 1)

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
    errors = []
    warnings = []

    # Check effect sizes
    if not hasattr(model, "effect_sizes_initiated") or not model.effect_sizes_initiated:
        if hasattr(model, "effects"):
            available = [info["name"] for info in model.effects.values()]
            errors.append(
                f"Effect sizes must be set using set_effects() before running analysis. "
                f"Available effects: {', '.join(available)}"
            )
        else:
            errors.append("Effect sizes must be set before running analysis")

    # Check other required attributes
    required_attrs = ["power", "alpha", "n_simulations"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            errors.append(f"Model missing required attribute: {attr}")

    return _ValidationResult(len(errors) == 0, errors, warnings)


def _validate_test_formula(
    test_formula: str, available_variables: List[str]
) -> _ValidationResult:
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
                f"Variables not found in original model: {', '.join(sorted(missing_vars))}. "
                f"Available: {', '.join(available_variables)}"
            )

        return _ValidationResult(len(errors) == 0, errors, [])

    except Exception as e:
        errors.append(f"Error parsing test_formula: {str(e)}")
        return _ValidationResult(False, errors, [])
