"""Validation utilities for model inputs, parameters, and mathematical constraints."""

import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

__all__ = [
    "_validate_cluster_config",
    "_validate_cluster_sample_size",
    "_validate_estimator",
]


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

    @classmethod
    def from_errors(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "_ValidationResult":
        """Create a result from error/warning lists, deriving ``is_valid`` automatically."""
        return cls(len(errors) == 0, errors, warnings or [])

    def raise_if_invalid(self):
        """Raise ``ValueError`` if the validation failed."""
        if not self.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(f"• {err}" for err in self.errors)
            raise ValueError(error_msg)

    def raise_or_warn(self, stacklevel: int = 3):
        """Raise on errors; otherwise emit any soft warnings as ``UserWarning``.

        ``raise_if_invalid`` alone drops ``warnings`` on the floor, so setters
        that carry non-fatal guidance (alpha/baseline-p bands) call this instead.
        ``stacklevel=3`` points the warning at the user's setter call."""
        self.raise_if_invalid()
        for w in self.warnings:
            warnings.warn(w, UserWarning, stacklevel=stacklevel)


def _cluster_limits():
    from ..config import get_cluster_limits
    return get_cluster_limits()


def _check_type(value: Any, expected_types: tuple, name: str) -> Optional[str]:
    if not isinstance(value, expected_types):
        actual_type = type(value).__name__
        expected = expected_types[0].__name__ if len(expected_types) == 1 else f"one of {[t.__name__ for t in expected_types]}"
        return f"{name} must be {expected}, got {actual_type}"
    return None


def _check_range(
    value: Union[int, float],
    min_val: Optional[float],
    max_val: Optional[float],
    name: str,
) -> Optional[str]:
    if min_val is not None and value < min_val:
        return f"{name} must be >= {min_val}, got {value}"
    if max_val is not None and value > max_val:
        return f"{name} must be <= {max_val}, got {value}"
    return None


def _validate_numeric_parameter(
    value: Any,
    name: str,
    expected_types: tuple = (int, float),
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_rounding: bool = False,
) -> _ValidationResult:
    """Type failure short-circuits range and rounding checks — returning early
    avoids confusing cascading errors on a value of the wrong type."""
    errors: List[str] = []
    warnings: List[str] = []

    type_error = _check_type(value, expected_types, name)
    if type_error:
        errors.append(type_error)
        return _ValidationResult(False, errors, warnings)

    range_error = _check_range(value, min_val, max_val, name)
    if range_error:
        errors.append(range_error)

    if allow_rounding and isinstance(value, float) and int in expected_types:
        rounded = int(round(value))
        if value != rounded:
            warnings.append(f"{name} rounded from {value} to {rounded}")

    return _ValidationResult.from_errors(errors, warnings)


def _validate_power(power: Any) -> _ValidationResult:
    """Validate power parameter (0-100%)."""
    return _validate_numeric_parameter(power, "Power", min_val=0, max_val=100)


def _validate_alpha(alpha: Any) -> _ValidationResult:
    """Validate alpha is numeric. The hard ``(0, 1)`` range is enforced by the
    engine (contract ``invariant_15``) at run, so no range reject lives here; a
    soft warning fires above ``max_alpha`` because power at such a high
    significance level is rarely meaningful."""
    from ..config import get_limits
    res = _validate_numeric_parameter(alpha, "Alpha")
    if res.is_valid and isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        max_alpha = get_limits()["max_alpha"]
        if alpha > max_alpha:
            res.warnings.append(
                f"Alpha = {alpha} is above the usual maximum of {max_alpha}; power at "
                f"such a high significance level is rarely meaningful."
            )
    return res


def _validate_simulations(n_simulations: Any) -> Tuple[int, _ValidationResult]:
    """Validate and process number of simulations."""
    result = _validate_numeric_parameter(n_simulations, "Number of simulations", min_val=1, allow_rounding=True)

    if result.is_valid:
        rounded = int(round(n_simulations))
        # 800 simulations threshold: below this, Monte Carlo standard error
        # exceeds ~1.5% for power near 50%, reducing result reliability.
        if rounded < 800:
            result.warnings.append(f"Low simulation count ({rounded}). Consider using at least 1000 for reliable results.")
        return rounded, result

    return 0, result


def _validate_sample_size(sample_size: Any) -> _ValidationResult:
    """Validate sample size parameter.

    Requires an integer >= 20 and <= 100,000.
    """
    errors = []

    if not isinstance(sample_size, int):
        errors.append(f"sample_size must be an integer, got {type(sample_size).__name__}")
        return _ValidationResult(False, errors, [])

    if sample_size < 20:
        errors.append(f"sample_size must be at least 20, got {sample_size}")
    elif sample_size > 100000:
        errors.append(
            f"sample_size too large ({sample_size:,}). Maximum recommended: 100,000. We cannot guarantee stability for such small p-values."
        )

    return _ValidationResult.from_errors(errors)


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
    # Green's rule of thumb: N >= 15 + p for adequate power in regression,
    # where p is the number of predictors (design matrix columns).
    min_required = 15 + n_variables

    if sample_size < min_required:
        errors.append(
            f"sample_size ({sample_size}) is too small for a model with {n_variables} "
            f"variables. Minimum required: {min_required} (15 + {n_variables} variables)."
        )

    return _ValidationResult.from_errors(errors)


def _validate_sample_size_range(from_size: Any, to_size: Any, by: Any) -> _ValidationResult:
    """Validate sample size range parameters."""
    errors: List[str] = []
    warnings: List[str] = []

    # `by` accepts the literal "auto" (engine auto-counts the grid)
    # in addition to a positive integer step/point-count.
    for param, name in [(from_size, "from_size"), (to_size, "to_size")]:
        if not isinstance(param, int) or param <= 0:
            errors.append(f"{name} must be a positive integer, got {param}")
    if not (by == "auto" or (isinstance(by, int) and by > 0)):
        errors.append(f"by must be a positive integer or \"auto\", got {by!r}")

    if errors:
        return _ValidationResult(False, errors, warnings)

    if from_size >= to_size:
        errors.append(f"from_size ({from_size}) must be less than to_size ({to_size})")

    if by != "auto" and by > (to_size - from_size):
        errors.append(f"Step size 'by' ({by}) is larger than range ({to_size - from_size}). This will only test one sample size.")

    if by != "auto":
        n_tests = len(range(from_size, to_size + 1, by))
        if n_tests > 100:
            warnings.append(f"Large number of sample sizes to test ({n_tests}). This may take significant time.")

    return _ValidationResult.from_errors(errors, warnings)


def _validate_correlation_matrix(
    corr_matrix: Optional[List[List[float]]],
) -> _ValidationResult:
    """Validate the structural matrix properties the wire format cannot preserve.

    Only the upper-triangle off-diagonal pairs cross the FFI boundary, so an
    asymmetric or non-unit-diagonal full matrix would otherwise be silently
    coerced to its upper triangle. These guards keep malformed full-matrix
    input erroring loudly. Range (``|r| <= 1``) and positive semi-definiteness
    are enforced downstream by the engine — a non-PSD input matrix is rejected
    at run entry, and any off-diagonal ``|r| > 1`` makes the matrix non-PSD —
    so they are not duplicated here.
    """
    errors = []

    if corr_matrix is None:
        errors.append("Correlation matrix is None")
        return _ValidationResult(False, errors, [])

    n = len(corr_matrix)
    if any(len(row) != n for row in corr_matrix):
        errors.append("Correlation matrix must be square")
        return _ValidationResult(False, errors, [])

    # The former allclose checks (atol=1e-8, rtol=1e-5) gave an effective
    # tolerance ~1e-5 for values near 1; tol=1e-8 would be ~1000x stricter and
    # newly reject ~1e-7 floating-point asymmetry the previous version accepted.
    tol = 1e-5
    if any(abs(corr_matrix[i][i] - 1.0) > tol for i in range(n)):
        errors.append("Diagonal elements of correlation matrix must be 1")

    if any(
        abs(corr_matrix[i][j] - corr_matrix[j][i]) > tol
        for i in range(n)
        for j in range(i + 1, n)
    ):
        errors.append("Correlation matrix must be symmetric")

    return _ValidationResult.from_errors(errors)


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


def _validate_family(family: Any) -> _ValidationResult:
    """Validate the GLM family kwarg.

    Accepts "ols", "logit", and "lme" (mixed-effects: random intercepts,
    random slopes, crossed/nested groupings, and cluster-level predictors,
    all configured at ``set_cluster`` time).
    """
    if not isinstance(family, str):
        return _ValidationResult(
            False,
            [f"family must be a string, got {type(family).__name__}"],
            [],
        )
    if family.lower() not in ("ols", "logit", "lme"):
        return _ValidationResult(
            False,
            [f"family must be 'ols', 'logit', or 'lme', got {family!r}"],
            [],
        )
    return _ValidationResult(True, [], [])


def _validate_estimator(estimator: Any) -> _ValidationResult:
    """Validate the optional estimator override kwarg.

    ``None`` means "derive the default from outcome_kind + cluster presence".
    The three accepted strings are the canonical wire names.
    """
    if estimator is None:
        return _ValidationResult(True, [], [])
    if not isinstance(estimator, str):
        return _ValidationResult(
            False,
            [f"estimator must be a string or None, got {type(estimator).__name__}"],
            [],
        )
    if estimator.lower() not in ("ols", "glm", "mle"):
        return _ValidationResult(
            False,
            [f"estimator must be 'ols', 'glm', 'mle', or None, got {estimator!r}"],
            [],
        )
    return _ValidationResult(True, [], [])


def _validate_baseline_probability(p: Any) -> _ValidationResult:
    """Validate baseline probability for family='logit'.

    Errors:
        - p not numeric
        - p outside the open interval (0, 1)

    Warnings:
        - p outside ``limits.baseline_p_warn`` — extreme baselines lead to
          near-separation and unstable power estimates.
    """
    from ..config import get_limits
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(p, (int, float)) or isinstance(p, bool):
        errors.append(f"baseline probability must be a number, got {type(p).__name__}")
        return _ValidationResult(False, errors, warnings)

    pv = float(p)
    if not (0.0 < pv < 1.0):
        errors.append(f"baseline probability must be in the open interval (0, 1), got {pv}")
        return _ValidationResult(False, errors, warnings)

    lo, hi = get_limits()["baseline_p_warn"]
    if pv < lo or pv > hi:
        warnings.append(f"baseline probability {pv} is extreme (outside [{lo}, {hi}]); expect near-separation and unstable power estimates")

    return _ValidationResult.from_errors(errors, warnings)


def _warn_logit_effect_scale(parsed_effects: dict, registry) -> List[str]:
    """Emit warnings for likely scale-mismatched logit effect inputs.

    Fires for any continuous-predictor effect with |β| > 3, suggesting a
    raw-scale OR may have been pasted instead of a standardized log-odds.
    Also fires for |β| > 5 as a strict superset.

    Args:
        parsed_effects: mapping ``effect_name -> float`` for effects in
            the current ``set_effects`` call.
        registry: ``VariableRegistry`` (used to identify continuous predictors).

    Returns:
        List of warning strings (empty when nothing to warn about).
    """
    warnings: List[str] = []

    # We can't always know var_type at set_effects time — _apply_variable_types
    # has not yet run. Default to treating predictors as continuous unless we
    # KNOW they are binary or factor. This matches the registry default
    # (var_type → "normal" after apply) and keeps the warning useful for the
    # common case where the user has not called set_variable_type.
    non_continuous_types = {"binary", "factor", "uploaded_binary", "uploaded_factor"}
    non_continuous_names: set = set()
    for name in registry.predictor_names:
        pred = registry.get_predictor(name)
        vt = getattr(pred, "var_type", None) if pred is not None else None
        if vt in non_continuous_types:
            non_continuous_names.add(name)

    def _all_continuous(effect_name: str) -> bool:
        return all(part.strip() not in non_continuous_names for part in effect_name.split(":"))

    for ename, beta in parsed_effects.items():
        if not isinstance(beta, (int, float)):
            continue
        if abs(beta) > 5.0:
            warnings.append(f"effect {ename}={beta} has |β|>5 (OR>~150); check for input error")
            continue  # don't double-warn
        # Single-predictor scale-mismatch warning fires for continuous mains
        # and for any interaction whose components are all continuous.
        if _all_continuous(ename) and abs(beta) > 3.0:
            warnings.append(
                f"effect {ename}={beta} has |β|>3 (OR>~20) on a standardized "
                f"continuous predictor — likely a raw-scale value pasted from R "
                f"or G*Power. Conversion: β_mcpower = log(OR) * sd(x_raw)"
            )

    return warnings


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
        # Unicode-aware identifier extraction
        from mcpower.spec.parsers import _IDENT

        variables_in_formula = set(re.findall(_IDENT, test_formula))

        if not variables_in_formula:
            errors.append(f"No variables found in test_formula: '{test_formula}'")
            return _ValidationResult(False, errors, [])

        missing_vars = variables_in_formula - set(available_variables)
        if missing_vars:
            missing_str = ", ".join(sorted(missing_vars))
            errors.append(
                f"test_formula references terms missing from the model formula: {missing_str}. "
                f"Every term you test must also exist in the model formula — add {missing_str} "
                f"to the formula and give it an effect (use 0 if you don't want it to influence "
                f"the generated data). Available terms: {', '.join(available_variables)}"
            )

        return _ValidationResult.from_errors(errors)

    except Exception as e:
        errors.append(f"Error parsing test_formula: {str(e)}")
        return _ValidationResult(False, errors, [])


def _validate_upload_data(
    columns_data: List[List[Any]], columns: Optional[List[str]] = None
) -> _ValidationResult:
    """Validate uploaded data (column-major) after normalization.

    Checks:
        - Row count within ``[upload.min_rows, upload.max_rows]``.
        - No NaN or Inf values in numeric columns (string columns skipped).
    """
    from ..config import get_upload
    errors = []

    upload = get_upload()
    min_rows, max_rows = upload["min_rows"], upload["max_rows"]
    n_rows = len(columns_data[0]) if columns_data else 0
    if n_rows < min_rows:
        errors.append(f"Need at least {min_rows} samples for reliable quantile matching, got {n_rows}")
    elif n_rows > max_rows:
        errors.append(f"Uploaded data has too many rows ({n_rows:,}); the maximum is {max_rows:,}.")

    # NaN/Inf check on numeric columns. Non-numeric (string) columns can't cast
    # to float, so they are detected and skipped column-by-column.
    nan_cols: List[str] = []
    inf_cols: List[str] = []
    for ci, col in enumerate(columns_data):
        floats: List[float] = []
        numeric = True
        for v in col:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                numeric = False
                break
        if not numeric:
            continue
        col_name = columns[ci] if columns is not None and ci < len(columns) else f"column_{ci + 1}"
        if any(math.isnan(x) for x in floats):
            nan_cols.append(col_name)
        if any(math.isinf(x) for x in floats):
            inf_cols.append(col_name)
    if nan_cols:
        errors.append(
            f"Uploaded data contains NaN values in columns: {', '.join(nan_cols)}. Remove or impute missing values before uploading."
        )
    if inf_cols:
        errors.append(
            f"Uploaded data contains Inf values in columns: {', '.join(inf_cols)}. Remove or replace infinite values before uploading."
        )

    return _ValidationResult.from_errors(errors)


def _validate_cluster_config(
    grouping_var: str,
    icc: float,
    n_clusters: Optional[int],
    cluster_size: Optional[int],
    parsed_grouping_vars: List[str],
) -> _ValidationResult:
    """Validate cluster configuration parameters.

    Random slopes, crossed/nested groupings, and cluster-level predictors are
    all supported and configured through ``set_cluster``.
    """
    from ..config import get_limits
    errors: List[str] = []
    warnings: List[str] = []

    if grouping_var not in parsed_grouping_vars:
        errors.append(
            f"Grouping variable '{grouping_var}' not found in formula random effects. Expected one of: {', '.join(parsed_grouping_vars)}"
        )

    # Strict ICC range for numerical stability
    icc_lo, icc_hi = get_limits()["icc_stability"]
    if not isinstance(icc, (int, float)):
        errors.append("ICC must be a number")
    elif icc < 0 or icc >= 1:
        errors.append(f"ICC must be between 0 and 1 (exclusive on upper end), got {icc}")
    elif icc != 0 and (icc < icc_lo or icc > icc_hi):
        errors.append(
            f"ICC must be 0 (no clustering) or between {icc_lo} and {icc_hi} for numerical stability. "
            f"Got {icc}. Extreme ICC values (< {icc_lo} or > {icc_hi}) cause convergence issues in mixed models."
        )

    if n_clusters is not None and cluster_size is not None:
        errors.append("Specify either n_clusters OR cluster_size, not both")
    elif n_clusters is None and cluster_size is None:
        errors.append("Must specify either n_clusters or cluster_size")

    if n_clusters is not None:
        if not isinstance(n_clusters, int) or n_clusters < 2:
            errors.append(f"n_clusters must be an integer >= 2, got {n_clusters}")

    if cluster_size is not None:
        reliable = _cluster_limits()["reliable_rows_per_cluster"]
        if not isinstance(cluster_size, int) or cluster_size < reliable:
            errors.append(f"cluster_size must be an integer >= {reliable} for reliable mixed model estimation. Got {cluster_size}.")

    return _ValidationResult.from_errors(errors, warnings)


def _validate_cluster_sample_size(
    sample_size: int,
    n_clusters: Optional[int],
    cluster_size: Optional[int],
) -> _ValidationResult:
    """
    Validate that cluster configuration with sample_size provides sufficient observations.

    Args:
        sample_size: Total number of observations
        n_clusters: Number of clusters
        cluster_size: Size per cluster (if fixed), or None (computed from sample_size)

    Returns:
        ValidationResult with errors if observations per cluster < 5
    """
    errors: List[str] = []
    warnings: List[str] = []

    if cluster_size is not None:
        effective_cluster_size = cluster_size
    else:
        effective_cluster_size = sample_size // n_clusters

    limits = _cluster_limits()
    reliable = limits["reliable_rows_per_cluster"]
    recommended = limits["recommended_rows_per_cluster"]
    if effective_cluster_size < reliable:
        hint = (
            f" Either increase sample_size to {n_clusters * reliable} or reduce n_clusters to {sample_size // reliable}."
            if n_clusters is not None
            else f" Increase cluster_size to at least {reliable}."
        )
        errors.append(
            f"Insufficient observations per cluster: {effective_cluster_size} (sample_size={sample_size}, "
            f"n_clusters={n_clusters}). Need at least {reliable} observations per cluster for reliable "
            f"mixed model estimation.{hint}"
        )
    elif effective_cluster_size < recommended:
        warnings.append(
            f"Low observations per cluster: {effective_cluster_size} (sample_size={sample_size}, "
            f"n_clusters={n_clusters}). Recommended: at least {recommended} observations per cluster. "
            f"Small cluster sizes may cause convergence issues or biased variance estimates."
        )

    return _ValidationResult.from_errors(errors, warnings)
