"""
Data Generator for Monte Carlo Power Analysis

OVERVIEW:
=========
Generates synthetic datasets with specified:
- Distributions (normal, binary, skewed, etc.)
- Correlation structures
- Variable transformations
- Uploaded data structure preservation

OPTIMIZATION STRATEGY:
=====================
Three-tier performance system:
1. AOT compiled (fastest) - pre-compiled native code
2. JIT compiled (fast) - runtime compilation via numba
3. Pure Python (slowest) - fallback when numba unavailable

CORE CONCEPT:
============
1. Generate correlated normal data using Cholesky decomposition
2. Transform to target distributions via lookup tables
3. Use quantile matching for uploaded data structure preservation

LOOKUP TABLES:
=============
Pre-computed CDF/PPF tables avoid repeated scipy calls:
- NORM_CDF_TABLE: Normal CDF for distribution transforms
- T3_PPF_TABLE: t distributed with df = 3 quantiles for heavy-tailed distributions
"""

import os
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

# Distribution resolution and ranges
DIST_RESOLUTION = 2048  # Lookup table size (accuracy vs memory)
PERCENTILE_RANGE = (0.001, 0.999)  # Avoid extreme quantiles
NORM_RANGE = (-6, 6)  # Normal distribution range
SQRT3 = np.sqrt(3)
SKEW_MEAN = np.exp(0.5)  # Lognormal standardization
SKEW_STD = np.sqrt(np.exp(2) - np.exp(1))
NORM_SCALE = (DIST_RESOLUTION - 1) / (NORM_RANGE[1] - NORM_RANGE[0])
PERC_SCALE = (DIST_RESOLUTION - 1) / (PERCENTILE_RANGE[1] - PERCENTILE_RANGE[0])
FLOAT_NEAR_ZERO = 1e-15

# Global lookup tables (initialized on import)
NORM_CDF_TABLE = None
T3_PPF_TABLE = None


def _init_tables():
    """Initialize/load cached distribution lookup tables for fast transforms."""
    global NORM_CDF_TABLE, T3_PPF_TABLE

    cache_file = os.path.join(os.path.dirname(__file__), ".generator_lookup_tables.pkl")

    try:
        with open(cache_file, "rb") as f:
            NORM_CDF_TABLE, T3_PPF_TABLE = pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        from scipy.stats import norm, t

        # Build lookup tables
        x_norm = np.linspace(*NORM_RANGE, DIST_RESOLUTION)
        NORM_CDF_TABLE = norm.cdf(x_norm)

        percentile_points = np.linspace(*PERCENTILE_RANGE, DIST_RESOLUTION)
        T3_PPF_TABLE = t.ppf(percentile_points, 3) / SQRT3

        # Cache for future use
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((NORM_CDF_TABLE, T3_PPF_TABLE), f)
        except:
            pass


_init_tables()

# AOT compilation setup (build-time only)
if os.environ.get("NUMBA_AOT_BUILD"):
    try:
        from numba.pycc import CC

        cc = CC("data_generation_compiled")
        compile_function = lambda sig: lambda func: cc.export(func.__name__, sig)(func)  # type: ignore
    except ImportError as e:
        print(f"Warning: AOT compilation not available: {e}")
        cc = None
        compile_function = None
else:
    cc = None
    compile_function = None


def create_uploaded_lookup_tables(
    data_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create quantile-matching tables for uploaded data variables.

    Maps normal quantiles to empirical data quantiles, preserving
    the original distribution shape when generating new samples.

    Args:
        data_matrix: (n_samples, n_vars) empirical data

    Returns:
        normal_values: (n_vars, n_samples) normal quantiles
        uploaded_values: (n_vars, n_samples) sorted empirical values
    """
    from scipy.stats import norm

    n_samples, n_vars = data_matrix.shape
    normal_values = np.zeros((n_vars, n_samples))
    uploaded_values = np.zeros((n_vars, n_samples))

    for var_idx in range(n_vars):
        data = data_matrix[:, var_idx]
        normalized = (data - np.mean(data)) / np.std(data)
        sorted_uploaded = np.sort(normalized)

        # Map to uniform then to normal quantiles
        percentiles = np.linspace(
            1 / (n_samples + 1), n_samples / (n_samples + 1), n_samples
        )
        normal_quantiles = norm.ppf(percentiles)

        normal_values[var_idx] = normal_quantiles
        uploaded_values[var_idx] = sorted_uploaded

    return normal_values, uploaded_values


def _generate_X_core(
    sample_size,
    n_vars,
    correlation_matrix,
    var_types,
    var_params,
    norm_cdf_table,
    t3_ppf_table,
    upload_normal_values,
    upload_data_values,
    sim_seed,
):
    """
    Core data generation: correlated normal → distribution transforms.

    ALGORITHM:
    1. Generate multivariate normal with specified correlations
    2. Transform each variable to target distribution via lookup tables
    3. Handle uploaded data via quantile matching

    Distribution types (var_types):
    0=normal, 1=binary, 2=right_skewed, 3=left_skewed,
    4=high_kurtosis, 5=uniform, 99=uploaded_data
    """

    def _vectorized_norm_cdf_lookup(x_array):
        """Fast normal CDF via linear interpolation in lookup table."""
        n = len(x_array)
        result = np.zeros(n)

        for i in range(n):
            x = x_array[i]
            if x < NORM_RANGE[0]:
                result[i] = 0.0
            elif x > NORM_RANGE[1]:
                result[i] = 1.0
            else:
                idx = (x - NORM_RANGE[0]) * NORM_SCALE
                idx_int = int(idx)

                if idx_int >= DIST_RESOLUTION - 1:
                    result[i] = norm_cdf_table[DIST_RESOLUTION - 1]
                else:
                    frac = idx - idx_int
                    result[i] = (
                        norm_cdf_table[idx_int] * (1 - frac)
                        + norm_cdf_table[idx_int + 1] * frac
                    )
        return result

    def _vectorized_t3_ppf_lookup(percentile_array):
        """Fast t(3) quantiles for heavy-tailed distributions."""
        n = len(percentile_array)
        result = np.zeros(n)

        for i in range(n):
            percentile = percentile_array[i]
            if percentile <= PERCENTILE_RANGE[0]:
                result[i] = t3_ppf_table[0]
            elif percentile >= PERCENTILE_RANGE[1]:
                result[i] = t3_ppf_table[DIST_RESOLUTION - 1]
            else:
                idx = (percentile - PERCENTILE_RANGE[0]) * PERC_SCALE
                idx_int = int(idx)

                if idx_int >= DIST_RESOLUTION - 1:
                    result[i] = t3_ppf_table[DIST_RESOLUTION - 1]
                else:
                    frac = idx - idx_int
                    result[i] = (
                        t3_ppf_table[idx_int] * (1 - frac)
                        + t3_ppf_table[idx_int + 1] * frac
                    )
        return result

    def _vectorized_uploaded_lookup(normal_array, normal_vals, uploaded_vals):
        """Map normal values to uploaded data via binary search + interpolation."""
        n_samples = len(normal_array)
        n_lookup = len(normal_vals)
        result = np.zeros(n_samples)

        for i in range(n_samples):
            normal_value = normal_array[i]

            if normal_value <= normal_vals[0]:
                result[i] = uploaded_vals[0]
            elif normal_value >= normal_vals[-1]:
                result[i] = uploaded_vals[-1]
            else:
                # Binary search for bracketing values
                left, right = 0, n_lookup - 1
                while left < right - 1:
                    mid = (left + right) // 2
                    if normal_vals[mid] <= normal_value:
                        left = mid
                    else:
                        right = mid

                # Linear interpolation
                frac = (normal_value - normal_vals[left]) / (
                    normal_vals[right] - normal_vals[left]
                )
                result[i] = (
                    uploaded_vals[left] * (1 - frac) + uploaded_vals[right] * frac
                )

        return result

    def _cholesky_decomposition(corr_matrix):
        """Robust Cholesky with eigenvalue correction for near-singular matrices."""
        try:
            return np.linalg.cholesky(corr_matrix)
        except:
            # Fallback: eigenvalue decomposition with regularization
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            eigenvals = np.maximum(eigenvals, FLOAT_NEAR_ZERO)
            return eigenvecs @ np.diag(np.sqrt(eigenvals))

    def _transform_distribution(data, dist_type, param, var_idx):
        """Transform standard normal to target distribution."""
        if dist_type == 0:  # normal
            return data.copy()
        elif dist_type == 1:  # binary
            percentiles = _vectorized_norm_cdf_lookup(data)
            binary_data = (percentiles < param).astype(np.float64)
            return binary_data - np.mean(binary_data)  # Center at 0
        elif dist_type == 2:  # right_skewed (exponential-like)
            percentiles = _vectorized_norm_cdf_lookup(data)
            return (-np.log(percentiles) - SKEW_MEAN) / SKEW_STD
        elif dist_type == 3:  # left_skewed
            percentiles = _vectorized_norm_cdf_lookup(data)
            return (np.log(1 - percentiles) + SKEW_MEAN) / SKEW_STD
        elif dist_type == 4:  # high_kurtosis (t-distribution)
            percentiles = _vectorized_norm_cdf_lookup(data)
            return _vectorized_t3_ppf_lookup(percentiles)
        elif dist_type == 5:  # uniform
            percentiles = _vectorized_norm_cdf_lookup(data)
            return SQRT3 * (2 * percentiles - 1)  # Uniform[-√3, √3], var=1
        elif dist_type == 99:  # uploaded_data
            if var_idx < upload_normal_values.shape[0]:
                normal_vals = upload_normal_values[var_idx]
                uploaded_vals = upload_data_values[var_idx]
                return _vectorized_uploaded_lookup(data, normal_vals, uploaded_vals)
            else:
                return data.copy()
        else:
            return data.copy()

    # Main generation algorithm
    if sim_seed >= 0:
        np.random.seed(sim_seed)
    base_normal = np.random.standard_normal((sample_size, n_vars))

    # Apply correlation structure
    cholesky_matrix = _cholesky_decomposition(correlation_matrix)
    correlated_data = base_normal @ cholesky_matrix.T

    # Transform to target distributions
    X = np.zeros((sample_size, n_vars))
    for j in range(n_vars):
        X[:, j] = _transform_distribution(
            correlated_data[:, j], var_types[j], var_params[j], j
        )

    return X


# COMPILATION SETUP: AOT → JIT → Python fallback
if os.environ.get("NUMBA_AOT_BUILD"):
    if compile_function and cc:
        _generate_X_core = compile_function(
            "f8[:,:](i8, i8, f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:,:], f8[:,:], i8)"
        )(_generate_X_core)
        cc.compile()
    _generate_X_runtime = _generate_X_core
else:
    try:
        # Try AOT compiled version first
        from .data_generation_compiled import _generate_X_core as _generate_X_core_aot

        _generate_X_runtime = _generate_X_core_aot
        print("Using AOT data generation")
    except (ImportError, AttributeError):
        try:
            # Fall back to JIT compilation
            from numba import njit

            _generate_X_runtime = njit(
                "f8[:,:](i8, i8, f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:,:], f8[:,:], i8)",
                cache=True,
            )(_generate_X_core)
            print("Using JIT data generation")
        except ImportError:
            # Pure Python fallback
            _generate_X_runtime = _generate_X_core
            print("Using Python data generation")
            print("Install numba for faster performance: pip install numba")


def _generate_X(
    sample_size: int,
    n_vars: int,
    correlation_matrix: Optional[np.ndarray] = None,
    var_types: Optional[np.ndarray] = None,
    var_params: Optional[np.ndarray] = None,
    normal_values: Optional[np.ndarray] = None,
    uploaded_values: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate design matrix with specified distributions and correlations.

    Args:
        sample_size: Number of observations
        n_vars: Number of variables
        correlation_matrix: Variable correlations (default: identity)
        var_types: Distribution types per variable (default: all normal)
        var_params: Distribution parameters (e.g., binary proportions)
        normal_values, uploaded_values: Lookup tables for uploaded data
        seed: Random seed for reproducibility

    Returns:
        X: (sample_size, n_vars) design matrix
    """
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_vars)

    if normal_values is None:
        normal_values = np.zeros((2, 2))
    if uploaded_values is None:
        uploaded_values = np.zeros((2, 2))

    return _generate_X_runtime(
        sample_size,
        n_vars,  # type: ignore
        correlation_matrix,
        var_types,
        var_params,
        NORM_CDF_TABLE,
        T3_PPF_TABLE,
        normal_values,
        uploaded_values,
        seed if seed is not None else -1,
    )


def _generate_factors(sample_size, factor_specs, seed):
    """
    Generate factor variables as dummy variables for linear regression.

    Args:
        sample_size: Number of observations
        factor_specs: List of factor specifications, each containing:
                     {'n_levels': int, 'proportions': [float, ...]}
        seed: Random seed or None

    Returns:
        X_factors: (sample_size, total_dummies) array of binary values
                  For each factor with n levels, creates n-1 dummy variables
                  Level 1 is reference (all dummies = 0)
    """
    if seed is not None:
        np.random.seed(seed)

    if not factor_specs:
        return np.empty((sample_size, 0), dtype=float)

    # Generate all factor columns efficiently
    factor_columns = []

    for spec in factor_specs:
        n_levels = spec["n_levels"]
        proportions = spec["proportions"]

        # Generate categorical data
        factor_data = np.random.choice(n_levels, size=sample_size, p=proportions)

        # Vectorized dummy variable creation using one-hot encoding
        # Create identity matrix and index into it
        dummies = np.eye(n_levels, dtype=float)[factor_data]

        # Remove reference level (first column) and append remaining columns
        factor_columns.append(dummies[:, 1:])  # Skip level 0 (reference)

    # Concatenate all factor columns horizontally
    return (
        np.hstack(factor_columns)
        if factor_columns
        else np.empty((sample_size, 0), dtype=float)
    )
