"""
Data Generator for Monte Carlo Power Analysis.

Generates synthetic datasets with:
- Distributions (normal, binary, skewed, etc.)
- Correlation structures
- Uploaded data structure preservation

Performance: JIT compiled via numba, falls back to pure Python.
Future: C++ backend via pybind11 (see mcpower/backends/native.py)
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Distribution constants
DIST_RESOLUTION = 2048
PERCENTILE_RANGE = (0.001, 0.999)
NORM_RANGE = (-6, 6)
SQRT3 = np.sqrt(3)
SKEW_MEAN = np.exp(0.5)
SKEW_STD = np.sqrt(np.exp(2) - np.exp(1))
NORM_SCALE = (DIST_RESOLUTION - 1) / (NORM_RANGE[1] - NORM_RANGE[0])
PERC_SCALE = (DIST_RESOLUTION - 1) / (PERCENTILE_RANGE[1] - PERCENTILE_RANGE[0])
FLOAT_NEAR_ZERO = 1e-15

# Global lookup tables
NORM_CDF_TABLE = None
T3_PPF_TABLE = None
T3_SD = 1.0  # Effective SD of t(3) values from the lookup pipeline


def _init_tables():
    """Initialize distribution lookup tables from the shared LookupTableManager.

    Both the Python and C++ backends load from the same source to guarantee
    identical table data and therefore identical distribution transforms.
    """
    global NORM_CDF_TABLE, T3_PPF_TABLE, T3_SD
    from ..tables import get_table_manager

    manager = get_table_manager()
    NORM_CDF_TABLE = manager.load_norm_cdf_table().astype(np.float64)
    T3_PPF_TABLE = manager.load_t3_ppf_table().astype(np.float64)

    # Compute the effective SD of t(3) values produced by the lookup pipeline.
    # The t(3) distribution has theoretical variance = 3, but the lookup table
    # clips percentiles to [0.001, 0.999] and uses interpolation, yielding a
    # lower empirical SD (~0.93). We normalise by this SD so that the
    # high-kurtosis distribution has Var ≈ 1, matching normal/uniform/skewed.
    T3_SD = _compute_t3_sd()


def _compute_t3_sd():
    """Compute the effective SD of high-kurtosis values from the lookup pipeline.

    Replicates the vectorised norm-CDF -> t(3)-PPF lookup chain on a large
    fixed-seed sample to get a stable SD estimate.
    """
    assert NORM_CDF_TABLE is not None
    assert T3_PPF_TABLE is not None

    rng_state = np.random.get_state()
    np.random.seed(999999)
    z = np.random.standard_normal(200000)
    np.random.set_state(rng_state)

    # Step 1: Normal CDF lookup (z -> percentile)
    z_clipped = np.clip(z, NORM_RANGE[0], NORM_RANGE[1])
    idx = (z_clipped - NORM_RANGE[0]) * NORM_SCALE
    idx_int = np.clip(idx.astype(int), 0, DIST_RESOLUTION - 2)
    frac = idx - idx_int
    cdf_vals = NORM_CDF_TABLE[idx_int] * (1 - frac) + NORM_CDF_TABLE[idx_int + 1] * frac

    # Step 2: t(3) PPF lookup (percentile -> t3 quantile)
    cdf_vals = np.clip(cdf_vals, PERCENTILE_RANGE[0], PERCENTILE_RANGE[1])
    idx2 = (cdf_vals - PERCENTILE_RANGE[0]) * PERC_SCALE
    idx2_int = np.clip(idx2.astype(int), 0, DIST_RESOLUTION - 2)
    frac2 = idx2 - idx2_int
    t3_vals = T3_PPF_TABLE[idx2_int] * (1 - frac2) + T3_PPF_TABLE[idx2_int + 1] * frac2

    return float(np.std(t3_vals))


_init_tables()


def create_uploaded_lookup_tables(
    data_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create quantile-matching tables for uploaded data.

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

        percentiles = np.linspace(1 / (n_samples + 1), n_samples / (n_samples + 1), n_samples)
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
    """Generate the predictor matrix ``X`` with correlated, typed columns.

    Algorithm:

    1. Draw ``(sample_size, n_vars)`` i.i.d. standard normals.
    2. Apply the Cholesky factor of *correlation_matrix* to induce
       the desired correlation structure.
    3. Transform each column to its target distribution via
       probability-integral transforms using pre-computed lookup tables.

    Distribution codes (``var_types``):
    0=normal, 1=binary, 2=right_skewed, 3=left_skewed,
    4=high_kurtosis (t with df=3), 5=uniform,
    97=uploaded_factor, 98=uploaded_binary, 99=uploaded_data.

    This function is also used as the body for the Numba JIT-compiled
    ``_generate_X_jit`` variant.
    """

    def _vectorized_norm_cdf_lookup(x_array):
        """Map standard-normal values to CDF probabilities via lookup table."""
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
                    result[i] = norm_cdf_table[idx_int] * (1 - frac) + norm_cdf_table[idx_int + 1] * frac
        return result

    def _vectorized_t3_ppf_lookup(percentile_array):
        """Map CDF probabilities to t(df=3) quantiles via lookup table."""
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
                    result[i] = t3_ppf_table[idx_int] * (1 - frac) + t3_ppf_table[idx_int + 1] * frac
        return result

    def _vectorized_uploaded_lookup(normal_array, normal_vals, uploaded_vals):
        """Interpolate uploaded empirical quantiles from normal values."""
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
                left, right = 0, n_lookup - 1
                while left < right - 1:
                    mid = (left + right) // 2
                    if normal_vals[mid] <= normal_value:
                        left = mid
                    else:
                        right = mid
                frac = (normal_value - normal_vals[left]) / (normal_vals[right] - normal_vals[left])
                result[i] = uploaded_vals[left] * (1 - frac) + uploaded_vals[right] * frac
        return result

    def _cholesky_decomposition(corr_matrix):
        """Compute Cholesky factor, falling back to eigen-decomposition if needed."""
        try:
            return np.linalg.cholesky(corr_matrix)
        except Exception:
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            eigenvals = np.maximum(eigenvals, FLOAT_NEAR_ZERO)
            return eigenvecs @ np.diag(np.sqrt(eigenvals))

    def _transform_distribution(data, dist_type, param, var_idx):
        """Transform a standard-normal column to the target distribution."""
        if dist_type == 0:  # normal
            return data.copy()
        elif dist_type == 1:  # binary
            percentiles = _vectorized_norm_cdf_lookup(data)
            binary_data = (percentiles < param).astype(np.float64)
            return binary_data - np.mean(binary_data)
        elif dist_type == 2:  # right_skewed
            percentiles = _vectorized_norm_cdf_lookup(data)
            percentiles = np.clip(percentiles, PERCENTILE_RANGE[0], PERCENTILE_RANGE[1])
            return (-np.log(percentiles) - SKEW_MEAN) / SKEW_STD
        elif dist_type == 3:  # left_skewed
            percentiles = _vectorized_norm_cdf_lookup(data)
            percentiles = np.clip(percentiles, PERCENTILE_RANGE[0], PERCENTILE_RANGE[1])
            return (np.log(1 - percentiles) + SKEW_MEAN) / SKEW_STD
        elif dist_type == 4:  # high_kurtosis
            percentiles = _vectorized_norm_cdf_lookup(data)
            return _vectorized_t3_ppf_lookup(percentiles) / T3_SD
        elif dist_type == 5:  # uniform
            percentiles = _vectorized_norm_cdf_lookup(data)
            return SQRT3 * (2 * percentiles - 1)
        elif dist_type == 97:  # uploaded_factor (handled via bootstrap, shouldn't reach here)
            return data.copy()
        elif dist_type == 98:  # uploaded_binary (handled via bootstrap, shouldn't reach here)
            return data.copy()
        elif dist_type == 99:  # uploaded_data (continuous with lookup tables)
            if var_idx < upload_normal_values.shape[0]:
                normal_vals = upload_normal_values[var_idx]
                uploaded_vals = upload_data_values[var_idx]
                return _vectorized_uploaded_lookup(data, normal_vals, uploaded_vals)
            return data.copy()
        else:
            return data.copy()

    # Main algorithm
    if sim_seed >= 0:
        np.random.seed(sim_seed)
    base_normal = np.random.standard_normal((sample_size, n_vars))

    cholesky_matrix = _cholesky_decomposition(correlation_matrix)
    correlated_data = base_normal @ cholesky_matrix.T

    X = np.zeros((sample_size, n_vars))
    for j in range(n_vars):
        X[:, j] = _transform_distribution(correlated_data[:, j], var_types[j], var_params[j], j)

    return X


# Try JIT compilation, fall back to pure Python
try:
    from numba import njit

    _generate_X_jit = njit(
        "f8[:,:](i8, i8, f8[:,:], i8[:], f8[:], f8[:], f8[:], f8[:,:], f8[:,:], i8)",
        cache=True,
    )(_generate_X_core)
    _USE_JIT = True
except ImportError:
    _generate_X_jit = _generate_X_core
    _USE_JIT = False


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
        var_types: Distribution types per variable
        var_params: Distribution parameters
        normal_values, uploaded_values: Lookup tables for uploaded data
        seed: Random seed

    Returns:
        X: (sample_size, n_vars) design matrix
    """
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_vars)
    if normal_values is None:
        normal_values = np.zeros((2, 2))
    if uploaded_values is None:
        uploaded_values = np.zeros((2, 2))

    return _generate_X_jit(  # type: ignore[no-any-return]
        sample_size,
        n_vars,
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
    Generate factor variables as dummy variables.

    Args:
        sample_size: Number of observations
        factor_specs: List of {'n_levels': int, 'proportions': [float, ...]}
        seed: Random seed

    Returns:
        X_factors: (sample_size, total_dummies) array
    """
    if seed is not None:
        np.random.seed(seed)

    if not factor_specs:
        return np.empty((sample_size, 0), dtype=float)

    factor_columns = []
    for spec in factor_specs:
        n_levels = spec["n_levels"]
        proportions = spec["proportions"]
        factor_data = np.random.choice(n_levels, size=sample_size, p=proportions)
        dummies = np.eye(n_levels, dtype=float)[factor_data]
        factor_columns.append(dummies[:, 1:])

    return np.hstack(factor_columns) if factor_columns else np.empty((sample_size, 0), dtype=float)


def bootstrap_uploaded_data(
    sample_size: int,
    raw_data: np.ndarray,
    var_metadata: dict,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap uploaded data for strict correlation preservation mode.

    Samples whole rows from the uploaded data (preserving exact relationships),
    then transforms binary and factor variables appropriately.

    Args:
        sample_size: Number of samples to generate
        raw_data: Original uploaded data (n_samples × n_vars), normalized for continuous
        var_metadata: Dict mapping var_name to {type, data_index, unique_values, ...}
        seed: Random seed

    Returns:
        X_non_factors: Non-factor variables (continuous + binary mapped to 0-1)
        X_factors: Factor dummy variables
    """
    if seed is not None:
        np.random.seed(seed)

    # Bootstrap whole rows
    n_samples = raw_data.shape[0]
    row_indices = np.random.choice(n_samples, size=sample_size, replace=True)
    bootstrapped_data = raw_data[row_indices, :]

    # Separate by type
    non_factor_columns = []
    factor_columns = []

    # Sort by data_index to maintain order
    sorted_vars = sorted(var_metadata.items(), key=lambda x: x[1]["data_index"])

    for _var_name, info in sorted_vars:
        idx = info["data_index"]
        col_data = bootstrapped_data[:, idx]

        if info["type"] == "binary":
            # Map to 0-1
            unique_vals = info["unique_values"]
            # Map first unique value to 0, second to 1
            binary_data = np.where(col_data == unique_vals[0], 0.0, 1.0)
            # Center: mean = 0
            binary_data = binary_data - np.mean(binary_data)
            non_factor_columns.append(binary_data)

        elif info["type"] == "factor":
            # Create dummy variables
            unique_vals = info["unique_values"]
            n_levels = len(unique_vals)

            # Map values to indices
            factor_indices = np.zeros(sample_size, dtype=int)
            for level_idx, val in enumerate(unique_vals):
                factor_indices[col_data == val] = level_idx

            # Create dummies (drop first level)
            dummies = np.eye(n_levels, dtype=float)[factor_indices]
            factor_columns.append(dummies[:, 1:])

        else:  # continuous
            # Already normalized, keep as-is
            non_factor_columns.append(col_data)

    # Stack results
    X_non_factors = np.column_stack(non_factor_columns) if non_factor_columns else np.empty((sample_size, 0), dtype=float)

    X_factors = np.hstack(factor_columns) if factor_columns else np.empty((sample_size, 0), dtype=float)

    return X_non_factors, X_factors


def _generate_cluster_effects(
    sample_size: int,
    cluster_specs: Dict[str, Dict],
    sim_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate cluster random effect columns for mixed model data.

    For each cluster spec:
    1. Generate cluster_id assignments (each ID repeated cluster_size times)
    2. Generate random intercepts b_i ~ N(0, tau²) for each cluster
    3. Create id_effect column where each observation gets its cluster's b_i

    Args:
        sample_size: Total number of observations (n_clusters * cluster_size)
        cluster_specs: Dict of {grouping_var: {n_clusters, cluster_size, tau_squared, ...}}
        sim_seed: Random seed

    Returns:
        X_cluster: (sample_size, n_cluster_vars) array of random effect columns
    """
    if sim_seed is not None:
        # Use a derived seed to avoid collision with X generation seed
        np.random.seed(sim_seed + 3)

    columns = []

    for _gv, spec in cluster_specs.items():
        n_clusters = spec["n_clusters"]
        cluster_size = spec["cluster_size"]
        tau_sq = spec["tau_squared"]

        # Compute cluster_size from sample_size if not provided
        if cluster_size is None:
            cluster_size = sample_size // n_clusters

        # Generate random intercepts for each cluster
        tau = np.sqrt(tau_sq)
        random_intercepts = np.random.normal(0, tau, size=n_clusters)

        # Create id_effect column: repeat each cluster's intercept
        # cluster_id assignment: [0,0,...,0, 1,1,...,1, ..., K-1,K-1,...,K-1]
        id_effect = np.repeat(random_intercepts, cluster_size)

        # Trim or pad to exact sample_size (handles rounding)
        if len(id_effect) > sample_size:
            id_effect = id_effect[:sample_size]
        elif len(id_effect) < sample_size:
            # Should not happen if sample_size = n_clusters * cluster_size
            id_effect = np.pad(id_effect, (0, sample_size - len(id_effect)))

        columns.append(id_effect)

    if not columns:
        return np.empty((sample_size, 0), dtype=float)

    return np.column_stack(columns)
