"""
Data Generator for Monte Carlo Power Analysis.

Generates synthetic datasets with:
- Distributions (normal, binary, skewed, etc.)
- Correlation structures
- Uploaded data structure preservation

Core computation is handled by the C++ backend (see mcpower/backends/native.py).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    from mcpower.stats.distributions import norm_ppf_array

    n_samples, n_vars = data_matrix.shape
    normal_values = np.zeros((n_vars, n_samples))
    uploaded_values = np.zeros((n_vars, n_samples))

    for var_idx in range(n_vars):
        data = data_matrix[:, var_idx]
        normalized = (data - np.mean(data)) / np.std(data)
        sorted_uploaded = np.sort(normalized)

        percentiles = np.linspace(1 / (n_samples + 1), n_samples / (n_samples + 1), n_samples)
        normal_quantiles = norm_ppf_array(percentiles)

        normal_values[var_idx] = normal_quantiles
        uploaded_values[var_idx] = sorted_uploaded

    return normal_values, uploaded_values


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


def _generate_non_normal_intercepts(
    n_clusters: int,
    tau: float,
    dist: str,
    df: int,
    rng_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate non-normal random intercepts for LME scenario perturbations.

    Args:
        n_clusters: Number of clusters.
        tau: Standard deviation of the random intercept distribution.
        dist: Distribution type (``"normal"``, ``"heavy_tailed"``, ``"skewed"``).
        df: Degrees of freedom for t or chi-squared distribution.
        rng_state: Optional ``RandomState`` for reproducibility. When ``None``,
            uses the global numpy random state.

    Returns:
        1-D array of random intercepts with mean ≈ 0 and SD ≈ tau.
    """
    gen = rng_state if rng_state is not None else np.random

    if dist == "heavy_tailed":
        df = max(df, 3)
        # t(df) has variance df/(df-2); scale so SD = tau
        scale = tau / np.sqrt(df / (df - 2))
        return np.asarray(gen.standard_t(df, size=n_clusters) * scale)
    elif dist == "skewed":
        df = max(df, 3)
        # Shifted chi-squared: mean=0, SD=tau
        raw = (gen.chisquare(df, size=n_clusters) - df) / np.sqrt(2 * df)
        return np.asarray(raw * tau)
    else:
        return gen.normal(0, tau, size=n_clusters)


def _generate_cluster_effects(
    sample_size: int,
    cluster_specs: Dict[str, Dict],
    sim_seed: Optional[int] = None,
    lme_perturbations: Optional[Dict] = None,
) -> np.ndarray:
    """
    Generate cluster random effect columns for mixed model data.

    For each cluster spec:
    1. Generate cluster_id assignments (each ID repeated cluster_size times)
    2. Generate random intercepts b_i ~ N(0, tau²) for each cluster
    3. Create id_effect column where each observation gets its cluster's b_i

    When *lme_perturbations* is provided, tau² is jittered and the random
    effect distribution may be non-normal.

    Args:
        sample_size: Total number of observations (n_clusters * cluster_size)
        cluster_specs: Dict of {grouping_var: {n_clusters, cluster_size, tau_squared, ...}}
        sim_seed: Random seed
        lme_perturbations: Optional dict from ``apply_lme_perturbations()``

    Returns:
        X_cluster: (sample_size, n_cluster_vars) array of random effect columns
    """
    if sim_seed is not None:
        # Use a derived seed to avoid collision with X generation seed
        np.random.seed(sim_seed + 3)

    columns = []

    for gv, spec in cluster_specs.items():
        n_clusters = spec["n_clusters"]
        cluster_size = spec["cluster_size"]
        tau_sq = spec["tau_squared"]

        # Compute cluster_size from sample_size if not provided
        if cluster_size is None:
            cluster_size = sample_size // n_clusters

        # Apply LME perturbations if present
        if lme_perturbations is not None:
            multiplier = lme_perturbations["tau_squared_multipliers"].get(gv, 1.0)
            tau_sq = tau_sq * multiplier

        tau = np.sqrt(tau_sq)

        # Generate random intercepts (possibly non-normal)
        if lme_perturbations is not None:
            re_dist = lme_perturbations.get("random_effect_dist", "normal")
            re_df = lme_perturbations.get("random_effect_df", 5)
            random_intercepts = _generate_non_normal_intercepts(n_clusters, tau, re_dist, re_df)
        else:
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


@dataclass
class RandomEffectsResult:
    """Result from random effects generation.

    Carries everything the simulation pipeline needs to:
    1. Build X (intercept_columns go into X at cluster column positions)
    2. Add slope contributions directly to y
    3. Pass cluster IDs, Z matrices, and nesting info to the LME solver
    """

    intercept_columns: np.ndarray
    """(N, n_intercept_terms) random intercept values per observation."""

    slope_contribution: np.ndarray
    """(N,) total random slope contribution to y."""

    cluster_ids_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    """Per-grouping-variable cluster membership arrays."""

    Z_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    """Per-grouping-variable (N, q) random-effects design matrices (slopes only)."""

    child_to_parent: Optional[np.ndarray] = None
    """Mapping from child-level cluster index to parent-level index (nested only)."""

    K_parent: int = 0
    """Number of parent-level clusters (nested only)."""

    K_child: int = 0
    """Number of child-level clusters (nested only)."""

    parent_var: Optional[str] = None
    """Parent grouping variable name (nested only)."""

    child_var: Optional[str] = None
    """Child grouping variable name (nested only)."""


def _generate_random_effects(
    sample_size: int,
    cluster_specs: Dict[str, Dict],
    X_non_factors: np.ndarray,
    non_factor_names: List[str],
    sim_seed: Optional[int] = None,
    lme_perturbations: Optional[Dict] = None,
) -> RandomEffectsResult:
    """Generate random effects for all grouping factors.

    Handles three model types:

    1. **Random intercepts** (q=1) — backward compatible with
       ``_generate_cluster_effects``.
    2. **Random slopes** (q>1) — draws correlated intercept+slope effects
       from MVN(0, G), builds the Z matrix, and computes the slope
       contribution to y.
    3. **Nested random intercepts** — generates hierarchical random effects
       for parent and child levels (e.g. school / classroom).

    Args:
        sample_size: Total number of observations.
        cluster_specs: Dict of ``{grouping_var: spec_dict}``.  Each
            *spec_dict* has keys ``n_clusters``, ``cluster_size``,
            ``tau_squared``, and for Phase 2 features optionally
            ``G_matrix``, ``random_slope_vars``, ``parent_var``,
            ``n_per_parent``, ``q``.
        X_non_factors: ``(N, n_vars)`` non-factor predictor matrix
            (needed for random slope contributions).
        non_factor_names: Ordered names of non-factor predictors (used
            to locate slope variable columns in *X_non_factors*).
        sim_seed: Random seed (a derived seed ``sim_seed + 3`` is used
            to avoid collision with X generation).
        lme_perturbations: Optional dict from ``apply_lme_perturbations()``
            with ICC jitter multipliers and non-normal RE distribution params.

    Returns:
        A :class:`RandomEffectsResult` with intercept columns, slope
        contributions, cluster IDs, Z matrices, and nesting metadata.
    """
    if sim_seed is not None:
        np.random.seed(sim_seed + 3)

    intercept_cols: List[np.ndarray] = []
    slope_contribution = np.zeros(sample_size)
    cluster_ids_dict: Dict[str, np.ndarray] = {}
    Z_matrices: Dict[str, np.ndarray] = {}

    # Nested model bookkeeping
    child_to_parent: Optional[np.ndarray] = None
    K_parent = 0
    K_child = 0
    parent_var_name: Optional[str] = None
    child_var_name: Optional[str] = None

    # Classify specs into parent, child (nested), and simple (non-nested)
    parent_specs: Dict[str, Dict] = {}
    child_specs: Dict[str, Dict] = {}
    simple_specs: Dict[str, Dict] = {}

    child_parent_set = {s.get("parent_var") for s in cluster_specs.values() if s.get("parent_var")}

    for gv, spec in cluster_specs.items():
        if spec.get("parent_var"):
            child_specs[gv] = spec
        elif gv in child_parent_set:
            parent_specs[gv] = spec
        else:
            simple_specs[gv] = spec

    # ------------------------------------------------------------------
    # Simple random intercepts and random slopes
    # ------------------------------------------------------------------
    for gv, spec in simple_specs.items():
        n_clusters = spec["n_clusters"]
        cluster_size = spec["cluster_size"]
        if cluster_size is None:
            cluster_size = sample_size // n_clusters
        q = spec.get("q", 1)

        # Cluster IDs (same for intercept-only and slope models)
        raw_ids = np.repeat(np.arange(n_clusters, dtype=np.intp), cluster_size)
        cluster_ids = _trim_or_pad(raw_ids, sample_size)
        cluster_ids_dict[gv] = cluster_ids

        # Apply LME perturbations: ICC jitter on tau_squared
        tau_sq = spec["tau_squared"]
        if lme_perturbations is not None:
            multiplier = lme_perturbations["tau_squared_multipliers"].get(gv, 1.0)
            tau_sq = tau_sq * multiplier

        if q == 1:
            # --- Random intercept only ---
            tau = np.sqrt(tau_sq)
            if lme_perturbations is not None:
                re_dist = lme_perturbations.get("random_effect_dist", "normal")
                re_df = lme_perturbations.get("random_effect_df", 5)
                random_intercepts = _generate_non_normal_intercepts(n_clusters, tau, re_dist, re_df)
            else:
                random_intercepts = np.random.normal(0, tau, size=n_clusters)
            id_effect = _trim_or_pad(np.repeat(random_intercepts, cluster_size), sample_size)
            intercept_cols.append(id_effect)

        else:
            # --- Random slopes (q > 1) ---
            G_matrix = spec["G_matrix"].copy()
            slope_vars = spec.get("random_slope_vars", [])

            # Apply ICC jitter to G_matrix intercept variance
            if lme_perturbations is not None:
                ratio = tau_sq / spec["tau_squared"] if spec["tau_squared"] > 0 else 1.0
                # Scale intercept row/column of G by sqrt(ratio)
                sqrt_ratio = np.sqrt(ratio)
                G_matrix[0, :] *= sqrt_ratio
                G_matrix[:, 0] *= sqrt_ratio

            # Draw correlated [b_int, b_slope1, ...] per cluster
            re_dist = lme_perturbations.get("random_effect_dist", "normal") if lme_perturbations else "normal"
            re_df = lme_perturbations.get("random_effect_df", 5) if lme_perturbations else 5

            if re_dist == "heavy_tailed" and lme_perturbations is not None:
                # Multivariate t: MVN(0, G * (df-2)/df) × sqrt(df / chi2(df))
                df = max(re_df, 3)
                G_scaled = G_matrix * ((df - 2.0) / df)
                b_normal = np.random.multivariate_normal(np.zeros(q), G_scaled, size=n_clusters)
                chi2_samples = np.random.chisquare(df, size=n_clusters)
                mixing = np.sqrt(df / chi2_samples)
                b = b_normal * mixing[:, np.newaxis]
            elif re_dist == "skewed" and lme_perturbations is not None:
                # Independent skewed marginals via shifted chi-squared, scaled by Cholesky
                df = max(re_df, 3)
                L = np.linalg.cholesky(G_matrix)
                raw = (np.random.chisquare(df, size=(n_clusters, q)) - df) / np.sqrt(2 * df)
                b = raw @ L.T
            else:
                b = np.random.multivariate_normal(np.zeros(q), G_matrix, size=n_clusters)

            # Intercept component
            intercept_effect = _trim_or_pad(np.repeat(b[:, 0], cluster_size), sample_size)
            intercept_cols.append(intercept_effect)

            # Slope contributions to y: sum_s b_slope_s[j] * x_s[i]
            for s_idx, slope_var in enumerate(slope_vars):
                if slope_var in non_factor_names:
                    col_idx = non_factor_names.index(slope_var)
                    x_col = X_non_factors[:, col_idx]
                    b_slope_repeated = _trim_or_pad(np.repeat(b[:, 1 + s_idx], cluster_size), sample_size)
                    slope_contribution += b_slope_repeated * x_col

            # Build Z matrix: [1, x_slope1, x_slope2, ...]
            Z_cols = [np.ones(sample_size)]
            for slope_var in slope_vars:
                if slope_var in non_factor_names:
                    col_idx = non_factor_names.index(slope_var)
                    Z_cols.append(X_non_factors[:, col_idx])
            Z_matrices[gv] = np.column_stack(Z_cols)

    # ------------------------------------------------------------------
    # Nested random intercepts  (1|parent) + (1|parent:child)
    # ------------------------------------------------------------------
    if parent_specs and child_specs:
        p_gv = next(iter(parent_specs))
        p_spec = parent_specs[p_gv]
        c_gv = next(iter(child_specs))
        c_spec = child_specs[c_gv]

        K_parent = p_spec["n_clusters"]
        n_per_parent = c_spec.get("n_per_parent", 1)
        K_child = K_parent * n_per_parent
        child_cluster_size = c_spec["cluster_size"]
        if child_cluster_size is None:
            child_cluster_size = sample_size // K_child

        tau_sq_parent = p_spec["tau_squared"]
        tau_sq_child = c_spec["tau_squared"]

        if lme_perturbations is not None:
            tau_sq_parent *= lme_perturbations["tau_squared_multipliers"].get(p_gv, 1.0)
            tau_sq_child *= lme_perturbations["tau_squared_multipliers"].get(c_gv, 1.0)

        tau_parent = np.sqrt(tau_sq_parent)
        tau_child = np.sqrt(tau_sq_child)

        if lme_perturbations is not None:
            re_dist = lme_perturbations.get("random_effect_dist", "normal")
            re_df = lme_perturbations.get("random_effect_df", 5)
            b_parent = _generate_non_normal_intercepts(K_parent, tau_parent, re_dist, re_df)
            b_child = _generate_non_normal_intercepts(K_child, tau_child, re_dist, re_df)
        else:
            b_parent = np.random.normal(0, tau_parent, size=K_parent)
            b_child = np.random.normal(0, tau_child, size=K_child)

        # IDs: parent_ids assigns each observation to a parent cluster,
        # child_ids assigns each observation to a child cluster.
        raw_parent_ids = np.repeat(
            np.arange(K_parent, dtype=np.intp),
            n_per_parent * child_cluster_size,
        )
        raw_child_ids = np.repeat(np.arange(K_child, dtype=np.intp), child_cluster_size)
        parent_ids = _trim_or_pad(raw_parent_ids, sample_size)
        child_ids = _trim_or_pad(raw_child_ids, sample_size)

        # child_to_parent: maps child cluster index → parent cluster index
        child_to_parent = np.repeat(np.arange(K_parent, dtype=np.intp), n_per_parent)

        # Intercept contributions (one column per level)
        intercept_cols.append(b_parent[parent_ids])
        intercept_cols.append(b_child[child_ids])

        cluster_ids_dict[p_gv] = parent_ids
        cluster_ids_dict[c_gv] = child_ids

        parent_var_name = p_gv
        child_var_name = c_gv

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    if intercept_cols:
        intercept_columns = np.column_stack(intercept_cols)
    else:
        intercept_columns = np.empty((sample_size, 0), dtype=float)

    return RandomEffectsResult(
        intercept_columns=intercept_columns,
        slope_contribution=slope_contribution,
        cluster_ids_dict=cluster_ids_dict,
        Z_matrices=Z_matrices,
        child_to_parent=child_to_parent,
        K_parent=K_parent,
        K_child=K_child,
        parent_var=parent_var_name,
        child_var=child_var_name,
    )


def _trim_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad *arr* to exactly *target_len* elements."""
    if len(arr) > target_len:
        return arr[:target_len]
    elif len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    return arr
