"""
Native C++ compute backend for MCPower.

This module wraps the pybind11-compiled C++ backend,
providing high-performance implementations of the core algorithms.
"""

import numpy as np

# Import the compiled module
try:
    from . import mcpower_native  # type: ignore[attr-defined]

    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False
    mcpower_native = None


class NativeBackend:
    """
    C++ compute backend using pybind11 bindings.

    Provides ~10-50x speedup over Python/Numba implementations.
    """

    def __init__(self):
        """Initialize native backend and load lookup tables."""
        if not _NATIVE_AVAILABLE:
            raise ImportError(
                "Native C++ backend not available. Install from PyPI for pre-compiled binaries: pip install --upgrade mcpower"
            )

        # Initialize tables if not already done
        if not mcpower_native.tables_initialized():
            self._initialize_tables()

    def _initialize_tables(self) -> None:
        """Load data generation tables into the C++ module."""
        from ..tables import get_table_manager

        manager = get_table_manager()

        # Only load data generation tables (OLS no longer needs lookup tables)
        norm_cdf = manager.load_norm_cdf_table()
        t3_ppf = manager.load_t3_ppf_table()

        # Ensure correct dtypes
        norm_cdf = np.ascontiguousarray(norm_cdf.astype(np.float64))
        t3_ppf = np.ascontiguousarray(t3_ppf.astype(np.float64))

        # Initialize C++ tables (generation tables only)
        mcpower_native.init_tables(norm_cdf, t3_ppf)

    def ols_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_indices: np.ndarray,
        f_crit: float,
        t_crit: float,
        correction_t_crits: np.ndarray,
        correction_method: int,
    ) -> np.ndarray:
        """
        Run OLS regression analysis.

        Args:
            X: Design matrix (n_samples, n_features)
            y: Response vector (n_samples,)
            target_indices: Indices of coefficients to test
            f_crit: Precomputed F critical value
            t_crit: Precomputed t critical value
            correction_t_crits: Precomputed correction critical values
            correction_method: Correction method (0=none, 1=Bonferroni, 2=FDR, 3=Holm)

        Returns:
            Array: [f_sig, uncorrected..., corrected...]
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        target_indices = np.ascontiguousarray(target_indices, dtype=np.int32)
        correction_t_crits = np.ascontiguousarray(correction_t_crits, dtype=np.float64)

        return mcpower_native.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)  # type: ignore[no-any-return]

    def generate_y(
        self,
        X: np.ndarray,
        effects: np.ndarray,
        heterogeneity: float,
        heteroskedasticity: float,
        seed: int,
    ) -> np.ndarray:
        """
        Generate dependent variable.

        Args:
            X: Design matrix (n_samples, n_features)
            effects: Effect sizes (n_features,)
            heterogeneity: Effect size variation SD
            heteroskedasticity: Error-predictor correlation
            seed: Random seed (-1 for random)

        Returns:
            Response vector (n_samples,)
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        effects = np.ascontiguousarray(effects, dtype=np.float64)

        return mcpower_native.generate_y(X, effects, heterogeneity, heteroskedasticity, seed)  # type: ignore[no-any-return]

    def generate_X(
        self,
        n_samples: int,
        n_vars: int,
        correlation_matrix: np.ndarray,
        var_types: np.ndarray,
        var_params: np.ndarray,
        upload_normal: np.ndarray,
        upload_data: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """
        Generate design matrix.

        Args:
            n_samples: Number of observations
            n_vars: Number of variables
            correlation_matrix: Variable correlations
            var_types: Distribution types
            var_params: Distribution parameters
            upload_normal: Normal quantiles for uploaded data
            upload_data: Uploaded data values
            seed: Random seed (-1 for random)

        Returns:
            Design matrix (n_samples, n_vars)
        """
        correlation_matrix = np.ascontiguousarray(correlation_matrix, dtype=np.float64)
        var_types = np.ascontiguousarray(var_types, dtype=np.int32)
        var_params = np.ascontiguousarray(var_params, dtype=np.float64)
        upload_normal = np.ascontiguousarray(upload_normal, dtype=np.float64)
        upload_data = np.ascontiguousarray(upload_data, dtype=np.float64)

        return mcpower_native.generate_X(  # type: ignore[no-any-return]
            n_samples,
            n_vars,
            correlation_matrix,
            var_types,
            var_params,
            upload_normal,
            upload_data,
            seed,
        )

    def lme_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_ids: np.ndarray,
        n_clusters: int,
        target_indices: np.ndarray,
        chi2_crit: float,
        z_crit: float,
        correction_z_crits: np.ndarray,
        correction_method: int,
        warm_lambda_sq: float = -1.0,
    ) -> np.ndarray:
        """
        Run LME analysis with precomputed critical values.

        Args:
            X: Design matrix (n_samples, n_features), no intercept
            y: Response vector (n_samples,)
            cluster_ids: Cluster membership (n_samples,)
            n_clusters: Number of clusters
            target_indices: Indices of coefficients to test
            chi2_crit: Precomputed chi-squared critical value
            z_crit: Precomputed z critical value
            correction_z_crits: Precomputed correction critical values
            correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
            warm_lambda_sq: Warm start lambda^2 (-1 for cold start)

        Returns:
            Array: [f_sig, uncorrected..., corrected..., wald_flag]
            or empty array on failure
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        cluster_ids = np.ascontiguousarray(cluster_ids, dtype=np.int32)
        target_indices = np.ascontiguousarray(target_indices, dtype=np.int32)
        correction_z_crits = np.ascontiguousarray(correction_z_crits, dtype=np.float64)

        return mcpower_native.lme_analysis(  # type: ignore[no-any-return]
            X,
            y,
            cluster_ids,
            n_clusters,
            target_indices,
            chi2_crit,
            z_crit,
            correction_z_crits,
            correction_method,
            warm_lambda_sq,
        )

    def lme_analysis_general(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,
        cluster_ids: np.ndarray,
        n_clusters: int,
        q: int,
        target_indices: np.ndarray,
        chi2_crit: float,
        z_crit: float,
        correction_z_crits: np.ndarray,
        correction_method: int,
        warm_theta: np.ndarray,
    ) -> np.ndarray:
        """
        Run LME analysis for random slopes (q>1).

        Args:
            X: Design matrix (n_samples, n_features), no intercept
            y: Response vector (n_samples,)
            Z: Random effects design matrix (n_samples, q)
            cluster_ids: Cluster membership (n_samples,)
            n_clusters: Number of clusters
            q: Random effects dimension
            target_indices: Indices of coefficients to test
            chi2_crit: Precomputed chi-squared critical value
            z_crit: Precomputed z critical value
            correction_z_crits: Precomputed correction critical values
            correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
            warm_theta: Warm start theta (empty for cold start)

        Returns:
            Array: [f_sig, uncorrected..., corrected..., wald_flag]
            or empty array on failure
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        cluster_ids = np.ascontiguousarray(cluster_ids, dtype=np.int32)
        target_indices = np.ascontiguousarray(target_indices, dtype=np.int32)
        correction_z_crits = np.ascontiguousarray(correction_z_crits, dtype=np.float64)
        warm_theta = np.ascontiguousarray(warm_theta, dtype=np.float64)

        return mcpower_native.lme_analysis_general(  # type: ignore[no-any-return]
            X,
            y,
            Z,
            cluster_ids,
            n_clusters,
            q,
            target_indices,
            chi2_crit,
            z_crit,
            correction_z_crits,
            correction_method,
            warm_theta,
        )

    def lme_analysis_nested(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_ids: np.ndarray,
        child_ids: np.ndarray,
        K_parent: int,
        K_child: int,
        child_to_parent: np.ndarray,
        target_indices: np.ndarray,
        chi2_crit: float,
        z_crit: float,
        correction_z_crits: np.ndarray,
        correction_method: int,
        warm_theta: np.ndarray,
    ) -> np.ndarray:
        """
        Run LME analysis for nested random intercepts.

        Args:
            X: Design matrix (n_samples, n_features), no intercept
            y: Response vector (n_samples,)
            parent_ids: Parent cluster membership (n_samples,)
            child_ids: Child cluster membership (n_samples,)
            K_parent: Number of parent clusters
            K_child: Number of child clusters
            child_to_parent: Mapping child -> parent (K_child,)
            target_indices: Indices of coefficients to test
            chi2_crit: Precomputed chi-squared critical value
            z_crit: Precomputed z critical value
            correction_z_crits: Precomputed correction critical values
            correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm
            warm_theta: Warm start [theta_parent, theta_child] (empty for cold start)

        Returns:
            Array: [f_sig, uncorrected..., corrected..., wald_flag]
            or empty array on failure
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        parent_ids = np.ascontiguousarray(parent_ids, dtype=np.int32)
        child_ids = np.ascontiguousarray(child_ids, dtype=np.int32)
        child_to_parent = np.ascontiguousarray(child_to_parent, dtype=np.int32)
        target_indices = np.ascontiguousarray(target_indices, dtype=np.int32)
        correction_z_crits = np.ascontiguousarray(correction_z_crits, dtype=np.float64)
        warm_theta = np.ascontiguousarray(warm_theta, dtype=np.float64)

        return mcpower_native.lme_analysis_nested(  # type: ignore[no-any-return]
            X,
            y,
            parent_ids,
            child_ids,
            K_parent,
            K_child,
            child_to_parent,
            target_indices,
            chi2_crit,
            z_crit,
            correction_z_crits,
            correction_method,
            warm_theta,
        )


def is_native_available() -> bool:
    """Check if native backend is available."""
    return _NATIVE_AVAILABLE
