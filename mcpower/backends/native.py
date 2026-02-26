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


def _prep(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Ensure array is contiguous with the expected dtype for C++ interop."""
    return np.ascontiguousarray(arr, dtype=dtype)


class NativeBackend:
    """
    C++ compute backend using pybind11 bindings.

    Provides high-performance implementations of the core algorithms.
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
        norm_cdf = _prep(norm_cdf)
        t3_ppf = _prep(t3_ppf)

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
        X = _prep(X)
        y = _prep(y)
        target_indices = _prep(target_indices, np.int32)
        correction_t_crits = _prep(correction_t_crits)

        return mcpower_native.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)  # type: ignore[no-any-return]

    def generate_y(
        self,
        X: np.ndarray,
        effects: np.ndarray,
        heterogeneity: float,
        heteroskedasticity: float,
        seed: int,
        residual_dist: int = 0,
        residual_df: float = 10.0,
    ) -> np.ndarray:
        """
        Generate dependent variable.

        Args:
            X: Design matrix (n_samples, n_features)
            effects: Effect sizes (n_features,)
            heterogeneity: Effect size variation SD
            heteroskedasticity: Error-predictor correlation
            seed: Random seed (-1 for random)
            residual_dist: Error distribution (0=normal, 1=heavy_tailed, 2=skewed)
            residual_df: Degrees of freedom for non-normal residuals

        Returns:
            Response vector (n_samples,)
        """
        X = _prep(X)
        effects = _prep(effects)

        return mcpower_native.generate_y(X, effects, heterogeneity, heteroskedasticity, seed, residual_dist, residual_df)  # type: ignore[no-any-return]

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
        correlation_matrix = _prep(correlation_matrix)
        var_types = _prep(var_types, np.int32)
        var_params = _prep(var_params)
        upload_normal = _prep(upload_normal)
        upload_data = _prep(upload_data)

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

            wald_flag: 1.0 if the Wald test was used as fallback for the overall
            significance test (instead of the likelihood ratio test), 0.0 otherwise.
        """
        X = _prep(X)
        y = _prep(y)
        cluster_ids = _prep(cluster_ids, np.int32)
        target_indices = _prep(target_indices, np.int32)
        correction_z_crits = _prep(correction_z_crits)

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

            wald_flag: 1.0 if the Wald test was used as fallback for the overall
            significance test (instead of the likelihood ratio test), 0.0 otherwise.
        """
        X = _prep(X)
        y = _prep(y)
        Z = _prep(Z)
        cluster_ids = _prep(cluster_ids, np.int32)
        target_indices = _prep(target_indices, np.int32)
        correction_z_crits = _prep(correction_z_crits)
        warm_theta = _prep(warm_theta)

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

            wald_flag: 1.0 if the Wald test was used as fallback for the overall
            significance test (instead of the likelihood ratio test), 0.0 otherwise.
        """
        X = _prep(X)
        y = _prep(y)
        parent_ids = _prep(parent_ids, np.int32)
        child_ids = _prep(child_ids, np.int32)
        child_to_parent = _prep(child_to_parent, np.int32)
        target_indices = _prep(target_indices, np.int32)
        correction_z_crits = _prep(correction_z_crits)
        warm_theta = _prep(warm_theta)

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
