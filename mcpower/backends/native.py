"""
Native C++ compute backend for MCPower.

This module wraps the pybind11-compiled C++ backend,
providing high-performance implementations of the core algorithms.
"""

import numpy as np

# Import the compiled module
try:
    from . import mcpower_native

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

        return mcpower_native.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)

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

        return mcpower_native.generate_y(X, effects, heterogeneity, heteroskedasticity, seed)

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

        return mcpower_native.generate_X(
            n_samples,
            n_vars,
            correlation_matrix,
            var_types,
            var_params,
            upload_normal,
            upload_data,
            seed,
        )


def is_native_available() -> bool:
    """Check if native backend is available."""
    return _NATIVE_AVAILABLE
