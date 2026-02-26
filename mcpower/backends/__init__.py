"""
Backend abstraction for MCPower framework.

This module provides a unified interface for compute backends.
The only supported backend is native C++ (compiled via pybind11).
"""

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol defining the compute backend interface."""

    def ols_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_indices: np.ndarray,
        f_crit: float,
        t_crit: float,
        correction_t_crits: np.ndarray,
        # correction_method encoding: 0=none, 1=Bonferroni, 2=FDR (BH), 3=Holm
        correction_method: int,
    ) -> np.ndarray:
        """Run OLS regression and return significance flags.

        Returns:
            1-D array ``[F_sig, uncorrected_1..n, corrected_1..n]``.
        """
        ...

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
        """Generate the dependent variable ``y = X @ effects + error``.

        Args:
            residual_dist: Error distribution (0=normal, 1=heavy_tailed, 2=skewed).
            residual_df: Degrees of freedom for non-normal residuals.

        Returns:
            1-D array of length ``n_samples``.
        """
        ...

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
        """Generate the predictor matrix ``X`` with the specified distributions.

        Returns:
            2-D array of shape ``(n_samples, n_vars)``.
        """
        ...

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
        warm_lambda_sq: float,
    ) -> np.ndarray:
        """Run LME analysis and return significance flags.

        Returns:
            1-D array ``[F_sig, uncorrected_1..n, corrected_1..n, wald_flag]``
            or empty array on failure.
        """
        ...


# Global backend instance
_backend_instance: Optional[ComputeBackend] = None


def get_backend() -> ComputeBackend:
    """
    Get the active compute backend.

    On first call, instantiates the C++ native backend.
    Subsequent calls return the cached instance.

    Raises:
        ImportError: If the C++ extension is not compiled/installed.
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    from .native import NativeBackend

    _backend_instance = NativeBackend()
    return _backend_instance


__all__ = [
    "ComputeBackend",
    "get_backend",
]
