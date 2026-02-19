"""
Backend abstraction for MCPower framework.

This module provides a unified interface for compute backends.
The only supported backend is native C++ (compiled via pybind11).

Users can override via set_backend('c++' | 'default') or pass a ComputeBackend instance.
"""

from typing import Optional, Protocol, Union, runtime_checkable

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
    ) -> np.ndarray:
        """Generate the dependent variable ``y = X @ effects + error``.

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


# Valid backend names for set_backend()
_BACKEND_NAMES = {"default", "c++"}

# Global backend instance
_backend_instance: Optional[ComputeBackend] = None
_backend_forced = False


def get_backend() -> ComputeBackend:
    """
    Get the active compute backend.

    On first call, instantiates the C++ native backend.
    Subsequent calls return the cached instance unless reset_backend() is called.

    Raises:
        ImportError: If the C++ extension is not compiled/installed.
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    from .native import NativeBackend

    _backend_instance = NativeBackend()
    return _backend_instance


def set_backend(backend: Union[str, ComputeBackend]) -> None:
    """
    Set the compute backend.

    Args:
        backend: One of:
            - 'default' -- use native C++ backend
            - 'c++'     -- force native C++ backend
            - A ComputeBackend instance

    Raises:
        ImportError: If the C++ backend is not available.
        ValueError: If the string is not recognized.
    """
    global _backend_instance, _backend_forced

    if isinstance(backend, str):
        name = backend.lower().strip()
        if name not in _BACKEND_NAMES:
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {', '.join(sorted(_BACKEND_NAMES))}")

        from .native import NativeBackend

        _backend_instance = NativeBackend()
        _backend_forced = name != "default"
    else:
        _backend_instance = backend
        _backend_forced = True


def reset_backend() -> None:
    """Reset backend to automatic selection."""
    global _backend_instance, _backend_forced
    _backend_instance = None
    _backend_forced = False


def get_backend_info() -> dict:
    """
    Get information about the current backend.

    Returns:
        Dictionary with backend name, type, and whether it was forced.
    """
    backend = get_backend()
    name = type(backend).__name__
    return {
        "name": name,
        "is_native": name == "NativeBackend",
        "module": type(backend).__module__,
        "forced": _backend_forced,
    }


__all__ = [
    "ComputeBackend",
    "get_backend",
    "set_backend",
    "reset_backend",
    "get_backend_info",
]
