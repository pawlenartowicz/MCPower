"""
Backend abstraction for MCPower framework.

This module provides a unified interface for compute backends,
allowing seamless switching between Python, JIT, and C++ implementations.

The automatic backend selection follows priority order:
1. Native (C++) - fastest, requires compiled extensions
2. JIT (Numba) - middle ground, requires numba
3. Python - slowest, always available

Users can override the selection via set_backend('c++' | 'jit' | 'python' | 'default').
"""

from typing import Protocol, Union, runtime_checkable

import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol defining the compute backend interface.

    All backends (Python, JIT, C++) must implement these three methods.
    """

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
_BACKEND_NAMES = {"default", "c++", "jit", "python"}

# Global backend instance
_backend_instance = None
_backend_forced = False
_warn_on_fallback = True


def _create_backend(name: str) -> ComputeBackend:
    """
    Instantiate a backend by name.

    Args:
        name: 'c++', 'jit', or 'python'

    Raises:
        ImportError: If the requested backend is not available.
    """
    if name == "c++":
        from .native import NativeBackend

        return NativeBackend()

    if name == "jit":
        from .jit import JITBackend

        return JITBackend()

    if name == "python":
        from .python import PythonBackend

        return PythonBackend()

    raise ValueError(f"Unknown backend: {name!r}")


def _auto_select() -> ComputeBackend:
    """Auto-select the best available backend: C++ > JIT > Python."""
    for name in ("c++", "jit", "python"):
        try:
            backend = _create_backend(name)
            if name == "python" and _warn_on_fallback:
                import warnings

                warnings.warn(
                    "No native C++ or Numba backend found — using pure Python (slower). "
                    "Install Numba for better performance: pip install MCPower[JIT]",
                    stacklevel=3,
                )
            return backend
        except ImportError:
            continue

    # Should never reach here — PythonBackend is always available
    from .python import PythonBackend

    return PythonBackend()


def get_backend() -> ComputeBackend:
    """
    Get the active compute backend.

    On first call, auto-selects the best available backend (C++ > JIT > Python).
    Subsequent calls return the cached instance unless reset_backend() is called.
    """
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    _backend_instance = _auto_select()
    return _backend_instance


def set_backend(backend: Union[str, ComputeBackend]) -> None:
    """
    Set the compute backend.

    Args:
        backend: One of:
            - 'default' — auto-select best available (C++ > JIT > Python)
            - 'c++'     — force native C++ backend
            - 'jit'     — force Numba JIT backend
            - 'python'  — force pure Python backend
            - A ComputeBackend instance

    Raises:
        ImportError: If the requested backend is not available.
        ValueError: If the string is not recognized.
    """
    global _backend_instance, _backend_forced

    if isinstance(backend, str):
        name = backend.lower().strip()
        if name not in _BACKEND_NAMES:
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {', '.join(sorted(_BACKEND_NAMES))}")
        if name == "default":
            _backend_instance = _auto_select()
            _backend_forced = False
        else:
            _backend_instance = _create_backend(name)
            _backend_forced = True
    else:
        _backend_instance = backend
        _backend_forced = True


def reset_backend() -> None:
    """Reset backend to automatic selection."""
    global _backend_instance, _backend_forced
    _backend_instance = None
    _backend_forced = False


def set_fallback_warning(enabled: bool = True) -> None:
    """Enable or disable the warning emitted when falling back to pure Python."""
    global _warn_on_fallback
    _warn_on_fallback = enabled


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
        "is_jit": name == "JITBackend",
        "module": type(backend).__module__,
        "forced": _backend_forced,
    }


__all__ = [
    "ComputeBackend",
    "get_backend",
    "set_backend",
    "reset_backend",
    "get_backend_info",
    "set_fallback_warning",
]
