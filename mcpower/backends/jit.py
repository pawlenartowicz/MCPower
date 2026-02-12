"""
Numba JIT compute backend for MCPower.

This module wraps the Numba JIT-compiled functions from utils,
providing a middle ground between the native C++ and pure Python backends.
"""

import numpy as np

try:
    from ..utils.data_generation import _USE_JIT as _DG_JIT
    from ..utils.data_generation import NORM_CDF_TABLE, T3_PPF_TABLE, _generate_X_jit
    from ..utils.ols import _USE_JIT as _OLS_JIT
    from ..utils.ols import _generate_y_jit, _ols_jit

    if not (_OLS_JIT and _DG_JIT):
        raise ImportError("Numba JIT compilation not available")

    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False


class JITBackend:
    """Numba JIT compute backend.

    Delegates to ``@njit``-compiled versions of the core functions,
    providing ~2-5x speedup over pure Python. Requires ``numba`` to
    be installed.
    """

    def __init__(self):
        """Verify that Numba JIT compilation is available."""
        if not _JIT_AVAILABLE:
            raise ImportError("Numba JIT backend not available. Install numba: pip install numba")

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
        """Run JIT-compiled OLS regression and return significance flags."""
        return _ols_jit(  # type: ignore[no-any-return]
            X,
            y,
            target_indices,
            f_crit,
            t_crit,
            correction_t_crits,
            correction_method,
        )

    def generate_y(
        self,
        X: np.ndarray,
        effects: np.ndarray,
        heterogeneity: float,
        heteroskedasticity: float,
        seed: int,
    ) -> np.ndarray:
        """Generate the dependent variable using JIT-compiled code."""
        return _generate_y_jit(X, effects, heterogeneity, heteroskedasticity, seed)  # type: ignore[no-any-return]

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
        """Generate the predictor matrix using JIT-compiled code."""
        return _generate_X_jit(  # type: ignore[no-any-return]
            n_samples,
            n_vars,
            correlation_matrix,
            var_types,
            var_params,
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            upload_normal,
            upload_data,
            seed,
        )
