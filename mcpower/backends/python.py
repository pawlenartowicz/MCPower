"""
Pure Python compute backend for MCPower.

This module provides a fallback implementation that uses no compiled
extensions — neither C++ nor Numba JIT. Always available.
"""

import numpy as np

from ..utils.data_generation import NORM_CDF_TABLE, T3_PPF_TABLE, _generate_X_core
from ..utils.ols import _generate_y_core, _ols_core


class PythonBackend:
    """Pure Python compute backend (no compiled extensions).

    Delegates to the uncompiled ``_ols_core``, ``_generate_y_core``, and
    ``_generate_X_core`` functions. Slowest backend but always available.
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
        """Run OLS regression and return significance flags."""
        return _ols_core(
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
        """Generate the dependent variable y = X @ effects + error."""
        return _generate_y_core(X, effects, heterogeneity, heteroskedasticity, seed)

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
        """Generate the predictor matrix X with correlated, typed columns."""
        return _generate_X_core(
            n_samples,
            n_vars,
            correlation_matrix,
            var_types,
            var_params,
            NORM_CDF_TABLE,
            T3_PPF_TABLE,
            upload_normal,
            upload_data,
            seed if seed >= 0 else -1,
        )
