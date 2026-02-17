"""
Pure Python compute backend for MCPower.

This module provides a fallback implementation that uses no compiled
extensions — neither C++ nor Numba JIT. Always available.
"""

import numpy as np

from ..stats.data_generation import NORM_CDF_TABLE, T3_PPF_TABLE, _generate_X_core
from ..stats.ols import _generate_y_core, _ols_core


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
        return _ols_core(  # type: ignore[no-any-return]
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
        return _generate_y_core(X, effects, heterogeneity, heteroskedasticity, seed)  # type: ignore[no-any-return]

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
        return _generate_X_core(  # type: ignore[no-any-return]
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
        """Run LME analysis using pure Python solver."""
        from ..stats.lme_solver import lme_analysis_full

        result = lme_analysis_full(
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
        return result if result is not None else np.empty(0)
