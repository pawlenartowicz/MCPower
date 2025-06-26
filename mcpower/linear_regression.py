import numpy as np
from typing import List, Dict, Optional, Tuple
from base import MCPowerBase
from utils.ols import _ols_analysis, _generate_y

class LinearRegression(MCPowerBase):
    """
    Monte Carlo power analysis for Linear Regression models.
    
    Supports continuous predictors, interactions, and correlations between predictors.
    Uses Ordinary Least Squares estimation with F-tests and t-tests.
    """

    @property
    def model_type(self) -> str:
        """Return model type name for display."""
        return 'Linear Regression'

    def _run_statistical_analysis(self, X_expanded: np.ndarray, y: np.ndarray, 
                             target_indices: np.ndarray, alpha: float, 
                             correction_method: int) -> np.ndarray:
        """Run linear regression analysis using compiled OLS if available."""
        return _ols_analysis(X_expanded, y, target_indices, correction_method, alpha)     

    def _generate_dependent_variable(self, X_expanded: np.ndarray, effect_sizes_expanded: np.ndarray,
                                     heterogeneity: float = 0.0, heteroskedasticity: float = 0.0,
                                     seed: Optional[int] = None) -> np.ndarray:
        """
        Generate continuous dependent variable for linear regression.

        Returns:
            y: Generated dependent variable
        """
        
        seed_val = int(seed) if seed is not None else -1

        # Generate y using compiled function
        y = _generate_y(X_expanded=X_expanded, effect_sizes=effect_sizes_expanded,
                        heterogeneity=heterogeneity, heteroskedasticity=heteroskedasticity,
                        seed=seed_val
        )
        return y
