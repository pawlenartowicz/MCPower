"""
Monte Carlo Power Analysis Package.

A flexible framework for conducting power analysis using Monte Carlo simulations
for various statistical models including Linear Regression, Logistic Regression,
ANOVA, and Mixed Effects models.

Main Classes:
    MCPowerBase: Base class for all power analysis models
    LinearRegression: Power analysis for linear regression models
    LogisticRegression: Power analysis for logistic regression models (coming soon)

Example Usage:
    >>> from mcpower import LinearRegression
    >>>
    >>> # Create model
    >>> mc = LinearRegression("y = x1 + x2 + x1:x2")
    >>>
    >>> # Set effect sizes
    >>> mc.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
    >>>
    >>> # Find power
    >>> mc.find_power(sample_size=100)
    >>>
    >>> # Find required sample size
    >>> mc.find_sample_size(target_test='x1', from_size=50, to_size=200)
"""

from .linear_regression import LinearRegression

# Version info
__version__ = "0.3.3"
__author__ = "Paweł Lenartowicz"
__email__ = "pawellenartowicz@europe.com"

# Public API
__all__ = [
    # Model classes
    "LinearRegression"
]


# Module metadata
def get_info():
    """Get package information."""
    return {
        "name": "mcpower",
        "version": __version__,
        "description": "Monte Carlo Power Analysis for Statistical Models",
        "models": ["LinearRegression"],
        "author": __author__,
        "email": __email__,
    }


# Check for updates
from .utils.updates import _check_for_updates

_check_for_updates(__version__)
