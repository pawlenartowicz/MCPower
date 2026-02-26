"""MCPower - Monte Carlo Power Analysis.

A simulation-based framework for statistical power analysis supporting
linear regression and linear mixed-effects models with interactions,
correlated predictors, non-normal distributions, and empirical data upload.

Example:
    >>> from mcpower import MCPower
    >>>
    >>> model = MCPower("y = x1 + x2 + x1:x2")
    >>> model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
    >>> model.find_power(sample_size=100)
    >>>
    >>> model.find_sample_size(target_test='x1', from_size=50, to_size=200)
"""

from importlib.metadata import version as _get_version

from .model import MCPower
from .progress import PrintReporter, ProgressReporter, SimulationCancelled, TqdmReporter

__version__ = _get_version("MCPower")
__author__ = "Pawel Lenartowicz"
__email__ = "pawellenartowicz@europe.com"

__all__ = [
    "MCPower",
    "SimulationCancelled",
    "ProgressReporter",
    "PrintReporter",
    "TqdmReporter",
]


import threading as _threading

from .utils.updates import _check_for_updates

_threading.Thread(target=_check_for_updates, args=(__version__,), daemon=True).start()
