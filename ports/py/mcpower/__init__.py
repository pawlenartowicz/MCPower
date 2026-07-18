"""MCPower — Monte Carlo power analysis with a native Rust engine."""

from importlib.metadata import PackageNotFoundError, version

from .model import MCPower
from .datasets import mtcars
from . import progress

try:
    __version__ = version("mcpower")
except PackageNotFoundError:  # running from an uninstalled source tree
    __version__ = "0.0.0"

__all__ = ["MCPower", "mtcars", "progress"]
