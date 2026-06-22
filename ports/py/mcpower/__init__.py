"""MCPower — Monte Carlo power analysis with a native Rust engine."""

from .model import MCPower
from .datasets import mtcars
from . import progress

__version__ = "1.0.0a2"
__all__ = ["MCPower", "mtcars", "progress"]
