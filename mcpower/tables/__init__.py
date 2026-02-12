"""Pre-computed lookup tables for MCPower data-generation transforms.

Re-exports ``LookupTableManager`` and the ``get_table_manager`` singleton
accessor used by both the Python and C++ backends.
"""

from .lookup import LookupTableManager, get_table_manager

__all__ = [
    "LookupTableManager",
    "get_table_manager",
]
