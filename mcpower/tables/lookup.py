"""Lookup table management for data-generation distribution transforms.

Pre-computes and caches normal-CDF and t(3)-PPF tables that are used
by the probability-integral-transform pipeline in ``data_generation.py``
and the C++ native backend.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class LookupTableManager:
    """Manages pre-computed lookup tables for data-generation transforms.

    Tables are lazily loaded from disk (``tables/data/*.npz``) on first
    access and generated from scipy if the cache files are missing.
    Both the Python backend and the C++ native backend consume the
    same tables to guarantee identical distribution transforms.

    Class constants:
        DIST_RESOLUTION: Number of points in each table (2048).
        PERCENTILE_RANGE: CDF range for percentile-based tables.
        NORM_RANGE: Standard-normal x-axis range for CDF tables.
    """

    # Distribution lookup table specifications
    DIST_RESOLUTION = 2048
    PERCENTILE_RANGE = (0.001, 0.999)
    NORM_RANGE = (-6, 6)

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialise the table manager.

        Args:
            data_dir: Directory for cached ``.npz`` table files.
                Defaults to ``tables/data/`` alongside this module.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"

        self.data_dir = data_dir
        self._tables: Dict[str, np.ndarray] = {}

    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_norm_cdf_table(self) -> np.ndarray:
        """Load (or generate and cache) the normal CDF lookup table.

        Returns:
            1-D float64 array of length ``DIST_RESOLUTION``.
        """
        if "norm_cdf" in self._tables:
            return self._tables["norm_cdf"]

        cache_file = self.data_dir / "norm_cdf.npz"

        try:
            data = np.load(cache_file)
            self._tables["norm_cdf"] = data["norm_cdf"]
            return self._tables["norm_cdf"]
        except (FileNotFoundError, KeyError):
            pass

        self._generate_norm_cdf_table()
        return self._tables["norm_cdf"]

    def load_t3_ppf_table(self) -> np.ndarray:
        """Load (or generate and cache) the t(df=3) PPF lookup table.

        Returns:
            1-D float64 array of length ``DIST_RESOLUTION``.
        """
        if "t3_ppf" in self._tables:
            return self._tables["t3_ppf"]

        cache_file = self.data_dir / "t3_ppf.npz"

        try:
            data = np.load(cache_file)
            self._tables["t3_ppf"] = data["t3_ppf"]
            return self._tables["t3_ppf"]
        except (FileNotFoundError, KeyError):
            pass

        self._generate_t3_ppf_table()
        return self._tables["t3_ppf"]

    def load_all_generation_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all tables needed for data generation.

        Returns:
            Tuple of (norm_cdf_table, t3_ppf_table)
        """
        norm_cdf = self.load_norm_cdf_table()
        t3_ppf = self.load_t3_ppf_table()
        return norm_cdf, t3_ppf

    def _generate_norm_cdf_table(self) -> None:
        """Generate normal CDF table for distribution transforms."""
        from mcpower.stats.distributions import generate_norm_cdf_table

        norm_cdf = generate_norm_cdf_table(self.NORM_RANGE[0], self.NORM_RANGE[1], self.DIST_RESOLUTION)
        x_norm = np.linspace(*self.NORM_RANGE, self.DIST_RESOLUTION)

        self._tables["norm_cdf"] = norm_cdf

        self.ensure_data_dir()
        try:
            np.savez_compressed(self.data_dir / "norm_cdf.npz", norm_cdf=norm_cdf, x_range=x_norm)
        except Exception:
            pass

    def _generate_t3_ppf_table(self) -> None:
        """Generate t(3) PPF table for heavy-tailed transforms."""
        from mcpower.stats.distributions import generate_t3_ppf_table

        percentile_points = np.linspace(*self.PERCENTILE_RANGE, self.DIST_RESOLUTION)
        t3_ppf = generate_t3_ppf_table(self.PERCENTILE_RANGE[0], self.PERCENTILE_RANGE[1], self.DIST_RESOLUTION)

        self._tables["t3_ppf"] = t3_ppf

        self.ensure_data_dir()
        try:
            np.savez_compressed(
                self.data_dir / "t3_ppf.npz",
                t3_ppf=t3_ppf,
                percentile_range=percentile_points,
            )
        except Exception:
            pass

    def generate_all_tables(self) -> None:
        """Generate all lookup tables and save to data directory."""
        print("Generating normal CDF table...")
        self._generate_norm_cdf_table()

        print("Generating t(3) PPF table...")
        self._generate_t3_ppf_table()

        print("All tables generated successfully.")

    def get_table_info(self) -> Dict[str, Dict]:
        """
        Get information about loaded tables.

        Returns:
            Dictionary with table names and shapes
        """
        return {
            name: {
                "shape": table.shape,
                "dtype": str(table.dtype),
                "size_mb": table.nbytes / (1024 * 1024),
            }
            for name, table in self._tables.items()
        }


# Global table manager instance
_table_manager: Optional[LookupTableManager] = None


def get_table_manager() -> LookupTableManager:
    """Return the global ``LookupTableManager`` singleton (created on first call)."""
    global _table_manager

    if _table_manager is None:
        _table_manager = LookupTableManager()

    return _table_manager
