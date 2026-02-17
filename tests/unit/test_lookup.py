"""
Tests for lookup table management.
"""

import numpy as np

from mcpower.tables.lookup import LookupTableManager, get_table_manager


class TestLookupTableManager:
    """Test LookupTableManager class."""

    def test_ensure_data_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "tables"
        manager = LookupTableManager(data_dir=subdir)
        manager.ensure_data_dir()
        assert subdir.exists()

    def test_load_norm_cdf_generates_table(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        table = manager.load_norm_cdf_table()
        assert isinstance(table, np.ndarray)
        assert len(table) == LookupTableManager.DIST_RESOLUTION
        # CDF values should be in [0, 1]
        assert table.min() >= 0
        assert table.max() <= 1

    def test_load_norm_cdf_caches(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        t1 = manager.load_norm_cdf_table()
        t2 = manager.load_norm_cdf_table()
        assert t1 is t2  # same object from cache

    def test_load_t3_ppf_generates_table(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        table = manager.load_t3_ppf_table()
        assert isinstance(table, np.ndarray)
        assert len(table) == LookupTableManager.DIST_RESOLUTION

    def test_load_t3_ppf_caches(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        t1 = manager.load_t3_ppf_table()
        t2 = manager.load_t3_ppf_table()
        assert t1 is t2

    def test_load_all_generation_tables(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        norm_cdf, t3_ppf = manager.load_all_generation_tables()
        assert len(norm_cdf) == LookupTableManager.DIST_RESOLUTION
        assert len(t3_ppf) == LookupTableManager.DIST_RESOLUTION

    def test_get_table_info(self, tmp_path):
        manager = LookupTableManager(data_dir=tmp_path)
        manager.load_norm_cdf_table()
        info = manager.get_table_info()
        assert "norm_cdf" in info
        assert "shape" in info["norm_cdf"]
        assert "dtype" in info["norm_cdf"]
        assert "size_mb" in info["norm_cdf"]

    def test_load_from_cache_file(self, tmp_path):
        # Generate tables → saved to cache
        manager1 = LookupTableManager(data_dir=tmp_path)
        t1 = manager1.load_norm_cdf_table()

        # New manager loads from cache file
        manager2 = LookupTableManager(data_dir=tmp_path)
        t2 = manager2.load_norm_cdf_table()
        assert np.allclose(t1, t2)

    def test_singleton_get_table_manager(self):
        m1 = get_table_manager()
        m2 = get_table_manager()
        assert m1 is m2
