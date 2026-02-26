"""Tests for mcpower.backends.native — import fallback and _prep utility."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mcpower.backends.native import _prep


class TestPrep:
    """Test _prep array coercion for C++ interop."""

    def test_contiguous_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _prep(arr)
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float64

    def test_non_contiguous_becomes_contiguous(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        col = arr[:, 1]  # non-contiguous column slice
        assert not col.flags["C_CONTIGUOUS"]
        result = _prep(col)
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_dtype_conversion_float32_to_float64(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = _prep(arr, np.float64)
        assert result.dtype == np.float64

    def test_dtype_conversion_int64_to_int32(self):
        arr = np.array([0, 1, 2], dtype=np.int64)
        result = _prep(arr, np.int32)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="F")
        assert not arr.flags["C_CONTIGUOUS"]
        result = _prep(arr)
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == np.float64


class TestNativeBackendImport:
    """Test NativeBackend init when C++ extension is unavailable."""

    def test_init_raises_when_unavailable(self):
        """NativeBackend() should raise ImportError when _NATIVE_AVAILABLE=False."""
        with patch("mcpower.backends.native._NATIVE_AVAILABLE", False):
            from mcpower.backends.native import NativeBackend
            with pytest.raises(ImportError, match="Native C\\+\\+ backend not available"):
                NativeBackend()

    def test_is_native_available_reflects_module_state(self):
        from mcpower.backends.native import is_native_available
        # Just verify it returns a bool
        result = is_native_available()
        assert isinstance(result, bool)
