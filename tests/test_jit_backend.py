"""
Tests for Numba JIT backend.
"""

import numpy as np
import pytest


def _jit_check():
    """Check if JIT is available at module level for skipif markers."""
    try:
        from mcpower.backends.jit import _JIT_AVAILABLE

        return _JIT_AVAILABLE
    except ImportError:
        return False


class TestJITBackend:
    """Test JIT backend availability and behavior."""

    def test_jit_unavailable_raises(self, monkeypatch):
        """When _JIT_AVAILABLE is False, constructor should raise ImportError."""
        import mcpower.backends.jit as jit_mod

        monkeypatch.setattr(jit_mod, "_JIT_AVAILABLE", False)

        with pytest.raises(ImportError, match="JIT backend not available"):
            jit_mod.JITBackend()

    def test_jit_available_flag_exists(self):
        """Module exposes _JIT_AVAILABLE boolean."""
        from mcpower.backends.jit import _JIT_AVAILABLE

        assert isinstance(_JIT_AVAILABLE, bool)

    @pytest.mark.skipif(
        not _jit_check(),
        reason="Numba JIT not available",
    )
    def test_jit_generate_X_parity(self):
        """JIT generate_X produces same shape as Python backend."""
        from mcpower.backends.jit import JITBackend
        from mcpower.backends.python import PythonBackend

        n, p = 50, 2
        corr = np.eye(p)
        var_types = np.zeros(p, dtype=np.int64)
        var_params = np.zeros(p, dtype=np.float64)
        upload_normal = np.zeros((2, 2), dtype=np.float64)
        upload_data = np.zeros((2, 2), dtype=np.float64)
        seed = 42

        py = PythonBackend()
        X_py = py.generate_X(n, p, corr, var_types, var_params, upload_normal, upload_data, seed)

        jit = JITBackend()
        X_jit = jit.generate_X(n, p, corr, var_types, var_params, upload_normal, upload_data, seed)

        assert X_py.shape == X_jit.shape

    @pytest.mark.skipif(
        not _jit_check(),
        reason="Numba JIT not available",
    )
    def test_jit_ols_analysis_parity(self):
        """JIT ols_analysis returns same result shape as Python backend."""
        from mcpower.backends.jit import JITBackend
        from mcpower.backends.python import PythonBackend

        n, p = 100, 3
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.array([0.5, 0.3, 0.2]) + np.random.randn(n) * 0.5
        target_indices = np.array([0, 1, 2], dtype=np.int64)
        f_crit = 3.0
        t_crit = 2.0
        correction_t_crits = np.array([2.5, 2.5, 2.5])
        correction_method = 0

        py = PythonBackend()
        r_py = py.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)

        jit = JITBackend()
        r_jit = jit.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, correction_method)

        assert r_py.shape == r_jit.shape
