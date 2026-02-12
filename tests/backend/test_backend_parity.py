"""
Cross-backend parity tests.

Verify that C++ and Python backends produce statistically equivalent results.
All tests are skipped if the native backend is not available.
"""

import numpy as np
import pytest

from tests.helpers.mc_margins import mc_proportion_margin
from tests.helpers.power_helpers import compute_crits


def _native_available():
    try:
        from mcpower.backends.native import NativeBackend

        NativeBackend()
        return True
    except (ImportError, Exception):
        return False


skip_if_no_native = pytest.mark.skipif(
    not _native_available(),
    reason="Native C++ backend not available",
)


@skip_if_no_native
class TestCrossFrontendParity:
    """
    Compare full MCPower runs with backend forcing.

    Verify power estimates agree within MC margins.
    """

    def _get_backends(self):
        from mcpower.backends.native import NativeBackend
        from mcpower.backends.python import PythonBackend

        return NativeBackend(), PythonBackend()

    @pytest.mark.slow
    def test_power_agreement_simple(self):
        """y = x1, beta=0.3, n=100, 2000 sims — power within MC margin."""
        from mcpower import MCPower
        from mcpower.backends import reset_backend, set_backend

        results = {}
        for backend_name in ["c++", "python"]:
            set_backend(backend_name)

            m = MCPower("y = x1")
            m.set_effects("x1=0.3")
            m.set_simulations(2000)
            m.set_seed(42)

            res = m.find_power(
                sample_size=100,
                target_test="all",
                print_results=False,
                return_results=True,
            )
            results[backend_name] = res
            reset_backend()

        cpp = results["c++"]["results"]["individual_powers"]
        py = results["python"]["results"]["individual_powers"]

        for test_name in cpp:
            p_cpp = cpp[test_name] / 100.0
            p_py = py[test_name] / 100.0
            margin = mc_proportion_margin(max(p_cpp, p_py, 0.01), 2000)
            assert abs(p_cpp - p_py) < margin, f"{test_name}: C++={p_cpp:.1%} vs Python={p_py:.1%}, margin={margin:.4f}"

    @pytest.mark.slow
    def test_rejection_rates_agree(self):
        """Per-predictor rejection rates agree within MC margin (2000 datasets)."""
        native, python = self._get_backends()
        n_datasets = 2000
        n = 100
        p = 2
        target_indices = np.array([0, 1], dtype=np.int64)

        nat_decisions = np.zeros((n_datasets, p))
        py_decisions = np.zeros((n_datasets, p))

        rng = np.random.RandomState(12345)
        for i in range(n_datasets):
            X = rng.randn(n, p).astype(np.float64)
            y = (0.3 * X[:, 0] + rng.randn(n)).astype(np.float64)

            f_crit, t_crit, crits = compute_crits(X, target_indices)

            nat_res = native.ols_analysis(X, y, target_indices, f_crit, t_crit, crits, 0)
            py_res = python.ols_analysis(X, y, target_indices, f_crit, t_crit, crits, 0)

            nat_decisions[i] = nat_res[1 : 1 + p]
            py_decisions[i] = py_res[1 : 1 + p]

        for j in range(p):
            nat_rate = nat_decisions[:, j].mean()
            py_rate = py_decisions[:, j].mean()
            margin = mc_proportion_margin(max(nat_rate, py_rate, 0.01), n_datasets)
            assert abs(nat_rate - py_rate) < margin, f"Predictor {j}: native={nat_rate:.4f} vs python={py_rate:.4f}, margin={margin:.4f}"
