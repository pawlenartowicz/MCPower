"""
OLS analysis isolation tests.

Generate data with numpy (no backend), then run OLS with each backend.
This isolates whether the OLS implementation differs between backends.
"""

import numpy as np
import pytest

from tests.config import N_SIMS, SEED
from tests.helpers.analytical import analytical_t_power
from tests.helpers.mc_margins import mc_accuracy_margin
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
class TestOLSAnalysisIsolation:
    """
    Generate data with plain numpy, then run OLS with each backend.
    Any difference in rejection rates is purely in the OLS code path.
    """

    def _get_backends(self):
        from mcpower.backends.native import NativeBackend
        from mcpower.backends.python import PythonBackend

        return NativeBackend(), PythonBackend()

    @pytest.mark.parametrize(
        "b1,b2,n",
        [
            (0.3, 0.2, 100),
            (0.5, 0.3, 80),
        ],
    )
    def test_ols_only_isolation(self, b1, b2, n):
        """
        numpy-generated data → each backend's ols_analysis → compare rates.
        """
        native, python = self._get_backends()
        rng = np.random.RandomState(SEED)
        p = 2
        target_indices = np.array([0, 1], dtype=np.int64)

        nat_reject = np.zeros((N_SIMS, p))
        py_reject = np.zeros((N_SIMS, p))

        X_template = np.zeros((n, p), dtype=np.float64)
        f_crit, t_crit, correction_t_crits = compute_crits(X_template, target_indices)

        for i in range(N_SIMS):
            X = rng.randn(n, p).astype(np.float64)
            y = (b1 * X[:, 0] + b2 * X[:, 1] + rng.randn(n)).astype(np.float64)

            nat = native.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, 0)
            py = python.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, 0)

            nat_reject[i] = nat[1 : 1 + p]
            py_reject[i] = py[1 : 1 + p]

        exact_vals = {
            "x1": analytical_t_power(b1, n, p=p, sigma_eps=1.0, vif_j=1.0),
            "x2": analytical_t_power(b2, n, p=p, sigma_eps=1.0, vif_j=1.0),
        }

        for j, var in enumerate(["x1", "x2"]):
            nat_rate = nat_reject[:, j].mean() * 100
            py_rate = py_reject[:, j].mean() * 100
            exact = exact_vals[var]
            margin = mc_accuracy_margin(exact, N_SIMS)

            agree = np.mean(nat_reject[:, j] == py_reject[:, j]) * 100

            print(
                f"  {var} (b={[b1, b2][j]}, n={n}): "
                f"native={nat_rate:.2f}%, python={py_rate:.2f}%, "
                f"analytical={exact:.2f}% ± {margin:.2f}%, "
                f"per-dataset agreement={agree:.1f}%"
            )

            nat_err = abs(nat_rate - exact)
            py_err = abs(py_rate - exact)

            assert nat_err < margin, f"[native OLS] {var}: MC={nat_rate:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"
            assert py_err < margin, f"[python OLS] {var}: MC={py_rate:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"

    @pytest.mark.parametrize(
        "b1,b2,n",
        [
            (0.3, 0.2, 100),
            (0.5, 0.3, 80),
        ],
    )
    def test_ols_disagreement_rate(self, b1, b2, n):
        """
        On identical data, how often do native and python OLS disagree?
        """
        native, python = self._get_backends()
        rng = np.random.RandomState(SEED)
        p = 2
        target_indices = np.array([0, 1], dtype=np.int64)

        nat_only = np.zeros(p)
        py_only = np.zeros(p)

        X_template = np.zeros((n, p), dtype=np.float64)
        f_crit, t_crit, correction_t_crits = compute_crits(X_template, target_indices)

        for _i in range(N_SIMS):
            X = rng.randn(n, p).astype(np.float64)
            y = (b1 * X[:, 0] + b2 * X[:, 1] + rng.randn(n)).astype(np.float64)

            nat = native.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, 0)
            py = python.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, 0)

            for j in range(p):
                n_rej = nat[1 + j] > 0.5
                p_rej = py[1 + j] > 0.5
                if n_rej and not p_rej:
                    nat_only[j] += 1
                elif p_rej and not n_rej:
                    py_only[j] += 1

        for j, var in enumerate(["x1", "x2"]):
            total_disagree = nat_only[j] + py_only[j]
            print(
                f"  {var}: total_disagree={total_disagree}/{N_SIMS} "
                f"({total_disagree / N_SIMS * 100:.1f}%), "
                f"native_only_rejects={nat_only[j]:.0f}, "
                f"python_only_rejects={py_only[j]:.0f}"
            )

            assert total_disagree / N_SIMS < 0.03, (
                f"[{var}] Excessive OLS disagreement: "
                f"{total_disagree / N_SIMS * 100:.1f}% "
                f"(native_only={nat_only[j]:.0f}, python_only={py_only[j]:.0f})"
            )
