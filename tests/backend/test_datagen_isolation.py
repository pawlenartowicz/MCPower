"""
Data generation isolation tests.

Use each backend's generate_X / generate_y separately, then run OLS
with a single (Python) backend. This isolates whether the bias comes
from data generation or OLS analysis.
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
@pytest.mark.slow
class TestDataGenerationIsolation:
    """
    Generate data with each backend, but always analyze with the Python
    backend's OLS. If the bias only appears with one backend's data
    generation, the problem is in generate_X or generate_y.
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
    def test_datagen_isolation_uncorrelated(self, b1, b2, n):
        """
        Generate X/y with backend A, analyze with Python OLS — per backend.
        """
        native, python = self._get_backends()
        rng_seeds = list(range(SEED, SEED + N_SIMS))
        p = 2
        target_indices = np.array([0, 1], dtype=np.int64)
        corr = np.eye(p, dtype=np.float64)
        var_types = np.zeros(p, dtype=np.int32)
        var_params = np.zeros(p, dtype=np.float64)
        effects = np.array([b1, b2], dtype=np.float64)
        upload_normal = np.zeros((2, 2), dtype=np.float64)
        upload_data = np.zeros((2, 2), dtype=np.float64)

        X_template = np.zeros((n, p), dtype=np.float64)
        f_crit, t_crit, correction_t_crits = compute_crits(X_template, target_indices)

        results = {}
        for gen_backend, gen_label in [(native, "native_gen"), (python, "python_gen")]:
            reject = np.zeros((N_SIMS, p))

            for i, s in enumerate(rng_seeds):
                X = gen_backend.generate_X(
                    n,
                    p,
                    corr,
                    var_types,
                    var_params,
                    upload_normal,
                    upload_data,
                    s,
                )
                y = gen_backend.generate_y(X, effects, 0.0, 0.0, s + 1)

                # Always analyze with Python backend
                res = python.ols_analysis(X, y, target_indices, f_crit, t_crit, correction_t_crits, 0)
                reject[i] = res[1 : 1 + p]

            results[gen_label] = reject

        for j, (var, beta) in enumerate([("x1", b1), ("x2", b2)]):
            exact = analytical_t_power(beta, n, p=p, sigma_eps=1.0, vif_j=1.0)
            margin = mc_accuracy_margin(exact, N_SIMS)

            for gen_label in ["native_gen", "python_gen"]:
                rate = results[gen_label][:, j].mean() * 100
                err = abs(rate - exact)
                print(f"  {var} ({gen_label}): MC={rate:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%  [err={err:.2f}]")
                assert err < margin, f"[{gen_label} → python_ols] {var}: MC={rate:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"

            # Cross-check: both gen backends should give similar rates
            nat_rate = results["native_gen"][:, j].mean() * 100
            py_rate = results["python_gen"][:, j].mean() * 100
            diff = abs(nat_rate - py_rate)
            assert diff < 2.0, f"[datagen cross-check] {var}: native_gen={nat_rate:.2f}% vs python_gen={py_rate:.2f}% (diff={diff:.2f}%)"
