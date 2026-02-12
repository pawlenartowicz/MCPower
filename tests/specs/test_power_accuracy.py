"""
Power accuracy tests — backend-agnostic.

Compare MC power estimates against exact analytical power from
non-central t / F distributions.
Tests run on ALL available backends via the backend fixture.
"""

import contextlib
import io

import numpy as np
import pytest

from tests.config import N_SIMS, SEED
from tests.helpers.analytical import analytical_f_power, analytical_t_power
from tests.helpers.mc_margins import mc_accuracy_margin
from tests.helpers.power_helpers import get_power


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestAccuracyVsAnalytical:
    """
    Compare MC power estimates against exact analytical power.

    DGP: y = Xβ + ε,  X ~ N(0, Σ),  ε ~ N(0, 1)  →  σ_ε = 1
    """

    @pytest.mark.parametrize(
        "beta,n",
        [
            (0.1, 200),
            (0.3, 80),
            (0.3, 200),
            (0.5, 50),
            (0.5, 150),
        ],
    )
    def test_single_predictor_t_test(self, backend, beta, n):
        """t-test power matches analytical non-central t."""
        from mcpower import MCPower

        m = MCPower("y = x1")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects(f"x1={beta}")
        result = m.find_power(
            sample_size=n,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        mc_power = get_power(result, "x1")
        exact_power = analytical_t_power(beta, n, p=1, sigma_eps=1.0, vif_j=1.0)
        margin = mc_accuracy_margin(exact_power, N_SIMS)
        assert abs(mc_power - exact_power) < margin, (
            f"[{backend}] β={beta}, n={n}: MC={mc_power:.2f}%, analytical={exact_power:.2f}% ± {margin:.2f}%"
        )

    @pytest.mark.parametrize(
        "b1,b2,n",
        [
            (0.3, 0.2, 100),
            (0.5, 0.3, 80),
            (0.2, 0.2, 200),
        ],
    )
    def test_two_predictors_uncorrelated(self, backend, b1, b2, n):
        """Each t-test and F-test with Σ = I."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects(f"x1={b1}, x2={b2}")
        result = m.find_power(
            sample_size=n,
            target_test="all",
            print_results=False,
            return_results=True,
        )

        Sigma = np.eye(2)
        for var, beta in [("x1", b1), ("x2", b2)]:
            mc_power = get_power(result, var)
            exact = analytical_t_power(beta, n, p=2, sigma_eps=1.0, vif_j=1.0)
            margin = mc_accuracy_margin(exact, N_SIMS)
            assert abs(mc_power - exact) < margin, f"[{backend}] {var}: MC={mc_power:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"

        mc_f = get_power(result, "overall")
        exact_f = analytical_f_power([b1, b2], n, Sigma, sigma_eps=1.0)
        margin_f = mc_accuracy_margin(exact_f, N_SIMS)
        assert abs(mc_f - exact_f) < margin_f, f"[{backend}] F-test: MC={mc_f:.2f}%, analytical={exact_f:.2f}% ± {margin_f:.2f}%"

    @pytest.mark.parametrize(
        "b1,b2,rho,n",
        [
            (0.3, 0.2, 0.3, 100),
            (0.3, 0.2, 0.5, 100),
            (0.3, 0.2, 0.7, 150),
            (0.5, 0.3, 0.5, 80),
        ],
    )
    def test_two_predictors_correlated_t_tests(self, backend, b1, b2, rho, n):
        """Individual t-tests with correlated predictors: VIF matters."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects(f"x1={b1}, x2={b2}")
        m.set_correlations(f"(x1,x2)={rho}")
        result = m.find_power(
            sample_size=n,
            target_test="x1, x2",
            print_results=False,
            return_results=True,
        )

        Sigma = np.array([[1, rho], [rho, 1]])
        Sigma_inv = np.linalg.inv(Sigma)

        for idx, (var, beta) in enumerate([("x1", b1), ("x2", b2)]):
            vif = Sigma_inv[idx, idx]
            mc_power = get_power(result, var)
            exact = analytical_t_power(beta, n, p=2, sigma_eps=1.0, vif_j=vif)
            margin = mc_accuracy_margin(exact, N_SIMS)
            assert abs(mc_power - exact) < margin, (
                f"[{backend}] rho={rho}, {var}: MC={mc_power:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"
            )
