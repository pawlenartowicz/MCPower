"""
Power accuracy tests for LME models.

Compare MC power estimates against exact analytical power from
z-test (Wald) and LR (chi-squared) distributions for mixed models.

Follows the exact pattern of test_power_accuracy.py for OLS models.
"""

import contextlib
import io

import numpy as np
import pytest

from tests.config import LME_N_SIMS_BENCHMARK, LME_THRESHOLD_STRICT, SEED
from tests.helpers.analytical import analytical_lr_power_lme, analytical_z_power_lme
from tests.helpers.mc_margins import mc_accuracy_margin
from tests.helpers.power_helpers import get_power

pytestmark = pytest.mark.lme


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@pytest.mark.slow
class TestLMEAccuracyVsAnalytical:
    """
    Compare MC power estimates against exact analytical power for LME models.

    DGP: y_ij = Xb + b_i + eps,  X ~ N(0, Sigma) iid,
         b_i ~ N(0, tau^2),  eps ~ N(0, 1),  tau^2 = ICC/(1-ICC)
    """

    # -----------------------------------------------------------------
    # Single predictor z-test (individual fixed effect)
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "beta,n_total,n_clusters,icc",
        [
            (0.3, 1000, 20, 0.1),  # m=50, Deff~1.019
            (0.5, 1000, 20, 0.1),  # m=50, Deff~1.019
            (0.3, 1500, 30, 0.2),  # m=50, Deff~1.038
            (0.5, 1000, 20, 0.2),  # m=50, Deff~1.038
            (0.3, 1000, 20, 0.3),  # m=50, Deff~1.060
            (0.5, 1500, 30, 0.3),  # m=50, Deff~1.060
            (0.2, 2500, 50, 0.2),  # m=50, Deff~1.038
        ],
    )
    def test_single_predictor_z_test(self, beta, n_total, n_clusters, icc):
        """z-test power for individual fixed effect matches analytical."""
        from mcpower import MCPower

        m = MCPower("y ~ x1 + (1|g)")
        m.set_simulations(LME_N_SIMS_BENCHMARK)
        m.set_seed(SEED)
        m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
        m.set_cluster("g", ICC=icc, n_clusters=n_clusters)
        m.set_effects(f"x1={beta}")
        result = m.find_power(
            sample_size=n_total,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        mc_power = get_power(result, "x1")
        exact_power = analytical_z_power_lme(
            beta,
            n_total,
            n_clusters,
            icc,
            sigma_eps=1.0,
            vif_j=1.0,
        )
        margin = mc_accuracy_margin(exact_power, LME_N_SIMS_BENCHMARK)
        assert abs(mc_power - exact_power) < margin, (
            f"beta={beta}, n={n_total}, K={n_clusters}, ICC={icc}: MC={mc_power:.2f}%, analytical={exact_power:.2f}% +/- {margin:.2f}%"
        )

    # -----------------------------------------------------------------
    # Overall LR test (chi-squared)
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "beta,n_total,n_clusters,icc",
        [
            (0.3, 1000, 20, 0.1),
            (0.5, 1000, 20, 0.2),
            (0.3, 1500, 30, 0.3),
        ],
    )
    def test_single_predictor_lr_test(self, beta, n_total, n_clusters, icc):
        """LR test (overall) power matches analytical chi-squared."""
        from mcpower import MCPower

        m = MCPower("y ~ x1 + (1|g)")
        m.set_simulations(LME_N_SIMS_BENCHMARK)
        m.set_seed(SEED)
        m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
        m.set_cluster("g", ICC=icc, n_clusters=n_clusters)
        m.set_effects(f"x1={beta}")
        result = m.find_power(
            sample_size=n_total,
            target_test="overall",
            print_results=False,
            return_results=True,
        )
        mc_power = get_power(result, "overall")
        Sigma = np.eye(1)
        exact_power = analytical_lr_power_lme(
            [beta],
            n_total,
            n_clusters,
            icc,
            corr_matrix=Sigma,
            sigma_eps=1.0,
        )
        margin = mc_accuracy_margin(exact_power, LME_N_SIMS_BENCHMARK)
        assert abs(mc_power - exact_power) < margin, (
            f"LR test beta={beta}, n={n_total}, K={n_clusters}, ICC={icc}: "
            f"MC={mc_power:.2f}%, analytical={exact_power:.2f}% +/- {margin:.2f}%"
        )

    # -----------------------------------------------------------------
    # Two predictors uncorrelated (Sigma = I)
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "b1,b2,n_total,n_clusters,icc",
        [
            (0.3, 0.2, 1000, 20, 0.1),
            (0.5, 0.3, 1000, 20, 0.2),
        ],
    )
    def test_two_predictors_uncorrelated(self, b1, b2, n_total, n_clusters, icc):
        """Individual z-tests and overall LR with Sigma = I."""
        from mcpower import MCPower

        m = MCPower("y ~ x1 + x2 + (1|g)")
        m.set_simulations(LME_N_SIMS_BENCHMARK)
        m.set_seed(SEED)
        m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
        m.set_cluster("g", ICC=icc, n_clusters=n_clusters)
        m.set_effects(f"x1={b1}, x2={b2}")
        result = m.find_power(
            sample_size=n_total,
            target_test="all",
            print_results=False,
            return_results=True,
        )

        # Check individual z-tests
        for var, beta in [("x1", b1), ("x2", b2)]:
            mc_power = get_power(result, var)
            exact = analytical_z_power_lme(
                beta,
                n_total,
                n_clusters,
                icc,
                sigma_eps=1.0,
                vif_j=1.0,
            )
            margin = mc_accuracy_margin(exact, LME_N_SIMS_BENCHMARK)
            assert abs(mc_power - exact) < margin, f"{var}: MC={mc_power:.2f}%, analytical={exact:.2f}% +/- {margin:.2f}%"

        # Check overall LR test
        Sigma = np.eye(2)
        mc_f = get_power(result, "overall")
        exact_f = analytical_lr_power_lme(
            [b1, b2],
            n_total,
            n_clusters,
            icc,
            corr_matrix=Sigma,
            sigma_eps=1.0,
        )
        margin_f = mc_accuracy_margin(exact_f, LME_N_SIMS_BENCHMARK)
        assert abs(mc_f - exact_f) < margin_f, f"LR test: MC={mc_f:.2f}%, analytical={exact_f:.2f}% +/- {margin_f:.2f}%"

    # -----------------------------------------------------------------
    # Two predictors correlated (VIF impact)
    # -----------------------------------------------------------------
    @pytest.mark.parametrize(
        "b1,b2,rho,n_total,n_clusters,icc",
        [
            (0.3, 0.2, 0.3, 1000, 20, 0.1),
            (0.5, 0.3, 0.5, 1500, 30, 0.2),
        ],
    )
    def test_two_predictors_correlated(self, b1, b2, rho, n_total, n_clusters, icc):
        """Individual z-tests with correlated predictors: VIF matters."""
        from mcpower import MCPower

        m = MCPower("y ~ x1 + x2 + (1|g)")
        m.set_simulations(LME_N_SIMS_BENCHMARK)
        m.set_seed(SEED)
        m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
        m.set_cluster("g", ICC=icc, n_clusters=n_clusters)
        m.set_effects(f"x1={b1}, x2={b2}")
        m.set_correlations(f"(x1,x2)={rho}")
        result = m.find_power(
            sample_size=n_total,
            target_test="x1, x2",
            print_results=False,
            return_results=True,
        )

        Sigma = np.array([[1, rho], [rho, 1]])
        Sigma_inv = np.linalg.inv(Sigma)

        for idx, (var, beta) in enumerate([("x1", b1), ("x2", b2)]):
            vif = Sigma_inv[idx, idx]
            mc_power = get_power(result, var)
            exact = analytical_z_power_lme(
                beta,
                n_total,
                n_clusters,
                icc,
                sigma_eps=1.0,
                vif_j=vif,
            )
            margin = mc_accuracy_margin(exact, LME_N_SIMS_BENCHMARK)
            assert abs(mc_power - exact) < margin, f"rho={rho}, {var}: MC={mc_power:.2f}%, analytical={exact:.2f}% +/- {margin:.2f}%"
