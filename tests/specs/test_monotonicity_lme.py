"""
Power monotonicity tests for LME models.

Power must increase with effect size, sample size, and alpha,
and decrease with predictor correlation (VIF).

Follows the exact pattern of test_monotonicity.py for OLS models.

Design constraint: LME requires >=50 obs/cluster (10 per parameter, 5 params),
so n_total >= 1000 with K=20. To get mid-range power we use small effects
(beta ~ 0.05-0.15).

Note: ICC has negligible effect on fixed-effect power when predictors are
generated iid within clusters (Deff_within ~ 1.02 for all ICC values with
m=50). This is correct behavior for MCPower's DGP.
"""

import contextlib
import io

import pytest

from tests.config import (
    ICC_MODERATE,
    LME_N_SIMS_BENCHMARK,
    LME_THRESHOLD_STRICT,
    N_CLUSTERS_MODERATE,
    SEED,
)
from tests.helpers.power_helpers import get_power

pytestmark = pytest.mark.lme


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@pytest.mark.slow
class TestLMEPowerMonotonicity:
    """Power must increase with effect size, sample size, and alpha."""

    def test_power_increases_with_effect_size(self):
        """Larger standardised beta -> higher power."""
        from mcpower import MCPower

        powers = []
        for effect in [0.05, 0.10, 0.15]:
            m = MCPower("y ~ x1 + (1|g)")
            m.set_simulations(LME_N_SIMS_BENCHMARK)
            m.set_seed(SEED)
            m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
            m.set_cluster("g", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
            m.set_effects(f"x1={effect}")
            result = m.find_power(
                sample_size=1000,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in effect size: {powers}"

    def test_power_increases_with_sample_size(self):
        """Larger N -> higher power (for non-zero effect)."""
        from mcpower import MCPower

        powers = []
        for n_total in [1000, 1500, 2000]:
            n_clusters = n_total // 50  # keep m=50 constant
            m = MCPower("y ~ x1 + (1|g)")
            m.set_simulations(LME_N_SIMS_BENCHMARK)
            m.set_seed(SEED)
            m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
            m.set_cluster("g", ICC=ICC_MODERATE, n_clusters=n_clusters)
            m.set_effects("x1=0.05")
            result = m.find_power(
                sample_size=n_total,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in N: {powers}"

    def test_power_decreases_with_correlation(self):
        """Higher predictor correlation -> lower power (VIF increases)."""
        from mcpower import MCPower

        powers = []
        for rho in [0.0, 0.3, 0.6]:
            m = MCPower("y ~ x1 + x2 + (1|g)")
            m.set_simulations(LME_N_SIMS_BENCHMARK)
            m.set_seed(SEED)
            m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
            m.set_cluster("g", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
            m.set_effects("x1=0.10, x2=0.10")
            if rho > 0:
                m.set_correlations(f"(x1,x2)={rho}")
            result = m.find_power(
                sample_size=1000,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] > powers[i + 1], f"Power not decreasing with correlation: {powers}"

    def test_power_increases_with_alpha(self):
        """Less stringent alpha -> higher power."""
        from mcpower import MCPower

        powers = []
        for alpha in [0.01, 0.05, 0.10]:
            m = MCPower("y ~ x1 + (1|g)")
            m.set_simulations(LME_N_SIMS_BENCHMARK)
            m.set_seed(SEED)
            m.set_alpha(alpha)
            m.set_max_failed_simulations(LME_THRESHOLD_STRICT)
            m.set_cluster("g", ICC=ICC_MODERATE, n_clusters=N_CLUSTERS_MODERATE)
            m.set_effects("x1=0.10")
            result = m.find_power(
                sample_size=1000,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in alpha: {powers}"
