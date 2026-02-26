"""
Power monotonicity tests.

Power must increase with effect size, sample size, and alpha.
"""

import contextlib
import io

import pytest

from tests.config import N_SIMS_ORDERING as N_SIMS, SEED
from tests.helpers.power_helpers import get_power


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestPowerMonotonicity:
    """Power must increase with effect size, sample size, and alpha."""

    def test_power_increases_with_effect_size(self):
        """Larger standardised beta → higher power."""
        from mcpower import MCPower

        powers = []
        for effect in [0.1, 0.3, 0.5]:
            m = MCPower("y = x1")
            m.set_simulations(N_SIMS)
            m.set_seed(SEED)
            m.set_effects(f"x1={effect}")
            result = m.find_power(
                sample_size=80,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in effect size: {powers}"

    def test_power_increases_with_sample_size(self):
        """Larger N → higher power (for non-zero effect)."""
        from mcpower import MCPower

        powers = []
        for n in [30, 80, 200]:
            m = MCPower("y = x1")
            m.set_simulations(N_SIMS)
            m.set_seed(SEED)
            m.set_effects("x1=0.3")
            result = m.find_power(
                sample_size=n,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in N: {powers}"

    def test_power_increases_with_alpha(self):
        """Less stringent alpha → higher power."""
        from mcpower import MCPower

        powers = []
        for alpha in [0.01, 0.05, 0.10]:
            m = MCPower("y = x1")
            m.set_simulations(N_SIMS)
            m.set_seed(SEED)
            m.set_alpha(alpha)
            m.set_effects("x1=0.3")
            result = m.find_power(
                sample_size=60,
                target_test="x1",
                print_results=False,
                return_results=True,
            )
            powers.append(get_power(result, "x1"))

        for i in range(len(powers) - 1):
            assert powers[i] < powers[i + 1], f"Power not monotonic in alpha: {powers}"


class TestPowerConvergence:
    """Power must approach 100% when signal is overwhelming."""

    def test_large_effect_high_power(self):
        """Very large effect → power near 100%."""
        from mcpower import MCPower

        m = MCPower("y = x1")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects("x1=1.0")
        result = m.find_power(
            sample_size=200,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        assert power > 99.0, f"Large-effect power should be ~100%, got {power:.2f}%"

    def test_large_n_moderate_effect(self):
        """Large N with moderate effect → power near 100%."""
        from mcpower import MCPower

        m = MCPower("y = x1")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects("x1=0.3")
        result = m.find_power(
            sample_size=500,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        assert power > 99.0, f"Large-N power should be ~100%, got {power:.2f}%"
