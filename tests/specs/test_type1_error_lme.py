"""
Type I error control tests for LME models.

Under H0 (all fixed effects = 0), rejection rate must equal alpha.
Follows the exact pattern of test_type1_error.py for OLS models.
"""

import contextlib
import io

import pytest

from tests.config import LME_N_SIMS_BENCHMARK, LME_THRESHOLD_STRICT, SEED
from tests.helpers.mc_margins import mc_margin
from tests.helpers.power_helpers import get_power

pytestmark = pytest.mark.lme


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _make_null_lme(equation, n_clusters, icc, n_sims=LME_N_SIMS_BENCHMARK, alpha=0.05, seed=SEED, max_failed=LME_THRESHOLD_STRICT):
    """Create an LME model with all fixed effects set to zero."""
    from mcpower import MCPower

    m = MCPower(equation)
    m.set_simulations(n_sims)
    m.set_seed(seed)
    m.set_alpha(alpha)
    m.set_max_failed_simulations(max_failed)
    m.set_cluster("g", ICC=icc, n_clusters=n_clusters)

    # Set all fixed effects to zero
    effect_names = [name for name in m._registry.effect_names if name != "g"]
    effects_str = ", ".join(f"{e}=0" for e in effect_names)
    m.set_effects(effects_str)
    return m


@pytest.mark.slow
class TestLMETypeIErrorControl:
    """Under H0 (all fixed effects = 0), rejection rate must equal alpha."""

    def test_single_predictor_null_overall(self):
        """LR test rejection rate ~ alpha with one predictor at zero effect."""
        m = _make_null_lme("y ~ x1 + (1|g)", n_clusters=20, icc=0.2)
        result = m.find_power(
            sample_size=1000,
            target_test="overall",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "overall")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, f"LR test power under H0: {power:.2f}%, expected {expected}% +/- {margin:.2f}%"

    def test_single_predictor_null_individual(self):
        """z-test rejection rate ~ alpha for a single zero-effect predictor."""
        m = _make_null_lme("y ~ x1 + (1|g)", n_clusters=20, icc=0.2)
        result = m.find_power(
            sample_size=1000,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, f"z-test power under H0: {power:.2f}%, expected {expected}% +/- {margin:.2f}%"

    def test_two_predictors_null_each(self):
        """Both predictors at zero -> each z-test rejects at ~alpha."""
        m = _make_null_lme("y ~ x1 + x2 + (1|g)", n_clusters=20, icc=0.2)
        result = m.find_power(
            sample_size=1000,
            target_test="x1, x2",
            print_results=False,
            return_results=True,
        )
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        for var in ["x1", "x2"]:
            power = get_power(result, var)
            assert abs(power - expected) < margin, f"{var} power under H0: {power:.2f}%, expected {expected}% +/- {margin:.2f}%"

    def test_large_sample_null_no_inflation(self):
        """
        Large N with zero effect must NOT inflate Type I error.

        Catches bugs where power grows with N even when effect = 0.
        """
        m = _make_null_lme("y ~ x1 + (1|g)", n_clusters=50, icc=0.2)
        result = m.find_power(
            sample_size=2500,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, (
            f"Large-N null power: {power:.2f}%, expected {expected}% +/- {margin:.2f}% (Type I error inflated with N?)"
        )


@pytest.mark.slow
class TestLMEAlphaCalibration:
    """Rejection rate tracks the nominal alpha across levels."""

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_null_rejection_matches_alpha(self, alpha):
        m = _make_null_lme("y ~ x1 + (1|g)", n_clusters=20, icc=0.2, alpha=alpha)
        result = m.find_power(
            sample_size=1000,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(alpha, m.n_simulations)
        expected = alpha * 100
        assert abs(power - expected) < margin, f"alpha={alpha}: observed {power:.2f}%, expected {expected}% +/- {margin:.2f}%"
