"""
Type I error control tests.

Under H0 (effect = 0), rejection rate must equal alpha.
"""

import contextlib
import io

import pytest

from tests.config import N_SIMS_STANDARD as N_SIMS, SEED
from tests.helpers.mc_margins import mc_margin
from tests.helpers.power_helpers import get_power, make_null_model


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestTypeIErrorControl:
    """Under H0 (effect = 0), rejection rate must equal alpha."""

    def test_single_predictor_null_overall(self):
        """F-test rejection rate ≈ alpha with one predictor at zero effect."""
        m = make_null_model("y = x1", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="overall",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "overall")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, f"F-test power under H0: {power:.2f}%, expected {expected}% ± {margin:.2f}%"

    def test_single_predictor_null_individual(self):
        """t-test rejection rate ≈ alpha for a single zero-effect predictor."""
        m = make_null_model("y = x1", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, f"t-test power under H0: {power:.2f}%, expected {expected}% ± {margin:.2f}%"

    def test_two_predictors_null_each(self):
        """Both predictors at zero → each t-test rejects at ~alpha."""
        m = make_null_model("y = x1 + x2", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1, x2",
            print_results=False,
            return_results=True,
        )
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        for var in ["x1", "x2"]:
            power = get_power(result, var)
            assert abs(power - expected) < margin, f"{var} power under H0: {power:.2f}%, expected {expected}% ± {margin:.2f}%"

    def test_large_sample_null(self):
        """
        Large N with zero effect must NOT inflate Type I error.

        This catches bugs where power grows with N even when effect = 0.
        """
        m = make_null_model("y = x1", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=500,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(m.alpha, m.n_simulations)
        expected = m.alpha * 100
        assert abs(power - expected) < margin, (
            f"Large-N null power: {power:.2f}%, expected {expected}% ± {margin:.2f}% (Type I error inflated with N?)"
        )


class TestAlphaCalibration:
    """Rejection rate tracks the nominal alpha across levels."""

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_null_rejection_matches_alpha(self, alpha):
        m = make_null_model("y = x1", n_sims=N_SIMS, alpha=alpha, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        power = get_power(result, "x1")
        margin = mc_margin(alpha, m.n_simulations)
        expected = alpha * 100
        assert abs(power - expected) < margin, f"alpha={alpha}: observed {power:.2f}%, expected {expected}% ± {margin:.2f}%"
