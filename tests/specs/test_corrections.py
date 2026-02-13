"""
Multiple comparison correction tests — backend-agnostic.

Tests run on ALL available backends via the backend fixture.
"""

import contextlib
import io

import pytest

from tests.config import N_SIMS, SEED
from tests.helpers.mc_margins import mc_margin
from tests.helpers.power_helpers import get_power, get_power_corrected, make_null_model


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestCorrectionConservativeness:
    """
    Under H0 all corrections must control the family-wise or
    false-discovery rate, hence corrected rejection ≤ uncorrected.
    """

    @pytest.mark.parametrize("correction", ["bonferroni", "holm", "fdr"])
    def test_corrected_leq_uncorrected_under_null(self, backend, correction):
        """Corrected power ≤ uncorrected power when all effects = 0."""
        m = make_null_model("y = x1 + x2 + x3", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction=correction,
            print_results=False,
            return_results=True,
        )
        for var in ["x1", "x2", "x3"]:
            uncorr = get_power(result, var)
            corr = get_power_corrected(result, var)
            assert corr <= uncorr + 0.5, (  # tiny tolerance for MC noise
                f"[{backend}] {correction}: corrected {corr:.2f}% > uncorrected {uncorr:.2f}% for {var}"
            )

    @pytest.mark.parametrize("correction", ["bonferroni", "holm"])
    def test_fwer_controlled_under_null(self, backend, correction):
        """
        Family-wise error rate under H0 should be ≤ alpha.

        Bonferroni and Holm both guarantee FWER control.
        """
        m = make_null_model("y = x1 + x2 + x3", n_sims=N_SIMS, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction=correction,
            print_results=False,
            return_results=True,
        )
        for var in ["x1", "x2", "x3"]:
            corr = get_power_corrected(result, var)
            # Under complete null, FWER-controlling methods should have
            # per-test rejection well below the nominal alpha
            assert corr < m.alpha * 100 + mc_margin(m.alpha, m.n_simulations), (
                f"[{backend}] {correction} FWER violation for {var}: corrected power = {corr:.2f}%"
            )

    @pytest.mark.slow
    def test_bonferroni_more_conservative_than_fdr(self, backend):
        """Bonferroni should reject ≤ FDR (BH) under non-null."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2 + x3")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_effects("x1=0.3, x2=0.2, x3=0.1")

        result_bonf = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction="bonferroni",
            print_results=False,
            return_results=True,
        )
        result_fdr = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction="fdr",
            print_results=False,
            return_results=True,
        )
        for var in ["x1", "x2", "x3"]:
            bonf = get_power_corrected(result_bonf, var)
            fdr = get_power_corrected(result_fdr, var)
            # Bonferroni ≤ BH-FDR (with MC tolerance)
            assert bonf <= fdr + 2.0, f"[{backend}] Bonferroni ({bonf:.2f}%) > FDR ({fdr:.2f}%) for {var}"
