"""
Non-default alpha level tests — backend-agnostic.

Validates that the full alpha pipeline (power accuracy, corrections,
null calibration) works correctly at alpha != 0.05.
Tests run on ALL available backends via the backend fixture.
"""

import contextlib
import io

import numpy as np
import pytest

from tests.config import N_SIMS, SEED
from tests.helpers.analytical import analytical_f_power, analytical_t_power
from tests.helpers.mc_margins import mc_accuracy_margin, mc_margin
from tests.helpers.power_helpers import get_power, get_power_corrected, make_null_model


@pytest.fixture(autouse=True)
def _quiet():
    """Suppress stdout for all tests in this module."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── Class 1: Power accuracy at alpha != 0.05 ────────────────────────


class TestAlphaAccuracyVsAnalytical:
    """
    Compare MC power against analytical power at non-default alpha levels.

    DGP: y = Xβ + ε,  X ~ N(0, Σ),  ε ~ N(0, 1)  →  σ_ε = 1
    """

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    @pytest.mark.parametrize(
        "beta,n",
        [
            (0.3, 80),
            (0.3, 200),
            (0.5, 100),
        ],
    )
    def test_single_predictor_t_test_alpha(self, backend, alpha, beta, n):
        """t-test power matches analytical non-central t at non-default alpha."""
        from mcpower import MCPower

        m = MCPower("y = x1")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_alpha(alpha)
        m.set_effects(f"x1={beta}")
        result = m.find_power(
            sample_size=n,
            target_test="x1",
            print_results=False,
            return_results=True,
        )
        mc_power = get_power(result, "x1")
        exact_power = analytical_t_power(beta, n, p=1, sigma_eps=1.0, vif_j=1.0, alpha=alpha)
        margin = mc_accuracy_margin(exact_power, N_SIMS)
        assert abs(mc_power - exact_power) < margin, (
            f"[{backend}] alpha={alpha}, β={beta}, n={n}: MC={mc_power:.2f}%, analytical={exact_power:.2f}% ± {margin:.2f}%"
        )

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    @pytest.mark.parametrize(
        "b1,b2,n",
        [
            (0.3, 0.2, 100),
            (0.5, 0.3, 80),
        ],
    )
    def test_two_predictors_uncorrelated_alpha(self, backend, alpha, b1, b2, n):
        """Each t-test and F-test with Σ = I at non-default alpha."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_alpha(alpha)
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
            exact = analytical_t_power(
                beta,
                n,
                p=2,
                sigma_eps=1.0,
                vif_j=1.0,
                alpha=alpha,
            )
            margin = mc_accuracy_margin(exact, N_SIMS)
            assert abs(mc_power - exact) < margin, (
                f"[{backend}] alpha={alpha}, {var}: MC={mc_power:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"
            )

        mc_f = get_power(result, "overall")
        exact_f = analytical_f_power([b1, b2], n, Sigma, sigma_eps=1.0, alpha=alpha)
        margin_f = mc_accuracy_margin(exact_f, N_SIMS)
        assert abs(mc_f - exact_f) < margin_f, (
            f"[{backend}] alpha={alpha}, F-test: MC={mc_f:.2f}%, analytical={exact_f:.2f}% ± {margin_f:.2f}%"
        )

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    @pytest.mark.parametrize(
        "b1,b2,rho,n",
        [
            (0.3, 0.2, 0.5, 100),
            (0.5, 0.3, 0.5, 80),
        ],
    )
    def test_two_predictors_correlated_alpha(self, backend, alpha, b1, b2, rho, n):
        """VIF-corrected t-tests with correlated predictors at non-default alpha."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_alpha(alpha)
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
            exact = analytical_t_power(
                beta,
                n,
                p=2,
                sigma_eps=1.0,
                vif_j=vif,
                alpha=alpha,
            )
            margin = mc_accuracy_margin(exact, N_SIMS)
            assert abs(mc_power - exact) < margin, (
                f"[{backend}] alpha={alpha}, rho={rho}, {var}: MC={mc_power:.2f}%, analytical={exact:.2f}% ± {margin:.2f}%"
            )


# ── Class 2: Corrections at alpha != 0.05 ───────────────────────────


class TestAlphaCorrectionAccuracy:
    """
    Multiple comparison corrections must behave correctly at non-default alpha.

    Fills the gap where test_corrections.py only validates at alpha=0.05.
    """

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    @pytest.mark.parametrize("correction", ["bonferroni", "holm", "fdr"])
    def test_corrected_leq_uncorrected_at_alpha(self, backend, alpha, correction):
        """Corrected power <= uncorrected power when all effects = 0."""
        m = make_null_model("y = x1 + x2 + x3", n_sims=N_SIMS, alpha=alpha, seed=SEED)
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
            assert corr <= uncorr + 0.5, (
                f"[{backend}] alpha={alpha}, {correction}: corrected {corr:.2f}% > uncorrected {uncorr:.2f}% for {var}"
            )

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    @pytest.mark.parametrize("correction", ["bonferroni", "holm"])
    def test_fwer_controlled_at_alpha(self, backend, alpha, correction):
        """FWER-controlling methods keep per-test rejection below nominal alpha."""
        m = make_null_model("y = x1 + x2 + x3", n_sims=N_SIMS, alpha=alpha, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction=correction,
            print_results=False,
            return_results=True,
        )
        for var in ["x1", "x2", "x3"]:
            corr = get_power_corrected(result, var)
            assert corr < alpha * 100 + mc_margin(alpha, N_SIMS), (
                f"[{backend}] alpha={alpha}, {correction} FWER violation for {var}: corrected power = {corr:.2f}%"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    def test_bonferroni_more_conservative_than_fdr_at_alpha(self, backend, alpha):
        """Bonferroni should reject <= FDR (BH) under non-null at non-default alpha."""
        from mcpower import MCPower

        m = MCPower("y = x1 + x2 + x3")
        m.set_simulations(N_SIMS)
        m.set_seed(SEED)
        m.set_alpha(alpha)
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
            assert bonf <= fdr + 2.0, f"[{backend}] alpha={alpha}: Bonferroni ({bonf:.2f}%) > FDR ({fdr:.2f}%) for {var}"


# ── Class 3: Null calibration at alpha != 0.05 (multi-predictor) ────


class TestAlphaCalibrationExtended:
    """
    Extends TestAlphaCalibration from test_type1_error.py (single-predictor)
    to multi-predictor models and corrected rejection under the null.
    """

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_null_rejection_multi_predictor(self, backend, alpha):
        """Two-predictor null: each t-test and overall F-test reject at ~alpha."""
        m = make_null_model("y = x1 + x2", n_sims=N_SIMS, alpha=alpha, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="all",
            print_results=False,
            return_results=True,
        )
        margin = mc_margin(alpha, N_SIMS)
        expected = alpha * 100
        for test_name in ["x1", "x2", "overall"]:
            power = get_power(result, test_name)
            assert abs(power - expected) < margin, (
                f"[{backend}] alpha={alpha}, {test_name}: observed {power:.2f}%, expected {expected}% ± {margin:.2f}%"
            )

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    @pytest.mark.parametrize("correction", ["bonferroni", "holm"])
    def test_null_rejection_corrected_at_alpha(self, backend, alpha, correction):
        """Corrected null rejection stays below alpha + MC margin for 3 predictors."""
        m = make_null_model("y = x1 + x2 + x3", n_sims=N_SIMS, alpha=alpha, seed=SEED)
        result = m.find_power(
            sample_size=100,
            target_test="x1, x2, x3",
            correction=correction,
            print_results=False,
            return_results=True,
        )
        margin = mc_margin(alpha, N_SIMS)
        for var in ["x1", "x2", "x3"]:
            corr = get_power_corrected(result, var)
            assert corr < alpha * 100 + margin, (
                f"[{backend}] alpha={alpha}, {correction}, {var}: corrected rejection {corr:.2f}% exceeds {alpha * 100}% + {margin:.2f}%"
            )
