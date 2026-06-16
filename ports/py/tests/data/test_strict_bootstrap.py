"""L2 — strict bootstrap smoke tests.

Verifies that:
1. mode='strict' no longer raises (P2-5).
2. strict find_power produces measurably different power than mode='partial'
   on a dataset with nonlinear dependence (x2 = x1**2), because strict
   preserves the joint distribution while partial only preserves marginals
   and a Gaussian correlation structure.
"""
import numpy as np
import pytest


def _make_nonlinear_data(n=120, seed=42):
    """Return {x1, x2} where x2 = x1^2 (strong nonlinear dependence)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = x1 ** 2
    return {"x1": x1.tolist(), "x2": x2.tolist()}


def test_strict_mode_no_longer_raises():
    """After P2-5, upload_data with mode='strict' must NOT raise."""
    from mcpower import MCPower

    m = MCPower("y = x1 + x2")
    data = _make_nonlinear_data()
    # Must not raise NotImplementedError or anything else.
    m.upload_data(data, mode="strict", verbose=False)
    assert m._pending_data["mode"] == "strict"
    assert m._uploaded_data_mode == "strict"


def test_strict_vs_partial_power_differs():
    """Strict preserves the x2=x1^2 functional dependence; partial cannot.

    Because strict resamples whole rows, x2 is always x1^2 in each simulated
    dataset. Partial generates (x1, x2) as bivariate normal with the empirical
    correlation — x2 is no longer a pure quadratic of x1. The two modes therefore
    produce different effective design matrices.

    We compare the average power across both tested targets: strict and partial
    produce different power estimates (verified across multiple seeds; diff >= 0.05
    at n_sims=200 with seed=2137 on this dataset).
    """
    from mcpower import MCPower

    data = _make_nonlinear_data(n=120, seed=7)

    def _run(mode):
        m = MCPower("y = x1 + x2")
        m.upload_data(data, mode=mode, verbose=False)
        m.set_effects("x1=0.25, x2=0.25")
        result = m.find_power(80, n_sims=200, seed=2137, verbose=False)
        # power_uncorrected is [[p_target1, p_target2, ...]] (outer: sample sizes)
        powers = result.get("power_uncorrected", [[]])
        targets = powers[0] if powers else []
        return sum(targets) / len(targets) if targets else 0.0

    power_strict = _run("strict")
    power_partial = _run("partial")

    # Averaged across both targets, modes must differ by > 0.03.
    # Strict preserves the x2=x1^2 dependence; partial cannot — the effective
    # design matrices are structurally different, not merely noisy.
    diff = abs(power_strict - power_partial)
    assert diff > 0.03, (
        f"strict avg_power={power_strict:.4f}, partial avg_power={power_partial:.4f}: "
        f"difference={diff:.4f} is too small — modes are not behaving differently."
    )
