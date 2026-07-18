"""Tests for ``MCPower.get_effects_from_data(y)`` (Task D4).

``get_effects_from_data`` fits the already-specified model against uploaded
data and returns a ``set_effects``-style string of standardized coefficients.
It is an *approximation* (standardization + random-X assumption + sampling
error) and does NOT auto-apply the recovered effects.

The required gate is recovery of a CONTINUOUS main effect to high precision.
Factor/interaction-string fidelity is a documented follow-up (the d-style
scaling is the documented approximation), so it is intentionally not gated
here.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from mcpower import MCPower


def test_get_effects_from_data_returns_parseable_string_no_autoapply():
    """A deterministic ``y = 2*x1`` yields a standardized slope of 1.0 (the
    correlation), the string parses back through ``set_effects``, and the call
    does NOT mutate the model's effect state."""
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=200)
    y = 2.0 * x1  # deterministic ⇒ standardized slope = corr = 1.0

    m = MCPower("y = x1")
    m.upload_data({"x1": x1.tolist(), "y": y.tolist()}, verbose=False)

    # No effects set as a side effect: confirm before the call.
    assert m._effects_set is False
    assert m._pending_effects == []  # accumulator empty (no set_effects call)

    s = m.get_effects_from_data("y")

    assert "x1=" in s
    val = float(s.split("x1=")[1].split(",")[0])
    assert abs(val - 1.0) < 1e-3

    # Not auto-applied: effect state must be untouched.
    assert m._effects_set is False
    assert m._pending_effects == []  # accumulator still empty

    # The returned string parses back through set_effects on a fresh model.
    m2 = MCPower("y = x1")
    m2.set_effects(s)  # must not raise


def test_get_effects_from_data_recovers_two_continuous_slopes():
    """With independent predictors and a deterministic linear y, both
    standardized slopes are recovered. Confirms canonical column ordering maps
    betas → names correctly when there is more than one predictor."""
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Standardize to known population scale so coefficients are interpretable.
    x1s = (x1 - x1.mean()) / x1.std()
    x2s = (x2 - x2.mean()) / x2.std()
    y = 0.6 * x1s + 0.3 * x2s  # deterministic, no noise

    m = MCPower("y = x1 + x2")
    m.upload_data({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}, verbose=False)
    s = m.get_effects_from_data("y")

    parsed = dict(
        (k.strip(), float(v))
        for k, v in (tok.split("=") for tok in s.split(","))
    )
    # y is standardized to unit variance, so slopes shrink proportionally but
    # their RATIO is preserved (0.6 : 0.3 == 2:1). Recover the standardized
    # slopes directly: with orthogonal-ish predictors they're close to the
    # correlation of y with each x, scaled by y's sd.
    sd_y = y.std()
    assert parsed["x1"] == pytest.approx(0.6 / sd_y, abs=1e-2)
    assert parsed["x2"] == pytest.approx(0.3 / sd_y, abs=1e-2)


def test_get_effects_from_data_missing_predictor_errors():
    """If a modeled main-effect predictor is absent from the upload, the
    multivariable fit can't estimate it — raise a clear error."""
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=100)
    y = x1 + rng.normal(size=100)

    m = MCPower("y = x1 + x2")
    # Upload only x1 + y; x2 is modeled but missing.
    m.upload_data({"x1": x1.tolist(), "y": y.tolist()}, verbose=False)

    with pytest.raises(ValueError, match="x2"):
        m.get_effects_from_data("y")


def test_get_effects_from_data_no_uploaded_data_errors():
    """Calling without any uploaded data is an error."""
    m = MCPower("y = x1")
    with pytest.raises((RuntimeError, ValueError)):
        m.get_effects_from_data("y")


def test_get_effects_from_data_missing_y_column_errors():
    """The ``y`` argument must name an uploaded column."""
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=100)
    y = x1.copy()

    m = MCPower("y = x1")
    m.upload_data({"x1": x1.tolist(), "y": y.tolist()}, verbose=False)
    with pytest.raises((ValueError, KeyError), match="nope"):
        m.get_effects_from_data("nope")


def test_get_effects_from_data_recovers_glm_log_odds():
    """Logit: a saturated 2x2 table with odds ratio 4 → the recovered log-odds
    coefficient of the binary predictor is ln(4). The GLM fit uses the native
    0/1 outcome (NOT a z-scored outcome)."""
    # x=1 group (30 rows): 20 successes / 10 failures → odds 2.0
    # x=0 group (30 rows): 10 successes / 20 failures → odds 0.5 → OR = 4.
    x = [1.0] * 30 + [0.0] * 30
    y = [1.0] * 20 + [0.0] * 10 + [1.0] * 10 + [0.0] * 20

    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.5)
    m.upload_data({"x": x, "y": y}, verbose=False)
    s = m.get_effects_from_data("y")

    assert "x=" in s
    val = float(s.split("x=")[1].split(",")[0])
    assert abs(val - np.log(4.0)) < 1e-3


def test_get_effects_from_data_recovers_mle_fixed_effect():
    """Mixed: clustered data ``y = β·x + u_cluster + ε``. The fixed effect is
    recovered on the z-scored-x scale (``β·sd(x)``), fitting on the native
    outcome and threading the uploaded grouping column as cluster IDs."""
    rng = np.random.default_rng(11)
    n, n_clusters = 200, 8
    csize = n // n_clusters
    x = rng.normal(size=n)
    group = np.repeat(np.arange(n_clusters), csize).astype(float)
    u = (group - (n_clusters - 1) / 2.0) * 0.4  # per-cluster intercept
    eps = rng.normal(scale=0.1, size=n)
    beta_true = 1.5
    y = beta_true * x + u + eps

    m = MCPower("y ~ x + (1|group)", family="lme")
    m.set_cluster("group", ICC=0.2, n_clusters=n_clusters)
    m.upload_data(
        {"x": x.tolist(), "group": group.tolist(), "y": y.tolist()},
        verbose=False,
    )
    s = m.get_effects_from_data("y")

    assert "x=" in s
    val = float(s.split("x=")[1].split(",")[0])
    expected = beta_true * x.std()  # recovered on the z-scored-x scale
    assert abs(val - expected) < 0.15


def _icc_from_note(captured: str) -> float:
    """Pull the estimated ICC value out of the verbose note's ``Estimated ICC``
    line (single-grouping snippet form). Raises if the line is absent."""
    line = next(ln for ln in captured.splitlines() if ln.startswith("Estimated ICC"))
    return float(line.split(":", 1)[1].split("(")[0].strip())


def test_get_effects_from_data_reports_icc_linear_mixed(capsys):
    """Clustered linear-mixed upload: the verbose note reports an estimated ICC
    in (0,1) recovered from τ̂²/(τ̂²+σ̂²). Data is a random-intercept model with
    true ICC ≈ 0.33, so the recovered value must sit in a discriminating band
    (the latent π²/3 formula would land near 0.13, outside it). Deterministic:
    a second call prints the identical value."""
    rng = np.random.default_rng(7)
    n_clusters, csize = 40, 25
    n = n_clusters * csize
    x = rng.normal(size=n)
    u = rng.normal(scale=np.sqrt(0.5), size=n_clusters)  # var(u)=0.5
    group = np.repeat(np.arange(n_clusters), csize).astype(float)
    eps = rng.normal(scale=1.0, size=n)  # var(eps)=1 ⇒ true ICC = 0.5/1.5 ≈ 0.33
    y = 0.8 * x + u[np.repeat(np.arange(n_clusters), csize)] + eps

    m = MCPower("y ~ x + (1|group)", family="lme")
    m.set_cluster("group", ICC=0.3, n_clusters=n_clusters)
    m.upload_data(
        {"x": x.tolist(), "group": group.tolist(), "y": y.tolist()},
        verbose=False,
    )

    m.get_effects_from_data("y")
    out1 = capsys.readouterr().out
    assert "Estimated ICC:" in out1
    assert "logit latent scale" not in out1
    assert "set_cluster('group', ICC=" in out1
    # Gaussian outcome: betas[0] is a mean offset, not a probability — no baseline line.
    assert "Estimated baseline probability" not in out1
    icc1 = _icc_from_note(out1)
    assert 0.15 < icc1 < 0.55  # band around the true 0.33; excludes a latent-formula slip

    # Deterministic — a second call reproduces the same value exactly.
    m.get_effects_from_data("y")
    icc2 = _icc_from_note(capsys.readouterr().out)
    assert icc1 == icc2


def test_get_effects_from_data_reports_icc_logistic_latent(capsys):
    """Clustered logistic upload: the verbose note reports a *latent-scale* ICC
    in (0,1), computed as τ̂²/(τ̂²+π²/3) — GLMM τ̂²-from-upload is noisy so only a
    loose bound is asserted."""
    rng = np.random.default_rng(13)
    n_clusters, csize = 30, 40
    n = n_clusters * csize
    x = rng.normal(size=n)
    u = rng.normal(scale=1.2, size=n_clusters)  # strong cluster log-odds offsets
    group = np.repeat(np.arange(n_clusters), csize).astype(float)
    eta = 0.7 * x + u[np.repeat(np.arange(n_clusters), csize)]
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)

    m = MCPower("y ~ x + (1|group)", family="logit")
    m.set_cluster("group", ICC=0.2, n_clusters=n_clusters)
    m.upload_data(
        {"x": x.tolist(), "group": group.tolist(), "y": y.tolist()},
        verbose=False,
    )

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    assert "Estimated ICC (logit latent scale):" in out
    icc = _icc_from_note(out)
    assert 0.0 < icc < 1.0
    # A binary outcome also reports the recovered baseline probability (a
    # probability ∈ (0,1)), independent of clustering.
    bline = next(
        ln for ln in out.splitlines() if ln.startswith("Estimated baseline probability:")
    )
    p_hat = float(bline.split(":", 1)[1].split("(")[0].strip())
    assert 0.0 < p_hat < 1.0
    assert "set_baseline_probability(" in out


def test_get_effects_from_data_probit_recovers_baseline_probability(capsys):
    """B1 round-trip: data generated from a probit model with true baseline
    p=0.30 must recover ≈0.30, not the logit-mistaken 0.372 (logistic applied
    to a probit intercept). The recovery is `Φ(β̂0)` — using `logistic` here
    would systematically overshoot because `Φ⁻¹(0.30) ≈ -0.524` and
    `logistic(-0.524) ≈ 0.372 != 0.30`."""
    nd = statistics.NormalDist()
    rng = np.random.default_rng(21)
    n = 8000
    x = rng.normal(size=n)
    beta_true = 0.5
    b0_true = nd.inv_cdf(0.30)
    eta = b0_true + beta_true * x
    p = np.array([nd.cdf(e) for e in eta])
    y = (rng.uniform(size=n) < p).astype(float)

    m = MCPower("y = x", family="probit")
    m.set_baseline_probability(0.30)
    m.upload_data({"x": x.tolist(), "y": y.tolist()}, verbose=False)

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    bline = next(
        ln for ln in out.splitlines() if ln.startswith("Estimated baseline probability:")
    )
    p_hat = float(bline.split(":", 1)[1].split("(")[0].strip())
    assert abs(p_hat - 0.30) < 0.05, (
        f"probit baseline recovery {p_hat} should be close to the true 0.30"
    )


def test_get_effects_from_data_logit_recovers_baseline_probability(capsys):
    """B1 companion: pin the logit path at the same time so the link branch
    cannot collapse silently (both arms recovering the same wrong number
    would otherwise pass a probit-only regression test)."""
    rng = np.random.default_rng(23)
    n = 8000
    x = rng.normal(size=n)
    beta_true = 0.5
    b0_true = math.log(0.30 / 0.70)
    eta = b0_true + beta_true * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)

    m = MCPower("y = x", family="logit")
    m.set_baseline_probability(0.30)
    m.upload_data({"x": x.tolist(), "y": y.tolist()}, verbose=False)

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    bline = next(
        ln for ln in out.splitlines() if ln.startswith("Estimated baseline probability:")
    )
    p_hat = float(bline.split(":", 1)[1].split("(")[0].strip())
    assert abs(p_hat - 0.30) < 0.05, (
        f"logit baseline recovery {p_hat} should be close to the true 0.30"
    )


def test_get_effects_from_data_reports_icc_probit_latent(capsys):
    """Clustered probit upload (M9): the verbose note must use the probit
    latent-scale formula (residual variance fixed at 1, not logit's π²/3) —
    mirrors driver.rs's cluster_icc branching for ``BinaryLink::Probit``."""
    nd = statistics.NormalDist()
    rng = np.random.default_rng(17)
    n_clusters, csize = 30, 40
    n = n_clusters * csize
    x = rng.normal(size=n)
    u = rng.normal(scale=1.0, size=n_clusters)
    group = np.repeat(np.arange(n_clusters), csize).astype(float)
    eta = 0.7 * x + u[np.repeat(np.arange(n_clusters), csize)]
    p = np.array([nd.cdf(e) for e in eta])
    y = (rng.uniform(size=n) < p).astype(float)

    m = MCPower("y ~ x + (1|group)", family="probit")
    m.set_cluster("group", ICC=0.2, n_clusters=n_clusters)
    m.upload_data(
        {"x": x.tolist(), "group": group.tolist(), "y": y.tolist()},
        verbose=False,
    )

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    assert "Estimated ICC (probit latent scale):" in out
    assert "logit latent scale" not in out
    icc = _icc_from_note(out)
    assert 0.0 < icc < 1.0

    # Discriminate against the (wrong) logit residual π²/3: the τ² implied by
    # the printed ICC under the correct residual=1 formula would print a
    # materially different ICC under the logit formula.
    tau_sq_implied = icc / (1.0 - icc)
    icc_if_logit_formula = tau_sq_implied / (tau_sq_implied + math.pi ** 2 / 3.0)
    assert abs(icc - icc_if_logit_formula) > 0.05


def test_get_effects_from_data_poisson_reports_no_icc_line(capsys):
    """Clustered Poisson upload (M9): the ICC line is suppressed entirely — a
    log-link count model has no latent-scale residual to form an ICC ratio
    against (raw τ², not ICC-derived). Mirrors driver.rs's cluster_icc, which
    returns ``None`` for ``MixedOutcome::Poisson``."""
    rng = np.random.default_rng(19)
    n_clusters, csize = 20, 20
    n = n_clusters * csize
    x = rng.normal(size=n)
    u = rng.normal(scale=np.sqrt(0.3), size=n_clusters)
    group = np.repeat(np.arange(n_clusters), csize).astype(float)
    eta = 0.3 * x + u[np.repeat(np.arange(n_clusters), csize)]
    y = rng.poisson(np.exp(eta)).astype(float)

    m = MCPower("y ~ x + (1|group)", family="poisson")
    m.set_cluster("group", tau_squared=0.3, n_clusters=n_clusters)
    m.upload_data(
        {"x": x.tolist(), "group": group.tolist(), "y": y.tolist()},
        verbose=False,
    )

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    assert "APPROXIMATION" in out
    assert "Estimated ICC" not in out


def test_get_effects_from_data_ols_reports_no_icc(capsys):
    """Non-clustered OLS: the approximation note is unchanged — no ICC line."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=100)
    y = 1.5 * x + rng.normal(scale=0.3, size=100)

    m = MCPower("y = x")
    m.upload_data({"x": x.tolist(), "y": y.tolist()}, verbose=False)

    m.get_effects_from_data("y")
    out = capsys.readouterr().out
    assert "APPROXIMATION" in out
    assert "Estimated ICC" not in out
    # Continuous OLS outcome: no baseline probability either.
    assert "Estimated baseline probability" not in out
