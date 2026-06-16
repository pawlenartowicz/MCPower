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
    assert m._pending_effects is None

    s = m.get_effects_from_data("y")

    assert "x1=" in s
    val = float(s.split("x1=")[1].split(",")[0])
    assert abs(val - 1.0) < 1e-3

    # Not auto-applied: effect state must be untouched.
    assert m._effects_set is False
    assert m._pending_effects is None

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
