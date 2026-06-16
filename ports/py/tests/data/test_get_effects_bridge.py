"""L2 tests for the ``fit_uploaded_data`` PyO3 bridge.

Verifies that the bridge correctly calls the orchestrator's ``debug_load_data``
and returns a dict with the expected fields.  Uses a trivial OLS model
``y = x`` (one normal predictor, intercept 0) with a known design so we can
assert exact beta recovery.

Design layout (column-major, i.e. flat list col0 then col1):
  "intercept" col (index 0) = [1, 1, 1, 1, 1]
  "col_1"     col (index 1) = [1, 2, 3, 4, 5]  (the x predictor)
  design = [1, 1, 1, 1, 1,  1, 2, 3, 4, 5]
  nrow=5, ncol=2

The engine names the intercept column "intercept" and predictors "col_1", "col_2", …
Betas are mapped by design_columns (authoritative), not positional index.

Outcome: y = 2*x → [2, 4, 6, 8, 10] (true intercept 0, true slope 2).
Expected: beta for col[1] ≈ 2.0, beta for col[0] ≈ 0.0 (within 1e-6).
"""

import json
import math

import numpy as np
import pytest

from mcpower import MCPower, _engine

# ── Minimal LinearSpec for "y = x" (one normal predictor, OLS) ──────────────
_LINEAR_SPEC_JSON = json.dumps(
    {
        "formula": "y = x",
        "predictors": [{"name": "x", "kind": "normal"}],
        "effects": [{"name": "x", "size": 0.5}],
        "correlations": [],
        "alpha": 0.05,
        "correction": "bonferroni",
        "targets": ["overall"],
        "heteroskedasticity": {},
        "residual": {"distribution": "normal"},
        "max_failed_fraction": 0.1,
        "scenarios": [],
    }
)

# ── Design: full matrix including intercept column, column-major ─────────────
# col_0 = intercept (all 1.0), col_1 = x = [1, 2, 3, 4, 5]
_DESIGN = [1.0, 1.0, 1.0, 1.0, 1.0,   # col_0 (intercept)
           1.0, 2.0, 3.0, 4.0, 5.0]   # col_1 (x)
_NROW = 5
_NCOL = 2
_OUTCOME = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2*x
_SEED = 2137


def _build_contracts() -> bytes:
    _names, contracts_bytes, _skeleton = _engine.build_contract_from_spec(
        _LINEAR_SPEC_JSON, "continuous", "ols", 0.0, "[]"
    )
    return contracts_bytes


# ── Main test ────────────────────────────────────────────────────────────────

def test_fit_uploaded_data_returns_dict():
    """``fit_uploaded_data`` returns a dict with betas, design_columns,
    converged, targets, variance_components, sigma_sq_hat, re_corr."""
    contracts_bytes = _build_contracts()
    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME, None
    )
    assert isinstance(out, dict)
    assert "betas" in out
    assert "design_columns" in out
    assert "converged" in out
    assert "targets" in out
    assert "variance_components" in out
    assert "sigma_sq_hat" in out
    assert "re_corr" in out


def test_fit_uploaded_data_lme_debug_fields_ols():
    """OLS path: variance_components and re_corr are empty lists; sigma_sq_hat
    is NaN (the Mle path is the only one that populates these fields)."""
    contracts_bytes = _build_contracts()
    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME, None
    )
    assert isinstance(out["variance_components"], list)
    assert out["variance_components"] == []
    assert isinstance(out["re_corr"], list)
    assert out["re_corr"] == []
    # OLS does not surface σ̂²; the engine signals this with NaN.
    assert math.isnan(out["sigma_sq_hat"])


def test_fit_uploaded_data_beta_recovery():
    """OLS on y=2x should recover intercept≈0 and slope≈2 (within 1e-6)."""
    contracts_bytes = _build_contracts()
    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME, None
    )
    assert out["converged"] is True

    cols = out["design_columns"]
    betas = out["betas"]
    assert isinstance(cols, list)
    assert len(cols) == _NCOL
    assert len(betas) == _NCOL

    # Map by column name (authoritative over positional index).
    col_to_beta = dict(zip(cols, betas))
    # Engine labels the intercept column "intercept" and predictors "col_1", "col_2", …
    # Use design_columns as authoritative — don't hard-code positions.
    intercept_col = cols[0]   # first column is always the intercept
    predictor_col = cols[1]   # second column is the x predictor
    assert abs(col_to_beta[intercept_col]) < 1e-6, (
        f"Intercept beta expected ~0.0, got {col_to_beta[intercept_col]} "
        f"(column {intercept_col!r})"
    )
    assert abs(col_to_beta[predictor_col] - 2.0) < 1e-6, (
        f"Slope beta expected ~2.0, got {col_to_beta[predictor_col]} "
        f"(column {predictor_col!r})"
    )


def test_fit_uploaded_data_targets_structure():
    """Each target dict has the expected keys and reasonable values."""
    contracts_bytes = _build_contracts()
    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME, None
    )
    targets = out["targets"]
    assert isinstance(targets, list)
    assert len(targets) >= 1

    for t in targets:
        assert isinstance(t, dict)
        for key in (
            "target_index", "target_label", "beta", "se",
            "statistic", "statistic_kind", "critical_value",
            "alpha", "df", "two_sided",
        ):
            assert key in t, f"Missing key {key!r} in target dict"
        assert t["statistic_kind"] in ("t", "f", "wald_chi2", "z")
        assert t["two_sided"] is True
        assert t["alpha"] == pytest.approx(0.05)


def test_fit_uploaded_data_without_clusters():
    """Passing ``cluster_ids=None`` is accepted without error."""
    contracts_bytes = _build_contracts()
    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME, None
    )
    assert out["converged"] is True


def test_fit_uploaded_data_rejects_bad_shape():
    """Invalid shapes must raise a clean ValueError, not panic across FFI."""
    contracts_bytes = _build_contracts()
    # Negative nrow.
    with pytest.raises(ValueError):
        _engine.fit_uploaded_data(
            contracts_bytes, 0, _SEED, _DESIGN, -1, _NCOL, _OUTCOME, None
        )
    # design length inconsistent with nrow*ncol.
    with pytest.raises(ValueError):
        _engine.fit_uploaded_data(
            contracts_bytes, 0, _SEED, _DESIGN[:-1], _NROW, _NCOL, _OUTCOME, None
        )
    # outcome length inconsistent with nrow.
    with pytest.raises(ValueError):
        _engine.fit_uploaded_data(
            contracts_bytes, 0, _SEED, _DESIGN, _NROW, _NCOL, _OUTCOME[:-1], None
        )


def test_fit_uploaded_data_lme_debug_fields_populated():
    """MLE general path: variance_components and sigma_sq_hat are populated when
    an LME model with a random slope is fit.

    The general MLE path (which surfaces variance components) is triggered when
    the primary grouping has slope terms OR extra groupings.  A random-slope
    model (1+x|group) gives q_p=2, so `variance_components` has [τ̂₀², τ̂₁²]
    and `re_corr` has one entry (the intercept–slope correlation).

    Data: y = 1.5·x + u₀_cluster + u₁_cluster·x + ε
    6 clusters, 30 observations each.
    """
    rng = np.random.default_rng(42)
    n_clusters, csize = 6, 30
    n = n_clusters * csize
    x_raw = rng.normal(size=n)
    x = (x_raw - x_raw.mean()) / x_raw.std()   # z-scored
    cluster_raw = np.repeat(np.arange(n_clusters), csize)
    # Per-cluster random intercepts and slopes (τ_int=0.65, τ_slp=0.55)
    u_int = rng.normal(scale=0.65, size=n_clusters)[cluster_raw]
    u_slp = rng.normal(scale=0.55, size=n_clusters)[cluster_raw]
    eps = rng.normal(scale=1.0, size=n)
    y = 1.5 * x + u_int + u_slp * x + eps

    # Use MCPower's spec-builder with random_slopes to trigger the general MLE
    # path that surfaces per-component variance.  Slope column 0 = x (0-based
    # predictor index after intercept; column parameter in SlopeTerm is the
    # generator column index, not the design column — matches _build_cluster_spec_dict).
    m = MCPower("y ~ x + (1+x|group)", family="lme")
    m.set_effects("x=0.5").set_cluster(
        "group", ICC=0.3, n_clusters=n_clusters,
        random_slopes=["x"], slope_variance=0.3
    )

    _, estimator_wire, intercept_arg, clusters_json = m._encode_outcome_and_clusters()
    # Pin n_clusters to the data so the fitter's per-cluster buffers are correct.
    clusters = json.loads(clusters_json)
    clusters[0]["sizing"] = {"FixedClusters": {"n_clusters": n_clusters}}
    clusters_json = json.dumps(clusters)

    spec_json = json.dumps({
        "formula": "y = x",
        "predictors": [{"name": "x", "kind": "normal"}],
        "effects": [{"name": "x", "size": 0.5}],
        "correlations": [],
        "alpha": 0.05,
        "correction": "bonferroni",
        "targets": ["overall"],
        "heteroskedasticity": {},
        "residual": {"distribution": "normal"},
        "max_failed_fraction": 0.1,
        "scenarios": [],
    })
    _names, contracts_bytes, _sk = _engine.build_contract_from_spec(
        spec_json, "continuous", estimator_wire, intercept_arg, clusters_json
    )

    # Design: intercept (col_0) + z-scored x (col_1), column-major.
    design_flat = [1.0] * n + x.tolist()
    cluster_ids = cluster_raw.tolist()

    out = _engine.fit_uploaded_data(
        contracts_bytes, 0, 2137, design_flat, n, 2, y.tolist(), cluster_ids
    )

    # General MLE path surfaces ≥2 variance components [τ̂₀², τ̂₁²] (intercept +
    # slope for the primary grouping).
    assert isinstance(out["variance_components"], list)
    assert len(out["variance_components"]) >= 2, (
        f"expected ≥2 variance components for a slope model, got {out['variance_components']}"
    )
    for vc in out["variance_components"]:
        assert vc >= 0.0, f"variance component must be non-negative, got {vc}"

    # sigma_sq_hat must be finite and positive for a converged MLE fit.
    assert math.isfinite(out["sigma_sq_hat"]), (
        f"sigma_sq_hat should be finite for the general MLE path, got {out['sigma_sq_hat']}"
    )
    assert out["sigma_sq_hat"] > 0.0, (
        f"sigma_sq_hat should be positive, got {out['sigma_sq_hat']}"
    )

    # re_corr has q_p(q_p−1)/2 = 1 entry (intercept–slope correlation) for q_p=2.
    assert isinstance(out["re_corr"], list)
    assert len(out["re_corr"]) == 1, (
        f"expected 1 re_corr entry for q_p=2, got {out['re_corr']}"
    )
    corr = out["re_corr"][0]
    assert math.isfinite(corr), f"re_corr entry must be finite, got {corr}"
    assert -1.0 <= corr <= 1.0, f"re_corr entry must be a valid correlation, got {corr}"
