"""Integration: sparse-factor exclusion end-to-end (Task 11).

y = x1 + g, g=(factor,0.95,0.05), N=40 → level 2 receives ~2 observations,
below the engine's minimum-5 threshold. Expected behaviour:
  - preflight UserWarning on grid_warnings channel;
  - no RuntimeError / convergence failure;
  - factor_exclusion_counts == [n_sims] (g excluded every sim);
  - g's dummy effect (label '2' — bare integer when no level labels are set)
    power == 0.0; x1 power > 0;
  - repr(result) carries a 'g excluded 100.0%' diagnostic line.
"""

from __future__ import annotations

import warnings

import pytest

from mcpower import MCPower


@pytest.fixture(scope="module")
def sparse_result():
    """Run find_power with g sparse at N=40, capture warnings, return result."""
    m = MCPower("y = x1 + g")
    m.set_variable_type("g=(factor,0.95,0.05)")
    m.set_effects("x1=0.5, g[2]=0.8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = m.find_power(sample_size=40, n_sims=200, seed=2137, verbose=False)
    return result, caught


def test_preflight_warning_emitted(sparse_result):
    """Engine emits a UserWarning containing 'excluded from every simulation' at N=40."""
    _, caught = sparse_result
    uw = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("excluded from every simulation" in w for w in uw), (
        f"Expected preflight exclusion UserWarning; got: {uw}"
    )


def test_no_convergence_failure(sparse_result):
    """Sparse factor exclusion must not cause a RuntimeError — convergence stays ~100%."""
    result, _ = sparse_result
    # convergence_rate is a list (one per N); all entries must be ~1.0.
    cr = result.get("convergence_rate")
    if cr is not None:
        assert min(cr) > 0.95, f"Unexpected convergence drop: {cr}"


def test_factor_exclusion_counts_all_sims(sparse_result):
    """factor_exclusion_counts must equal [n_sims] — g excluded every simulation."""
    result, _ = sparse_result
    n_sims = result.get("n_sims")
    excl = result.get("factor_exclusion_counts")
    assert excl is not None, "factor_exclusion_counts key missing from result"
    assert len(excl) == 1, f"Expected 1 factor, got {len(excl)}: {excl}"
    assert excl[0] == n_sims, (
        f"Expected g excluded in all {n_sims} sims, got {excl[0]}"
    )


def test_g_power_is_zero_x1_power_positive(sparse_result):
    """g's dummy effect (label '2' when no level labels set) power must be 0
    (excluded every sim); x1 power must be positive."""
    result, _ = sparse_result
    from mcpower.output.tables import build_rows

    # Retrieve meta from the result object (PowerResult stores ._meta).
    meta = result._meta
    inner = result  # single-scenario result is flat (no "scenarios" envelope)
    power = inner.get("power_uncorrected")
    assert power is not None

    rows = build_rows(inner.get("target_indices", []), meta)
    label_power = {
        r["label"]: power[0][r["pos"]]
        for r in rows
        if r["kind"] != "factor_header"
    }
    # g is the sole factor; its non-reference dummy effect(s) should be 0.
    # Integer-indexed labels (no level_labels set) render as bare integers '2'.
    g_effects = {k: v for k, v in label_power.items() if k.startswith("g[") or k == "2"}
    x1_power = label_power.get("x1")
    assert g_effects, f"No g dummy effects found in rows: {list(label_power)}"
    for name, pwr in g_effects.items():
        assert pwr == 0.0, f"{name} power expected 0.0 (excluded), got {pwr}"
    assert x1_power is not None and x1_power > 0, (
        f"x1 power expected > 0, got {x1_power}"
    )


def test_repr_carries_diagnostic_line(sparse_result):
    """repr(result) must contain a 'g excluded 100.0%' diagnostic warning line,
    pinning factor-name ordering end-to-end from meta['factors'] to render_short."""
    result, _ = sparse_result
    text = repr(result)
    assert "g excluded 100.0%" in text, (
        f"Expected 'g excluded 100.0%' in repr; got:\n{text}"
    )
