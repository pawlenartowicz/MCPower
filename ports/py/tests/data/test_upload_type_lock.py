"""Upload type-lock guard: uploaded column types are authoritative and cannot be overridden.

When a column matched to a model predictor is uploaded, its detected type class
(continuous / binary / factor) comes from the data. Declaring a conflicting class
via set_variable_type raises a clear ValueError naming the column, instead of
letting a confusing ColumnId error bubble up from the engine.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal upload arrays
# ---------------------------------------------------------------------------

def _continuous_data(n=40):
    """40-row continuous column (normal-ish)."""
    rng = np.random.default_rng(7)
    return rng.normal(size=n).tolist()


def _binary_data(n=40):
    """40-row binary column (0/1)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 2, size=n).tolist()


def _factor_data(n=42):
    """42-row factor column (3 string levels, each appearing 14 times → ratio=14 ≥ 15 fails,
    but string columns are always factor regardless of ratio)."""
    return (["cat", "dog", "bird"] * (n // 3))[:n]


# ---------------------------------------------------------------------------
# 1. Continuous detected + declared factor → ValueError
# ---------------------------------------------------------------------------

def test_continuous_upload_declared_factor_raises():
    """Detected continuous + set_variable_type factor → clear ValueError naming the column."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": _continuous_data()}, verbose=False)
    # "(factor,3)" is the tuple syntax for factor with 3 levels
    m.set_variable_type("x=(factor,3)")
    m.set_effects("x[2]=0.3, x[3]=0.3")

    with pytest.raises(ValueError) as exc_info:
        m.find_power(100)

    msg = str(exc_info.value)
    # Must name the column
    assert "'x'" in msg or "\"x\"" in msg, f"Column name missing from error: {msg}"
    # Must mention the detected type
    assert "continuous" in msg, f"Detected type missing from error: {msg}"
    # Must NOT be the engine's ColumnId error (which says "DirectOnFactor" or "ColumnId")
    assert "DirectOnFactor" not in msg, f"Got engine's internal error instead of our guard: {msg}"
    assert "ColumnId" not in msg, f"Got engine's internal ColumnId error: {msg}"


# ---------------------------------------------------------------------------
# 2. Binary detected + declared continuous → ValueError
# ---------------------------------------------------------------------------

def test_binary_upload_declared_continuous_raises():
    """Detected binary + set_variable_type normal → clear ValueError naming the column."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": _binary_data()}, verbose=False)
    # "normal" is the continuous class — conflicts with detected binary
    m.set_variable_type("x=normal")
    m.set_effects("x=0.3")

    with pytest.raises(ValueError) as exc_info:
        m.find_power(100)

    msg = str(exc_info.value)
    assert "'x'" in msg or "\"x\"" in msg, f"Column name missing from error: {msg}"
    assert "binary" in msg, f"Detected type missing from error: {msg}"
    assert "DirectOnFactor" not in msg, f"Got engine's internal error: {msg}"


# ---------------------------------------------------------------------------
# 3a. Matching class declaration → no error (factor with same data levels)
# ---------------------------------------------------------------------------

def test_matching_factor_declaration_no_error():
    """factor with wrong k on a detected factor column: no error, levels come from data."""
    from mcpower import MCPower

    factor_vals = _factor_data()  # cat/dog/bird → 3 levels
    m = MCPower("y = breed")
    m.upload_data({"breed": factor_vals}, verbose=False)
    # "(factor,5)" declares 5 levels — same factor class but wrong k; data wins on details
    m.set_variable_type("breed=(factor,5)")
    # data labels sorted → ["bird","cat","dog"], ref="bird"; non-ref dummies: cat, dog
    m.set_effects("breed[cat]=0.3, breed[dog]=0.3")

    # Must NOT raise on _apply
    m._apply()

    # After apply, factor info must reflect the data's 3 levels, not declared k=5
    factor_info = m._registry._factors.get("breed")
    assert factor_info is not None, "breed should be in _factors after apply"
    assert factor_info["n_levels"] == 3, (
        f"Expected 3 levels from data, got {factor_info['n_levels']}"
    )
    assert sorted(factor_info["level_labels"]) == ["bird", "cat", "dog"], (
        f"Expected data labels, got {factor_info['level_labels']}"
    )


# ---------------------------------------------------------------------------
# 3b. No declaration at all on uploaded column → no error
# ---------------------------------------------------------------------------

def test_no_declaration_on_uploaded_column_no_error():
    """No set_variable_type at all → upload path works without raising."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": _continuous_data()}, verbose=False)
    m.set_effects("x=0.3")

    # Should reach _apply cleanly
    m._apply()
    pred = m._registry.get_predictor("x")
    assert pred is not None
    # The type detector classifies a rng.normal sample as "normal"; bare `is not None`
    # would pass even if the predictor were silently mistyped.
    assert pred.var_type == "normal", (
        f"Expected 'normal' for a continuous upload with no declaration, got {pred.var_type!r}"
    )


def test_no_declaration_binary_no_error():
    """Binary upload with no type declaration → no error."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": _binary_data()}, verbose=False)
    m.set_effects("x=0.3")
    m._apply()
    pred = m._registry.get_predictor("x")
    assert pred is not None
    assert pred.var_type == "binary"


# ---------------------------------------------------------------------------
# 4. Continuous + declared continuous distribution → no error, distribution kept
# ---------------------------------------------------------------------------

def test_continuous_upload_declared_right_skewed_no_error():
    """Detected continuous + set_variable_type right_skewed → no error AND distribution kept."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": _continuous_data()}, verbose=False)
    # right_skewed is still the continuous class — not a conflict
    m.set_variable_type("x=right_skewed")
    m.set_effects("x=0.3")

    # Must NOT raise
    m._apply()

    pred = m._registry.get_predictor("x")
    assert pred is not None
    # The declared distribution must be preserved (continuous re-apply is a no-op)
    assert pred.var_type == "right_skewed", (
        f"Expected 'right_skewed' to be preserved, got {pred.var_type!r}"
    )
