"""D3 — upload_data rename (mode=), detect types, remove guard, emit upload block."""
import numpy as np
import pytest


def test_mode_default_is_partial_and_validates():
    from mcpower import MCPower

    m = MCPower("y = x")
    with pytest.raises(ValueError, match="none.*partial.*strict"):
        m.upload_data({"x": list(range(30))}, mode="bad")


def test_strict_mode_is_accepted_and_stored():
    """strict (bootstrap) is now fully supported — upload_data must accept it and
    store the mode in _pending_data and _uploaded_data_mode."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": list(range(30))}, mode="strict", verbose=False)
    assert m._pending_data["mode"] == "strict"
    assert m._uploaded_data_mode == "strict"


def test_upload_data_accepts_valid_mode_strings():
    from mcpower import MCPower

    m = MCPower("y = x")
    data = list(range(30))
    m.upload_data({"x": data}, mode="none", verbose=False)
    m.upload_data({"x": data}, mode="partial", verbose=False)
    m.upload_data({"x": data}, mode="strict", verbose=False)


def test_upload_reaches_spec_without_raising():
    from mcpower import MCPower

    rng = np.random.default_rng(0)
    m = MCPower("y = x1 + x2")
    m.upload_data({"x1": rng.normal(size=40).tolist()})  # x2 stays synthetic
    m.set_effects("x1=0.3, x2=0.3")
    contract = m.to_simulation_spec()  # must NOT raise
    # The uploaded frame should be present in the generation part of the contract
    assert contract is not None
    gen = contract.get("generation", {})
    assert gen.get("uploaded_frame") is not None, (
        f"expected uploaded_frame in generation; keys: {list(gen.keys())}"
    )


def test_upload_block_has_matched_column():
    """The upload block emitted to the engine contains x1 with raw values."""
    from mcpower import MCPower
    from mcpower.data.upload import normalize_upload_input

    rng = np.random.default_rng(1)
    vals = rng.normal(size=40).tolist()
    m = MCPower("y = x1 + x2")
    m.upload_data({"x1": vals})
    m.set_effects("x1=0.3, x2=0.3")

    pending = m._pending_data
    assert pending is not None
    assert "columns_typed" in pending
    col_names = [c[0] for c in pending["columns_typed"]]
    assert "x1" in col_names
    # x2 is NOT in upload — only matched predictors stored in columns_typed
    # (raw_columns holds all including unmatched for D4)
    assert "mode" in pending
    assert pending["mode"] == "partial"  # default
    assert "uploaded_n" in pending
    assert pending["uploaded_n"] == 40


def test_pending_data_stores_raw_columns_for_d4():
    """_pending_data must store raw columns for ALL uploaded cols (including y) for D4."""
    from mcpower import MCPower

    rng = np.random.default_rng(2)
    m = MCPower("y = x1")
    x1_vals = rng.normal(size=30).tolist()
    y_vals = rng.normal(size=30).tolist()
    m.upload_data({"x1": x1_vals, "y": y_vals})
    pending = m._pending_data
    assert "raw_columns" in pending
    raw = pending["raw_columns"]
    assert "x1" in raw
    assert "y" in raw


def test_type_detection_sets_registry():
    """After upload, the registry has the detected type for matched predictors."""
    from mcpower import MCPower

    rng = np.random.default_rng(3)
    m = MCPower("y = x1")
    vals = rng.normal(size=30).tolist()
    m.upload_data({"x1": vals})
    pred = m._registry.get_predictor("x1")
    assert pred is not None
    # The type detector classifies a normal-ish sample as "normal" (the continuous
    # sub-type). The former 3-option tuple admitted raw-passthrough fallbacks that
    # wouldn't reflect real detection; pin to the single correct type.
    assert pred.var_type == "normal", (
        f"Expected 'normal' for a random normal sample, got {pred.var_type!r}"
    )


def test_mode_none_partial_strict_accepted():
    """mode='none', 'partial', and 'strict' are all accepted."""
    from mcpower import MCPower

    m = MCPower("y = x")
    m.upload_data({"x": list(range(30))}, mode="none", verbose=False)
    assert m._pending_data["mode"] == "none"
    m.upload_data({"x": list(range(30))}, mode="partial", verbose=False)
    assert m._pending_data["mode"] == "partial"
    m.upload_data({"x": list(range(30))}, mode="strict", verbose=False)
    assert m._pending_data["mode"] == "strict"


def test_upload_data_second_call_replaces_first():
    """upload_data called twice keeps only the second frame in _pending_data.

    _pending_data is fully replaced each call (no merge). Distinguishable by
    uploaded_n: frame A has 30 rows, frame B has 60 rows.
    """
    from mcpower import MCPower

    m = MCPower("y = x1 + x2")
    m.set_effects("x1=0.3, x2=0.3")

    data_a = {"x1": list(range(30)), "x2": list(range(30, 60))}
    data_b = {"x1": list(range(100, 160))}

    m.upload_data(data_a, verbose=False)
    m.upload_data(data_b, verbose=False)

    assert m._pending_data["uploaded_n"] == 60  # only frame B survives


def test_upload_column_order_assigns_effect_to_correct_predictor():
    """Column order must map x1's data to the x1 predictor slot.

    Deliberate effect asymmetry: x1=0.5 >> x2=0.1 so power(x1) > power(x2)
    at N=200.  A column-order swap would assign x2's data to x1's slot,
    flipping the inequality.
    """
    from mcpower import MCPower

    rng = np.random.default_rng(7)
    x1_vals = rng.normal(size=60).tolist()
    x2_vals = rng.normal(size=60).tolist()

    m = MCPower("y = x1 + x2")
    # Upload x2 first, then x1 — deliberate inversion of formula order.
    m.upload_data({"x2": x2_vals, "x1": x1_vals}, verbose=False)
    m.set_effects("x1=0.5, x2=0.1")

    r = m.find_power(200, n_sims=400, seed=2137, verbose=False)
    p = r["power_uncorrected"][0]
    assert p[0] > p[1], (
        f"Expected power(x1=0.5) > power(x2=0.1) but got x1={p[0]:.3f}, x2={p[1]:.3f}; "
        "column-to-predictor assignment may be incorrect."
    )
