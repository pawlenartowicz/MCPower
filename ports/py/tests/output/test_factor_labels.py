"""Port-side factor labels: reference-by-value resolution + result-name
rendering from the engine's index-only effect skeleton.

These pin the port's half of the single-source-in-Rust design: the engine
returns a `FactorLevel { factor, level }` skeleton (level = index into the
factor's FULL label list, reference included), and the port renders
`factor[labels[level]]` — never re-deriving the factor-expansion layout.
"""

import pytest

from mcpower.output.tables import build_rows
from mcpower.spec.variables import VariableRegistry, _resolve_reference


# ── reference-by-value resolution ──────────────────────────────────────────

def test_reference_string_matches_label_exactly():
    assert _resolve_reference("6", ["4", "6", "8"]) == "6"


def test_reference_numeric_matches_via_value_to_label():
    # 6 / 6.0 render to "6" (value_to_label drops the trailing .0).
    assert _resolve_reference(6, ["4", "6", "8"]) == "6"
    assert _resolve_reference(6.0, ["4", "6", "8"]) == "6"


def test_reference_default_is_first_label():
    assert _resolve_reference(None, ["4", "6", "8"]) == "4"


def test_reference_string_not_a_label_raises():
    with pytest.raises(ValueError):
        _resolve_reference("nope", ["4", "6", "8"])


def test_reference_numeric_not_a_label_raises():
    with pytest.raises(ValueError):
        _resolve_reference(99, ["4", "6", "8"])


def test_set_variable_type_stores_resolved_reference_and_full_levels():
    reg = VariableRegistry("y = cyl")
    reg.set_variable_type("cyl", "factor", n_levels=3, labels=["4", "6", "8"], reference=6)
    # reference resolved by value → the canonical label string.
    assert reg._factors["cyl"]["reference_level"] == "6"
    # full ordered label store the skeleton indexes into (reference included).
    assert reg.factor_levels("cyl") == ["4", "6", "8"]


# ── skeleton → display name rendering ──────────────────────────────────────

def test_build_rows_renders_string_data_labels():
    # origin with data labels; baseline "Europe" (index 0), dummies Japan/USA.
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "factor_level", "factor": "origin", "level": 1},
            {"kind": "factor_level", "factor": "origin", "level": 2},
        ],
        "factors": {"origin": {"baseline": "Europe",
                               "levels": ["Europe", "Japan", "USA"]}},
    }
    rows = build_rows([1, 2], meta)
    assert rows[0] == {"kind": "factor_header", "label": "origin", "baseline": "Europe"}
    assert rows[1] == {"kind": "factor_level", "label": "Japan", "factor": "origin", "pos": 0}
    assert rows[2] == {"kind": "factor_level", "label": "USA", "factor": "origin", "pos": 1}


def test_build_rows_renders_numeric_labels_and_interaction():
    # cyl[6] (numeric label) + the continuous×factor interaction x1:cyl[6]
    # (a flat row, label joined by ":").
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "continuous", "predictor": "x1"},
            {"kind": "factor_level", "factor": "cyl", "level": 1},
            {"kind": "interaction", "components": [
                {"kind": "continuous", "predictor": "x1"},
                {"kind": "factor_level", "factor": "cyl", "level": 1}]},
        ],
        "factors": {"cyl": {"baseline": "4", "levels": ["4", "6", "8"]}},
    }
    rows = build_rows([1, 2, 3], meta)
    assert rows[0] == {"kind": "continuous", "label": "x1", "pos": 0}
    assert rows[1] == {"kind": "factor_header", "label": "cyl", "baseline": "4"}
    assert rows[2] == {"kind": "factor_level", "label": "6", "factor": "cyl", "pos": 1}
    assert rows[3] == {"kind": "continuous", "label": "x1:cyl[6]", "pos": 2}


def test_build_rows_falls_back_to_integers_without_labels():
    # No labels stored → an unnamed factor renders 1..k (never crashes).
    meta = {
        "effect_skeleton": [
            {"kind": "intercept"},
            {"kind": "factor_level", "factor": "g", "level": 1},
        ],
        "factors": {"g": {"baseline": "1", "levels": []}},
    }
    rows = build_rows([1], meta)
    assert rows[1] == {"kind": "factor_level", "label": "2", "factor": "g", "pos": 0}


# ── end-to-end: uploaded factor labels reach the rendered table ────────────

def test_uploaded_factor_renders_data_labels_end_to_end():
    from mcpower import MCPower

    breeds = (["cat", "dog", "bird"] * 14)[:42]  # sorted → bird (ref), cat, dog
    m = MCPower("y = breed")
    m.upload_data({"breed": breeds}, verbose=False)
    m.set_effects("breed[cat]=0.5, breed[dog]=0.5")
    res = m.find_power(80, verbose=False)
    s = str(res)
    # The rendered table names the DATA labels, not breed[1]/breed[2], and the
    # reference (first sorted label) is the baseline.
    assert "breed" in s
    assert "baseline: bird" in s
    assert "cat" in s and "dog" in s
    assert "breed[1]" not in s and "breed[2]" not in s
