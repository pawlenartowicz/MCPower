"""Cross-port golden harness for engine-spec-builder's formula + assignment parsers."""
import json
import re
from pathlib import Path

import pytest

from mcpower import _engine

ROOT = Path(__file__).resolve().parents[4] / "configs" / "formula-fixtures"
CASES = json.loads((ROOT / "cases.json").read_text())["cases"]
ASSIGN = sorted((ROOT / "assignments").glob("*.json"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_formula_fixture(case):
    formula = case["formula"]
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            _engine.parse_formula(formula)
    else:
        assert _engine.parse_formula(formula) == case["expected"]


@pytest.mark.parametrize("case", ASSIGN, ids=[c.stem for c in ASSIGN])
def test_assignment_fixture(case):
    fixture = json.loads(case.read_text())
    input_str = fixture["input"]
    kind = fixture["kind"]
    known = fixture["known"]
    expected = fixture["expected"]
    if "error" in expected and set(expected.keys()) == {"error"}:
        # Top-level hard error — parse_assignments must raise ValueError
        with pytest.raises(ValueError, match=expected["error"]):
            _engine.parse_assignments(input_str, kind, known)
    else:
        assert _engine.parse_assignments(input_str, kind, known) == expected


# ---------------------------------------------------------------------------
# Structural assertion: parsers.py must carry no grammar regex literals
# ---------------------------------------------------------------------------

PARSERS_PY = Path(__file__).resolve().parents[2] / "mcpower" / "spec" / "parsers.py"

FORBIDDEN_GRAMMAR_PATTERNS = [
    # Active use of re.* with formula grammar patterns — these indicate
    # parsers.py is implementing grammar itself rather than delegating to Rust.
    # We look for actual re-module calls (not docstring mentions of formulas).
    r"re\.(compile|match|search|finditer|sub|findall)\s*\(",
    # Note: a plain module-level ``_IDENT = r"..."`` identifier pattern is
    # allowed — it is a re-exported variable-name matcher for validators.py,
    # not formula grammar. parsers.py must still implement no grammar via re.*.
]


def test_parsers_py_holds_no_grammar():
    """All formula/assignment grammar must live in Rust (engine-spec-builder).

    The Python parsers.py shim re-shapes the Rust output for legacy callers but must
    not call re.* functions to implement formula grammar.

    Note: docstrings may contain formula examples like ``(1|g)`` — those are fine.
    The check targets active parsing code (re.compile/match/search/sub/etc calls).
    """
    text = PARSERS_PY.read_text()
    for pat in FORBIDDEN_GRAMMAR_PATTERNS:
        flags = re.MULTILINE if pat.startswith("^") else 0
        assert not re.search(pat, text, flags), (
            f"forbidden grammar pattern {pat!r} survives in parsers.py — "
            f"grammar must live in mcpower/crates/engine-spec-builder/src/"
        )
