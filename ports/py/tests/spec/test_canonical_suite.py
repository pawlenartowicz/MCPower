"""Canonical formula-effect suite — Python port verification.

Loads configs/formula-fixtures/canonical-suite.json, parses each formula via the
Rust engine bridge (_engine.parse_formula), normalizes to the port-neutral
canonical shape, and asserts it matches `expected`. Error entries assert the
parser rejects with a message containing the `error` substring.
"""
import json
import re
from pathlib import Path

import pytest

import mcpower._engine as _engine

SUITE = json.loads(
    (Path(__file__).resolve().parents[4] / "configs" / "formula-fixtures" / "canonical-suite.json").read_text()
)
CASES = SUITE["cases"]


def _canonical(parsed: dict) -> dict:
    fixed = []
    for t in parsed["terms"]:
        if t["kind"] == "main":
            fixed.append(t["name"])
        else:  # interaction
            fixed.append(":".join(t["vars"]))
    res = []
    for r in parsed["random_effects"]:
        if r["kind"] == "intercept":
            res.append(f"intercept|{r['group']}")
        else:  # slope
            res.append(f"slope({','.join(r['vars'])})|{r['group']}")
    return {"outcome": parsed["dependent"], "fixed_effects": fixed, "random_effects": res}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_canonical_suite(case):
    if "error" in case:
        with pytest.raises(ValueError, match=re.escape(case["error"])):
            _engine.parse_formula(case["formula"])
    else:
        parsed = _engine.parse_formula(case["formula"])
        assert _canonical(parsed) == case["expected"]
