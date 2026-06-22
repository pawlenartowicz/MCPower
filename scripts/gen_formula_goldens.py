"""Generate goldens for the formula fixture corpus.

Refreshes the `expected` block of every positive case in
`configs/formula-fixtures/cases.json` by calling `_engine.parse_formula` on its `formula`.
Cases carrying an `error` key (negative cases) keep their hand-written pattern and are skipped.

For assignments, this script regenerates the `expected` block in each `assignments/*.json`
fixture by calling `_engine.parse_assignments(input, kind, known)`.

Re-run when the parser shape changes intentionally.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "configs" / "formula-fixtures"


def regenerate_formula_cases():
    from mcpower import _engine
    path = ROOT / "cases.json"
    data = json.loads(path.read_text())
    for case in data["cases"]:
        if "error" in case:
            # Negative cases keep their hand-written error pattern.
            continue
        try:
            out = _engine.parse_formula(case["formula"])
        except Exception as e:
            print(f"[skip] {case['id']}: parse_formula raised {type(e).__name__}: {e}")
            continue
        # deep-sort keys so goldens stay stable regardless of engine field order
        case["expected"] = json.loads(json.dumps(out, sort_keys=True))
        print(f"[ok] {case['id']}")
    path.write_text(json.dumps(data, indent=2) + "\n")


def regenerate_assignments():
    from mcpower import _engine
    assignments_dir = ROOT / "assignments"
    for json_path in sorted(assignments_dir.glob("*.json")):
        fixture = json.loads(json_path.read_text())
        if "error" in fixture.get("expected", {}):
            continue   # negative case
        if json_path.stem.startswith("err_"):
            continue   # error case
        try:
            out = _engine.parse_assignments(fixture["input"], fixture["kind"], fixture["known"])
        except Exception as e:
            print(f"[skip] {json_path.name}: parse_assignments raised {type(e).__name__}: {e}")
            continue
        fixture["expected"] = out
        json_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
        print(f"[ok] {json_path.name}")


if __name__ == "__main__":
    regenerate_formula_cases()
    regenerate_assignments()
