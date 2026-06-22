"""Central validation executor for the examples book (Python chunks).

Runs each chunks/<id>.py exactly once in a fresh namespace, records pass/fail,
and dumps results.json. Validation only — no output is captured or displayed.
Must run inside the workspace .venv so the editable mcpower build is imported,
not a stale installed copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
CHUNKS = BOOK / "chunks"
RESULTS = BOOK / "results.json"


def _require_mcpower() -> None:
    try:
        import mcpower  # noqa: F401
    except Exception as e:  # pragma: no cover - environment gate
        raise SystemExit(
            f"PRECONDITION FAILED: cannot import editable mcpower in this "
            f"interpreter ({e!r}). Activate the workspace .venv and "
            f"`maturin develop` the py port before validating."
        )


def discover_and_run(chunks_dir: Path, glob: str, only: set[str] | None = None) -> dict:
    results: dict[str, dict] = {}
    for path in sorted(chunks_dir.glob(glob)):
        cid = path.stem
        if only and cid not in only:
            continue
        try:
            exec(compile(path.read_text(), str(path), "exec"), {})  # noqa: S102 - trusted, author-written
            results[cid] = {"ok": True, "err": None}
        except Exception as e:  # noqa: BLE001 - validation harness records every failure
            results[cid] = {"ok": False, "err": repr(e)}
    return results


def write_results(path: Path, results: dict, only: set[str] | None) -> None:
    """Persist results. A subset run (``only`` set) MERGES into the existing file so a
    fix-loop re-run of a few ids does not clobber the full record; a full run (``only``
    None) replaces, pruning ids no longer present."""
    merged = dict(results)
    if only and path.exists():
        prior = json.loads(path.read_text())
        prior.update(results)
        merged = prior
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def main() -> None:
    _require_mcpower()
    only = set(sys.argv[1:]) or None
    results = discover_and_run(CHUNKS, "*.py", only=only)
    write_results(RESULTS, results, only)
    n_fail = sum(1 for r in results.values() if not r["ok"])
    print(f"validate.py: {len(results)} chunks, {n_fail} failed -> {RESULTS}")


if __name__ == "__main__":
    main()
