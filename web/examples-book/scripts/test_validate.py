import json
from pathlib import Path
import validate


def test_runs_good_and_bad_chunks(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "good-01.py").write_text("x = 1 + 1\nassert x == 2\n")
    (chunks / "bad-01.py").write_text("raise ValueError('boom')\n")

    results = validate.discover_and_run(chunks, "*.py")

    assert results["good-01"] == {"ok": True, "err": None}
    assert results["bad-01"]["ok"] is False
    assert "boom" in results["bad-01"]["err"]


def test_subset_filter(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "a.py").write_text("pass\n")
    (chunks / "b.py").write_text("raise RuntimeError('x')\n")

    results = validate.discover_and_run(chunks, "*.py", only={"a"})

    assert set(results) == {"a"}


def test_subset_run_merges_into_existing(tmp_path):
    # A fix-loop re-run of a few ids must NOT clobber the full record.
    out = tmp_path / "results.json"
    out.write_text(json.dumps({"a": {"ok": True, "err": None}, "b": {"ok": False, "err": "x"}}))

    validate.write_results(out, {"b": {"ok": True, "err": None}}, only={"b"})

    data = json.loads(out.read_text())
    assert set(data) == {"a", "b"}        # a preserved
    assert data["b"]["ok"] is True        # b updated


def test_full_run_replaces(tmp_path):
    # A full run (only=None) is authoritative: it prunes ids no longer present.
    out = tmp_path / "results.json"
    out.write_text(json.dumps({"stale": {"ok": True, "err": None}}))

    validate.write_results(out, {"a": {"ok": True, "err": None}}, only=None)

    data = json.loads(out.read_text())
    assert set(data) == {"a"}
