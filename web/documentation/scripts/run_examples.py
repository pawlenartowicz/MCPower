#!/usr/bin/env python3
"""Run the Python documentation examples and cache their real output.

Reads ``examples-python.json``, executes each entry's ``code`` in a fresh
namespace, captures stdout into ``output``, and stamps ``captured_at``.

If an entry sets ``"plot"``, the code must leave the result object to be plotted
in a variable named ``result``; the runner renders it to
``../assets/examples/<plot>`` via that result's public ``save_plot`` (PNG, light-print
theme — the default). This script also doubles as the shared renderer: on every run it sweeps
``../assets/examples`` for any ``*.vl.json`` specs left by ``run_examples.R``
(the R port has no native PNG renderer) and converts them to PNG with
``vl_convert``, so both ports' charts come from one renderer. Run the R runner
first, then this one, to materialise every plot.

This fills the cache that ``inject_examples.py`` then pastes into the tutorial
pages — author ``code``, run this to fill ``output``/``captured_at`` and render
plots, then run ``inject_examples.py`` to update the pages. Nothing here is wired
into the leyline build; it is a one-shot cache refresher.

Usage:
    python run_examples.py            # refresh every entry
    python run_examples.py <id> ...   # refresh only the named entries
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "examples-python.json"
ASSETS = HERE.parent / "assets" / "examples"


def _render_spec_file(spec_path: Path) -> None:
    """Convert a Vega-Lite ``.vl.json`` spec to a same-stem PNG, then drop it."""
    import vl_convert as vlc

    png = spec_path.with_suffix("").with_suffix(".png")  # foo.vl.json -> foo.png
    png.write_bytes(vlc.vegalite_to_png(spec_path.read_text(), scale=2))
    spec_path.unlink()
    print(f"  rendered {png.name} (from R spec)")


def run_entry(entry: dict) -> None:
    namespace: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(entry["code"], namespace)  # noqa: S102 — trusted, author-written
    entry["output"] = buf.getvalue().rstrip("\n")
    entry["captured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    plot_name = entry.get("plot")
    if plot_name:
        result = namespace.get("result")
        if result is None:
            raise SystemExit(
                f"entry {entry['id']!r} sets 'plot' but left no 'result' variable"
            )
        ASSETS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / plot_name
            result.save_plot(str(tmp_path))
            shutil.copy2(tmp_path, ASSETS / plot_name)
        print(f"  plot: {plot_name}")


def main() -> None:
    entries = json.loads(CACHE.read_text())
    wanted = set(sys.argv[1:])
    for entry in entries:
        if wanted and entry["id"] not in wanted:
            continue
        print(f"running {entry['id']} ...")
        run_entry(entry)
    CACHE.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {CACHE.name}")

    # Shared render step: turn any R-emitted specs into PNGs with one renderer.
    for spec_path in sorted(ASSETS.glob("*.vl.json")):
        _render_spec_file(spec_path)


if __name__ == "__main__":
    main()
