#!/usr/bin/env python3
"""Inject cached example chunks into the tutorial pages at their markers.

Tutorial pages carry managed regions delimited by

    <!-- example:<id> -->
    ...anything here is overwritten...
    <!-- /example -->

This script rewrites the inside of every such region from the example cache
(``examples-{python,r}.json``): the fenced ``code`` block, the captured
``output`` block, and a plot embed when the entry rendered one. The marker
lines themselves are preserved, so the author keeps full control of the prose
*around* each region while the code+output *inside* it never drifts from the
cache.

It is the cache -> page half of the pipeline and is deliberately kept separate
from ``run_examples.*`` (which is the run -> cache half): the runner stays a
single-purpose output capturer, this stays a single-purpose page rewriter.
Neither runs at leyline build time — published pages stay static.

Run it once per port, after refreshing the cache with the matching runner:

    python inject_examples.py examples-python.json ../tutorial-python
    python inject_examples.py examples-r.json      ../tutorial-r

The fenced-code language is inferred from the cache filename (``python`` vs
``r``). Re-running on an up-to-date page is a no-op (byte-identical output).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Marker pair; the id matches example slugs (word chars, dot, slash, dash). The
# body between the markers (anything, including empty) is replaced wholesale.
REGION = re.compile(
    r"(?P<open><!-- example:(?P<id>[\w./-]+) -->).*?(?P<close><!-- /example -->)",
    re.DOTALL,
)


def _lang_from_cache(cache: Path) -> str:
    """Fence language for a cache file: examples-python.json -> python, else r."""
    return "python" if "python" in cache.name else "r"


def _render(entry: dict, lang: str) -> str:
    """The managed body for one region: fenced code, captured output, plot embed."""
    if not entry.get("captured_at"):
        raise SystemExit(
            f"example {entry['id']!r} has no captured output — run the matching "
            f"runner before injecting"
        )
    parts = [f"```{lang}", entry["code"].rstrip("\n"), "```"]
    output = entry.get("output", "").rstrip("\n")
    if output:
        parts += ["", "```", output, "```"]
    plot = entry.get("plot")
    if plot:
        # Embed the rendered image (PNG); .vl.json fallbacks are not embeddable.
        parts += ["", f"![[examples/{plot}|600]]"]
    return "\n".join(parts)


def inject_page(path: Path, by_id: dict, lang: str) -> int:
    text = path.read_text()
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        eid = m.group("id")
        seen.add(eid)
        if eid not in by_id:
            raise SystemExit(f"{path.name}: marker references unknown example {eid!r}")
        body = _render(by_id[eid], lang)
        return f"{m.group('open')}\n{body}\n{m.group('close')}"

    new_text = REGION.sub(repl, text)
    if new_text != text:
        path.write_text(new_text)
    if seen:
        print(f"  {path.name}: {len(seen)} region(s) [{', '.join(sorted(seen))}]")
    return len(seen)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: inject_examples.py <cache.json> <pages_dir>")
    cache = Path(sys.argv[1])
    pages_dir = Path(sys.argv[2])
    lang = _lang_from_cache(cache)
    by_id = {e["id"]: e for e in json.loads(cache.read_text())}

    print(f"injecting {cache.name} ({lang}) into {pages_dir}/ ...")
    total = sum(inject_page(p, by_id, lang) for p in sorted(pages_dir.glob("*.md")))
    print(f"done — {total} region(s) injected")


if __name__ == "__main__":
    main()
