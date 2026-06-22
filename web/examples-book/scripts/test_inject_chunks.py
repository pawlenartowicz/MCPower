from pathlib import Path
import inject_chunks


def test_injects_py_and_r(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "ex-1.py").write_text("from mcpower import MCPower\nm = MCPower('y = x')\n")
    (chunks / "ex-1.R").write_text("library(mcpower)\nm <- MCPower$new('y ~ x')\n")

    page = (
        "# Title\n\n"
        "<!-- chunk:py:ex-1 -->\n```python\nSTALE\n```\n<!-- /chunk:py:ex-1 -->\n\n"
        "<!-- chunk:r:ex-1 -->\n```r\n```\n<!-- /chunk:r:ex-1 -->\n"
    )
    out = inject_chunks.inject(page, chunks)

    assert "from mcpower import MCPower" in out
    assert "STALE" not in out
    assert "m <- MCPower$new('y ~ x')" in out
    # idempotent
    assert inject_chunks.inject(out, chunks) == out
