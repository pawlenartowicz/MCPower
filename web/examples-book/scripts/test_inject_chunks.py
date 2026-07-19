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


def test_injects_inside_accordion_details(tmp_path):
    """The <details> accordion wrapper is inert to injection — the fences
    inside it fill exactly as if unwrapped."""
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "ex-2.py").write_text("from mcpower import MCPower\nm = MCPower('y = x')\n")
    (chunks / "ex-2.R").write_text("library(mcpower)\nm <- MCPower$new('y ~ x')\n")

    page = (
        "# Title\n\n"
        "## Copy-paste setup\n\n"
        "<details><summary>Python setup</summary>\n\n"
        "<!-- chunk:py:ex-2 -->\n```python\n```\n<!-- /chunk:py:ex-2 -->\n\n"
        "</details>\n\n"
        "<details><summary>R setup</summary>\n\n"
        "<!-- chunk:r:ex-2 -->\n```r\n```\n<!-- /chunk:r:ex-2 -->\n\n"
        "</details>\n"
    )
    out = inject_chunks.inject(page, chunks)

    assert "from mcpower import MCPower" in out
    assert "m <- MCPower$new('y ~ x')" in out
    assert "<details><summary>Python setup</summary>" in out
    assert "<details><summary>R setup</summary>" in out
    # idempotent
    assert inject_chunks.inject(out, chunks) == out
