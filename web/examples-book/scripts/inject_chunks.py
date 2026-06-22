"""Single-source the displayed code block from the validated chunk file.

For each paired marker <!-- chunk:<lang>:<id> --> ... <!-- /chunk:<lang>:<id> -->
on a page, replace the fenced body with chunks/<id>.<ext>. Re-runnable.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
CHUNKS = BOOK / "chunks"
FAMILIES = ["anova", "ols", "glm", "lmm", "glmm"]

_LANG = {"py": ("python", ".py"), "r": ("r", ".R")}


def inject(page_text: str, chunks_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        lang, cid = m.group("lang"), m.group("id")
        fence, ext = _LANG[lang]
        code = (chunks_dir / f"{cid}{ext}").read_text().rstrip("\n")
        return (
            f"<!-- chunk:{lang}:{cid} -->\n"
            f"```{fence}\n{code}\n```\n"
            f"<!-- /chunk:{lang}:{cid} -->"
        )

    pattern = re.compile(
        r"<!-- chunk:(?P<lang>py|r):(?P<id>[\w-]+) -->.*?<!-- /chunk:(?P=lang):(?P=id) -->",
        re.DOTALL,
    )
    return pattern.sub(repl, page_text)


def main() -> None:
    n = 0
    for fam in FAMILIES:
        for page in (BOOK / fam).glob("*.md"):
            text = page.read_text()
            new = inject(text, CHUNKS)
            if new != text:
                page.write_text(new)
                n += 1
    print(f"inject_chunks.py: rewrote {n} page(s)")


if __name__ == "__main__":
    main()
