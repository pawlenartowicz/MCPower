# Contributing to MCPower

Thanks for your interest. Bug reports, feature ideas, and pull requests are
all welcome.

## Reporting and getting in touch

- **Open an issue:** [bug report](https://github.com/pawlenartowicz/mcpower/issues/new?template=bug_report.yml)
  or [feature request](https://github.com/pawlenartowicz/mcpower/issues/new?template=feature_request.yml).
  For bugs, include the port (Python / R / desktop / web), the version, and a
  minimal reproducer.
- **Email me:** pawellenartowicz@europe.com.
- **Find me on Bluesky:** [@freestylerscientist.pl](https://bsky.app/profile/freestylerscientist.pl).
- **No GitHub account?** Use https://mcpower.app/report — it funnels into the
  same tracker.

Please check you're on the latest version first; your issue may already be fixed.

## Building and testing

MCPower is **one Rust engine with four ports** (Python, R, desktop, web); a
contract change touches all four. See the README and `CLAUDE.md` for the layout.
You need a recent stable Rust toolchain.

| Part | Build + test |
|------|--------------|
| Rust engine | `cargo test --workspace` |
| Python | `cd ports/py && maturin develop --release && pytest tests/` |
| R | `cd ports/r && R CMD INSTALL --no-multiarch . && Rscript -e 'library(mcpower); testthat::test_dir("tests/testthat")'` then `cd ../../validation && Rscript regression.R` |
| Web / WASM | `cd ports/wasm && pnpm build:wasm && pnpm test` |
| Desktop app | `cd ports/app && pnpm install && pnpm test` (dev: `pnpm dev:tauri`) |

## Pull requests

1. Branch off `main`; keep the blast radius minimal.
2. Add or update tests for any behavior change.
3. Rust must pass `cargo fmt --all` and `cargo clippy --workspace --all-targets`;
   format Python with `ruff`. Match the surrounding style.
4. Open a PR describing **what** changed and **why**, linking any related issue.

If you alter a public orchestrator type or entry point, expose it consistently
in every port that surfaces it.

By contributing, you agree your work is licensed under the project's
[GPL-3.0-or-later](LICENSE).
