# MCPower documentation

Power analysis by simulation — any design from t-test to mixed models, in
your browser, on your desktop, or in Python and R.

Describe the study you plan to run, and MCPower generates thousands of
synthetic datasets that match it, fits your model to each, and counts how often
the effect comes out significant. That count is your power — no lookup tables,
no closed-form formulas that only cover textbook designs.

> [!note] Work in progress
> Nearly everything is written. Three stubs remain: the tool comparison (with
> speed benchmarks), the desktop install guide (awaits the app release), and
> the contribution page.

## Start here

- [[about/index|About MCPower]] — what it is, who it's for, and how it compares.
- [[concepts/index|Concepts]] — the statistical walkthrough, idea to power number.
- [[about/roadmap|Roadmap]] — what's coming next and what's being weighed.

## Use it

- [[tutorial-app/index|The app]] — desktop (Tauri) and browser (WASM), one GUI.
- [[tutorial-python/index|Python]] — the `mcpower` package.
- [[tutorial-r/index|R]] — the R package.
- [[internals/debug|Debug mode]] — pipeline introspection in R.

## Under the hood

- [[internals/index|What's inside]] — engine architecture and optimizations.
- [[validation/index|Validation]] — how we know the numbers are right.
