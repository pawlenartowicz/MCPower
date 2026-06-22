---
title: "Contributing to MCPower"
description: "How to file issues, suggest improvements, or contribute code to MCPower - an open-source power analysis tool with Python, R, desktop, and browser ports."
---
# Contributing

MCPower is open source — curious users and researchers are welcome to file
issues, suggest improvements, or contribute code. This page is the starting
point.

## Where to find it

The source lives at **[github.com/pawlenartowicz/mcpower](https://github.com/pawlenartowicz/mcpower)**
(the `mcpower/` directory is its own git repository). Issues, bug reports, and
feature requests all go there.

## Filing an issue

The most useful thing you can do if something seems wrong is open a GitHub issue
with:

- what you were trying to compute (model, effect size, design),
- the code or app steps that reproduce it, and
- what you expected vs what you got.

You do not need to know the cause — a clear reproduction is enough. Enhancement
requests are also welcome; describing your use-case is more useful than naming
a specific API.

## The four-ports structure

MCPower ships as four things — a Python package, an R package, a desktop app
(Windows / macOS / Linux), and a browser app — all driven by a single compiled
native engine. That means a fix or improvement to the engine benefits every
port at once, and the per-port layers are deliberately thin.

If you want to contribute code, the relevant areas are:

- **Python port** (`ports/py/`) — the `MCPower` class and its fluent API.
- **R port** (`ports/r/`) — the R6-based MCPower object and formula interface.
- **Desktop / web app** (`ports/app/`) — the shared Svelte UI, which builds for
  both Tauri (desktop) and the browser (WASM).
- **Engine** (`crates/`) — the Rust simulation kernel; the most impactful layer
  but also the steepest on-ramp.

## Building from source

Each port builds from its own directory under `ports/` — Python with `maturin
develop`, R with `R CMD INSTALL`, the desktop/web app with `pnpm`, and the Rust
workspace checks with `cargo check --workspace`. See each port's directory for
the exact commands.

> [!note]
> The engine crates require a Rust toolchain; the Python port requires
> `maturin`; the app requires `pnpm` and a Tauri toolchain. All are standard
> tools available through their respective package managers.

## Good places to start

If you are new to the codebase, the easiest contributions are documentation
improvements, example corrections, and filing reproducible bug reports. Code
contributions that add a new predictor distribution, improve an error message,
or extend an example script are well-scoped first patches.

We appreciate careful, well-tested changes over large sweeping ones — the
simulation engine has an extensive validation suite, and new numerical code is
expected to pass it.
