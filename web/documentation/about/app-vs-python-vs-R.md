---
title: "App, Python, or R - Which MCPower Face to Use"
description: "Compare MCPower's four interfaces - browser app, desktop app, Python package, and R package - to pick the right face for your power analysis workflow."
---
# App, Python, or R — which face to use

Pick by what you are doing right now. To **try MCPower this minute without installing anything**, open the browser app — it runs the full engine in a browser tab. For **point-and-click analysis on your own machine**, with bigger uploads and no internet needed, install the desktop app. To **build a reproducible pipeline, script your power analysis, or run it against your own data files**, use the Python or R package. It is one engine underneath all of them, so a study you set up in one maps cleanly onto the others — switching later costs nothing.

The browser build is the desktop app *without installing* — a way to reach the same point-and-click tool from a link, not a separate product. So the real choice is three: **app, Python, or R**.

## What each face can do

| | App (desktop) | App (browser) | Python | R |
|---|---|---|---|---|
| **Install** | One installer (Windows/macOS/Linux) | None — open a link | `pip install mcpower` | `install.packages("mcpower")` |
| **Upload row cap** | 1,000,000 rows | 10,000 rows | 1,000,000 rows | 1,000,000 rows |
| **How you drive it** | Point and click | Point and click | Write a short script | Write a short script |
| **What you save** | A saved session | A saved session | A `.py` script | A `.R` script |
| **Works offline** | Yes | After first load | Yes | Yes |
| **Plotting** | Built in, nothing to install | Built in, nothing to install | Built in (optional extras for export) | Built in (optional extras for export) |

The browser app's lower upload cap (10,000 vs 1,000,000 rows) is the one capability gap that follows from *where* it runs — a browser tab is a tighter memory budget than a desktop install. Everything else is the same engine, the same models, the same numbers. If your pilot dataset is larger than 10,000 rows, use the desktop app or a package rather than the browser.

## App, Python, and R give the same answer

Within any one face, a seeded run reproduces *exactly* — same seed, same inputs, same numbers, every time. Across faces (app vs Python vs R) the results are identical for every practical purpose: the only difference is floating-point rounding in the last bits, far below anything you would report. The single observable consequence would be a significance call sitting *exactly* on the α boundary flipping by one decision between two faces — a coincidence on the order of once in a billion years of running. It is noted here for honesty, not because it is something to plan around. See [[concepts/limitations|the limitations page]] for the full statement.

One caveat sits a level below this: a run split across a *different number of workers* (most visibly the browser app, which parallelises across browser threads) draws a different random path and lands on a slightly different estimate — statistically equivalent, within Monte Carlo noise, but not byte-identical. That is the same seed reproducing per-machine, not across every degree of parallelism. The reassurance above still holds: the faces agree to the last decimal that matters.

## Where next

- [[about/comparison|How it compares]] — MCPower vs G*Power, superpower, simr, pwr, and WebPower. That is a different axis: this page is which MCPower face to use; that one is MCPower versus other tools.
- [[internals/engine-architecture|One engine, four ports]] — the engineering behind "same design, same seed, same answer" across all four faces.
