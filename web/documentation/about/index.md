---
title: "About MCPower - Statistical Power Analysis by Simulation"
description: "MCPower estimates statistical power by Monte Carlo simulation for OLS, logistic, mixed-effects, and ANOVA designs. Available in Python, R, desktop, and browser."
---
# About MCPower

MCPower estimates **statistical power by simulation**. Describe the study you
plan — the model, the predictors, the effect sizes you want to detect — and the
native engine generates thousands of synthetic datasets from that description,
fits your model to each one, and counts how often the effect comes out
significant. That count is the power: no lookup tables, no closed-form
approximations, just your study run many times before you run it once.

Simulation is what lets MCPower cover designs that formula-based calculators
struggle with — correlated predictors, skewed distributions, factors with
unequal groups, clustered observations, real pilot data — all in one tool.

## Who it's for

Researchers **planning studies** — an OLS regression, a logistic model, a
mixed-effects design, a factorial ANOVA — who need a defensible sample size or
an honest power estimate. MCPower is a planning tool, not a statistics
workbench: it doesn't analyse your collected data, and it isn't a framework for
building new methods.

## One engine, four ports

MCPower ships as a **Python package**, an **R package**, a **desktop app**, and
a **browser app**. All four are thin layers over the same compiled native
engine, so a power number computed in one is produced by the exact same
calculation in the others — same design, same seed, same answer. See
[[internals/engine-architecture|one engine, four ports]] for how that works.

## App or packages?

The **app** (desktop or browser) is the no-code path: pick a model family, fill
in the panels, press *Find power* — best for exploring a design, teaching, or
getting a quick answer. The **packages** are the scripted path: an analysis is
a short script you can version, rerun, and paste into a grant or
preregistration — best for reproducibility and pipelines. It's the same engine
underneath, and a session in one maps cleanly onto the other, so switching
later costs nothing.

## Where next

- [[about/app-vs-python-vs-R|App, Python, or R?]] — which face fits your situation.
- [[about/comparison|How it compares]] — MCPower vs G*Power, superpower, simr, pwr, and WebPower.
- [[about/roadmap|Roadmap]] — what's coming next and what's deliberately out of scope.
- [[about/acknowledgements|Acknowledgements]] — the prior work and open-source software the engine builds on.
- [[about/AI-usage|AI usage]] — where AI contributed to MCPower, and where it did not.
- [[tutorial-app/index|The app]] — the GUI, panel by panel.
- [[tutorial-python/index|Python]] — install to first analysis in minutes.
- [[tutorial-r/index|R]] — the same ladder in R.
- [[validation/index|Validation]] — how we know the numbers are right.
