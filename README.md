# MCPower

```
███╗   ███╗  ██████╗ ██████╗ 
████╗ ████║ ██╔════╝ ██╔══██╗ ██████╗ ██╗    ██╗███████╗██████╗ 
██╔████╔██║ ██║      ██║  ██║██╔═══██╗██║    ██║██╔════╝██╔══██╗
██║╚██╔╝██║ ██║      ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝
██║ ╚═╝ ██║ ██║      ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
██║     ██║ ╚██████╗ ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═════╝ ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
```

[![CI](https://github.com/pawlenartowicz/mcpower/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/mcpower/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16502734-blue)](https://doi.org/10.5281/zenodo.16502734)

**Power analysis by simulation — any design from t-test to mixed models, in your browser, on your desktop, or in Python and R.**

## Why MCPower?

- **MCPower covers anything from ANOVA to generalized linear to mixed
  models.** Analytical power formulas exist for a few textbook designs and are
  correct only when all their assumptions are met (they aren't). Monte Carlo
  is the ground truth they approximate.
- **Fast enough to mean it.** A purpose-built engine, 100–1000× faster
  than a hand-written R/Python simulation loop — even the most complex power
  analysis runs in seconds, not hours or even days for mixed models. Speed
  stops being the reason to avoid simulation.
- **Robustness built in.** Stress-tests your design against the messy,
  non-ideal data that formulas assume away, so you catch under-powering before
  you collect.
- **Easy, and everywhere.** A few-line API across four bindings — Python, R,
  desktop app, browser. Free and open source.

## Use it — pick your language

| Language | Install | Tutorial |
|---|---|---|
| **Online** | [open mcpower.app/online](https://mcpower.app/online) — no install | [App tutorial](https://docs.mcpower.app/tutorial-app/index) |
| **Desktop app** | [download](https://mcpower.app/downloads) | [App tutorial](https://docs.mcpower.app/tutorial-app/index) |
| **Python** | `pip install mcpower` | [Python tutorial](https://docs.mcpower.app/tutorial-python/index) |
| **R** | `install.packages("mcpower", repos = "https://r.mcpower.app")` | [R tutorial](https://docs.mcpower.app/tutorial-r/index) |

> **Fedora/RHEL:** the desktop download includes an `.rpm` — `sudo dnf install ./MCPower-*.rpm` (unsigned in v1; accept the GPG prompt or pass `--nogpgcheck`).

## More than a single power number

One run gives you the whole picture: power curves across a range of sample
sizes, automatic sample-size search, multiple-comparison corrections — and
power for many p-values at once: the chance that *all* your key tests come out
significant in the same study, which is what a multi-hypothesis paper actually
stands on. On top of that, built-in robustness scenarios stress-test the
design: flip a switch and the same analysis reruns with heterogeneous effects,
non-normal residuals, and outliers, so you see the power you'd get from messy
real-world data, not just the textbook case.

## Bring your own data (optional)

You don't need any data to start — describe the predictors and effect sizes
and MCPower generates everything. But if you have a pilot or a previous study,
upload it and the simulation inherits its real correlations and distributions
instead of idealized ones.

## No speed-for-accuracy trade-off

Monte Carlo has always been the better way to estimate power: simulate the
study as it will actually run, instead of trusting a formula whose assumptions
your design doesn't meet. The only reason to avoid it was speed. That reason
is gone.

The speed is an engineering result, not a statistical shortcut. Every model
uses the standard solver: normal equations for OLS, IRLS for GLMs (like
statsmodels and R's `glm`), REML optimized with BOBYQA for mixed models (like
`lme4`). Same algorithms, same convergence tolerances, nothing approximated to
run faster. The speed comes from the low-level details — no allocation in the
hot loop, memory batched to stay in cache, data generated efficiently.

This holds at extreme significance levels, too: in the range almost every
analysis lives in, power estimates are accurate to within a tenth of a
percentage point, and they stay within one point out to the edges — down to
the 5-sigma threshold (α = 0.0000005), or equivalently a Bonferroni
correction across 100,000 simultaneous tests at 5%. Past that (genome-wide
GWAS scans and the like) you're probably still fine, but I can't certify it —
those edge cases are untested.

## Nothing leaves your machine

The online version runs entirely in your browser — the engine is compiled to
WebAssembly and executes locally, so your design and any uploaded data never
touch a server. The desktop app works fully offline. No account, no uploads.

## Checked against the tools everybody trusts

Every estimator is validated against the standard references — R, statsmodels,
and `lme4` — by fitting the same data in both and comparing the numbers. They
match. See [Validation](https://docs.mcpower.app/validation/index) for how
this is done.

## Start here

- [About MCPower](https://docs.mcpower.app/about/index) — what it is, how it compares, app vs packages.
- [Concepts](https://docs.mcpower.app/concepts/index) — the statistical walkthrough, idea to power number.

## Under the hood

- [What's inside](https://docs.mcpower.app/internals/index) — engine architecture and optimizations.
- [Validation](https://docs.mcpower.app/validation/index) — how we know the numbers are right.

## Citation & License

GPL v3. If you use MCPower in research, please cite:

Lenartowicz, P. (2025). MCPower: Monte Carlo Power Analysis for Complex Statistical Models [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.16502734

```bibtex
@software{mcpower2025,
  author    = {Lenartowicz, Pawe{\l}},
  title     = {{MCPower}: Monte Carlo Power Analysis for Complex Statistical Models},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16502734},
  url       = {https://doi.org/10.5281/zenodo.16502734}
}
```

## Issues

- [GitHub Issues](https://github.com/pawlenartowicz/mcpower/issues)

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
