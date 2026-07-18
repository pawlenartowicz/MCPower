# Changelog

All notable user-facing changes to MCPower are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each language
port keeps its own version, so a version listed here may reach Python, R, the
desktop app, and the web app on different dates.

## [1.1.0] — 2026-07-08

### Changed
- Clustered binary/count models (logit, probit, Poisson GLMMs) now default to a
  faster Wald standard-error method (`wald_se="rx"`, fastmode) instead of the
  old per-fit Hessian. Results shift slightly from 1.0.x runs of the same
  design; pass `wald_se="hessian"` to restore the previous behaviour exactly.

## [1.0.3] — 2026-06-25

### Added
- More visual helpers for reading power results.

## [1.0.2] — 2026-06-16

### Changed
- Improved plotting.
- Linux desktop builds are now `.deb`/`.rpm` only; the AppImage was dropped.

## [1.0.0] — 2026-03-01

Initial public release. A single Monte Carlo power-analysis engine covering OLS,
GLM, mixed-effects, and ANOVA designs, shipped across four language ports —
Python, R, the desktop app, and the in-browser web app — from the same core.

### Added
- Power at a fixed sample size and automatic sample-size search.
- Power curves across a range of sample sizes.
- Multiple-comparison corrections and joint power (the chance all key tests are
  significant in the same study).
- Robustness scenarios: rerun the same design with heterogeneous effects,
  non-normal residuals, and outliers.
- Bring-your-own-data: seed the simulation from a pilot or previous study so it
  inherits real correlations and distributions.

[1.1.0]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.3]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.2]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.0]: https://github.com/pawlenartowicz/mcpower/releases
