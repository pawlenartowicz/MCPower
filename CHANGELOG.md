# Changelog

All notable user-facing changes to MCPower are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each language
port keeps its own version, so a version listed here may reach Python, R, the
desktop app, and the web app on different dates.

## [1.1.2] — 2026-08-24

### Changed
- Internal upgrade of the mixed-model fitting kernel. No change to the API,
  the arguments, or the meaning of any result.

## [1.1.1] — 2026-07-23

### Added
- Two new outcome families. `family="probit"` fits binary outcomes through a
  normal-latent link, and `family="poisson"` fits counts and event rates
  through a log link. Poisson designs set their reference point with
  `set_baseline_rate(lambda)`; a clustered Poisson sizes its random intercept
  with `tau_squared=` rather than `ICC=`, since a count model has no natural
  latent-scale ICC.
- Adaptive Gauss-Hermite quadrature for clustered binary and count models, via
  the new `agq=` argument on `find_power` / `find_sample_size` (the app shows it
  as an **AGQ** estimation mode). The default `agq=1` is the previous Laplace
  approximation; an odd value up to 25 is more accurate at small cluster sizes
  and slower. Allowed only for a single grouping factor with at most three
  random effects per group — any other design warns and runs on Laplace.
- Random slopes on crossed and nested extra groupings. A mixed model can now
  carry varying slopes on a second grouping — for example
  `(1 + treatment | subject) + (1 + treatment | item)`. Until now slopes were
  restricted to the primary grouping and every other factor entered as a random
  intercept only.
- Data upload accepts TSV, Excel (`.xlsx`), and SPSS (`.sav`) files alongside
  CSV. Separator and quote characters are auto-detected for delimited files and
  can be overridden, multi-sheet workbooks get a sheet picker, and a preview
  dialog shows the parsed data before it is committed to the model.
- An acknowledgements page recording the prior work the engine's methods build
  on.

### Changed
- The simulation now runs in double precision throughout; data generation was
  previously single precision. Results from the same design and seed can differ
  slightly from 1.1.0.

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

[1.1.2]: https://github.com/pawlenartowicz/mcpower/releases
[1.1.1]: https://github.com/pawlenartowicz/mcpower/releases
[1.1.0]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.3]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.2]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.0]: https://github.com/pawlenartowicz/mcpower/releases
