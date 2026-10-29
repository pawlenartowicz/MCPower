# Changelog

All notable user-facing changes to MCPower are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each language
port keeps its own version, so a version listed here may reach Python, R, the
desktop app, and the web app on different dates.

## [1.5.0]

### Added
- **Small-cluster warning for clustered binary and count models (GLMM).**
  `find_power` and `find_sample_size` warn when a cluster-level effect is
  tested with fewer than 20 clusters under the default
  `df_method="between_within"`. There the between-within *t* test rejects a
  true null more often than α (7.5–9.6% at 5 clusters, 6.6–6.8% at 10 in the
  logit calibration), so part of the reported power is excess Type I error. A
  sample-size search names the grid N below which the design has fewer than
  20 clusters, or says that every grid point has fewer. The app and web app
  show the warning in the diagnostics panel. The threshold is
  `limits.glmm_reliable_clusters` in `configs/config.json`.
- `set_factor_sampling(name, mode)` (Python and R) sets how one factor's group
  sizes are realised, overriding the scenario-wide `sampled_factor_proportions`
  for that factor only. `"exact"` holds the group sizes at the requested
  shares in every simulation, `"sampled"` draws each observation's level at
  random so the sizes vary between simulations, and `None` (`NULL` in R)
  removes the override. It works for declared factors and for factors from
  uploaded data.
- App and web app: for a mixed model, Find power's n can be given in the
  same units as the sample-size search: total observations, or the number of
  clusters or observations per cluster, whichever the cluster settings leave
  free. One unit switch now serves both cards, and an echo line restates n
  in the other unit.

### Changed
- MCPower now builds on version 0.7.0 of its mixed-model fitting library
  (glmm), which fits logistic and Poisson models on nearly or exactly
  collinear designs more robustly. Some fits that used to stop unconverged,
  on a nearly collinear design or from a start far from the estimate, now
  converge. A clustered fit that converges but cannot produce standard
  errors is now reported as unconverged, as unclustered fits already were.
  Before, it counted as converged but was never significant. This happens in
  degenerate designs, such as an interaction column that is all zero or a
  rare factor level in few clusters: there the convergence rate can drop
  sharply, and a run can now stop at the default convergence-failure
  threshold where 1.4.1 finished. Elsewhere, power and convergence rates
  for binary and count outcomes can differ slightly from 1.4.1 on nearly
  collinear designs. Moves results: clustered binary and count models on
  degenerate designs.
- Tukey HSD thresholds come from a more accurate studentized-range
  distribution and are solved far less often. A run solves a handful of
  thresholds per α and factor size, however many sample sizes, scenarios or
  draws it has. Before, it solved them again for every sample size and
  scenario, and a crossed or nested linear mixed model tested at
  Satterthwaite df evaluated the distribution once per target in every draw.
  Thresholds for factors with three or more levels match the exact
  distribution to a relative error of 1e-10 or better (the old distribution
  code was accurate to about 1e-9 to 1e-5), so Tukey-corrected power can
  differ from 1.4.1 only in rare draws that fall at the threshold.
- With a nested grouping such as `(1|school/class)` and a fixed
  `cluster_size`, the sample-size grid steps in whole top-level clusters
  (multiples of `cluster_size`). Before, it stepped in multiples of
  `cluster_size` times `n_per_parent`. A search has finer steps and can
  settle on a different N. Moves results: sample-size searches on such
  designs.
- `get_effects_from_data` (Python, R, app and web app) reports the effects of
  a continuous outcome (OLS and linear mixed models) in residual-SD units,
  the scale `set_effects` reads, since the simulation fixes the residual SD
  at 1. Before, OLS effects were in units of the outcome's total SD and
  linear mixed model effects in the outcome's raw units. A model that fits
  the outcome exactly is now an error. With a cluster declared but an OLS
  fit, it says the ICC is unavailable. Binary and count effects are
  unchanged. Moves results: effects estimated for continuous outcomes.
- Sample-size results carry the overall test and the estimator details for
  every grid N. The Python and R power-curve plots draw the overall test as
  one more curve, as the power-at-N chart already did. The text report of a
  sample-size search leaves out the estimator-details block, which describes
  a single run. A sample-size search checks the warning about Laplace bias
  with small clusters in binary GLMMs at every grid N.
- Pairwise contrast targets (such as `"g[2] vs g[3]"`) are tested on OLS
  models only: regression with a continuous outcome, and ANOVA. On a
  logistic, Poisson or mixed model a contrast target is now an error in
  every port ("contrasts are only supported for estimator=Ols"). Before, the
  run went ahead and reported 0% power for a contrast it never tested. The
  app and web app offer custom contrasts only where they can be tested, and
  drop contrasts left from a continuous outcome when the model moves to one
  that cannot test them.
- A non-normal residual chosen with `set_residual_distribution`
  (`high_kurtosis`, `right_skewed`, `left_skewed`) uses 10 df in every
  scenario: Student *t*(10) or a scaled χ²(10), at unit variance. Before, its
  df came from the scenario: 10 under optimistic, 8 under realistic, 5 under
  doomer.
- Realistic and doomer results change for continuous outcomes (OLS and
  linear mixed models) and for clustered models, including clustered binary
  and count models, because the scenario residual swap is removed (see
  Removed). The removal changes them in two ways. The residual no longer
  swaps to *t*(8)/χ²(8) under realistic or *t*(5)/χ²(5) under doomer, and the
  ICC jitter of a clustered model is now drawn one or two scenario draws
  earlier, so the same seed gives different draws. The removal itself leaves
  optimistic results for every model, and unclustered binary and count
  results under every preset, unchanged. Moves results: realistic and doomer
  runs of continuous and clustered models.

### Removed
- **The scenario residual swap.** The scenario keys `residual_change_prob`,
  `residual_dists` and `residual_df` are gone from every preset and from
  `configs/scenarios.json`; passing one to `set_scenario_configs` (Python
  and R) raises an "unknown config key" error. The swap replaced the normal
  residual with a heavy-tailed or skewed one of the same variance. The
  variance of a coefficient estimate depends on the residual variance, not
  its shape, so the swap never made the realistic or doomer preset more
  pessimistic.
- The `residual_pinned` key of `summary()` (Python and R).
- App and web app: the "Allowed residual swap distributions" control, the
  two residual-swap scenario settings (change probability and df), and the
  default unpinned residual option. The residual select defaults to
  `normal`.

### Fixed
- Exact factor levels (the optimistic default; see `set_factor_sampling`)
  are balanced jointly. Several factors are assigned over their crossed
  cells, so every combination of levels gets its share, and cluster-level
  factors are assigned over clusters the same way. With a fixed number of
  clusters (`n_clusters`), a within-cluster factor takes its levels inside
  every cluster, and with a nested grouping such as `(1|school/class)`,
  inside every nested group. Before, each factor was assigned on its own,
  and the levels could line up with another factor or with the clusters:
  two 50/50 factors were identical, so neither effect could be tested (power
  0) and the overall test was inflated; a 50/30/20 factor in 30 clusters put
  a single level in every cluster; and a 50/50 factor with two classes per
  school put a single level in every class, so the factor acted as a
  cluster-level treatment. The overall group sizes do not change. Moves
  results: designs with two or more exact factors; every design with a
  fixed number of clusters and an exact within-cluster factor, except a
  two-level 50/50 factor; and every design with a nested grouping and an
  exact within-cluster factor.
- Heteroskedasticity standardises its driver with the exact mean and SD of
  the linear predictor, counting interaction terms and the dummies of a
  factor with three or more levels. Before, it left out interactions and
  treated a factor's dummies as uncorrelated, so in such models the residual
  variance did not follow the ratio λ that was set. Moves results: runs with
  heteroskedasticity (the realistic and doomer presets, or any
  `heteroskedasticity_ratio` other than 1) of models with interactions or a
  factor with three or more levels.
- A heteroskedasticity driver set with `set_heteroskedasticity_driver`
  (Python and R) is matched to its predictor when the model runs. Before, it
  was matched when the setter was called, so declaring factors or uploading
  data afterwards could point it at a different predictor.
- Holm and Benjamini-Hochberg corrections of post-hoc contrasts rank each
  factor's contrasts within that factor, matching the per-factor thresholds.
  Before, the contrasts of all factors were ranked together. Moves results:
  Holm or Benjamini-Hochberg post-hoc contrasts on designs with two or more
  factors.
- A `test_formula` may write an interaction with its terms in another order
  (`b:a` for the model's `a:b`); before, this stopped with an internal
  error. A test interaction the model does not contain is a clear error.
- Python: a cluster given only by `cluster_size` is simulated with that
  cluster size, as in R and the app. Before, `find_power` turned it into a
  fixed number of clusters (sample size divided by `cluster_size`), and later
  runs on the same model kept that number, so a `find_sample_size` after a
  `find_power` searched with the cluster count fixed. Moves results: Python
  models whose cluster is set by `cluster_size`.
- When a cluster is given by `cluster_size`, the number of clusters a sample
  size implies must be at least `limits.min_clusters` (5); fewer is an error
  that names the count (Python and R). Before, Python required 2 and R did
  not check. R now also checks observations per cluster at the requested
  sample size, as Python does: fewer than 5
  (`limits.reliable_rows_per_cluster`) is an error, fewer than 10
  (`limits.recommended_rows_per_cluster`) a warning.
- R runs the model checks Python already ran. These are now errors:
  `family = "lme"` without a random-effect term, or without a `set_cluster()`
  call for every grouping in the formula; `set_cluster()` on a grouping the
  formula does not have, or with neither `n_clusters` nor `cluster_size` for
  a top-level grouping; an `n_per_parent` that is not an integer of at least
  2; and `set_simulations()` below 1.
- R: a zero or negative `sample_size`, `n_sims`, `from_size`, `to_size` or
  `by` is an error. Before, it could crash the R session.
- Python: Ctrl-C stops a running `find_power` or `find_sample_size` at the
  engine's next checkpoint, with or without progress output. Before, it
  took effect only when the run had finished.
- Uploaded data (`upload_data`, Python and R) is typed and encoded the same
  way on every upload and every run. A two-value column is now sent as 0/1
  (the lower value as 0). Before, its raw values were sent, so a column coded
  other than 0/1, such as 1/2, was simulated with the wrong share of each
  value. In R, a data frame is read column by column instead of through
  `as.matrix()`, so one character column no longer turns every column into
  text, which changed how the other columns were detected and rounded their
  numbers to 7 significant digits. A setter or upload called after a run now
  sees the formula's predictors; before, it saw the factor dummies the run
  had created (such as `g[2]`), so the next run could differ from a fresh
  model with the same settings. Moves results: first uploads with a
  two-value column not coded 0/1 or (R) a data frame with a character column,
  and setters or uploads called after a run.
- Uploads with missing values are rejected. Python checks every column,
  text columns included, for blank, NA and NaN cells; before, it checked
  numeric columns only. R rejects missing and infinite values, which it did
  not check before. The app and web app reject a missing cell in any column,
  and a non-numeric or infinite cell in a numeric or two-value column;
  before, they read it as 0.
- A predictor taken from uploaded data can no longer be a cluster-level
  variable: it is resampled from the upload row by row, so it was never drawn
  once per cluster. This is an error in every port, and the app and web app
  disable its cluster-level checkbox.
- Under the optimistic preset, which starts each fit from the true values,
  the reduced fit behind factor-exclusion and `test_formula` tests on a
  linear mixed model with crossed random intercepts, where a later grouping
  has more levels than the first, now starts from the right variance values.
  Optimistic results for such designs can change.
- App and web app: a probit model's intercept is the normal quantile of the
  baseline probability, Φ⁻¹(p), as in Python and R. Before, the app used the
  logit, log(p/(1 − p)), so probit models ran at the wrong baseline rate.
  Moves results: app and web probit models.
- App and web app: a factor whose chosen reference level is not its first
  level gets the right group sizes. Before, the shares were applied in the
  declared order, so the reference took the first level's share. Uploaded
  factors are coded the same way, and effects estimated from uploaded data
  name and code factor levels as the power run does, including an
  undeclared predictor that matches an uploaded factor column. Moves
  results: app and web factors with a non-first reference level and unequal
  shares.
- App and web app: a request to start a run while one is running is
  ignored.

## [1.4.1] — 2026-10-22

### Added
- Sample-size searches (`find_sample_size`, Python and R) report a required N
  for each post-hoc pairwise contrast (`target_test="all-contrasts"`): one
  "(pairwise)" span per factor in the printed Required N table, and one row per
  contrast in the Python `to_dataframe()` / R `as_tibble()` required-N frames.
- Config key `simulation.nested_n_per_parent` (default 2): a nested `(1|A/B)`
  grouping with no `n_per_parent` uses it.
- `simulation.n_sims.glmm` (default 400) in `configs/config.json`.

### Changed
- **Default simulation count for clustered binary and count (GLMM) models is
  400.** In 1.4.0 a Python/R `MCPower` with a logit or Poisson outcome used
  1,600 even when clustered. Pass `set_simulations(1600)` to reproduce 1.4.0
  run sizes. The app and web app switch the default when the outcome moves
  between Continuous and Logit/Poisson, unless the count was edited.
- **`MCPower.n_simulations` is `None` until `set_simulations()` is called**
  (Python and R). The count a run uses is resolved at run time from the design;
  reports show the resolved value.
- **Crossed or nested extra grouping under `df_method="between_within"`:** an
  LME is tested against a *t* at each fit's Satterthwaite df, as lmerTest
  computes it; a fit whose Satterthwaite pass fails falls back to Wald z. A
  GLMM still uses Wald z. Holm and Benjamini-Hochberg decisions for these fits
  use p-values, matching R's `p.adjust`. Power can differ from 1.4.0;
  `df_method="asymptotic"` gives the 1.4.0 numbers.
- Heteroskedasticity applies to all continuous outcomes, including linear mixed
  models; the "no effect" warning fires only for binary and count outcomes.
- A nested child grouping given `n_clusters` or `cluster_size` is an error
  asking for `n_per_parent` (Python and R).
- AGQ (`agq`/`nagq` > 1) is eligible only with the GLM estimator; otherwise it
  is stripped to Laplace with a warning.
- The convergence-failure error reads "Convergence failure rate ..."; the
  τ-boundary breakdown and `n_clusters` hint appear only for clustered models.
- App and web app: the GLM extras line reads "event rate = X% · baseline Y%".
- Building the R package from source needs rustc >= 1.85.

### Fixed
- A factor that appears only inside interactions no longer gets spurious
  main-effect rows for its dummy columns (Python and R).
- `debug_report` / `load_data`: the first simulated dataset uses the same
  heteroskedasticity coefficients as the fits.
- App and web app: the run menu no longer starts a run when the form has
  blocking errors.
- App and web app: the correlation editor shows the measured correlation for
  untouched uploaded continuous pairs.

### Removed
- The "GLM baseline drift" warning, the app's matching convergence notice, and
  `glm_baseline_drift_max` in `configs/config.json`; `baseline_prob_requested`
  is no longer in the report meta.

## [1.4.0] — 2026-09-06

### Added
- `df_method` on `find_power` / `find_sample_size` (Python and R) selects the
  reference distribution for fixed-effect tests in any mixed model (LME or
  clustered GLMM). `"between_within"` tests each fixed effect against a *t*
  with between-within denominator degrees of freedom — between-cluster df for
  a predictor declared cluster-level, within-cluster df otherwise — and
  `"asymptotic"` keeps the Wald z. It has no effect on OLS or unclustered GLM,
  and falls back to Wald z with a crossed or nested extra grouping. A fit whose
  random-effect variances all collapse to zero is tested against the collapsed
  model's own reference instead (residual-df *t* for an LME, Wald z for a GLMM).
- App and web app: a **Denominator df** selector on the run panel for mixed
  models, next to the estimation-mode switch; exported Python/R scripts carry
  the chosen value.
- `debug_report` / `load_data` report the critical value and degrees of
  freedom per target, since they now differ across targets under the
  between-within reference.
- The Python `to_dataframe()` and R `as_tibble()` power frames include one row
  per post-hoc pairwise contrast (`target_test="all-contrasts"`), labelled as in
  the printed table, and always carry the full column set even with zero rows.
- Validation: a denominator-df calibration page comparing both `df_method`
  settings against lmerTest Satterthwaite on LMMs and against the empirical
  Type I rate on logit GLMMs.

### Changed
- **Mixed-model tests default to `df_method="between_within"`.** Power for
  any LME or clustered GLMM can differ from 1.3.x, most visibly with few
  clusters or cluster-level predictors, where the Wald z was optimistic. Pass
  `df_method="asymptotic"` to reproduce the 1.3.x numbers.
- Under the between-within default a design that leaves fewer than one
  between-cluster df (`g − p_between < 1`, i.e. at least as many cluster-level
  columns, intercept included, as clusters) or fewer than one within-cluster df
  is refused with an error naming the counts, instead of running on the Wald
  z. Pick `df_method="asymptotic"` or add clusters. In `find_sample_size` the
  check applies to every candidate N in the searched range.

## [1.3.0] — 2026-08-28

### Added
- Python: `find_sample_size` takes a new `unit=` argument for mixed models —
  `"clusters"` searches the number of participants (with a fixed cluster size)
  and `"cluster_size"` searches observations per participant (with a fixed
  number of clusters), instead of always converting to/from total N by hand.
- R: `$find_sample_size(..., unit = "observations")` mirrors the Python option.
- App: the mixed-family Find sample size card gets two unit chips
  (observations, and either "clusters" or "per `<cluster>`" depending on which
  cluster field is fixed), with an echo line restating the searched range in
  the other unit.
- Web: the browser app carries the same unit chips as the desktop app, since
  both share the same UI.

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

[1.4.0]: https://github.com/pawlenartowicz/mcpower/releases
[1.3.0]: https://github.com/pawlenartowicz/mcpower/releases
[1.1.2]: https://github.com/pawlenartowicz/mcpower/releases
[1.1.1]: https://github.com/pawlenartowicz/mcpower/releases
[1.1.0]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.3]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.2]: https://github.com/pawlenartowicz/mcpower/releases
[1.0.0]: https://github.com/pawlenartowicz/mcpower/releases
