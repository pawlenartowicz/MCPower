# Changelog

All notable user-facing changes to MCPower are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each language
port keeps its own version, so a version listed here may reach Python, R, the
desktop app, and the web app on different dates.

## [1.5.1]

### Changed
- **Clustered binary and count models (GLMM) use the per-fit Hessian standard
  error by default.** `wald_se` defaults to `"hessian"`, the standard error
  glmer's summary reports (lme4 `use.hessian = TRUE`). `"rx"`, the Schur
  standard error (`use.hessian = FALSE`), is still available and faster: a
  Hessian run takes 1.24–1.35× as long. The app and web app's estimation
  switch defaults to Accurate. GLMM results change; `wald_se="rx"` gives the
  1.5.0 standard error. In the logit calibration under the new default, the
  between-within *t* test of a cluster-level effect rejects a true null at
  7.1–9.6% with 5 clusters and 6.2–6.7% with 10, and the Wald *z* at
  14.4–19.5% with 5 and 10.6% with 10.

### Fixed
- **Correlation noise had a smaller SD than set.** Each perturbed correlation
  is the average of two noise draws, so `correlation_noise_sd` reached the
  correlations divided by √2 (0.106 instead of 0.15 under `realistic`, 0.21
  instead of 0.30 under `doomer`). The noise now has the set SD. Results under
  any scenario with correlation noise change, including the `realistic` and
  `doomer` presets.
- **Sample-size grid for nested designs.** The lowest grid N counted rows per
  primary cluster only, so with a nested grouping such as `(1|school/class)`
  it could put one row in each class, which gives as many class levels as
  rows and cannot be fitted. Each child cluster now gets at least 2 rows, plus
  one per random slope on the nested grouping. The app and web app's Find
  power limits use the same floor.

## [1.5.0]

### Added
- **Small-cluster warning for clustered binary and count models (GLMM).**
  `find_power` and `find_sample_size` warn when a cluster-level effect is
  tested with fewer than 20 clusters, under either test these models use.
  There the between-within *t* test rejects a true null more often than α
  (7.5–9.6% at 5 clusters, 6.6–6.8% at 10 in the logit calibration), and the
  Wald *z* (`df_method = "asymptotic"`, and crossed or nested designs) more so
  (14.6–20.3% at 5, 10.9–11.2% at 10), so part of the reported power is excess
  Type I error. A
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
- **Linear mixed models with a random slope are tested at Satterthwaite df
  by default.** `df_method` gains two values, `"auto"` and
  `"satterthwaite"`, and `"auto"` is the new default. It picks the test from
  the design: between-within df for a single grouping factor without a
  random slope and for any single-grouping binary or count mixed model (both
  as before); per-fit Satterthwaite df, as lmerTest computes them, for a
  linear mixed model with a random slope on its single grouping (new) or
  with crossed or nested groupings (as before); the Wald z for a binary or
  count model with crossed or nested groupings (as before). Under
  between-within df a random-slope variable was tested at the within-cluster
  df however few clusters there were, so with few clusters the test rejected
  a true null more often than lmerTest's and power was overstated. Every
  other value names one test: `"between_within"` with crossed or nested
  groupings and `"satterthwaite"` on a binary or count mixed model are now
  errors that name `"auto"`. Before, `"between_within"` on a crossed or
  nested design ran Satterthwaite df (linear) or the Wald z (binary or
  count) without saying so. The app and web app label the values
  "Automatic (default)", "Between-within df", "Satterthwaite df" and "Wald z
  (asymptotic)", list only those the design supports, and report a saved
  configuration whose value the design does not support. Moves results:
  linear mixed models
  with a random slope, where power falls most with few clusters. On a
  single grouping factor `df_method="between_within"` gives the previous
  default's results, except for a model with random slopes under a
  `test_formula` or with an excluded factor level, which now keeps its
  slopes (see Fixed).
- **Mixed models accept smaller clusters.** Every port now takes a cluster of
  2 observations, plus 1 for each random slope on the primary grouping, in
  `find_power`, `find_sample_size` and `set_cluster(cluster_size=)`, and fewer
  is an error. Before, Python and R refused fewer than 5 observations per
  cluster and warned below 10, so a pre/post design could not be planned, and
  the app warned below 10. Those cluster-size warnings are gone in every port;
  the Laplace-bias warning for binary and count models stays.
  `limits.reliable_rows_per_cluster` is removed.
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
  models only: regression with a continuous outcome, and ANOVA. On a logistic,
  Poisson or mixed model a contrast target is now an error in every port
  ("contrasts ('g[2] vs g[3]') are available for OLS models only; ...").
  Before, the run went ahead and reported 0% power for a contrast it never
  tested. The app and web app offer custom contrasts only where they can be
  tested, and drop contrasts left from a continuous outcome when the model
  moves to one that cannot test them.
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
- A model of any family other than `family="lme"` (the default `"ols"`, or
  `"logit"`, `"probit"` or `"poisson"`) whose formula has a random-effect
  term such as `(1|g)` or `(1 + x|g)` but no `set_cluster` call is now an
  error that names the grouping, as it already was for `family="lme"`.
  Before, it ran as an unclustered model and dropped the random intercept
  and slopes without notice.
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
  not check.
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
- A mixed model with random slopes keeps its slopes when the fitted model
  is smaller than the generating one: under a `test_formula` that leaves out
  some predictors, and when a rare factor level is excluded
  (`factor_min_level_count`). Before, a linear mixed model was refitted
  without its random slopes, so the slope variance was ignored and power
  came out too high (in one example 74% instead of 24%). A binary or count
  mixed model (GLMM) put the slope on whichever predictor took its place
  in the smaller model, which gave wrong power, or stopped with an internal
  error. On a mixed model, a `test_formula` that leaves out a predictor with
  a random slope is now an error that names it, since the test model keeps
  the model's random effects. OLS and GLM fit no random effects, so there
  it may leave one out. Moves results: mixed models with random slopes and a
  `test_formula` that leaves out a predictor, and linear mixed models with
  random slopes in which a rare factor level is excluded.
- Python and R: a random slope must be declared both in the formula
  (`(1 + x|g)`) and in `set_cluster` (`random_slopes`); a slope in only one
  of them is an error that names the variable. Before, a slope written only
  in the formula was silently dropped, so the model was simulated and fitted
  with a random intercept only and power came out too high, and a slope
  given only in `set_cluster` was simulated and fitted although the formula
  did not show it. The groupings of a nested term `(1|A/B)` are the
  exception: the formula cannot write a slope on them, so their slopes come
  from `set_cluster` alone, as before. The app and web app take the slopes
  from the formula and were not affected.
- The uncorrelated random-effect syntax `(1 + x || g)` is refused with a
  message that says so and suggests `(1 + x | g)` with a zero slope-intercept
  correlation. Before, it stopped with a formula syntax error.
- A random-effect term in `test_formula`, such as `(1|g)`, is an error that
  says `test_formula` sets the fixed effects of the test model only (the test
  model keeps the model's random effects). Before, Python told the user to
  add the grouping to the model formula as a predictor, and R and the app
  dropped the term without notice, even one naming a grouping the model does
  not have.
- A post-hoc request (`target_test="all-contrasts"`) or a named contrast
  (`"f[2] vs f[3]"`) on a mixed model or a binary or count model is an error
  that says post-hoc comparisons are available for OLS models only and
  suggests testing the factor's dummies instead. Before, a post-hoc request
  stopped with "internal contract validation failed: posthoc tests require
  estimator == Ols", and a named contrast ran with 0% power (see Changed).
- The colon spelling of a nested intercept, `(1|school:classroom)` (alone or
  beside `(1|school)`), and a random slope inside a nested term, such as
  `(1 + x|school/classroom)`, are refused with messages that point to
  `(1|school/classroom)` and, for the slope, to `set_cluster`. A grouping
  name the parser does not take, such as `(1|school.id)`, is refused with a
  message naming it, and any other random-effect term the parser does not
  take is refused as an unsupported random-effect term. Before, each stopped
  with "formula syntax error at position 0".
- Settings that break a model rule stop with a user error in the model's own
  terms: random-slope correlations that do not form a valid correlation
  matrix, a random slope with a zero intercept variance, too many random
  slopes or groupings, a crossed grouping beside a primary grouping sized by
  `cluster_size`, a nested `n_per_parent` that does not divide
  `cluster_size`, more than one nested term (such as
  `(1|a/b) + (1|c/d)`, now refused when the formula is read, naming both
  terms), and alpha or a scenario setting out of range. Before, they
  stopped with "internal contract validation failed: ..." and the names of
  internal fields (a `RuntimeError` in Python; now `ValueError`), and in R a
  nested `n_per_parent` that does not divide `cluster_size` aborted the R
  session.
- An exact factor with equal level shares, such as 50/50 or a third each
  (exact is the optimistic default, or `set_factor_sampling(..., "exact")`),
  in a design with a crossed grouping, such as `(1|subject) + (1|item)`, takes
  its levels in the same shares within every crossed level. Before, the factor
  could follow the crossed level when the number of primary clusters (or
  nested groups) was odd: a 50/50 factor over 5 subjects and 2 items put its
  second level on 40% of one item's rows and 60% of the other's, and an equal
  three-level factor over 5 subjects and 3 items put its third level on 60% of
  one item's rows. Moves results: designs with a crossed grouping and an exact
  row-level factor. Realistic and doomer, which sample factor levels, do not
  move.
- A mixed model with a single cluster (`n_clusters = 1`) is an error,
  "n_clusters must be an integer >= 2, got 1", which also says that uploaded
  data needs at least 2 distinct groups. `set_cluster` already refused it in
  Python and R; `get_effects_from_data` on data with one group (Python and
  R), and a contract built directly, reached the engine, which stopped with a
  between-within df message or, with `df_method = "asymptotic"`, fitted a
  model with no between-cluster variance to estimate.
- Python and R: an OLS model fitted to clustered data (`(1|g)` with
  `set_cluster`, `family="ols"` or `estimator="ols"`) runs with
  `scenarios`. Before, it stopped with "internal contract validation failed:
  scenario.lme requires estimator == Mle". Its realistic and doomer
  scenarios perturb the random effects as a mixed model's do.
- R `MCPowerDebug$create_data()` and the other debug stages floor `debug_n`
  to the cluster atom with the warning `find_power` gives ("sample_size 105
  not a multiple of the cluster atom 10; using 100"), and refuse an N with
  fewer complete clusters than `find_power` accepts. Before, a fixed
  `cluster_size` with an N off the atom stopped with "index out of bounds".
- R `find_power` checks `sample_size` as Python does: one whole number (a
  double such as `200` is accepted) between 20 and 100,000, and at least 15
  plus the number of design columns, with Python's messages. Before, R ran
  `find_power(10)`, `find_power(150.5)` and `find_power(200000)`, and
  refused `NA` or `c(100, 200)` with messages that did not name
  `sample_size`. The app and web app check the total observations Find
  power's n stands for (for a mixed model planned in clusters, the number of
  clusters times the cluster size) against the same limits: the n input
  stays inside them, and an n outside them keeps Find power from starting
  and says under the input how many observations it gives, for example "n =
  10 gives 10 observations; Find power needs 20 to 100,000 observations". A
  mixed model's lower limit also holds the design to what the sample-size
  search accepts: at least 5 clusters under a fixed cluster size, and at least
  2 observations per cluster, plus 1 for each random slope, under a fixed
  cluster count. A fractional n is
  refused too. Before, the app took any n of 1 or more. The limits live in `configs/config.json` (`limits.sample_size`), which Python
  and the app now read too.
- R `set_cluster` with two or more random slopes defaults a missing
  `corr_with` to zeros, one per earlier slope, as Python and the app do.
  Before, the second slope needed `corr_with` and `find_power` stopped with
  "internal contract validation failed: slope 1 corr_with must list
  correlations with all earlier slopes; expected 1, got 0". A `corr_with` of
  the wrong length now stops in `set_cluster`, naming the slope and the
  length it needs.
- R: an error message that contains '%' (for example a grouping name such as
  `g%d`) reaches R as written. Before, R read the message as a format
  string: the text came out garbled, and a '%n' aborted the R session.
- Python `set_cluster` refuses a NaN or infinite `ICC`, `slope_variance`,
  `slope_intercept_corr` or `tau_squared` with a message that names it, and
  R `set_cluster` does the same for `ICC`, `tau_squared`, `n_clusters`,
  `cluster_size`, `n_per_parent` and each slope's `variance`,
  `corr_with_intercept` and `corr_with`. Before, Python stopped
  with "invalid clusters JSON" and R with "missing value where TRUE/FALSE
  needed" or an engine message.
- App and web app: a run on uploaded data is not kept in History, and the
  History panel says so under its search box. Before, every such entry held
  a full copy of the dataset, and each new run rewrote all of them (up to
  25) to `history.json` or the browser store, which slowed saving and could
  run out of storage.
- Under `correction="tukey"`, a target outside every factor (a continuous or
  binary predictor, an interaction) reports its corrected power as not
  applicable: NaN in Python, `NA` in R, "n/a" in the tables of every port,
  and no point on the corrected chart. Before, it read 0.0. Its corrected
  required N is not applicable too. With such a target, the corrected views
  over all tests are not applicable as a whole: "First N achieving all
  targets", the joint required-N table and the at-least-k and exactly-k
  charts read n/a or are left out, and `success_count_histogram_corrected`,
  `first_joint_achieved`, `first_joint_achieved_unit` and `fitted_joint` are
  `None` / `NULL`. Results carry a new `corrected_not_applicable` flag per
  target; factor levels and contrasts keep the values they had, and the
  Tukey warning now says so.
- A crossed or nested grouping with random slopes follows the primary
  grouping's rules: correlations that do not form a valid correlation
  matrix, and a random slope with a zero intercept variance, are errors.
  Before, both ran without notice: the first drew a slope at a variance
  other than the one set (4.26 times the set value in one example), and the
  second ignored the correlation.
- A variable given two random slopes on one grouping, such as
  `(1 + x + x|g)` with `random_slopes=["x", "x"]`, is an error. Before, it
  ran with twice the slope variance, so power came out too low (in one
  example 69% instead of 93%).
- Python `set_cluster` refuses `random_slopes` without a `slope_variance`
  above 0. Before, the slope was fitted but never simulated, so power came
  out as for a random-intercept model (in one example 100% instead of 93%).
- R `set_cluster` refuses an argument it does not know, as Python does.
  Before, a misspelled `icc = 0.5` was dropped without notice and the model
  ran without clustering (in one example 99.5% power instead of 38%).
- R `set_cluster` refuses `n_clusters` together with `cluster_size`, as
  Python does. Before, the run used `n_clusters` while the check on rows per
  cluster read `cluster_size`.
- R `set_cluster` refuses an `n_clusters`, `cluster_size` or `n_per_parent`
  past R's integer range with a message that names it. Before, a run stopped
  with "missing value where TRUE/FALSE needed".
- R `find_sample_size` checks `from_size`, `to_size` and `by` as Python
  does, with Python's messages. Before, it truncated a fractional value
  (`by = 2.5` ran as 2) and ran a `by` larger than the range.
- R `MCPowerDebug`: a `debug_n` or `debug_n_sims` below 1 is an error that
  names it. Before, a negative `debug_n_sims` aborted the R session.
- `estimator="glm"` on a continuous outcome and a stray colon in a formula
  (`y ~ x1 + x2:`) stop with a user error that says what to change. Before,
  each stopped with "internal contract validation failed: ..." (a
  `RuntimeError` in Python; now `ValueError`).
- Python: `df_method=""` is refused as not recognised. Before, it stopped
  with "invalid LinearSpec JSON".
- App and web app: rerunning a History entry whose n is outside the limits
  says so and does not start the run.

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
