# mcpower 1.5.1

* **Clustered binary and count models (GLMM) use the per-fit Hessian standard
  error by default.** `wald_se` defaults to `"hessian"` (lme4
  `use.hessian = TRUE`), the standard error glmer's summary reports for a fit
  that is not singular. At a singular fit glmer reports the Schur standard
  error, and MCPower keeps the Hessian. `"rx"`, the Schur
  standard error (`use.hessian = FALSE`), is still available and faster: a
  Hessian run takes 1.24–1.35× as long. GLMM results change;
  `wald_se = "rx"` gives the 1.5.0 standard error. In the logit calibration under the new
  default, the between-within *t* test of a cluster-level effect rejects a
  true null at 7.1–9.6% with 5 clusters and 6.2–6.7% with 10, and the Wald *z*
  at 14.4–19.5% with 5 and 10.6% with 10.
* Correlation noise now has the SD you set. Each perturbed correlation is the
  average of two noise draws, so `correlation_noise_sd` reached the
  correlations divided by √2 (0.106 instead of 0.15 under `realistic`, 0.21
  instead of 0.30 under `doomer`). Results under any scenario with
  correlation noise change, including the `realistic` and `doomer` presets.
* The sample-size grid for a nested design such as `(1|school/class)` starts
  at 2 rows per child cluster, plus one per random slope on the nested
  grouping. Before, it counted rows per primary cluster only and could put one
  row in each class, which cannot be fitted.

# mcpower 1.5.0

* `find_power()` and `find_sample_size()` warn when a clustered binary or
  count model (GLMM) tests a cluster-level effect with fewer than 20 clusters,
  under either test these models use: the between-within *t* test then rejects
  a true null more often than α (7.5–9.6% at 5 clusters in the logit
  calibration), and the Wald *z* (`df_method = "asymptotic"`, and crossed or
  nested designs) more so (14.6–20.3%), so part of the reported power is excess
  Type I error. A sample-size search names the grid N below which the design has
  fewer than 20 clusters, or says that every grid point has fewer.
* New `set_factor_sampling(name, mode)` sets how one factor's group sizes are
  realised, overriding the scenario-wide `sampled_factor_proportions` for
  that factor only. `"exact"` holds the group sizes at the requested shares
  in every simulation, `"sampled"` draws each observation's level at random
  so the sizes vary between simulations, and `NULL` removes the override. It
  works for declared factors and for factors from uploaded data.
* **Linear mixed models with a random slope are tested at Satterthwaite df
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
  count) without saying so. Moves results: linear mixed models with a
  random slope, where power falls most with few clusters. On a single
  grouping factor `df_method = "between_within"` gives the previous
  default's results, except for a model with random slopes under a
  `test_formula` or with an excluded factor level, which now keeps its
  slopes.
* A model of any family other than `family = "lme"` (the default `"ols"`, or
  `"logit"`, `"probit"` or `"poisson"`) whose formula has a random-effect
  term such as `(1|g)` or `(1 + x|g)` but no `set_cluster()` call is now an
  error that names the grouping, as it already was for `family = "lme"`.
  Before, it ran as an unclustered model and dropped the random intercept
  and slopes without notice.
* MCPower now builds on version 0.7.0 of its mixed-model fitting library
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
* Tukey HSD thresholds come from a more accurate studentized-range
  distribution and are solved far less often. A run solves a handful of
  thresholds per α and factor size, however many sample sizes, scenarios or
  draws it has. Before, it solved them again for every sample size and
  scenario, and a crossed or nested linear mixed model tested at
  Satterthwaite df evaluated the distribution once per target in every draw.
  Thresholds for factors with three or more levels match the exact
  distribution to a relative error of 1e-10 or better (the old distribution
  code was accurate to about 1e-9 to 1e-5), so Tukey-corrected power can
  differ from 1.4.1 only in rare draws that fall at the threshold.
* With a nested grouping such as `(1|school/class)` and a fixed
  `cluster_size`, the sample-size grid steps in whole top-level clusters
  (multiples of `cluster_size`). Before, it stepped in multiples of
  `cluster_size` times `n_per_parent`. A search has finer steps and can
  settle on a different N. Moves results: sample-size searches on such
  designs.
* `get_effects_from_data()` reports the effects of a continuous outcome (OLS
  and linear mixed models) in residual-SD units, the scale `set_effects()`
  reads, since the simulation fixes the residual SD at 1. Before, OLS effects
  were in units of the outcome's total SD and linear mixed model effects in
  the outcome's raw units. A model that fits the outcome exactly is now an
  error. With a cluster declared but an OLS fit, it says the ICC is
  unavailable. Binary and count effects are unchanged. Moves results:
  effects estimated for continuous outcomes.
* Sample-size results carry the overall test and the estimator details for
  every grid N. The power-curve plot draws the overall test as one more
  curve, as the power-at-N chart already did. The text report of a
  sample-size search leaves out the estimator-details block, which describes
  a single run. A sample-size search checks the warning about Laplace bias
  with small clusters in binary GLMMs at every grid N.
* Pairwise contrast targets (such as `"g[2] vs g[3]"`) are tested on OLS
  models only: regression with a continuous outcome, and ANOVA. On a logistic,
  Poisson or mixed model a contrast target is now an error ("contrasts ('g[2]
  vs g[3]') are available for OLS models only; ..."). Before, the run went
  ahead and reported 0% power for a contrast it never tested.
* Exact factor levels (the optimistic default; see `set_factor_sampling()`)
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
* Heteroskedasticity standardises its driver with the exact mean and SD of
  the linear predictor, counting interaction terms and the dummies of a
  factor with three or more levels. Before, it left out interactions and
  treated a factor's dummies as uncorrelated, so in such models the residual
  variance did not follow the ratio λ that was set. Moves results: runs with
  heteroskedasticity (the realistic and doomer presets, or any
  `heteroskedasticity_ratio` other than 1) of models with interactions or a
  factor with three or more levels.
* A heteroskedasticity driver set with `set_heteroskedasticity_driver()` is
  matched to its predictor when the model runs. Before, it was matched when
  the setter was called, so declaring factors or uploading data afterwards
  could point it at a different predictor.
* Holm and Benjamini-Hochberg corrections of post-hoc contrasts rank each
  factor's contrasts within that factor, matching the per-factor thresholds.
  Before, the contrasts of all factors were ranked together. Moves results:
  Holm or Benjamini-Hochberg post-hoc contrasts on designs with two or more
  factors.
* A `test_formula` may write an interaction with its terms in another order
  (`b:a` for the model's `a:b`); before, this stopped with an internal
  error. A test interaction the model does not contain is a clear error.
* When a cluster is given by `cluster_size`, the number of clusters a sample
  size implies must be at least `limits.min_clusters` (5); fewer is an
  error that names the count. Before, the implied cluster count was not
  checked.
* Mixed models accept smaller clusters: 2 observations per cluster, plus 1
  for each random slope on the primary grouping; fewer is an error. Before,
  fewer than 5 was refused and fewer than 10 warned, so a pre/post design could
  not be planned. `limits.reliable_rows_per_cluster` is removed.
* The model checks the Python package already ran are now errors in R too:
  `family = "lme"` without a random-effect term, or without a `set_cluster()`
  call for every grouping in the formula; `set_cluster()` on a grouping the
  formula does not have, or with neither `n_clusters` nor `cluster_size` for
  a top-level grouping; an `n_per_parent` that is not an integer of at least
  2; and `set_simulations()` below 1.
* A zero or negative `sample_size`, `n_sims`, `from_size`, `to_size` or `by`
  is an error. Before, it could crash the R session.
* `upload_data()` types and encodes the data the same way on every upload
  and every run. A two-value column is now sent as 0/1 (the lower value as
  0). Before, its raw values were sent, so a column coded other than 0/1,
  such as 1/2, was simulated with the wrong share of each value. A data
  frame is read column by column instead of through `as.matrix()`, so one
  character column no longer turns every column into text, which changed
  how the other columns were detected and rounded their numbers to 7
  significant digits. A setter or upload called after a run now sees the
  formula's predictors; before, it saw the factor dummies the run had
  created (such as `g[2]`), so the next run could differ from a fresh model
  with the same settings. Moves results: first uploads with a two-value
  column not coded 0/1 or a data frame with a character column, and setters
  or uploads called after a run.
* `upload_data()` rejects data with missing or infinite values, which it
  did not check before.
* A predictor taken from uploaded data can no longer be a cluster-level
  variable: it is resampled from the upload row by row, so it was never
  drawn once per cluster. This is now an error.
* Under the optimistic preset, which starts each fit from the true values,
  the reduced fit behind factor-exclusion and `test_formula` tests on a
  linear mixed model with crossed random intercepts, where a later grouping
  has more levels than the first, now starts from the right variance values.
  Optimistic results for such designs can change.
* The scenario residual swap is removed. The scenario keys
  `residual_change_prob`, `residual_dists` and `residual_df` are gone from
  every preset; passing one to `set_scenario_configs()` is an "unknown
  config key" error, and `summary()` no longer has a `residual_pinned`
  entry. The swap replaced the normal residual with a heavy-tailed or skewed
  one of the same variance. The variance of a coefficient estimate depends
  on the residual variance, not its shape, so the swap never made the
  realistic or doomer preset more pessimistic.
* A non-normal residual chosen with `set_residual_distribution()`
  (`"high_kurtosis"`, `"right_skewed"`, `"left_skewed"`) uses 10 df in every
  scenario: Student *t*(10) or a scaled χ²(10), at unit variance. Before,
  its df was 10 under optimistic, 8 under realistic and 5 under doomer.
* Realistic and doomer results change for continuous outcomes (OLS and
  linear mixed models) and for clustered models, because of the residual
  swap removal. The removal changes them in two ways. The residual no longer
  swaps to *t*(8)/χ²(8) under realistic or *t*(5)/χ²(5) under doomer, and the
  ICC jitter of a clustered model is now drawn one or two scenario draws
  earlier, so the same seed gives different draws. The removal itself leaves
  optimistic results for every model, and unclustered binary and count
  results under every preset, unchanged. Moves results: realistic and
  doomer runs of continuous and clustered models.
* A mixed model with random slopes keeps its slopes when the fitted model is
  smaller than the generating one: under a `test_formula` that leaves out
  some predictors, and when a rare factor level is excluded
  (`factor_min_level_count`). Before, a linear mixed model was refitted
  without its random slopes, so the slope variance was ignored and power
  came out too high (in one example 74% instead of 24%). A binary or count
  mixed model (GLMM) put the slope on whichever predictor took its place in
  the smaller model, which gave wrong power, or stopped with an internal
  error. On a mixed model, a `test_formula` that leaves out a predictor with
  a random slope is now an error that names it, since the test model keeps
  the model's random effects. OLS and GLM fit no random effects, so there it
  may leave one out. Moves results: mixed models with random slopes and a
  `test_formula` that leaves out a predictor, and linear mixed models with
  random slopes in which a rare factor level is excluded.
* A random slope must be declared both in the formula (`(1 + x|g)`) and in
  `set_cluster(random_slopes = ...)`; a slope in only one of them is an
  error that names the variable. Before, a slope written only in the
  formula was silently dropped, so the model was simulated and fitted with
  a random intercept only and power came out too high, and a slope given
  only in `set_cluster()` was simulated and fitted although the formula did
  not show it. The groupings of a nested term `(1|A/B)` are the exception:
  the formula cannot write a slope on them, so their slopes come from
  `set_cluster()` alone, as before.
* The uncorrelated random-effect syntax `(1 + x || g)` is refused with a
  message that says so and suggests `(1 + x | g)` with a zero
  slope-intercept correlation. Before, it stopped with a formula syntax
  error.
* A random-effect term in `test_formula`, such as `(1|g)`, is an error that
  says `test_formula` sets the fixed effects of the test model only (the
  test model keeps the model's random effects). Before, it was dropped
  without notice, even one naming a grouping the model does not have.
* A post-hoc request (`target_test = "all-contrasts"`) or a named contrast
  (`"f[2] vs f[3]"`) on a mixed model or a binary or count model is an error
  that says post-hoc comparisons are available for OLS models only and
  suggests testing the factor's dummies instead. Before, a post-hoc request
  stopped with "internal contract validation failed: posthoc tests require
  estimator == Ols", and a named contrast ran with 0% power (see above).
* The colon spelling of a nested intercept, `(1|school:classroom)` (alone or
  beside `(1|school)`), and a random slope inside a nested term, such as
  `(1 + x|school/classroom)`, are refused with messages that point to
  `(1|school/classroom)` and, for the slope, to `set_cluster()`. A grouping
  name the parser does not take, such as `(1|school.id)`, is refused with a
  message naming it, and any other random-effect term the parser does not
  take is refused as an unsupported random-effect term. Before, each stopped
  with "formula syntax error at position 0".
* Settings that break a model rule stop with an error in the model's own
  terms: random-slope correlations (`corr_with_intercept`, `corr_with`) that
  do not form a valid correlation matrix, a `corr_with` of the wrong length,
  a random slope with a zero intercept variance, too many random slopes or
  groupings, a crossed grouping beside a primary grouping sized by
  `cluster_size`, more than one nested term (such as `(1|a/b) + (1|c/d)`,
  now refused when the formula is read, naming both terms), and `alpha` or a
  scenario setting out of range. Before, they stopped with "internal
  contract validation failed: ..." and the names of internal fields. A
  nested `n_per_parent` that does not divide `cluster_size` aborted the R
  session; it is now an error too.
* An exact factor with equal level shares, such as 50/50 or a third each
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
* A mixed model with a single cluster is an error, "n_clusters must be an
  integer >= 2, got 1", which also says that uploaded data needs at least 2
  distinct groups. `set_cluster()` already refused `n_clusters = 1`;
  `get_effects_from_data()` on data with one group reached the engine,
  which stopped with a between-within df message or, with
  `df_method = "asymptotic"`, fitted a model with no between-cluster
  variance to estimate.
* An OLS model fitted to clustered data (`(1|g)` with `set_cluster()`,
  `family = "ols"` or `estimator = "ols"`) runs with `scenarios`. Before, it
  stopped with "internal contract validation failed: scenario.lme requires
  estimator == Mle". Its realistic and doomer scenarios perturb the random
  effects as a mixed model's do.
* `MCPowerDebug$create_data()` and the other debug stages floor `debug_n`
  to the cluster atom with the warning `find_power()` gives ("sample_size
  105 not a multiple of the cluster atom 10; using 100"). Before, a fixed
  `cluster_size` with an N off the atom stopped with "index out of bounds".
* `find_power()` checks `sample_size`: one whole number (a double such as
  `200` is accepted) between 20 and 100,000, and at least 15 plus the number
  of design columns, with the Python port's messages. Before, it ran
  `find_power(10)`, `find_power(150.5)` and `find_power(200000)`, and refused
  `NA` or `c(100, 200)` with messages that did not name `sample_size`.
* `set_cluster()` with two or more random slopes defaults a missing
  `corr_with` to zeros, one per earlier slope, as Python and the app do.
  Before, the second slope needed `corr_with` and `find_power()` stopped
  with "internal contract validation failed: slope 1 corr_with must list
  correlations with all earlier slopes; expected 1, got 0". A `corr_with`
  of the wrong length now stops in `set_cluster()`, naming the slope and
  the length it needs.
* An error message that contains '%' (for example a grouping name such as
  `g%d`) reaches R as written. Before, R read the message as a format
  string: the text came out garbled, and a '%n' aborted the R session.
* `set_cluster()` refuses a NaN or infinite `ICC`, `tau_squared`,
  `n_clusters`, `cluster_size`, `n_per_parent`, or slope `variance`,
  `corr_with_intercept` or `corr_with` with a message that names it.
  Before, it stopped with "missing value where TRUE/FALSE needed" or an
  engine message.
* Under `correction = "tukey"`, a target outside every factor (a continuous
  or binary predictor, an interaction) reports its corrected power and CI
  as `NA`, printed as "n/a", with no point on the corrected chart. Before,
  it read 0. Its corrected required N is "n/a" too. With such a target, the
  corrected views over all tests are not applicable as a whole: "First N
  achieving all targets", the joint required-N table and the at-least-k and
  exactly-k charts read "n/a" or are left out, and
  `success_count_histogram_corrected`, `first_joint_achieved`,
  `first_joint_achieved_unit` and `fitted_joint` are `NULL`. Results carry
  a new `corrected_not_applicable` flag per target; factor levels and
  contrasts keep the values they had, and the Tukey warning now says so.
* A crossed or nested grouping with random slopes follows the primary
  grouping's rules: correlations that do not form a valid correlation
  matrix, and a random slope with a zero intercept variance, are errors.
  Before, both ran without notice: the first drew a slope at a variance
  other than the one set (4.26 times the set value in one example), and the
  second ignored the correlation.
* A variable given two random slopes on one grouping is an error. Before, it
  ran with twice the slope variance, so power came out too low (in one
  example 69% instead of 93%).
* `set_cluster()` refuses an argument it does not know, as Python does.
  Before, a misspelled `icc = 0.5` was dropped without notice and the model
  ran without clustering (in one example 99.5% power instead of 38%).
* `set_cluster()` refuses `n_clusters` together with `cluster_size`, as
  Python does. Before, the run used `n_clusters` while the check on rows per
  cluster read `cluster_size`.
* `set_cluster()` refuses an `n_clusters`, `cluster_size` or `n_per_parent`
  past R's integer range with a message that names it. Before, a run stopped
  with "missing value where TRUE/FALSE needed".
* `find_sample_size()` checks `from_size`, `to_size` and `by` as Python
  does, with Python's messages. Before, it truncated a fractional value
  (`by = 2.5` ran as 2) and ran a `by` larger than the range.
* `MCPowerDebug`: a `debug_n` or `debug_n_sims` below 1 is an error that
  names it. Before, a negative `debug_n_sims` aborted the R session.
* `estimator = "glm"` on a continuous outcome and a stray colon in a formula
  (`y ~ x1 + x2:`) stop with a user error that says what to change. Before,
  each stopped with "internal contract validation failed: ...".

# mcpower 1.0.0

* First R release of MCPower on the native Rust simulation engine.
* Monte Carlo power and sample-size analysis for OLS, GLM, mixed-effects, and
  ANOVA designs through a single `MCPower` interface (formula + data-frame API).
* Distributed via R-universe; install with
  `install.packages("mcpower", repos = "https://pawlenartowicz.r-universe.dev")`.
