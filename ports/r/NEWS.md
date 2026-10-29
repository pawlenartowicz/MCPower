# mcpower 1.5.0

* `find_power()` and `find_sample_size()` warn when a clustered binary or
  count model (GLMM) tests a cluster-level effect with fewer than 20 clusters
  under the default `df_method = "between_within"`: the between-within *t*
  test then rejects a true null more often than α (7.5–9.6% at 5 clusters in
  the logit calibration), so part of the reported power is excess Type I
  error. A sample-size search names the grid N below which the design has
  fewer than 20 clusters, or says that every grid point has fewer.
* New `set_factor_sampling(name, mode)` sets how one factor's group sizes are
  realised, overriding the scenario-wide `sampled_factor_proportions` for
  that factor only. `"exact"` holds the group sizes at the requested shares
  in every simulation, `"sampled"` draws each observation's level at random
  so the sizes vary between simulations, and `NULL` removes the override. It
  works for declared factors and for factors from uploaded data.
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
  models only: regression with a continuous outcome, and ANOVA. On a
  logistic, Poisson or mixed model a contrast target is now an error
  ("contrasts are only supported for estimator=Ols"). Before, the run went
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
  error that names the count. `find_power()` and `find_sample_size()` also
  check observations per cluster at the requested sample size: fewer than 5
  (`limits.reliable_rows_per_cluster`) is an error, fewer than 10
  (`limits.recommended_rows_per_cluster`) a warning. Before, the implied
  cluster count was not checked, and observations per cluster were checked
  only for a given `cluster_size`, when the cluster was set.
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

# mcpower 1.0.0

* First R release of MCPower on the native Rust simulation engine.
* Monte Carlo power and sample-size analysis for OLS, GLM, mixed-effects, and
  ANOVA designs through a single `MCPower` interface (formula + data-frame API).
* Distributed via R-universe; install with
  `install.packages("mcpower", repos = "https://pawlenartowicz.r-universe.dev")`.
