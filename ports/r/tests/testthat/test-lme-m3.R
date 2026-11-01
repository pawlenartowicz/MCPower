library(mcpower)

# M3 random slopes — public set_cluster path.
# Mirrors Python tests/spec/test_set_cluster.py (tasks 1.8, 1.9).
# R interface: random_slopes = list(list(predictor="x1", variance=0.05,
# corr_with_intercept=0.3)), one list entry per slope, per-slope variance and
# corr_with_intercept (richer than Python's shared slope_variance/slope_intercept_corr).

# ---------------------------------------------------------------------------
# Gate removal — acceptance without error
# ---------------------------------------------------------------------------

# (1) random_slopes accepted without error and stored (absorbs former smoke-only check)
test_that("set_cluster with random_slopes does not error", {
  m <- expect_no_error(
    MCPower$new("y ~ x1 + (1 + x1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L,
                  random_slopes = list(
                    list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.3)
                  ))
  )
  expect_equal(m$.__enclos_env__$private$pending_clusters$school$raw_slopes[[1]]$predictor, "x1")
})

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# (2) raw_slopes stored in pending_clusters
test_that("set_cluster stores random_slopes in pending_clusters as raw_slopes", {
  m <- MCPower$new("y ~ x1 + (1 + x1|school)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L,
                random_slopes = list(
                  list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.3)
                ))
  slp <- m$.__enclos_env__$private$pending_clusters$school$raw_slopes
  expect_equal(length(slp), 1L)
  expect_equal(slp[[1]]$predictor, "x1")
  expect_equal(slp[[1]]$variance, 0.05)
  expect_equal(slp[[1]]$corr_with_intercept, 0.3)
})

# (3) No slopes configured — raw_slopes is NULL
test_that("set_cluster without random_slopes stores NULL raw_slopes", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L)
  expect_null(m$.__enclos_env__$private$pending_clusters$school$raw_slopes)
})

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# (4) Unknown predictor in random_slopes is rejected
test_that("set_cluster rejects unknown predictor in random_slopes", {
  expect_error(
    MCPower$new("y ~ x1 + (1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L,
                  random_slopes = list(
                    list(predictor = "z_unknown", variance = 0.05, corr_with_intercept = 0.0)
                  )),
    regexp = "not a modeled predictor"
  )
})

# (5) Non-positive variance is rejected
test_that("set_cluster rejects non-positive variance in random_slopes", {
  expect_error(
    MCPower$new("y ~ x1 + (1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L,
                  random_slopes = list(
                    list(predictor = "x1", variance = -0.01, corr_with_intercept = 0.0)
                  )),
    regexp = "variance must be a finite positive number"
  )
})

# ---------------------------------------------------------------------------
# clusters_json shape — via encoder called with pre-resolved slopes
# ---------------------------------------------------------------------------

# (6) slopes block in clusters_json has correct shape
# We manually set $slopes on the pending entry (bypassing build_contract_bytes)
# to test the encoder directly, mirroring the debug-seam pattern.
test_that("clusters_json slopes block has column/variance/corr_with_intercept/corr_with", {
  m <- MCPower$new("y ~ x1 + (1 + x1|school)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L,
                random_slopes = list(
                  list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.3)
                ))
  m$.__enclos_env__$private$apply()
  reg     <- m$.__enclos_env__$private$registry
  pending <- m$.__enclos_env__$private$pending_clusters
  col_idx <- reg$get_effect("x1")$column_index  # 0-based (main effect: column_index singular)
  # Inject resolved slopes into pending (mirrors build_contract_bytes resolution).
  pending$school$slopes <- list(
    list(column = col_idx, variance = 0.05,
         corr_with_intercept = 0.3, corr_with = I(numeric(0)))
  )
  enc <- mcpower:::.encode_outcome_and_clusters(
    m$family, "canonical", m$estimator, m$intercept, pending)
  cj  <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  slp <- cj[[1]]$slopes[[1]]
  expect_equal(slp$column, col_idx)
  expect_equal(slp$variance, 0.05)
  expect_equal(slp$corr_with_intercept, 0.3)
  # corr_with is empty for the first slope. An empty JSON array round-trips
  # through fromJSON(simplifyVector = FALSE) as an empty list(), not numeric(0).
  expect_equal(length(slp$corr_with), 0L)
})

# (7) corr_with serializes as empty JSON array "[]", not omitted
test_that("corr_with serializes as empty JSON array, not omitted", {
  pending <- list(
    school = list(icc = 0.1, n_clusters = 20L, cluster_size = 15L,
                  slopes = list(
                    list(column = 0L, variance = 0.05,
                         corr_with_intercept = 0.3, corr_with = I(numeric(0)))
                  ))
  )
  enc <- mcpower:::.encode_outcome_and_clusters("gaussian", "canonical", "lme", 0.0, pending)
  raw <- as.character(enc$clusters_json)
  expect_true(grepl('"corr_with":\\s*\\[\\]', raw))
})

# (8) No slopes configured — slopes key absent from clusters_json
test_that("no random_slopes → slopes key absent from clusters_json", {
  pending <- list(
    school = list(icc = 0.1, n_clusters = 20L, cluster_size = 15L)
  )
  enc <- mcpower:::.encode_outcome_and_clusters("gaussian", "canonical", "lme", 0.0, pending)
  cj  <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_null(cj[[1]]$slopes)
})

# (9) Two slopes: second slope's corr_with carries correlation with first slope
test_that("two slopes produce correctly chained corr_with entries", {
  pending <- list(
    school = list(
      icc = 0.1, n_clusters = 20L, cluster_size = 15L,
      slopes = list(
        list(column = 0L, variance = 0.05, corr_with_intercept = 0.3,
             corr_with = I(numeric(0))),
        list(column = 1L, variance = 0.03, corr_with_intercept = 0.2,
             corr_with = I(c(0.1)))
      )
    )
  )
  enc <- mcpower:::.encode_outcome_and_clusters("gaussian", "canonical", "lme", 0.0, pending)
  cj  <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  slp2 <- cj[[1]]$slopes[[2]]
  expect_equal(slp2$corr_with[[1]], 0.1, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# Formula and set_cluster must agree on the random slopes
# ---------------------------------------------------------------------------

# A slope written only in the formula is refused, naming the variable and the
# set_cluster fix, rather than being fitted as a random intercept.
test_that("a formula-only random slope is refused before the run", {
  m <- MCPower$new("y ~ x1 + x2 + (1 + x2|g)", family = "lme")$
    set_effects("x1=0.3, x2=0.3")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L)
  expect_error(
    m$find_power(sample_size = 200L, n_sims = 20L, progress_callback = FALSE,
                 verbose = FALSE),
    "random slope on 'x2'.*add list\\(predictor = \"x2\", variance = <variance>\\) to random_slopes"
  )
})

# The reverse: set_cluster declares a slope the formula's (1|g) does not.
test_that("a set_cluster-only random slope is refused before the run", {
  m <- MCPower$new("y ~ x1 + x2 + (1|g)", family = "lme")$
    set_effects("x1=0.3, x2=0.3")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
      list(predictor = "x2", variance = 0.1)))
  expect_error(
    m$find_power(sample_size = 200L, n_sims = 20L, progress_callback = FALSE,
                 verbose = FALSE),
    "write '(1 + x2|g)' in the formula", fixed = TRUE
  )
})

# The formula cannot write a slope on a nested child, so set_cluster is the
# only route: a nested-child slope with a plain (1|school/classroom) runs, and
# its variance reaches the contract.
test_that("a nested-child slope from set_cluster alone runs", {
  build <- function(variance) {
    MCPower$new("y ~ x + (1|school/classroom)", family = "lme")$
      set_effects("x=0.4")$
      set_cluster("school", ICC = 0.15, n_clusters = 10L)$
      set_cluster("school:classroom", ICC = 0.10, n_per_parent = 3L,
                  random_slopes = list(list(predictor = "x", variance = variance)))
  }
  contract <- function(m) {
    m$.__enclos_env__$private$apply()
    m$.__enclos_env__$private$build_contract_bytes("optimistic")
  }
  expect_false(identical(contract(build(0.1)), contract(build(0.2))))
  res <- build(0.1)$find_power(sample_size = 300L, n_sims = 20L, seed = 2137,
                               progress_callback = FALSE, verbose = FALSE)
  p <- res$power_uncorrected[[1]][[1]]
  expect_true(is.finite(p) && p >= 0 && p <= 1)
})

# The test model keeps the random effects, so test_formula must keep x2.
test_that("a test_formula that drops the slope variable is refused with its name", {
  m <- MCPower$new("y ~ x1 + x2 + (1 + x2|g)", family = "lme")$
    set_effects("x1=0.3, x2=0.3")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
      list(predictor = "x2", variance = 0.1)))
  expect_error(
    m$find_power(sample_size = 200L, n_sims = 20L, test_formula = "y ~ x1",
                 progress_callback = FALSE, verbose = FALSE),
    "keep 'x2' in test_formula", fixed = TRUE
  )
})

# test_formula sets fixed effects only; R used to drop a random-effect term in
# it silently, even one naming a grouping the model does not have.
test_that("a random-effect term in test_formula is refused", {
  m <- MCPower$new("y ~ x1 + x2 + (1|g)", family = "lme")$
    set_effects("x1=0.3, x2=0.3")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L)
  for (term in c("(1|g)", "(1|h)")) {
    expect_error(
      m$find_power(sample_size = 200L, n_sims = 20L,
                   test_formula = paste("y ~ x1 +", term),
                   progress_callback = FALSE, verbose = FALSE),
      paste0("test_formula sets the fixed effects of the test model only; the ",
             "test model keeps the model's random effects. Remove '", term,
             "' from test_formula"),
      fixed = TRUE
    )
  }
})

# One intercept correlation over two uncorrelated slopes: a user error with the
# contract's own message, not an internal one.
test_that("non-PSD random-slope correlations stop without the internal prefix", {
  m <- MCPower$new("y ~ x1 + x2 + (1 + x1 + x2|g)", family = "lme")$
    set_effects("x1=0.3, x2=0.2")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
      list(predictor = "x1", variance = 0.1, corr_with_intercept = 0.9),
      list(predictor = "x2", variance = 0.1, corr_with_intercept = 0.9,
           corr_with = 0)))
  err <- tryCatch(
    m$find_power(sample_size = 400L, n_sims = 20L, progress_callback = FALSE,
                 verbose = FALSE),
    error = function(e) conditionMessage(e))
  expect_true(startsWith(
    err, "the random-slope correlations are not a valid correlation matrix"), info = err)
  expect_false(grepl("internal", err, fixed = TRUE), info = err)
})

# lme4's uncorrelated '||' syntax gets its own message, not a syntax error.
test_that("the double-bar random-effect syntax is refused", {
  expect_error(
    MCPower$new("y ~ x1 + (1 + x1 || g)", family = "lme"),
    "uncorrelated random effects ('||')", fixed = TRUE
  )
})

# ---------------------------------------------------------------------------
# End-to-end (engine smoke)
# ---------------------------------------------------------------------------

# (10) find_power with random slopes completes without error
test_that("find_power with random slopes completes without error", {
  m <- MCPower$new("y ~ x1 + (1 + x1|school)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L,
                random_slopes = list(
                  list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.3)
                ))
  res <- m$find_power(sample_size = 100L, n_sims = 50L, seed = 2137,
                      progress_callback = FALSE, verbose = FALSE)
  # x1 effect 0.3 with a random slope at N=100 is detectable; pin a power floor so a
  # no-op/zero-power slope path fails where expect_no_error passed. Robust floor
  # (not capture-pinned — the R package is not installable in this WIP tree).
  expect_true(res$power_uncorrected[[1]][[1]] > 0.05)
})

test_that("a random-effect term without set_cluster is refused in every family", {
  # A non-lme family used to run unclustered, dropping the random intercept and
  # slope without notice. Mirrors test_set_cluster.py.
  term_msg <- "the random-effect term on 'g' has no set_cluster call"
  cases <- list(
    list(family = "logit", formula = "y ~ x + (1 + x|g)", msg = term_msg),
    list(family = "logit", formula = "y ~ x + (1|g)", msg = term_msg),
    list(family = "poisson", formula = "y ~ x + (1|g)", msg = term_msg),
    list(family = "lme", formula = "y ~ x + (1|g)",
         msg = "family='lme' requires set_cluster(...) for each random effect")
  )
  for (cs in cases) {
    m <- MCPower$new(cs$formula, family = cs$family)$set_effects("x=0.5")
    if (cs$family == "logit") m$set_baseline_probability(0.3)
    if (cs$family == "poisson") m$set_baseline_rate(1.0)
    expect_error(
      m$find_power(sample_size = 200L, n_sims = 20L, progress_callback = FALSE,
                   verbose = FALSE),
      cs$msg, fixed = TRUE, info = paste(cs$family, cs$formula))
  }
})

# corr_with defaults to zeros, one per earlier slope (as Python and the app).
test_that("a second random slope without corr_with gets corr_with 0", {
  contract <- function(cw2) {
    s2 <- list(predictor = "x2", variance = 0.1)
    s2[["corr_with"]] <- cw2
    m <- MCPower$new("y ~ x1 + x2 + (1 + x1 + x2|g)", family = "lme")$
      set_effects("x1=0.3, x2=0.3")$
      set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
        list(predictor = "x1", variance = 0.1), s2))
    m$.__enclos_env__$private$apply()
    m$.__enclos_env__$private$build_contract_bytes("optimistic")
  }
  expect_identical(contract(NULL), contract(0))
  # The bytes carry corr_with: a nonzero value changes them.
  expect_false(identical(contract(0), contract(0.3)))
})

test_that("set_cluster refuses a corr_with of the wrong length", {
  m <- MCPower$new("y ~ x1 + x2 + (1 + x1 + x2|g)", family = "lme")$set_effects("x1=0.3, x2=0.3")
  expect_error(
    m$set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
      list(predictor = "x1", variance = 0.1),
      list(predictor = "x2", variance = 0.1, corr_with = c(0.1, 0.2)))),
    "random_slopes[[2]]$corr_with gives 2 value(s); it needs 1, one correlation with each earlier slope in random_slopes",
    fixed = TRUE)
})
