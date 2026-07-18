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
      set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_size = 15L,
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
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_size = 15L,
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
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_size = 15L)
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
    regexp = "variance must be a positive number"
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
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_size = 15L,
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
# End-to-end (engine smoke)
# ---------------------------------------------------------------------------

# (10) find_power with random slopes completes without error
test_that("find_power with random slopes completes without error", {
  m <- MCPower$new("y ~ x1 + (1 + x1|school)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_size = 15L,
                random_slopes = list(
                  list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.3)
                ))
  res <- m$find_power(sample_size = 60L, n_sims = 50L, seed = 2137,
                      progress_callback = FALSE, verbose = FALSE)
  # x1 effect 0.3 with a random slope at N=60 is detectable; pin a power floor so a
  # no-op/zero-power slope path fails where expect_no_error passed. Robust floor
  # (not capture-pinned — the R package is not installable in this WIP tree).
  expect_true(res$power_uncorrected[[1]][[1]] > 0.05)
})
