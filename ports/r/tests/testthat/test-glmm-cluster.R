library(mcpower)

# ---------------------------------------------------------------------------
# 2.1  Latent-scale ICC -> tau^2 (binary path multiplies by pi^2/3; Gaussian unchanged)
# ---------------------------------------------------------------------------

# (1) logit + cluster: tau_squared carries the latent-scale factor
test_that("logit+cluster: tau_squared == icc/(1-icc) * pi^2/3", {
  icc <- 0.2
  m <- MCPower$new("y ~ x1 + (1|g)", family = "logit")$
    set_effects("x1=0.3")$
    set_baseline_probability(0.3)$
    set_cluster("g", ICC = icc, n_clusters = 20L, cluster_size = 10L)
  enc <- mcpower:::.encode_outcome_and_clusters(
    family            = "logit",
    link              = "canonical",
    estimator         = "glm",
    intercept         = -0.847,
    pending_clusters  = m$.__enclos_env__$private$pending_clusters
  )
  expected_tau <- icc / (1 - icc) * (pi^2 / 3)
  parsed <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_equal(parsed[[1]]$tau_squared, expected_tau, tolerance = 1e-10)
})

# (2) Gaussian (lme) + cluster: tau_squared is the plain Gaussian formula, unchanged
test_that("lme+cluster: tau_squared == icc/(1-icc) (no pi^2/3 factor)", {
  icc <- 0.2
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("g", ICC = icc, n_clusters = 20L, cluster_size = 10L)
  enc <- mcpower:::.encode_outcome_and_clusters(
    family            = "lme",
    link              = "canonical",
    estimator         = "mle",
    intercept         = 0.0,
    pending_clusters  = m$.__enclos_env__$private$pending_clusters
  )
  expected_tau <- icc / (1 - icc)
  parsed <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_equal(parsed[[1]]$tau_squared, expected_tau, tolerance = 1e-10)
})

# (2b) probit + cluster: tau_squared is the plain (unscaled) formula, no pi^2/3
test_that("probit+cluster: tau_squared == icc/(1-icc) (no pi^2/3 factor)", {
  icc <- 0.2
  m <- MCPower$new("y ~ x1 + (1|g)", family = "probit")$
    set_effects("x1=0.3")$
    set_baseline_probability(0.3)$
    set_cluster("g", ICC = icc, n_clusters = 20L, cluster_size = 10L)
  enc <- mcpower:::.encode_outcome_and_clusters(
    family            = "probit",
    link              = "probit",
    estimator         = "glm",
    intercept         = -0.847,
    pending_clusters  = m$.__enclos_env__$private$pending_clusters
  )
  expected_tau <- icc / (1 - icc)
  parsed <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_equal(parsed[[1]]$tau_squared, expected_tau, tolerance = 1e-10)
  # Confirm the logit pi^2/3 factor was NOT applied.
  expect_true(abs(parsed[[1]]$tau_squared - icc / (1 - icc) * (pi^2 / 3)) > 1e-6)
})

# (3) logit + two groupings: extra grouping also gets the latent-scale factor
test_that("logit + two set_cluster calls: extra grouping tau_squared also scaled by pi^2/3", {
  icc_primary <- 0.2
  icc_extra   <- 0.1
  pc <- list(
    g1 = list(icc = icc_primary, n_clusters = 20L, cluster_size = 10L),
    g2 = list(icc = icc_extra,   n_clusters = 15L, cluster_size = NULL,
              n_per_parent = NULL)
  )
  pc$g2$n_clusters <- 15L
  enc <- mcpower:::.encode_outcome_and_clusters(
    family           = "logit",
    link             = "canonical",
    estimator        = "glm",
    intercept        = -0.847,
    pending_clusters = pc
  )
  parsed <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expected_primary <- icc_primary / (1 - icc_primary) * (pi^2 / 3)
  expected_extra   <- icc_extra   / (1 - icc_extra)   * (pi^2 / 3)
  expect_equal(parsed[[1]]$tau_squared, expected_primary, tolerance = 1e-10)
  expect_equal(parsed[[1]]$extra_groupings[[1]]$tau_squared, expected_extra, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# 2.2  Laplace-bias warning (host-side, fires from find_power result)
# ---------------------------------------------------------------------------

# Helper: build a mock raw result list with a specific tau_squared_hat_mean
# and estimator tag, structured as .unwrap_scenario_result returns for a
# single scenario.
.mock_glmm_raw <- function(tau_hat, estimator = "glm") {
  list(
    estimator_extras = list(
      estimator            = estimator,
      tau_squared_hat_mean = tau_hat
    ),
    power_uncorrected   = list(0.8),
    power_corrected     = list(0.8),
    convergence_rate    = 1.0,
    n                   = 50L,
    n_sims              = 100L,
    target_indices      = list(0L),
    ci_uncorrected      = list(list(lo = 0.75, hi = 0.85)),
    ci_corrected        = list(list(lo = 0.75, hi = 0.85))
  )
}

# (4) warning fires when tau_hat > threshold AND cluster_size < recommended rows
test_that("Laplace-bias warning fires when tau_hat exceeds threshold with small clusters", {
  thr      <- mcpower:::.config()$report$thresholds$glmm_tau_sq_warn
  min_size <- mcpower:::.config()$limits$recommended_rows_per_cluster
  raw      <- .mock_glmm_raw(tau_hat = thr + 0.5)
  cluster_size <- min_size - 1L
  expect_warning(
    mcpower:::.check_glmm_laplace_bias_warning(raw, cluster_size),
    regexp = "Laplace-approximation bias likely"
  )
})

# (5) no warning when tau_hat is at or below the threshold
test_that("no Laplace-bias warning when tau_hat <= threshold", {
  thr      <- mcpower:::.config()$report$thresholds$glmm_tau_sq_warn
  min_size <- mcpower:::.config()$limits$recommended_rows_per_cluster
  raw      <- .mock_glmm_raw(tau_hat = thr)
  expect_no_warning(
    mcpower:::.check_glmm_laplace_bias_warning(raw, min_size - 1L)
  )
})

# (6) no warning when cluster_size >= recommended_rows_per_cluster, even with high tau_hat
test_that("no Laplace-bias warning when cluster_size >= recommended_rows_per_cluster", {
  thr      <- mcpower:::.config()$report$thresholds$glmm_tau_sq_warn
  min_size <- mcpower:::.config()$limits$recommended_rows_per_cluster
  raw      <- .mock_glmm_raw(tau_hat = thr + 1.0)
  expect_no_warning(
    mcpower:::.check_glmm_laplace_bias_warning(raw, min_size)
  )
})

# (7) warning message contains the cluster size
test_that("Laplace-bias warning message contains the cluster size", {
  thr      <- mcpower:::.config()$report$thresholds$glmm_tau_sq_warn
  min_size <- mcpower:::.config()$limits$recommended_rows_per_cluster
  cs       <- min_size - 1L
  raw      <- .mock_glmm_raw(tau_hat = thr + 0.5)
  expect_warning(
    mcpower:::.check_glmm_laplace_bias_warning(raw, cs),
    regexp = as.character(cs)
  )
})

# (8) no warning for a non-GLMM result (estimator != "glm")
test_that("no Laplace-bias warning for OLS result", {
  thr      <- mcpower:::.config()$report$thresholds$glmm_tau_sq_warn
  min_size <- mcpower:::.config()$limits$recommended_rows_per_cluster
  raw      <- .mock_glmm_raw(tau_hat = thr + 1.0, estimator = "ols")
  expect_no_warning(
    mcpower:::.check_glmm_laplace_bias_warning(raw, min_size - 1L)
  )
})

# ---------------------------------------------------------------------------
# 2.3  GLMM smoke + scenario-knob audit
# ---------------------------------------------------------------------------

# (9) 05a prereq guard: set_scenario_configs accepted and stored (absorbs former smoke-only check)
test_that("05a prereq: set_scenario_configs accepts icc_noise_sd on logit+cluster", {
  m <- expect_no_error(
    MCPower$new("y ~ x1 + (1|g)", family = "logit")$
      set_effects("x1=0.4")$
      set_baseline_probability(0.3)$
      set_cluster("g", ICC = 0.2, n_clusters = 20L, cluster_size = 10L)$
      set_scenario_configs(list(optimistic = list(icc_noise_sd = 0.05)))
  )
  expect_equal(m$.__enclos_env__$private$scenario_configs$optimistic$icc_noise_sd, 0.05)
})

# (10) logit + set_cluster reaches the GLMM kernel
test_that("find_power on logit+cluster returns estimator_extras with estimator='glm'", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "logit")$
    set_effects("x1=0.4")$
    set_baseline_probability(0.3)$
    set_cluster("g", ICC = 0.2, n_clusters = 20L, cluster_size = 10L)
  result <- m$find_power(sample_size = 100L, n_sims = 50L, seed = 2137,
                         progress_callback = FALSE, verbose = FALSE)
  expect_equal(result$estimator_extras$estimator, "glm")
})

# (11) icc_noise_sd override does not error on the GLMM path (Phase 0 Task 0.1)
test_that("set_scenario_configs(icc_noise_sd) does not error on logit+cluster find_power", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "logit")$
    set_effects("x1=0.4")$
    set_baseline_probability(0.3)$
    set_cluster("g", ICC = 0.2, n_clusters = 20L, cluster_size = 10L)$
    set_scenario_configs(list(optimistic = list(icc_noise_sd = 0.05)))
  result <- m$find_power(sample_size = 100L, n_sims = 50L, seed = 2137,
                         progress_callback = FALSE, verbose = FALSE)
  expect_true("power_uncorrected" %in% names(result))
  expect_equal(result$estimator_extras$estimator, "glm")
})
# Note: a former test (12) ran the identical spec to (10) above asserting only
# expect_no_error — dropped as a same-layer duplicate; (10)'s estimator=="glm"
# already proves the spec reaches the kernel with no port gate rejecting it.
