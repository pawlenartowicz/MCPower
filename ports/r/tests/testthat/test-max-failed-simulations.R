# test-max-failed-simulations.R — G-B: unit + end-to-end tests for
# .check_failure_threshold and its wiring into find_power / find_sample_size.
# Mirrors Python test_max_failed_simulations.py structure.

# ── Unit tests for .check_failure_threshold helper ───────────────────────────

test_that(".check_failure_threshold raises above threshold", {
  expect_error(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.8),
      boundary_hit_rate_tau_zero = c(0.15),
      boundary_hit_rate_high_tau = c(0.05),
      threshold                  = 0.1),
    regexp = "failure rate")
})

test_that(".check_failure_threshold passes below threshold", {
  expect_no_error(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.95),
      boundary_hit_rate_tau_zero = c(0.05),
      boundary_hit_rate_high_tau = c(0.0),
      threshold                  = 0.1))
})

test_that(".check_failure_threshold passes exactly at threshold (strict >)", {
  # failure_rate = 1 - 0.9 = 0.1 == threshold; strict > means no error
  expect_no_error(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.9),
      boundary_hit_rate_tau_zero = c(0.1),
      boundary_hit_rate_high_tau = c(0.0),
      threshold                  = 0.1))
})

test_that(".check_failure_threshold: threshold=1.0 never raises", {
  expect_no_error(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.0),
      boundary_hit_rate_tau_zero = c(0.5),
      boundary_hit_rate_high_tau = c(0.5),
      threshold                  = 1.0))
})

test_that(".check_failure_threshold: multi-N worst triggers error", {
  # N points 1 & 2 fine, N point 3 has failure_rate=0.3 > threshold=0.2
  expect_error(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.99, 0.99, 0.7),
      boundary_hit_rate_tau_zero = c(0.0,  0.0,  0.25),
      boundary_hit_rate_high_tau = c(0.0,  0.0,  0.05),
      threshold                  = 0.2))
})

test_that(".check_failure_threshold: error message includes rate and threshold", {
  err <- tryCatch(
    mcpower:::.check_failure_threshold(
      convergence_rate           = c(0.75),
      boundary_hit_rate_tau_zero = c(0.20),
      boundary_hit_rate_high_tau = c(0.05),
      threshold                  = 0.10),
    error = function(e) conditionMessage(e))
  expect_true(grepl("failure rate|Failure rate", err, ignore.case = TRUE))
  # worst rate = 1-0.75 = 0.25 = 25%; threshold = 10%
  expect_true(grepl("25", err))
  expect_true(grepl("10", err))
})

# ── End-to-end: OLS never raises at max_failed_simulations=0.0 ───────────────

test_that("find_power OLS never raises at max_failed_simulations=0.0", {
  # OLS convergence_rate is always 1.0; threshold 0.0 must never trigger
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$set_simulations(50)
  m$set_max_failed_simulations(0.0)
  expect_no_error(m$find_power(100, seed = 2137, progress_callback = FALSE))
})
