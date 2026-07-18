test_that("find_power runs end-to-end for OLS and returns a power >= 0.5", {
  # G-A: strengthen from [0,1] to >= 0.5 — x1=0.5 at N=200 must be well above chance.
  # The low-level raw call uses no scenario noisiness, so power can be close to 1.0
  # for easy configs; a lower-bound floor is the meaningful assertion here.
  out <- mcpower:::build_contract_from_spec(
    .ols_spec_json("y ~ x1", x1 = 0.5), "continuous", "canonical", "ols", 0.0, "[]")
  res <- mcpower:::find_power(out$contracts, sample_size = 200L, n_sims = 200L,
                               base_seed = 2137, progress = NULL)
  expect_true("scenarios" %in% names(res))
  opt <- res$scenarios[["optimistic"]]
  expect_equal(length(opt$power_uncorrected[[1]]), opt$n_targets)
  pwr <- opt$power_uncorrected[[1]][[1]]
  expect_true(pwr >= 0.5,
    info = sprintf("OLS power = %.3f; expected >= 0.5", pwr))
})

test_that("seed f64 round-trips: same seed -> identical power", {
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.3),
                                            "continuous", "canonical", "ols", 0.0, "[]")$contracts
  a <- mcpower:::find_power(cj, 150L, 300L, 2137, NULL)$scenarios[["optimistic"]]
  b <- mcpower:::find_power(cj, 150L, 300L, 2137, NULL)$scenarios[["optimistic"]]
  expect_identical(a$power_uncorrected, b$power_uncorrected)  # determinism pinned
})

test_that("progress callback receives (current, total) reports reaching total", {
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.3),
                                            "continuous", "canonical", "ols", 0.0, "[]")$contracts
  seen <- list()
  cb <- function(current, total) {
    seen[[length(seen) + 1L]] <<- c(current = current, total = total)
    TRUE
  }
  res <- mcpower:::find_power(cj, 200L, 400L, 2137, cb)
  expect_true("scenarios" %in% names(res))
  # The engine delivered events on the R main thread (background-run + poll).
  expect_true(length(seen) >= 1L, info = "progress callback never fired")
  currents <- vapply(seen, function(e) e[["current"]], numeric(1))
  totals   <- vapply(seen, function(e) e[["total"]],   numeric(1))
  # The reported denominator is the run's sim count, and the run completes the
  # full count. Per-event order is NOT asserted: under multiple worker threads
  # the engine's progress counter is reported out of order (the same
  # worker-count non-determinism that makes RNG streams not bit-equal), so only
  # the reached maximum is a stable invariant.
  expect_true(all(totals == 400))
  expect_equal(max(currents), 400)
})

test_that("progress callback spans a 2-scenario run (per-scenario events fire)", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$set_simulations(200)
  seen <- list()
  cb <- function(current, total) {
    seen[[length(seen) + 1L]] <<- c(current = current, total = total)
    TRUE
  }
  # scenarios = TRUE runs every configured scenario; the (current,total)
  # translation rolls completed_so_far forward at each ScenarioCompleted, so the
  # reached maximum spans every scenario's sims (n_scenarios * n_sims).
  res <- m$find_power(sample_size = 150, n_sims = 200, seed = 2137,
                      scenarios = TRUE, progress_callback = cb, verbose = FALSE)
  expect_true("scenarios" %in% names(res))
  n_scen <- length(res$scenarios)
  expect_true(n_scen >= 2L, info = "expected a multi-scenario envelope")
  expect_true(length(seen) >= 1L, info = "progress callback never fired for multi-scenario run")
  currents <- vapply(seen, function(e) e[["current"]], numeric(1))
  # Cross-scenario roll-forward: the max reported current is the grand total
  # across scenarios, proving the per-scenario events were folded in.
  expect_equal(max(currents), n_scen * 200)
})

test_that("a pre-cancelled run surfaces the 'cancelled by user' error (cancel seam)", {
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.3),
                                            "continuous", "canonical", "ols", 0.0, "[]")$contracts
  # find_power_precancelled flips the CancellationToken before the engine runs;
  # the orchestrator returns OrchestratorError::Cancelled at its first
  # checkpoint and the bridge maps it to this exact message (mirrors the live
  # Ctrl-C path, which flips the same token via R_CheckUserInterrupt).
  expect_error(
    mcpower:::find_power_precancelled(cj, 200L, 400L, 2137),
    "cancelled by user"
  )
})

test_that("find_sample_size grid converges within search bounds for large effect", {
  # G-A: add first_achieved convergence assertion — was key-presence only
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.5),
                                            "continuous", "canonical", "ols", 0.0, "[]")$contracts
  res <- mcpower:::find_sample_size(cj, target_power = 0.8, lo = 30L, hi = 120L,
            n_sims = 200L, base_seed = 2137, method = "grid", by = 30L,
            by_kind = "fixed", mode = "linear", tol_n = 1L, progress = NULL)
  opt <- res$scenarios[["optimistic"]]
  expect_true(opt$n_sample_sizes >= 2)
  expect_equal(length(opt$sample_sizes), opt$n_sample_sizes)
  # Must converge: first_achieved must be non-NULL and within [30, 120]
  fa <- opt$first_achieved[[1]]
  expect_false(is.null(fa), info = "grid failed to reach 80% power for x1=0.5 up to N=120")
  expect_true(fa >= 30L && fa <= 120L,
    info = sprintf("first_achieved = %s; expected in [30, 120]", fa))
})

test_that("MCPower OLS find_power runs end-to-end and setters chain", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$set_alpha(0.05)$set_simulations(300)
  res <- m$find_power(sample_size = 200, seed = 2137, progress_callback = FALSE)
  # APIC-69: a single-scenario call returns the *unwrapped* result (no "scenarios"
  # envelope) carrying the power-result keys. The former OR-form accepted either an
  # envelope or a bare power key, so a wrongly-wrapped result passed — assert the
  # unwrapped shape directly.
  expect_false("scenarios" %in% names(res))
  expect_true("power_uncorrected" %in% names(res))
  expect_true("n_targets" %in% names(res))
})

test_that("MCPower find_sample_size grid runs end-to-end", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")$set_simulations(200)
  res <- m$find_sample_size(from_size = 30, to_size = 120, by = 30,
                            seed = 2137, progress_callback = FALSE)
  # APIC-54: single-scenario grid find_sample_size returns the unwrapped sample-size
  # result (no "scenarios" envelope) with a sample_sizes curve. Assert the unwrapped
  # shape directly rather than the looser "either key present" OR.
  expect_false("scenarios" %in% names(res))
  expect_true("sample_sizes" %in% names(res))
})

# ---------------------------------------------------------------------------
# L2 binding: crossing fields (fitted / fitted_joint / cluster_atom)
# Verify the engine emits the new model-based crossing result fields at the
# R boundary (the host walker strips nothing — these must survive the round-trip).
# ---------------------------------------------------------------------------

test_that("L2 binding: find_sample_size result carries fitted / fitted_joint / cluster_atom", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$set_simulations(400)
  res <- m$find_sample_size(from_size = 40, to_size = 200, by = 20,
                            seed = 2137, progress_callback = FALSE)
  # fitted: non-empty, length == n_targets; all statuses in the allowed set
  expect_true(!is.null(res$fitted))
  expect_equal(length(res$fitted), 2L,
    info = "fitted length must equal n_targets (2 for y ~ x1 + x2)")
  allowed <- c("fitted", "at_or_below_min", "not_reached", "non_monotone")
  for (i in seq_len(2L)) {
    fj <- res$fitted[[as.character(i - 1L)]]
    expect_true(fj$status %in% allowed,
      info = sprintf("fitted[[%d]]$status must be in the allowed set", i - 1L))
  }
  # fitted_joint: non-empty; length == n_targets + sum(posthoc contrasts) — for
  # a two-target model with no posthoc that is 2.
  expect_true(!is.null(res$fitted_joint))
  expect_equal(length(res$fitted_joint), 2L,
    info = "fitted_joint length must equal n_targets for a no-posthoc model")
  for (i in seq_len(2L)) {
    fj <- res$fitted_joint[[as.character(i - 1L)]]
    expect_true(fj$status %in% allowed,
      info = sprintf("fitted_joint[[%d]]$status must be in the allowed set", i - 1L))
  }
  # cluster_atom: 1 for unclustered OLS
  expect_equal(res$cluster_atom, 1L,
    info = "cluster_atom must be 1 for unclustered OLS")
})

test_that("build_contract_bytes hook returns raw msgpack bytes", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")
  blob <- m$.__enclos_env__$private$build_contract_bytes("optimistic")
  expect_true(is.raw(blob))
  expect_true(length(blob) > 0L)
})

# ---------------------------------------------------------------------------
# A12: Guards (mirror Python verbatim)
# ---------------------------------------------------------------------------

test_that("random_slopes and n_per_parent are now accepted (gates removed)", {
  # random_slopes is now stored in pending_clusters$raw_slopes — no error.
  # A valid random_slopes list requires a modeled predictor. Use a formula that
  # has "x1" as a modeled predictor so the validator passes.
  m <- MCPower$new("y ~ x1 + (1|g)")$set_effects("x1=0.5")
  expect_no_error(m$set_cluster("g", n_clusters = 10L, random_slopes = list(
    list(predictor = "x1", variance = 0.05, corr_with_intercept = 0.0)
  )))
  # n_per_parent is now stored — no error.
  m2 <- MCPower$new("y ~ x1 + (1|a/b)")$set_effects("x1=0.5")
  expect_no_error(m2$set_cluster("a", n_clusters = 10L))
  expect_no_error(m2$set_cluster("a:b", n_per_parent = 4L))
  expect_equal(m2$.__enclos_env__$private$pending_clusters[["a:b"]]$n_per_parent, 4L)
  # between_vars is silently ignored (absorbed by `...`) — verified in
  # test-lme-cluster-level-vars.R; here just confirm no error.
  expect_no_error(MCPower$new("y ~ x1")$set_cluster("g", between_vars = "x1"))
})

test_that("logit runtime guards", {
  expect_error(
    MCPower$new("y ~ x1", family = "logit")$set_effects("x1=0.5")$find_power(100, progress_callback = FALSE),
    "baseline probability required for family='logit'; call set_baseline_probability(p) before find_power",
    fixed = TRUE)
  # "y ~ 1" is rejected by the Rust parser (as in Python); the intercept-only guard is a dead-code
  # safety net. Verify the parser-level rejection matches both R and Python behaviour.
  expect_error(MCPower$new("y ~ 1", family = "logit"))
})

test_that("logit scenario analysis runs (gate removed)", {
  # Gate-removal regression: scenario analysis is supported for binary
  # outcomes (β-jitter applies as log-odds heterogeneity). scenarios = TRUE
  # returns the 3-scenario envelope.
  res <- MCPower$new("y ~ x1", family = "logit")$set_effects("x1=0.5")$set_baseline_probability(0.3)$find_power(
    100, n_sims = 50, scenarios = TRUE, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "optimistic")
  expect_match(out, "realistic")
  expect_match(out, "doomer")
})

test_that("tukey is now accepted (guard removed)", {
  # tukey/tukey_hsd is no longer blocked — it routes to the engine's tukey_hsd correction.
  # A model with only continuous predictors (no factor) still runs; the posthoc block is empty.
  res <- MCPower$new("y ~ x1")$set_effects("x1=0.5")$find_power(100, correction = "tukey",
    verbose = FALSE, progress_callback = FALSE)
  # Assert it returned a real result, not just "did not error" — a crash-on-return
  # masked by expect_no_error would slip past.
  expect_false(is.null(res$power_uncorrected))
  expect_true(length(res$power_uncorrected) >= 1L)
})

test_that("set_parallel tombstone", {
  expect_error(MCPower$new("y ~ x1")$set_parallel(4),
    "mcpower has no set_parallel — parallelism is automatic and controlled by mcpower::set_n_threads(n).",
    fixed = TRUE)
})

test_that("find_sample_size on cluster_size-only LME errors", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$set_effects("x1=0.5")$set_cluster("g", ICC = 0.1, cluster_size = 20)
  expect_error(m$find_sample_size(progress_callback = FALSE),
    "find_sample_size with cluster_size-only LME specs is not yet supported; pass n_clusters to set_cluster",
    fixed = TRUE)
})

# ---------------------------------------------------------------------------
# A12: Logit + LME light-up
# ---------------------------------------------------------------------------

test_that("logit find_power runs with power floor > 0.3", {
  # G-A: add power-band floor for logit (large effect at N=300 must clear 30%)
  m <- MCPower$new("y ~ x1", family = "logit")$set_effects("x1=0.5")$set_baseline_probability(0.3)$set_simulations(300)
  res <- m$find_power(300, seed = 2137, progress_callback = FALSE)
  # Single-scenario: .unwrap_scenario_result returns the inner result directly.
  expect_equal(res$estimator_extras$estimator, "glm")
  expect_true(res$power_uncorrected[[1]] > 0.3,
    info = sprintf("logit power = %.3f", res$power_uncorrected[[1]]))
})

test_that("LME find_power runs with power floor > 0.3 and convergence >= 0.7", {
  # G-A: add power-band floor + convergence rate check for LME
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$set_effects("x1=0.5")$set_cluster("g", ICC = 0.1, n_clusters = 40)$set_simulations(200)
  res <- m$find_power(400, seed = 2137, progress_callback = FALSE)
  # Single-scenario: .unwrap_scenario_result returns the inner result directly.
  expect_equal(res$estimator_extras$estimator, "mle")
  expect_true(res$power_uncorrected[[1]] > 0.3,
    info = sprintf("LME power = %.3f", res$power_uncorrected[[1]]))
  expect_true(res$convergence_rate[[1]] >= 0.7,
    info = sprintf("convergence rate = %.3f", res$convergence_rate[[1]]))
  # boundary_rate_per_component: one entry per variance component (random
  # intercept only → length 1); each rate in [0, 1].
  brc <- res$estimator_extras$boundary_rate_per_component
  expect_true(!is.null(brc), info = "boundary_rate_per_component must be present for Mle")
  expect_true(length(brc) >= 1L,
    info = "expected at least one component for random-intercept LME")
  expect_true(all(brc >= 0 & brc <= 1),
    info = sprintf("rates out of [0,1]: %s", paste(brc, collapse = ", ")))
})

# ---------------------------------------------------------------------------
# G-C Tier-1-A: Interactions end-to-end (mirrors test_interactions.py)
# ---------------------------------------------------------------------------

test_that("find_power with interaction term x1:x2 returns non-trivial power", {
  # G-C T1-A: formula expansion and column-index targeting at the FFI boundary.
  # x1*x2 expands to x1 + x2 + x1:x2; target the interaction directly.
  m <- MCPower$new("y ~ x1 * x2")$
    set_effects("x1=0.3, x2=0.3, x1:x2=0.5")$
    set_simulations(200)
  res <- m$find_power(sample_size = 300, target_test = "x1:x2",
                      seed = 2137, progress_callback = FALSE)
  expect_false("scenarios" %in% names(res))
  expect_true("power_uncorrected" %in% names(res))
  # Medium-to-large interaction effect at N=300: power must be > 0 (non-trivial)
  expect_true(res$power_uncorrected[[1]] > 0,
    info = sprintf("interaction power = %.3f", res$power_uncorrected[[1]]))
})

# ---------------------------------------------------------------------------
# T2-A: Multi-scenario determinism — two find_power(scenarios=TRUE, seed=2137)
# calls must produce bit-identical power results across all scenarios.
# Mirrors Python tests/integration/test_reproducibility.py::test_g2_scenarios_byte_stable_two_runs
# ---------------------------------------------------------------------------

test_that("scenarios=TRUE with same seed gives identical results across two runs", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$set_simulations(200)

  a <- m$find_power(sample_size = 100, scenarios = TRUE, seed = 2137,
                    verbose = FALSE, progress_callback = FALSE)
  b <- m$find_power(sample_size = 100, scenarios = TRUE, seed = 2137,
                    verbose = FALSE, progress_callback = FALSE)

  # Both must be multi-scenario envelopes (scenarios wrapper present)
  expect_true("scenarios" %in% names(a))
  expect_true("scenarios" %in% names(b))

  # All three scenario slices must match bit-for-bit
  for (sc in names(a$scenarios)) {
    expect_identical(
      a$scenarios[[sc]]$power_uncorrected,
      b$scenarios[[sc]]$power_uncorrected,
      info = sprintf("power_uncorrected mismatch for scenario '%s'", sc)
    )
    expect_identical(
      a$scenarios[[sc]]$power_corrected,
      b$scenarios[[sc]]$power_corrected,
      info = sprintf("power_corrected mismatch for scenario '%s'", sc)
    )
    expect_identical(
      a$scenarios[[sc]]$ci_uncorrected,
      b$scenarios[[sc]]$ci_uncorrected,
      info = sprintf("ci_uncorrected mismatch for scenario '%s'", sc)
    )
    expect_identical(
      a$scenarios[[sc]]$n_sims,
      b$scenarios[[sc]]$n_sims,
      info = sprintf("n_sims mismatch for scenario '%s'", sc)
    )
  }
})

