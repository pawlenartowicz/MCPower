library(mcpower)

# df_method = "between_within" | "asymptotic" per-call arg surface test.
# Mirrors ports/py/tests/test_df_method.py. Validated at find_power entry,
# normalised in spec-builder.R, forwarded into LinearSpec -> project_contract.
# Only affects mixed-model (Mle, or clustered Glm) fixed-effect tests; OLS and
# unclustered GLM ignore it.

# Few-cluster LMM fixture with a cluster-level predictor -- where
# between-within df diverges most sharply from the asymptotic Wald z.
.few_cluster_lmm <- function() {
  MCPower$new("outcome ~ treat + baseline_score + (1|clinic)", family = "lme")$
    set_variable_type("treat=binary")$
    set_effects("treat=0.40, baseline_score=0.25")$
    set_cluster("clinic", ICC = 0.20, n_clusters = 6L, cluster_level_vars = "treat")
}

.ols <- function() {
  MCPower$new("y ~ x1")$set_effects("x1=0.30")
}

test_that("df_method rejects garbage values", {
  m <- .few_cluster_lmm()
  expect_error(
    m$find_power(sample_size = 120L, n_sims = 100L, df_method = "wald",
                 progress_callback = FALSE, verbose = FALSE))
  expect_error(
    m$find_power(sample_size = 120L, n_sims = 100L, df_method = "nonsense",
                 progress_callback = FALSE, verbose = FALSE))
})

test_that("df_method default is between_within", {
  expect_identical(mcpower:::.config()$estimation$df_method, "between_within")
  m <- .few_cluster_lmm()
  r_default <- m$find_power(sample_size = 120L, target_test = "treat", n_sims = 300L,
                            seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  r_bw <- m$find_power(sample_size = 120L, target_test = "treat", df_method = "between_within",
                       n_sims = 300L, seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  # Default (no arg) must equal explicit "between_within" -- same seed/data.
  expect_equal(r_default$power_uncorrected[[1]], r_bw$power_uncorrected[[1]])
})

test_that("df_method reaches the engine", {
  # The value threads through to the engine: asymptotic vs between_within at
  # the same seed produce different power for the few-cluster LMM.
  m <- .few_cluster_lmm()
  r_asym <- m$find_power(sample_size = 120L, target_test = "treat", df_method = "asymptotic",
                         n_sims = 500L, seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  r_bw <- m$find_power(sample_size = 120L, target_test = "treat", df_method = "between_within",
                       n_sims = 500L, seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  expect_false(isTRUE(all.equal(r_asym$power_uncorrected[[1]], r_bw$power_uncorrected[[1]])))
})

test_that("between_within power is <= asymptotic power for few-cluster LMM", {
  # Between-within df is the conservative choice for a design with few
  # clusters -- power should be no higher than the asymptotic Wald z at the
  # same seed.
  m <- .few_cluster_lmm()
  r_asym <- m$find_power(sample_size = 120L, target_test = "treat", df_method = "asymptotic",
                         n_sims = 1000L, seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  r_bw <- m$find_power(sample_size = 120L, target_test = "treat", df_method = "between_within",
                       n_sims = 1000L, seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  expect_true(r_bw$power_uncorrected[[1]] <= r_asym$power_uncorrected[[1]])
})

test_that("OLS ignores df_method", {
  # OLS ignores df_method: asymptotic vs between_within give identical power
  # at the same seed.
  m <- .ols()
  r_asym <- m$find_power(sample_size = 200L, df_method = "asymptotic", n_sims = 300L,
                         seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  r_bw <- m$find_power(sample_size = 200L, df_method = "between_within", n_sims = 300L,
                       seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  expect_equal(r_asym$power_uncorrected[[1]], r_bw$power_uncorrected[[1]])
})

# M10: R's .df_method_for_rust(NULL) reads config.json's estimation.df_method --
# no hardcoded per-port default. Mirrors the wald_se parity test above.
test_that("df_method: NULL resolves to the config estimation.df_method default", {
  expect_identical(mcpower:::.df_method_for_rust(NULL), mcpower:::.config()$estimation$df_method)
})

test_that("a collapsed (singular) between-within LMM fit reports the residual df", {
  # ICC 0.05 (the validator floor) with 5 clusters pins tau^2 at 0 on a good
  # share of draws; such a fit is the OLS on the same
  # columns and is tested against t(N - p), not the between df g - 2. A draw
  # that keeps tau^2 > 0 stays on the between df. Both kinds must show up.
  m <- MCPowerDebug$new("y ~ x + z + (1|g)", family = "lme",
                        debug_n = 100L, debug_n_sims = 1L)
  m$set_effects("x=0, z=0")
  m$set_cluster("g", ICC = 0.05, n_clusters = 5L, cluster_level_vars = "z")
  seen <- c(collapsed = FALSE, live = FALSE)
  for (k in 1:40) {
    m$.debug_seed <- 7000 + k
    f <- m$load_data(m$create_data(), df_method = "between_within")
    if (!isTRUE(f$converged)) next
    z <- f$targets[[2]]
    if (f$variance_components[[1]] == 0) {
      seen[["collapsed"]] <- TRUE
      expect_equal(z$df, 97)                    # N - p = 100 - 3
      expect_equal(z$critical_value, qt(0.975, 97))
    } else {
      seen[["live"]] <- TRUE
      expect_equal(z$df, 3)                     # g - p_between = 5 - 2
      expect_equal(z$critical_value, qt(0.975, 3))
    }
    if (all(seen)) break
  }
  expect_true(all(seen))
})
