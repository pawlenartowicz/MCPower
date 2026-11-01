library(mcpower)

# df_method = "auto" | "between_within" | "satterthwaite" | "asymptotic"
# per-call arg surface test. Mirrors ports/py/tests/test_df_method.py.
# Validated by name at find_power entry, normalised in spec-builder.R,
# forwarded into LinearSpec -> project_contract, where a named method the
# design does not support is refused. Only affects mixed-model (Mle, or
# clustered Glm) fixed-effect tests; OLS and unclustered GLM ignore it.

# Few-cluster LMM fixture with a cluster-level predictor -- where
# between-within df diverges most sharply from the asymptotic Wald z.
.few_cluster_lmm <- function() {
  MCPower$new("outcome ~ treat + baseline_score + (1|clinic)", family = "lme")$
    set_variable_type("treat=binary")$
    set_effects("treat=0.40, baseline_score=0.25")$
    set_cluster("clinic", ICC = 0.20, n_clusters = 6L, cluster_level_vars = "treat")
}

# Few-cluster random-slope LMM: the slope variable's between-within df is the
# within-cluster df, its Satterthwaite df rest on the 8 clusters.
.slope_lmm <- function() {
  MCPower$new("y ~ x1 + (1 + x1|g)", family = "lme")$
    set_effects("x1=0.30")$
    set_cluster("g", ICC = 0.20, n_clusters = 8L,
                random_slopes = list(list(predictor = "x1", variance = 0.10)))
}

.crossed_lmm <- function() {
  MCPower$new("y ~ x1 + (1|school) + (1|rater)", family = "lme")$
    set_effects("x1=0.4")$
    set_cluster("school", ICC = 0.2, n_clusters = 10L)$
    set_cluster("rater", ICC = 0.1, n_clusters = 6L)
}

.glmm <- function() {
  MCPower$new("y ~ x + (1|group)", family = "logit")$
    set_effects("x=0.5")$
    set_baseline_probability(0.3)$
    set_cluster("group", ICC = 0.3, n_clusters = 20L)
}

.ols <- function() {
  MCPower$new("y ~ x1")$set_effects("x1=0.30")
}

.power <- function(m, df_method, n = 120L, n_sims = 300L, ...) {
  r <- m$find_power(sample_size = n, df_method = df_method, n_sims = n_sims,
                    seed = 2137L, progress_callback = FALSE, verbose = FALSE, ...)
  r$power_uncorrected[[1]]
}

test_that("df_method rejects garbage values", {
  m <- .few_cluster_lmm()
  for (bad in c("wald", "nonsense", "kenward_roger")) {
    expect_error(
      m$find_power(sample_size = 120L, n_sims = 100L, df_method = bad,
                   progress_callback = FALSE, verbose = FALSE),
      "unknown df_method")
  }
})

test_that("df_method default is auto", {
  expect_identical(mcpower:::.config()$estimation$df_method, "auto")
  m <- .few_cluster_lmm()
  r_default <- m$find_power(sample_size = 120L, target_test = "treat", n_sims = 300L,
                            seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  expect_equal(r_default$power_uncorrected[[1]], .power(m, "auto", target_test = "treat"))
})

test_that("auto is between_within without a random slope", {
  m <- .few_cluster_lmm()
  expect_equal(.power(m, "auto", target_test = "treat"),
               .power(m, "between_within", target_test = "treat"))
})

test_that("auto is satterthwaite with a random slope", {
  # At 8 clusters Satterthwaite rejects less often than between-within.
  m <- .slope_lmm()
  p_auto <- .power(m, "auto", n = 160L, n_sims = 400L)
  expect_equal(p_auto, .power(m, "satterthwaite", n = 160L, n_sims = 400L))
  expect_lt(p_auto, .power(m, "between_within", n = 160L, n_sims = 400L))
})

test_that("df_method reaches the engine", {
  # asymptotic vs between_within at the same seed produce different power for
  # the few-cluster LMM.
  m <- .few_cluster_lmm()
  expect_false(isTRUE(all.equal(.power(m, "asymptotic", n_sims = 500L, target_test = "treat"),
                                .power(m, "between_within", n_sims = 500L, target_test = "treat"))))
})

test_that("between_within power is <= asymptotic power for few-cluster LMM", {
  # Between-within df is the conservative choice for a design with few
  # clusters -- power should be no higher than the asymptotic Wald z at the
  # same seed.
  m <- .few_cluster_lmm()
  expect_true(.power(m, "between_within", n_sims = 1000L, target_test = "treat") <=
                .power(m, "asymptotic", n_sims = 1000L, target_test = "treat"))
})

test_that("between_within is refused on crossed groupings", {
  m <- .crossed_lmm()
  expect_error(.power(m, "between_within", n = 120L, n_sims = 20L),
               "'between_within' needs a single grouping.*'auto'")
  # auto runs the same design, at Satterthwaite df
  expect_equal(.power(m, "auto", n = 120L, n_sims = 40L),
               .power(m, "satterthwaite", n = 120L, n_sims = 40L))
})

test_that("satterthwaite is refused on a GLMM", {
  m <- .glmm()
  expect_error(suppressWarnings(.power(m, "satterthwaite", n = 100L, n_sims = 20L)),
               "'satterthwaite' needs a continuous outcome.*'auto'")
})

test_that("OLS ignores df_method", {
  # Every value gives identical power at the same seed.
  m <- .ols()
  p <- vapply(c("auto", "asymptotic", "between_within", "satterthwaite"),
              function(d) .power(m, d, n = 200L), numeric(1))
  expect_equal(length(unique(p)), 1L)
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
