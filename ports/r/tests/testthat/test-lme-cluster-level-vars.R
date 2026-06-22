library(mcpower)

# Mirrors the Python cluster_level_vars contract (tests/spec/test_set_cluster.py).
# Canonical R LME construction: `(1|group)` random-effect syntax + family = "lme";
# a cluster-level covariate is a SEPARATE modeled predictor (e.g. x2), never the
# grouping variable itself.

# (a) cluster_level_vars accepted and stored (absorbs former smoke-only check)
test_that("set_cluster accepts cluster_level_vars without error", {
  m <- expect_no_error(
    MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
      set_effects("x1=0.3, x2=0.2")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = "x2")
  )
  expect_equal(m$.__enclos_env__$private$pending_clusters[["school"]]$cluster_level_vars, "x2")
})

# (b) old name between_vars is no longer a parameter. set_cluster has `...`, so
# between_vars is silently absorbed and has no effect (it does NOT populate
# cluster_level_vars). The supported parameter is cluster_level_vars.
test_that("between_vars is no longer a parameter — silently ignored, no effect", {
  m <- MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
    set_effects("x1=0.3, x2=0.2")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L, between_vars = "x2")
  # cluster_level_vars live per grouping var; between_vars must not populate them.
  expect_length(m$.__enclos_env__$private$pending_clusters[["school"]]$cluster_level_vars, 0L)
})

# (c) cluster_level_vars must be a character vector
test_that("set_cluster rejects non-character cluster_level_vars", {
  expect_error(
    MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
      set_effects("x1=0.3, x2=0.2")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = 42L),
    regexp = "cluster_level_vars"
  )
})

# (d) cluster_level_vars must name modeled predictors
test_that("set_cluster rejects cluster_level_vars not in formula", {
  expect_error(
    MCPower$new("y ~ x1 + (1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L,
                  cluster_level_vars = "z_not_in_formula"),
    regexp = "cluster_level_vars"
  )
})

# (d2) the grouping variable itself may not be a cluster-level covariate
test_that("set_cluster rejects the grouping variable as a cluster_level_var", {
  expect_error(
    MCPower$new("y ~ x1 + (1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = "school"),
    regexp = "grouping variable"
  )
})

# (e) cluster_level_vars may be NULL (backward compat)
test_that("set_cluster with cluster_level_vars = NULL still works", {
  expect_no_error(
    MCPower$new("y ~ x1 + (1|school)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = NULL)
  )
})

# (f) stored cluster_level_vars appear in the serialized LinearSpec payload
test_that("cluster_level_vars appears in to_linear_spec_list payload", {
  m <- MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
    set_effects("x1=0.3, x2=0.2")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = "x2")
  m$.__enclos_env__$private$apply()
  reg <- m$.__enclos_env__$private$registry
  # Source cluster_level_vars from per-group storage (the union the engine sees).
  clv <- unlist(lapply(m$.__enclos_env__$private$pending_clusters, `[[`, "cluster_level_vars"),
                use.names = FALSE)
  payload <- mcpower:::.to_linear_spec_list(
    reg, "optimistic",
    alpha = 0.05, correction = NULL, target_test = NULL,
    heteroskedasticity = list(driver_var_index = 0L),
    residual_name = "normal", max_failed = 0.1,
    test_formula = NULL,
    scenario_configs = mcpower:::.scenario_defaults(),
    pending_data = NULL,
    cluster_level_vars = clv
  )
  expect_true("cluster_level_vars" %in% names(payload))
  # Emitted via I() so it serialises as a JSON array; strip the AsIs class.
  expect_equal(as.character(payload$cluster_level_vars), "x2")
})

# (f2) a second set_cluster for a different grouping var must NOT drop the first
# group's cluster_level_vars — both survive (regression for the flat-field bug).
test_that("cluster_level_vars accumulate across crossed groupings", {
  m <- MCPower$new("y ~ x1 + x2 + x3 + (1|school) + (1|district)", family = "lme")$
    set_effects("x1=0.3, x2=0.2, x3=0.1")$
    set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_level_vars = "x2")$
    set_cluster("district", ICC = 0.05, n_clusters = 10L, cluster_level_vars = "x3")
  priv <- m$.__enclos_env__$private
  expect_equal(priv$pending_clusters[["school"]]$cluster_level_vars, "x2")
  expect_equal(priv$pending_clusters[["district"]]$cluster_level_vars, "x3")
  clv <- unlist(lapply(priv$pending_clusters, `[[`, "cluster_level_vars"), use.names = FALSE)
  expect_setequal(clv, c("x2", "x3"))
})

# (g) end-to-end: find_power succeeds with cluster_level_vars set
test_that("find_power succeeds with cluster_level_vars", {
  m <- MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
    set_effects("x1=0.3, x2=0.2")$
    set_cluster("school", ICC = 0.1, n_clusters = 20L, cluster_level_vars = "x2")
  res <- m$find_power(sample_size = 60L, n_sims = 50L,
                      seed = 2137, progress_callback = FALSE)
  # x1 effect 0.3 with a cluster-level covariate at N=60 is detectable; pin a power
  # floor so a no-op path fails where expect_no_error passed. Robust floor (not
  # capture-pinned — the R package is not installable in this WIP tree).
  expect_true(res$power_uncorrected[[1]][[1]] > 0.05)
})
