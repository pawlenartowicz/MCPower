library(mcpower)

# M2 multiple/nested groupings — public set_cluster path.
# Mirrors Python tests/spec/test_set_cluster.py (tasks 1.5, 1.6).
# Canonical construction: family="lme" + (1|g) formula + set_cluster("g", ...).

# ---------------------------------------------------------------------------
# Gate removal — acceptance without error
# ---------------------------------------------------------------------------

# (1) Two crossed groupings accepted without error and both stored (absorbs former smoke-only check)
test_that("two crossed set_cluster calls succeed without error", {
  m <- expect_no_error(
    MCPower$new("y ~ x1 + (1|school) + (1|district)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
      set_cluster("district", ICC = 0.05, n_clusters = 10L)
  )
  expect_equal(length(m$.__enclos_env__$private$pending_clusters), 2L)
})

# (2) n_per_parent gate removed — nested spec accepted at set_cluster time
test_that("n_per_parent no longer raises a reserved-feature error", {
  expect_no_error(
    MCPower$new("y ~ x1 + (1|school/district)", family = "lme")$
      set_effects("x1=0.3")$
      set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
      set_cluster("school:district", ICC = 0.05, n_per_parent = 5L)
  )
})

# (3) Nested formula — validate_lme_nested gate removed: (1|A/B) no longer errors
# at find_power. (We only check that set_cluster + apply don't raise; the engine
# smoke test is in test (7) below.)
test_that("formula with nested '/' no longer raises at apply time", {
  m <- MCPower$new("y ~ x1 + (1|school/district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
    set_cluster("school:district", ICC = 0.05, n_per_parent = 5L)
  expect_no_error(m$.__enclos_env__$private$apply())
})

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# (4) n_per_parent stored in pending_clusters
test_that("n_per_parent is stored in pending_clusters", {
  m <- MCPower$new("y ~ x1 + (1|school/district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",         ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
    set_cluster("school:district", ICC = 0.05, n_per_parent = 5L)
  expect_equal(m$.__enclos_env__$private$pending_clusters[["school:district"]]$n_per_parent, 5L)
})

# (5) Two set_cluster calls stored as two entries in pending_clusters
test_that("two set_cluster calls produce two entries in pending_clusters", {
  m <- MCPower$new("y ~ x1 + (1|school) + (1|district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
    set_cluster("district", ICC = 0.05, n_clusters = 10L)
  expect_equal(length(m$.__enclos_env__$private$pending_clusters), 2L)
  expect_true("school"   %in% names(m$.__enclos_env__$private$pending_clusters))
  expect_true("district" %in% names(m$.__enclos_env__$private$pending_clusters))
})

# ---------------------------------------------------------------------------
# clusters_json shape — direct call to internal encoder
# ---------------------------------------------------------------------------

# (6) Two crossed groupings produce extra_groupings with Crossed relation
test_that("two public set_cluster calls produce correct extra_groupings in clusters_json", {
  m <- MCPower$new("y ~ x1 + (1|school) + (1|district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",   ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
    set_cluster("district", ICC = 0.05, n_clusters = 10L)
  m$.__enclos_env__$private$apply()
  enc <- mcpower:::.encode_outcome_and_clusters(
    m$family, m$estimator, m$intercept,
    m$.__enclos_env__$private$pending_clusters)
  cj <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_equal(length(cj), 1L)          # one cluster spec (the primary)
  eg <- cj[[1]]$extra_groupings
  expect_equal(length(eg), 1L)          # one extra grouping (district)
  expect_false(is.null(eg[[1]]$relation$Crossed))
  expect_equal(eg[[1]]$relation$Crossed$n_clusters, 10L)
  district_tau <- 0.05 / (1 - 0.05)
  expect_equal(eg[[1]]$tau_squared, district_tau, tolerance = 1e-12)
})

# (7) Nested grouping: n_per_parent produces NestedWithin relation in extra_groupings
test_that("nested grouping with n_per_parent emits NestedWithin relation in clusters_json", {
  m <- MCPower$new("y ~ x1 + (1|school/district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",         ICC = 0.10, n_clusters = 20L, cluster_size = 15L)$
    set_cluster("school:district", ICC = 0.05, n_per_parent = 5L)
  m$.__enclos_env__$private$apply()
  enc <- mcpower:::.encode_outcome_and_clusters(
    m$family, m$estimator, m$intercept,
    m$.__enclos_env__$private$pending_clusters)
  cj <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  eg <- cj[[1]]$extra_groupings
  expect_equal(length(eg), 1L)
  expect_false(is.null(eg[[1]]$relation$NestedWithin))
  expect_equal(eg[[1]]$relation$NestedWithin$n_per_parent, 5L)
})

# (8) No extra groupings when only one set_cluster call — extra_groupings absent
test_that("single set_cluster produces no extra_groupings key in clusters_json", {
  pending <- list(
    school = list(icc = 0.1, n_clusters = 20L, cluster_size = 15L)
  )
  enc <- mcpower:::.encode_outcome_and_clusters("gaussian", "lme", 0.0, pending)
  cj <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_null(cj[[1]]$extra_groupings)
})

# (9) Debug seam (set_extra_groupings_debug) takes precedence over public path
test_that("debug seam extra_groupings takes precedence over public-path extra_groupings", {
  # Hand-craft a pending list with both public extra groupings AND debug-seam
  # cfg$extra_groupings. The debug value should win (harness invariant).
  pending <- list(
    school = list(
      icc         = 0.10, n_clusters = 20L, cluster_size = 15L,
      # debug seam: a two-entry extra_groupings list
      extra_groupings = list(
        list(relation = list(Crossed = list(n_clusters = 99L)), tau_squared = 0.01)
      )
    ),
    # second entry in pending_clusters (would normally trigger the public path)
    district = list(icc = 0.05, n_clusters = 10L)
  )
  enc <- mcpower:::.encode_outcome_and_clusters("gaussian", "lme", 0.0, pending)
  cj  <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  eg  <- cj[[1]]$extra_groupings
  # Debug seam wins — n_clusters=99 from the debug value, not 10 from public path
  expect_equal(eg[[1]]$relation$Crossed$n_clusters, 99L)
})

# ---------------------------------------------------------------------------
# End-to-end (engine smoke)
# ---------------------------------------------------------------------------

# (10) find_power with two crossed groupings completes without error
test_that("find_power with two crossed groupings completes without error", {
  m <- MCPower$new("y ~ x1 + (1|school) + (1|district)", family = "lme")$
    set_effects("x1=0.3")$
    set_cluster("school",   ICC = 0.10, n_clusters = 20L)$
    set_cluster("district", ICC = 0.05, n_clusters = 10L)
  # sample_size is a multiple of the crossed cluster atom (20 x 10) so the
  # engine does not snap-and-warn.
  res <- m$find_power(sample_size = 200L, n_sims = 50L, seed = 2137,
                      progress_callback = FALSE, verbose = FALSE)
  # x1 effect 0.3 over crossed groupings at N=200 is detectable; pin a power floor so
  # a no-op/zero-power crossed path fails where expect_no_error passed. Robust floor
  # (not capture-pinned — the R package is not installable in this WIP tree).
  expect_true(res$power_uncorrected[[1]][[1]] > 0.05)
})
