library(mcpower)

# (1) .scenario_dict emits random_effect_dist as integer (not NULL)
test_that(".scenario_dict emits random_effect_dist as an integer code", {
  d <- mcpower:::.scenario_dict("optimistic")
  expect_true("random_effect_dist" %in% names(d))
  expect_true(is.integer(d$random_effect_dist))
  # "normal" -> code 0 per .residual_codes()
  expect_equal(d$random_effect_dist, 0L)
})

# (2) .scenario_dict emits random_effect_df as numeric
test_that(".scenario_dict emits random_effect_df as numeric", {
  d <- mcpower:::.scenario_dict("optimistic")
  expect_true("random_effect_df" %in% names(d))
  expect_true(is.numeric(d$random_effect_df))
})

# (3) .scenario_dict emits icc_noise_sd as numeric
test_that(".scenario_dict emits icc_noise_sd as numeric", {
  d <- mcpower:::.scenario_dict("optimistic")
  expect_true("icc_noise_sd" %in% names(d))
  expect_true(is.numeric(d$icc_noise_sd))
})

# (4) lme = NULL key is gone
test_that(".scenario_dict no longer emits lme = NULL", {
  d <- mcpower:::.scenario_dict("optimistic")
  expect_false("lme" %in% names(d))
})

# (5) realistic scenario: random_effect_dist = heavy_tailed -> non-zero code
# Uses .re_dist_codes() (normal/heavy_tailed vocabulary), not .residual_codes().
test_that(".scenario_dict realistic emits heavy_tailed code for random_effect_dist", {
  d <- mcpower:::.scenario_dict("realistic")
  codes <- mcpower:::.re_dist_codes()
  expect_equal(d$random_effect_dist, as.integer(codes[["heavy_tailed"]]))
  expect_true(d$random_effect_dist > 0L)
})

# (6) set_scenario_configs now accepts AND stores icc_noise_sd (no longer rejected)
test_that("set_scenario_configs stores icc_noise_sd", {
  m <- MCPower$new("y ~ x1 + (1|g)")$
    set_effects("x1=0.3")$
    set_cluster("g", ICC = 0.1, n_clusters = 20L, cluster_size = 25L)
  m$set_scenario_configs(list(optimistic = list(icc_noise_sd = 0.05)))
  # Assert the value landed in private$scenario_configs — a silent no-op setter would
  # pass expect_no_error. Also de-duplicates the smoke loop in test-apic-surface.R.
  expect_equal(
    m$.__enclos_env__$private$scenario_configs[["optimistic"]][["icc_noise_sd"]],
    0.05
  )
})

# (7) set_scenario_configs accepts AND stores random_effect_dist
test_that("set_scenario_configs stores random_effect_dist", {
  m <- MCPower$new("y ~ x1 + (1|g)")$
    set_effects("x1=0.3")$
    set_cluster("g", ICC = 0.1, n_clusters = 20L, cluster_size = 25L)
  m$set_scenario_configs(list(optimistic = list(random_effect_dist = "heavy_tailed")))
  expect_equal(
    m$.__enclos_env__$private$scenario_configs[["optimistic"]][["random_effect_dist"]],
    "heavy_tailed"
  )
})

# (8) set_scenario_configs accepts AND stores random_effect_df
test_that("set_scenario_configs stores random_effect_df", {
  m <- MCPower$new("y ~ x1 + (1|g)")$
    set_effects("x1=0.3")$
    set_cluster("g", ICC = 0.1, n_clusters = 20L, cluster_size = 25L)
  m$set_scenario_configs(list(optimistic = list(random_effect_df = 5.0)))
  expect_equal(
    m$.__enclos_env__$private$scenario_configs[["optimistic"]][["random_effect_df"]],
    5.0
  )
})

# (9) unknown keys still rejected
test_that("set_scenario_configs still rejects unknown keys", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")
  expect_error(
    m$set_scenario_configs(list(optimistic = list(not_a_real_key = 1.0))),
    regexp = "unknown config key"
  )
})
