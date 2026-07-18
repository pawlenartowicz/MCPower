library(mcpower)

# truth_start is a scenario ASSUMPTION ("estimation is well-behaved"), not a
# generic knob: only the three named presets set it, and a brand-new custom
# scenario stays cold-start (FALSE) even though it inherits every other key
# from optimistic.

# (1) optimistic preset defaults to truth_start = TRUE
test_that(".scenario_dict optimistic emits truth_start = TRUE", {
  d <- mcpower:::.scenario_dict("optimistic")
  expect_true("truth_start" %in% names(d))
  expect_true(d$truth_start)
})

# (2) realistic/doomer presets default to truth_start = FALSE
test_that(".scenario_dict realistic and doomer emit truth_start = FALSE", {
  expect_false(mcpower:::.scenario_dict("realistic")$truth_start)
  expect_false(mcpower:::.scenario_dict("doomer")$truth_start)
})

# (3) a brand-new custom scenario does NOT inherit truth_start = TRUE from
# optimistic, even though it inherits every other unset key.
test_that("set_scenario_configs: custom scenario does not inherit truth_start", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")
  m$set_scenario_configs(list(my_case = list(heterogeneity = 0.5)))
  cfg <- m$.__enclos_env__$private$scenario_configs[["my_case"]]
  expect_false(cfg$truth_start)
  # Confirms the exclusion is targeted, not a break in the general inheritance:
  # heteroskedasticity_ratio (unset by the user) still comes from optimistic.
  expect_equal(
    cfg$heteroskedasticity_ratio,
    mcpower:::.scenario_defaults()[["optimistic"]][["heteroskedasticity_ratio"]]
  )
})

# (4) a custom scenario that sets truth_start explicitly keeps the user value.
test_that("set_scenario_configs: custom scenario honours explicit truth_start", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")
  m$set_scenario_configs(list(my_case = list(heterogeneity = 0.5, truth_start = TRUE)))
  cfg <- m$.__enclos_env__$private$scenario_configs[["my_case"]]
  expect_true(cfg$truth_start)
})

# (5) updating an existing preset (optimistic) keeps its truth_start unless overridden.
test_that("set_scenario_configs: updating optimistic preserves its truth_start", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")
  m$set_scenario_configs(list(optimistic = list(heterogeneity = 0.1)))
  cfg <- m$.__enclos_env__$private$scenario_configs[["optimistic"]]
  expect_true(cfg$truth_start)
})
