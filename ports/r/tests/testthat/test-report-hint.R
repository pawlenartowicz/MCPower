# The internal-error "report this" hint is appended to engine panics (surfaced
# as an R error) but never to validation or cancellation errors.
#
# panic_for_test is an internal test seam (mirrors engine-py's panic_for_test);
# see crates/engine-r/src/orchestrator_bridge.rs.

test_that("a caught engine panic surfaces with the report hint", {
  err <- tryCatch(mcpower:::panic_for_test(), error = function(e) conditionMessage(e))
  expect_true(grepl("mcpower.app/report", err, fixed = TRUE))
  expect_true(grepl("port=r", err, fixed = TRUE))
})

test_that("a validation error does not carry the report hint", {
  # Malformed contracts bytes → engine-layer validation error, no hint.
  err <- tryCatch(
    mcpower:::find_power(as.raw(c(0xff, 0xff, 0xff)), 100L, 100L, 2137, NULL),
    error = function(e) conditionMessage(e)
  )
  expect_false(grepl("mcpower.app/report", err, fixed = TRUE))
})

test_that("a cancellation error does not carry the report hint", {
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.3),
                                           "continuous", "canonical", "ols", 0.0, "[]")$contracts
  err <- tryCatch(
    mcpower:::find_power_precancelled(cj, 200L, 400L, 2137),
    error = function(e) conditionMessage(e)
  )
  expect_false(grepl("mcpower.app/report", err, fixed = TRUE))
})
