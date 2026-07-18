test_that("set_n_threads rejects non-numeric input", {
  expect_error(set_n_threads("4"), "n must be a single integer")
  expect_error(set_n_threads(NULL), "n must be a single integer")
  expect_error(set_n_threads(c(2L, 4L)), "n must be a single integer")
  expect_error(set_n_threads(NA_integer_), "n must be a single integer")
})

test_that("set_n_threads rejects n < 1 from the engine", {
  # The Rust side enforces n >= 1; the error propagates as a stop() from the bridge.
  expect_error(set_n_threads(0L), "n must be >= 1")
})

test_that("set_n_threads raises once the pool is already initialised", {
  # The rayon pool is a process-global OnceLock. Trigger lazy initialisation with
  # a minimal find_power call, then verify a second set_n_threads call errors.
  # Mirrors Python test_e3_set_n_threads_double_set_raises.
  cj <- mcpower:::build_contract_from_spec(
    .ols_spec_json("y ~ x1", x1 = 0.3), "continuous", "canonical", "ols", 0.0, "[]")$contracts
  mcpower:::find_power(cj, 50L, 50L, 2137, NULL)  # initialises the pool
  expect_error(set_n_threads(2L),
    regexp = "already initialized|already initialised",
    ignore.case = TRUE
  )
})
