# L1 — unit tests for .reuse_fraction and .strict_reuse_warning.
# Mirrors mcpower/ports/py/tests/test_reuse_diagnostics.py exactly.

test_that("reuse_fraction closed-form golden values", {
  expect_lt(abs(mcpower:::.reuse_fraction(1000L, 1000L) - 26.0), 1.0)
  expect_lt(abs(mcpower:::.reuse_fraction(1000L, 2000L) - 59.0), 1.0)
  expect_gte(mcpower:::.reuse_fraction(1000L, 999L), 0.0)
})

test_that("reuse_fraction edge cases", {
  expect_equal(mcpower:::.reuse_fraction(0L,  100L), 0.0)
  expect_equal(mcpower:::.reuse_fraction(-5L, 100L), 0.0)
  expect_equal(mcpower:::.reuse_fraction(1L,  100L), 100.0)
})

test_that("strict_reuse_warning fires above ratio (strict >)", {
  # (U=100, N=201, ratio=2.0): 201 > 2.0*100 → fires
  expect_false(is.null(mcpower:::.strict_reuse_warning(100L, 201L, 2.0)))
  # (U=100, N=200, ratio=2.0): 200 == 2.0*100, NOT strictly greater → does not fire
  expect_null(mcpower:::.strict_reuse_warning(100L, 200L, 2.0))
})

test_that("strict_reuse_warning message content", {
  msg <- mcpower:::.strict_reuse_warning(50L, 200L, 2.0)
  expect_false(is.null(msg))
  expect_true(grepl("partial|none", msg, ignore.case = TRUE))
  expect_true(grepl("200", msg))
  expect_true(grepl("50", msg))
})

# (strict-mode upload acceptance + storage is covered by test-api-surface.R's
# "mode='strict' is accepted and stores the upload" — this pure-smoke duplicate
# was merged away in the L1/L2 test audit.)

# --- G-C Tier-1-C: strict-bootstrap reuse warning end-to-end ----------------
# Mirrors test_strict_bootstrap.py: upload 40 rows, find_power with N=200
# (> 2x uploaded rows) -> expect reuse warning to fire above ratio threshold.

test_that("G-C T1-C: strict-bootstrap find_power warns when N >> uploaded rows", {
  set.seed(1)
  df <- data.frame(x1 = rnorm(40), stringsAsFactors = FALSE)
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")$set_simulations(50)
  m$upload_data(df, mode = "strict", verbose = FALSE)
  # N=200 >> 2 * 40 = 80 rows -> reuse warning fires (strict > threshold)
  expect_warning(
    m$find_power(200, seed = 2137, progress_callback = FALSE),
    regexp = "reuse|partial|none",
    ignore.case = TRUE
  )
})
