# D6 — upload type-lock guard: uploaded column types are authoritative.
# Mirrors Python test_upload_type_lock.py; covers the four spec cases.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Trigger apply() without running the full simulation engine.
# Accesses private$apply() directly via R6's .__enclos_env__$private.
# This fires all the type-lock validation without reaching the Rust engine.
.trigger_apply <- function(model) {
  model$.__enclos_env__$private$apply()
}

# Build a single-column data.frame suitable for upload_data().
.make_df <- function(col_name, values) {
  df <- data.frame(v = values, stringsAsFactors = FALSE)
  names(df) <- col_name
  df
}

# ---------------------------------------------------------------------------
# Case 1: detected continuous + declared factor → error naming the column
# ---------------------------------------------------------------------------

test_that("continuous column declared as factor raises type-lock error", {
  # Upload many distinct numeric values → detected as continuous.
  vals <- as.numeric(seq(1, 100))  # 100 distinct → continuous
  df   <- .make_df("x1", vals)

  m <- MCPower$new("y ~ x1")$
    set_variable_type("x1=(factor,3)")$   # user declares factor — conflicts
    set_effects("x1=0.5")$
    upload_data(df, verbose = FALSE)

  expect_error(
    .trigger_apply(m),
    regexp = "Column 'x1' was detected as continuous from your uploaded data; it can't be modeled as factor\\. Uploaded columns take their type from the data\\.",
    fixed  = FALSE
  )
})

# ---------------------------------------------------------------------------
# Case 2: detected binary + declared continuous → error
# ---------------------------------------------------------------------------

test_that("binary column declared as continuous raises type-lock error", {
  # Exactly 2 distinct values → detected as binary.
  vals <- rep(c(0, 1), 50)  # 100 rows, 2 distinct
  df   <- .make_df("x1", vals)

  m <- MCPower$new("y ~ x1")$
    set_variable_type("x1=normal")$       # user declares continuous — conflicts
    set_effects("x1=0.5")$
    upload_data(df, verbose = FALSE)

  expect_error(
    .trigger_apply(m),
    regexp = "Column 'x1' was detected as binary from your uploaded data; it can't be modeled as continuous\\. Uploaded columns take their type from the data\\.",
    fixed  = FALSE
  )
})

# ---------------------------------------------------------------------------
# Case 3: matching declaration / absent declaration → no error;
#         a factor declared with wrong k has levels overridden by data
# ---------------------------------------------------------------------------

test_that("factor column with wrong k: data wins on levels, no error", {
  # 3 distinct string values repeated → detected as factor with 3 levels.
  raw  <- rep(c("Japan", "USA", "Europe"), 34)[seq_len(100)]
  df   <- .make_df("origin", raw)

  # User declares (factor,5) — k=5 conflicts with data k=3, but CLASS matches.
  # Type-lock should NOT error; it should re-apply and override to data levels.
  # Effects are set after upload to use data-derived level names.
  m <- MCPower$new("y ~ origin")$
    set_variable_type("origin=(factor,5)")$
    upload_data(df, verbose = FALSE)

  # Manually set effects after upload so they use data-driven dummy names.
  # (trigger_apply expands factors using data labels: reference=Europe, dummies=Japan,USA)
  m$set_effects("origin[Japan]=0.3, origin[USA]=0.4")

  expect_no_error(
    .trigger_apply(m)
  )

  # After apply the registry factor info must reflect data (3 levels, not 5).
  reg     <- m$.__enclos_env__$private$registry
  finfo   <- reg$`_factors`[["origin"]]
  expect_equal(finfo$n_levels, 3L)
  expect_equal(sort(as.character(unlist(finfo$level_labels))),
               c("Europe", "Japan", "USA"))
})

test_that("no set_variable_type on uploaded column — no error", {
  # Upload a binary column without any type declaration.
  vals <- rep(c(0, 1), 50)
  df   <- .make_df("x1", vals)

  m <- MCPower$new("y ~ x1")$
    set_effects("x1=0.5")$
    upload_data(df, verbose = FALSE)

  expect_no_error(.trigger_apply(m))
})

# ---------------------------------------------------------------------------
# Case 4: detected continuous + declared continuous distribution → no error,
#         declared distribution is preserved
# ---------------------------------------------------------------------------

test_that("continuous column declared as right_skewed: no error and dist preserved", {
  # 100 distinct floats → continuous.
  set.seed(1)
  vals <- rnorm(100)
  df   <- .make_df("x1", vals)

  m <- MCPower$new("y ~ x1")$
    set_variable_type("x1=right_skewed")$
    set_effects("x1=0.5")$
    upload_data(df, verbose = FALSE)

  expect_no_error(.trigger_apply(m))

  # The declared distribution must remain right_skewed (not reset to normal).
  reg  <- m$.__enclos_env__$private$registry
  pred <- reg$get_predictor("x1")
  expect_equal(pred$var_type, "right_skewed")
})
