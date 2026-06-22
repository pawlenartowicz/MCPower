# test-factor-sampling.R — R mirror of Python test_factor_sampling.py.
#
# Tests that sampled_proportions flows from set_variable_type through to the
# wire predictor dict emitted by .to_linear_spec_list.

# Helper: call .to_linear_spec_list on a minimal "y ~ g" registry; returns the
# factor predictor entry.
.factor_wire_entry <- function(reg) {
  spec <- mcpower:::.to_linear_spec_list(
    reg, "optimistic",
    alpha = 0.05, correction = NULL, target_test = NULL,
    heteroskedasticity = list(driver_var_index = NULL),
    residual_name = "normal", max_failed = 1.0,
    test_formula = NULL
  )
  pred_list <- spec$predictors
  for (p in pred_list) {
    if (p$name == "g") return(p)
  }
  stop("predictor 'g' not found in wire spec")
}

test_that("sampled_proportions = TRUE reaches wire predictor", {
  reg <- mcpower:::RVariableRegistry$new("y ~ g")
  reg$set_variable_type("g", "factor", n_levels = 2L, proportions = list(0.5, 0.5),
                        sampled_proportions = TRUE)
  entry <- .factor_wire_entry(reg)
  expect_true(isTRUE(entry$sampled_proportions))
})

test_that("sampled_proportions = FALSE flows through to wire predictor", {
  reg <- mcpower:::RVariableRegistry$new("y ~ g")
  reg$set_variable_type("g", "factor", n_levels = 2L, proportions = list(0.5, 0.5),
                        sampled_proportions = FALSE)
  entry <- .factor_wire_entry(reg)
  expect_false(isTRUE(entry$sampled_proportions))
  # Must be explicitly present (not merely absent).
  expect_true("sampled_proportions" %in% names(entry))
})

test_that("sampled_proportions omitted from wire when not set (inherit/NULL)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ g")
  reg$set_variable_type("g", "factor", n_levels = 2L, proportions = list(0.5, 0.5))
  entry <- .factor_wire_entry(reg)
  expect_false("sampled_proportions" %in% names(entry))
})
