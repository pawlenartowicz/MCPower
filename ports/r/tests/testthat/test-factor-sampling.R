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

# Public set_factor_sampling: the flags a model's registry holds after one
# apply(), as list(name = flag); NULL = no override.
.applied_flags <- function(m) {
  priv <- m$.__enclos_env__$private
  priv$apply()
  lapply(priv$registry$`_factors`, `[[`, "sampled_proportions")
}

.two_factor_model <- function() {
  MCPower$new("y = g + h")$
    set_variable_type("g=(factor,0.2,0.3,0.5), h=(factor,3)")$
    set_effects("g[2]=0.3, g[3]=0.4, h[2]=0.2, h[3]=0.2")
}

test_that("set_factor_sampling modes reach the factor's wire entry", {
  expect_identical(.applied_flags(.two_factor_model()), list(g = NULL, h = NULL))
  m <- .two_factor_model()$set_factor_sampling("g", "exact")$set_factor_sampling("h", "sampled")
  expect_identical(.applied_flags(m), list(g = FALSE, h = TRUE))
  entry <- .factor_wire_entry(m$.__enclos_env__$private$registry)
  expect_identical(entry$sampled_proportions, FALSE)
  # NULL clears the override: the factor inherits the scenario again.
  m <- .two_factor_model()$set_factor_sampling("g", "exact")$set_factor_sampling("g", NULL)
  expect_identical(.applied_flags(m), list(g = NULL, h = NULL))
})

test_that("set_factor_sampling survives the upload re-apply", {
  # The upload re-apply rebuilds an uploaded factor from its data; the
  # override must still be on the rebuilt entry.
  m <- MCPower$new("y = breed")
  m$upload_data(data.frame(breed = rep(c("cat", "dog", "bird"), 10)), verbose = FALSE)
  m$set_effects("breed[cat]=0.3, breed[dog]=0.3")
  m$set_factor_sampling("breed", "sampled")
  expect_identical(.applied_flags(m), list(breed = TRUE))
})

test_that("set_factor_sampling rejects a bad mode and a non-factor", {
  m <- MCPower$new("y = g + x")$set_variable_type("g=(factor,3)")$
    set_effects("g[2]=0.3, g[3]=0.3, x=0.2")
  for (bad in list("Exact", "random", TRUE, c("exact", "sampled"))) {
    expect_error(m$set_factor_sampling("g", bad), "'exact', 'sampled' or NULL")
  }
  m$set_factor_sampling("x", "exact")
  expect_error(m$.__enclos_env__$private$apply(), "'x' is not a factor predictor")
})
