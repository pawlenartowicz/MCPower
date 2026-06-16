# test-posthoc.R — testthat cases for ANOVA post-hoc (Tukey HSD) feature.
# Mirrors Python tests in test_posthoc.py, test_set_tests.py, test_correction_handling.py.

# ---------------------------------------------------------------------------
# Helper: integer-indexed 3-level factor model (mirrors Python simple_anova_model)
# ---------------------------------------------------------------------------
.simple_anova_model <- function() {
  MCPower$new("pain_reduction = dose_group")$
    set_variable_type("dose_group=(factor,0.34,0.33,0.33)")$
    set_effects("dose_group[2]=0.5, dose_group[3]=0.8")$
    set_simulations(200)
}

# ---------------------------------------------------------------------------
# Helper: named-level 3-level factor model (mirrors Python _named_three_level_factor_model)
# ---------------------------------------------------------------------------
.named_three_level_factor_model <- function() {
  m <- MCPower$new("y = baseline_pain + treatment")
  m$set_variable_type("treatment=(factor,0.33,0.33,0.34)")
  m$set_effects("baseline_pain=0.3")
  m$set_simulations(200)
  # Force apply so registry state is initialised before we patch it.
  m$.__enclos_env__$private$apply()

  # Patch level_labels and reference_level on the registry factor.
  m$.__enclos_env__$private$registry$`_factors`[["treatment"]]$level_labels <-
    list("placebo", "low_dose", "high_dose")
  m$.__enclos_env__$private$registry$`_factors`[["treatment"]]$reference_level <-
    "placebo"

  # Rename the integer-indexed dummy effects (treatment[2], treatment[3])
  # to named-level effects (treatment[low_dose], treatment[high_dose]) so the
  # Rust spec builder accepts them in find_power.
  reg  <- m$.__enclos_env__$private$registry
  priv <- reg$.__enclos_env__$private
  old_names  <- c("treatment[2]", "treatment[3]")
  new_labels <- c("low_dose", "high_dose")
  for (i in seq_along(old_names)) {
    old_nm  <- old_names[[i]]
    new_nm  <- sprintf("treatment[%s]", new_labels[[i]])
    # Rename in private$effects.
    eff <- priv$effects[[old_nm]]
    if (!is.null(eff)) {
      eff$name <- new_nm
      priv$effects[[new_nm]] <- eff
      priv$effects[[old_nm]] <- NULL
    }
    # Rename in private$predictors.
    pred <- priv$predictors[[old_nm]]
    if (!is.null(pred)) {
      pred$name <- new_nm
      priv$predictors[[new_nm]] <- pred
      priv$predictors[[old_nm]] <- NULL
    }
  }
  # Mark as applied so find_power does not re-run apply() (which would undo our patches).
  m$.__enclos_env__$private$applied <- TRUE
  m
}

# ---------------------------------------------------------------------------
# TASK 6.1 — DSL: .resolve_tests all-contrasts / all-posthoc
# ---------------------------------------------------------------------------

test_that(".resolve_tests: all-contrasts produces posthoc_factors, empty targets/contrast_pairs", {
  reg <- mcpower:::RVariableRegistry$new("pain_reduction = dose_group")
  reg$set_variable_types("dose_group=(factor,3)")
  reg$expand_factors()
  reg$set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

  resolved <- mcpower:::.resolve_tests(reg, "all-contrasts")
  expect_equal(resolved$targets, character(0))
  expect_equal(resolved$contrast_pairs, list())
  expect_false(resolved$report_overall)
  expect_equal(resolved$posthoc_factors, "dose_group")
})

test_that(".resolve_tests: all-posthoc is alias for all-contrasts", {
  reg <- mcpower:::RVariableRegistry$new("pain_reduction = dose_group")
  reg$set_variable_types("dose_group=(factor,3)")
  reg$expand_factors()
  reg$set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

  r1 <- mcpower:::.resolve_tests(reg, "all-contrasts")
  r2 <- mcpower:::.resolve_tests(reg, "all-posthoc")
  expect_equal(r1$posthoc_factors, r2$posthoc_factors)
})

test_that(".resolve_tests: overall, all-contrasts leaves targets empty + report_overall=TRUE", {
  reg <- mcpower:::RVariableRegistry$new("pain_reduction = dose_group")
  reg$set_variable_types("dose_group=(factor,3)")
  reg$expand_factors()
  reg$set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

  resolved <- mcpower:::.resolve_tests(reg, "overall, all-contrasts")
  expect_equal(resolved$targets, character(0))
  expect_equal(resolved$contrast_pairs, list())
  expect_true(resolved$report_overall)
  expect_equal(resolved$posthoc_factors, "dose_group")
})

test_that(".resolve_tests: two-factor model produces two posthoc_factors", {
  reg <- mcpower:::RVariableRegistry$new("y = f1 + f2")
  reg$set_variable_types("f1=(factor,3), f2=(factor,2)")
  reg$expand_factors()
  reg$set_effects("f1[2]=0.3, f1[3]=0.5, f2[2]=0.2")

  resolved <- mcpower:::.resolve_tests(reg, "all-contrasts")
  expect_equal(length(resolved$posthoc_factors), 2L)
  expect_true("f1" %in% resolved$posthoc_factors)
  expect_true("f2" %in% resolved$posthoc_factors)
  expect_equal(resolved$targets, character(0))
  expect_equal(resolved$contrast_pairs, list())
})

test_that(".resolve_tests: all-contrasts with no factors stops with informative error", {
  reg <- mcpower:::RVariableRegistry$new("y = x1")
  reg$set_effects("x1=0.5")

  expect_error(
    mcpower:::.resolve_tests(reg, "all-contrasts"),
    "all-contrasts.*no factor|no factor.*all-contrasts",
    perl = TRUE
  )
})

# ---------------------------------------------------------------------------
# TASK 6.1 — .to_linear_spec_list emits posthoc_requests
# ---------------------------------------------------------------------------

test_that(".to_linear_spec_list emits posthoc_requests list for all-contrasts", {
  reg <- mcpower:::RVariableRegistry$new("pain_reduction = dose_group")
  reg$set_variable_types("dose_group=(factor,3)")
  reg$expand_factors()
  reg$set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

  payload <- mcpower:::.to_linear_spec_list(
    reg, scenario_names = "optimistic",
    alpha = 0.05, correction = NULL, target_test = "overall, all-contrasts",
    heteroskedasticity = list(driver_var_index = NULL),
    residual_name = "normal", max_failed = 1.0,
    test_formula = NULL)

  expect_true("posthoc_requests" %in% names(payload))
  expect_equal(length(payload$posthoc_requests), 1L)
  expect_equal(payload$posthoc_requests[[1]]$factor, "dose_group")
  # targets must be empty (engine handles n_targets==0 natively)
  expect_equal(payload$targets, list())
})

test_that(".to_linear_spec_list: posthoc_requests is empty list when no posthoc requested", {
  reg <- mcpower:::RVariableRegistry$new("y = x1")
  reg$set_effects("x1=0.5")

  payload <- mcpower:::.to_linear_spec_list(
    reg, scenario_names = "optimistic",
    alpha = 0.05, correction = NULL, target_test = NULL,
    heteroskedasticity = list(driver_var_index = NULL),
    residual_name = "normal", max_failed = 1.0,
    test_formula = NULL)

  expect_true("posthoc_requests" %in% names(payload))
  expect_equal(payload$posthoc_requests, list())
})

# ---------------------------------------------------------------------------
# TASK 6.1 — Correction alias: tukey / tukey_hsd map to "tukey_hsd"
# ---------------------------------------------------------------------------

test_that(".correction_for_rust maps tukey and tukey_hsd to tukey_hsd", {
  expect_equal(mcpower:::.correction_for_rust("tukey"),     "tukey_hsd")
  expect_equal(mcpower:::.correction_for_rust("tukey_hsd"), "tukey_hsd")
  expect_equal(mcpower:::.correction_for_rust("Tukey"),     "tukey_hsd")
  expect_equal(mcpower:::.correction_for_rust("TUKEY_HSD"), "tukey_hsd")
})

# ---------------------------------------------------------------------------
# TASK 6.1 — validate_correction_arg no longer rejects tukey
# ---------------------------------------------------------------------------

test_that("validate_correction_arg accepts tukey without error", {
  m <- .simple_anova_model()
  # Should not throw.
  expect_silent(m$.__enclos_env__$private$validate_correction_arg("tukey"))
})

# ---------------------------------------------------------------------------
# TASK 6.1 — D4 rule: Tukey + marginal beta targets warns, not errors
# ---------------------------------------------------------------------------

test_that("find_power: tukey + marginal beta targets warns (D4 rule)", {
  m <- .simple_anova_model()
  expect_warning(
    m$find_power(sample_size = 200,
                 target_test = "all, all-contrasts",
                 correction = "tukey",
                 verbose = FALSE, progress_callback = FALSE),
    "Tukey HSD"
  )
})

test_that("find_power: tukey + posthoc-only does NOT warn", {
  m <- .simple_anova_model()
  expect_no_warning(
    m$find_power(sample_size = 200,
                 target_test = "overall, all-contrasts",
                 correction = "tukey",
                 verbose = FALSE, progress_callback = FALSE)
  )
})

# ---------------------------------------------------------------------------
# TASK 6.1 — Engine surfaces posthoc block (end-to-end)
# ---------------------------------------------------------------------------

test_that("find_power with all-contrasts returns posthoc block with n_levels=3", {
  m <- .simple_anova_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "overall, all-contrasts",
                      verbose = FALSE, progress_callback = FALSE)
  ph <- res[["posthoc"]]
  expect_true(!is.null(ph) && length(ph) >= 1L)
  expect_equal(ph[[1]]$n_levels, 3L)
  expect_equal(length(ph[[1]]$power_uncorrected), 3L)  # C(3,2) = 3 pairs
})

test_that("tukey correction: corrected power <= uncorrected power per contrast", {
  m <- .simple_anova_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "overall, all-contrasts",
                      correction = "tukey",
                      verbose = FALSE, progress_callback = FALSE)
  ph <- res[["posthoc"]][[1]]
  for (i in seq_along(ph$power_uncorrected)) {
    expect_lte(ph$power_corrected[[i]], ph$power_uncorrected[[i]] + 1e-9)
  }
})

test_that("pure all-contrasts (no overall): posthoc block present, main power empty", {
  m <- .simple_anova_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "all-contrasts",
                      verbose = FALSE, progress_callback = FALSE)
  ph <- res[["posthoc"]]
  expect_true(!is.null(ph) && length(ph) >= 1L)
  expect_equal(ph[[1]]$n_levels, 3L)
  pu <- res[["power_uncorrected"]]
  # No marginal beta targets: main power_uncorrected should be empty
  expect_true(is.null(pu) || length(pu) == 0L || length(pu[[1]]) == 0L)
})

# ---------------------------------------------------------------------------
# TASK 6.2 — .build_report_meta carries posthoc_factors
# ---------------------------------------------------------------------------

test_that(".build_report_meta adds posthoc_factors with correct levels", {
  m <- .named_three_level_factor_model()
  meta <- m$.__enclos_env__$private$.build_report_meta(
    "tukey", posthoc_factors = c("treatment"))
  expect_true("posthoc_factors" %in% names(meta))
  pf <- meta$posthoc_factors
  expect_equal(length(pf), 1L)
  expect_equal(pf[[1]]$name, "treatment")
  expect_equal(pf[[1]]$levels, c("placebo", "low_dose", "high_dose"))
})

test_that(".build_report_meta synthesises integer labels when no level_labels", {
  m <- .simple_anova_model()
  m$.__enclos_env__$private$apply()
  meta <- m$.__enclos_env__$private$.build_report_meta(
    NULL, posthoc_factors = c("dose_group"))
  pf <- meta$posthoc_factors
  expect_equal(length(pf), 1L)
  expect_equal(pf[[1]]$name, "dose_group")
  expect_equal(pf[[1]]$levels, c("1", "2", "3"))
})

test_that(".build_report_meta: empty posthoc_factors when not requested", {
  m <- .simple_anova_model()
  m$.__enclos_env__$private$apply()
  meta <- m$.__enclos_env__$private$.build_report_meta(NULL)
  expect_equal(meta$posthoc_factors, list())
})

# ---------------------------------------------------------------------------
# TASK 6.2 — Rendering: short-form print (mirror Python test_short_form_renders_posthoc_factor_grouped)
# ---------------------------------------------------------------------------

test_that("print.mcpower_result: short form nests posthoc contrasts under a (pairwise) header", {
  m <- .named_three_level_factor_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "overall, all-contrasts",
                      correction = "tukey",
                      verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")

  expect_match(out, "treatment  (pairwise)", fixed = TRUE)
  expect_match(out, "low_dose vs placebo")
  expect_match(out, "high_dose vs low_dose")
  # Tukey correction -> the main table carries both columns
  expect_match(out, "uncorrected")
  expect_match(out, "corrected")
  # No standalone post-hoc section any more
  expect_false(grepl("Post-hoc pairwise contrasts", out))
  expect_false(grepl("factor: treatment", out))
})

test_that("print.mcpower_result: no correction -> single Power column for nested contrasts", {
  m <- .simple_anova_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "all-contrasts",
                      verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "dose_group  (pairwise)", fixed = TRUE)
  # Integer-indexed: expect canonical pair labels
  expect_match(out, "2 vs 1")
  expect_match(out, "3 vs 1")
  expect_match(out, "3 vs 2")
  # No correction -> no uncorrected/corrected split
  expect_false(grepl("uncorrected", out))
  expect_false(grepl("Post-hoc pairwise contrasts", out))
})

test_that("print.mcpower_report: long form nests posthoc contrasts in the per-test table", {
  m <- .named_three_level_factor_model()
  res <- m$find_power(sample_size = 200,
                      target_test = "overall, all-contrasts",
                      correction = "tukey",
                      verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(summary(res))), collapse = "\n")
  expect_match(out, "treatment  (pairwise)", fixed = TRUE)
  expect_match(out, "low_dose vs placebo")
  expect_match(out, "high_dose vs low_dose")
  expect_false(grepl("Post-hoc pairwise contrasts", out))
})

# ---------------------------------------------------------------------------
# Item #8 — Posthoc joint-section sim-count text (regression guard)
# ---------------------------------------------------------------------------

test_that("summary(posthoc result): joint section renders n_sims correctly", {
  # Uses .simple_anova_model() helper (defined above in this file).
  m <- .simple_anova_model()$set_simulations(200)
  res <- m$find_power(sample_size = 200,
                      target_test = "overall, all-contrasts",
                      verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(summary(res))), collapse = "\n")
  # Joint section header must be present (not "unavailable" fallback).
  expect_match(out, "Joint significance distribution")
  expect_false(grepl("unavailable", out))
  # CI footnote carries n_sims=200 — proves inner0$n_sims resolved correctly.
  expect_match(out, "n_sims=200")
  # Joint distribution renders as a table now (minimal-table format, mirrors
  # Python): the "Exactly | At least" column headers sit adjacent in the header.
  expect_match(out, "Exactly\\s+At least")
})
