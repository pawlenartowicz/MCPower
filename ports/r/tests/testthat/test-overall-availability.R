# Overall-test availability gate (Choice 1): the omnibus (OLS F / GLM LRT) is
# exposed only for OLS / unclustered GLM fits. Mixed-effects fits (LME,
# clustered GLMM) suppress it; an OLS fit on clustered data keeps the F-test.
# Mirrors the Python tests in tests/spec/test_set_tests.py.

# Build the LinearSpec payload list for a model, driving target_test and the
# estimator exactly as the find_power path does.
.payload_for <- function(m, target_test = NULL) {
  m$.__enclos_env__$private$apply()
  reg <- m$.__enclos_env__$private$registry
  mcpower:::.to_linear_spec_list(
    reg, "optimistic",
    alpha = 0.05, correction = NULL, target_test = target_test,
    heteroskedasticity = list(driver_var_index = 0L),
    residual_name = "normal", max_failed = 0.1, test_formula = NULL,
    scenario_configs = mcpower:::.scenario_defaults(),
    pending_data = NULL,
    cluster_level_vars = unlist(lapply(m$.__enclos_env__$private$pending_clusters, "[[", "cluster_level_vars"), use.names = FALSE),
    estimator = m$estimator
  )
}

.lme <- function() {
  MCPower$new("y ~ x + (1|group)", family = "lme")$
    set_effects("x=0.5")$set_cluster("group", ICC = 0.3, n_clusters = 20L)
}

.glmm <- function() {
  MCPower$new("y ~ x + (1|group)", family = "logit")$
    set_baseline_probability(0.3)$set_effects("x=0.5")$
    set_cluster("group", ICC = 0.3, n_clusters = 20L)
}

.clustered_ols <- function() {
  MCPower$new("y ~ x + (1|group)", family = "lme", estimator = "ols")$
    set_effects("x=0.5")$set_cluster("group", ICC = 0.3, n_clusters = 20L)
}

test_that(".overall_test_available follows the estimator/clustering rule", {
  reg_ols <- {
    m <- MCPower$new("y = x"); m$set_effects("x=0.5")
    m$.__enclos_env__$private$apply(); m$.__enclos_env__$private$registry
  }
  reg_lme <- { m <- .lme(); m$.__enclos_env__$private$apply(); m$.__enclos_env__$private$registry }
  expect_true(mcpower:::.overall_test_available("ols", reg_ols))
  expect_true(mcpower:::.overall_test_available("glm", reg_ols))         # unclustered GLM
  expect_false(mcpower:::.overall_test_available("glm", reg_lme))        # clustered → GLMM
  expect_false(mcpower:::.overall_test_available("mle", reg_lme))        # LME
})

test_that("LME family default suppresses the omnibus but keeps marginals", {
  p <- .payload_for(.lme())
  expect_false(p$report_overall)
  expect_equal(p$targets, list("overall"))  # sentinel = every β, no omnibus
})

test_that("clustered-logistic GLMM default suppresses the omnibus", {
  expect_false(.payload_for(.glmm())$report_overall)
})

test_that("explicit target_test='overall' on a mixed model errors", {
  expect_error(.payload_for(.lme(), target_test = "overall"),
               regexp = "not available for mixed-effects")
})

test_that("the 'y'/dep-var alias for the omnibus is also rejected on a mixed model", {
  expect_error(.payload_for(.lme(), target_test = "y"),
               regexp = "not available for mixed-effects")
})

test_that("'all' on a mixed model drops the omnibus and keeps fixed-effect betas", {
  p <- .payload_for(.glmm(), target_test = "all")
  expect_false(p$report_overall)
  expect_true("x" %in% unlist(p$targets))
})

test_that("an OLS fit on clustered data keeps the F-test", {
  expect_true(.payload_for(.clustered_ols())$report_overall)
  expect_true(.payload_for(.clustered_ols(), target_test = "overall")$report_overall)
})
