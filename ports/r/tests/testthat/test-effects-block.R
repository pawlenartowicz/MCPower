# The effects block (input β, standardized β, Cohen's d, OR/RR, model R²):
# wiring from the engine and rendering. The numbers themselves are checked in
# engine-core (test_standardized_effects.rs). Mirrors
# ports/py/tests/output/test_effects_block.py.

.run_fp <- function(m, ...) {
  m$find_power(sample_size = 100, n_sims = 20L, verbose = FALSE, progress_callback = FALSE, ...)
}

# The effects block lines: from the "Effect" header to the R² line, inclusive.
.block <- function(lines) lines[grep("^Effect ", lines)[1]:grep("^R²", lines)[1]]

test_that("values reach the result and both printed forms", {
  # One normal predictor: std β = β / √(β² + 1), R² = β² / (β² + 1).
  res <- .run_fp(MCPower$new("y ~ x1")$set_effects("x1=0.5"))
  se <- res$standardized_effects
  expect_equal(se$effects$x1$std_beta, 0.5 / sqrt(1.25), tolerance = 1e-12)
  expect_null(se$effects$x1$cohen_d)
  expect_equal(se$r2, 0.25 / 1.25, tolerance = 1e-12)
  for (out in list(capture.output(print(res)), capture.output(print(summary(res))))) {
    block <- .block(out)
    expect_true(any(grepl("^x1 +0\\.50 +0\\.447 +—$", block)))
    expect_true(any(grepl("^R² = 0\\.200$", block)))
  }
})

test_that("binary and factor rows carry d", {
  m <- MCPower$new("y ~ treat + group")
  m$set_variable_type("treat=binary, group=(factor,3)")
  m$set_effects("treat=0.5, group[2]=0.2, group[3]=0.4")
  se <- .run_fp(m)$standardized_effects$effects
  expect_false(is.null(se$treat$cohen_d))
  expect_false(is.null(se$`group[2]`$cohen_d))
})

test_that("a test_formula run still lists every simulated effect", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.3, x2=0.4")
  res <- .run_fp(m, test_formula = "y ~ x2", target_test = "x2")
  expect_setequal(names(res$standardized_effects$effects), c("x1", "x2"))
  expect_true(any(grepl("^x1 +0\\.30 ", .block(capture.output(print(res))))))
})

test_that("the OR prints only on the logit link", {
  logit <- .run_fp(MCPower$new("y ~ x1", family = "logit")$set_effects("x1=0.5")$
                     set_baseline_probability(0.3))
  probit <- .run_fp(MCPower$new("y ~ x1", family = "probit")$set_effects("x1=0.5")$
                      set_baseline_probability(0.3))
  expect_true(any(grepl("^Effect +Input +Std\\. β +d +OR$", .block(capture.output(print(logit))))))
  for (out in list(capture.output(print(probit)), capture.output(print(summary(probit))))) {
    expect_false("OR" %in% unlist(strsplit(out, " +")))
  }
})

test_that("a count outcome prints RR and dashes", {
  res <- .run_fp(MCPower$new("y ~ x1", family = "poisson")$set_effects("x1=0.3")$
                   set_baseline_rate(2.0))
  block <- .block(capture.output(print(res)))
  expect_true(any(grepl("^Effect +Input +Std\\. β +d +RR$", block)))
  expect_true(any(grepl("^x1 +0\\.30 +— +— +1\\.35$", block)))
  expect_true(any(grepl("^R² = —$", block)))
  expect_null(res$standardized_effects$r2)
})

test_that("a clustered run labels the fixed-effects R²", {
  m <- MCPower$new("y ~ x1 + (1|grp)")$set_effects("x1=0.5")$
    set_cluster("grp", ICC = 0.2, n_clusters = 10)
  block <- .block(capture.output(print(.run_fp(m))))
  expect_true(any(grepl("^R² \\(fixed effects\\) = 0\\.[0-9]{3}$", block)))
})
