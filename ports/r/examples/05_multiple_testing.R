# Multiple Testing Correction Example
# ===================================
#
# Controlling Type-I error when testing many hypotheses at once. This is the
# natural home for the joint-significance distribution: "how many of my
# effects will I detect?" is exactly the question multiple-testing raises.
#
# Run with:  Rscript 05_multiple_testing.R

suppressMessages(library(mcpower))

# 1. Five predictors — two are true nulls (effect = 0).
model <- MCPower$new(
  "treatment_response ~ biomarker1 + biomarker2 + biomarker3 + biomarker4 + age"
)
model$set_effects(
  "biomarker1=0.3, biomarker2=0.0, biomarker3=0.25, biomarker4=0.0, age=0.2"
)

# 2. No correction vs corrected — summary() shows the joint distribution
#    ("exactly k" / "at least k" significant), the headline number for a
#    screening study: how many discoveries should you expect?
cat(">>> uncorrected <- model$find_power(sample_size = 140, target_test = 'all', correction = NULL, verbose = FALSE)\n")
cat(">>> print(summary(uncorrected))\n")
uncorrected <- model$find_power(
  sample_size = 140, target_test = "all", correction = NULL, verbose = FALSE
)
print(summary(uncorrected))

cat("\n>>> bh <- model$find_power(sample_size = 140, target_test = 'all', correction = 'bh', verbose = FALSE)\n")
cat(">>> print(summary(bh))\n")
bh <- model$find_power(
  sample_size = 140, target_test = "all", correction = "bh", verbose = FALSE
)
print(summary(bh))

# 3. Side-by-side per-test power across correction methods.
methods <- list(none = NULL, bonferroni = "bonferroni", holm = "holm", bh = "bh")
if (requireNamespace("tibble", quietly = TRUE)) {
  acc <- NULL
  for (label in names(methods)) {
    res <- model$find_power(
      sample_size = 140, target_test = "all",
      correction = methods[[label]], verbose = FALSE
    )
    col <- tibble::as_tibble(res)[, c("test", "power")]
    names(col)[2] <- label
    acc <- if (is.null(acc)) col else merge(acc, col, by = "test")
  }
  cat("\n>>> acc\n")
  print(acc)
}

# 4. Sample size to reach target power under a correction.
cat("\n>>> model$find_sample_size(target_test = 'all', from_size = 50, to_size = 400, by = 20, correction = 'bonferroni')\n")
invisible(model$find_sample_size(
  target_test = "all", from_size = 50, to_size = 400, by = 20,
  correction = "bonferroni"
))

# 5. Focused, pre-registered hypotheses only.
cat("\n>>> model$find_power(sample_size = 140, target_test = 'biomarker1, biomarker3', correction = 'bonferroni')\n")
invisible(model$find_power(
  sample_size = 140, target_test = "biomarker1, biomarker3",
  correction = "bonferroni"
))
