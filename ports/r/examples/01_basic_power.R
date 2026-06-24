# Basic Power Analysis Example
# ============================
#
# Power calculation for a simple treatment study, and a tour of the result
# object returned by find_power(): short form, long form (summary()), the
# tidy tibble (as_tibble()), and the Vega-Lite plot (plot()).
#
# Run with:  Rscript 01_basic_power.R

library(mcpower)

# 1. Define the model with an R formula.
model <- MCPower$new("patient_outcome ~ treatment + baseline_score")

# 2. Expected effect sizes (standardised).
#    treatment=0.5      -> therapy shifts outcomes by 0.5 SD (a medium effect).
#    baseline_score=0.3 -> baseline moderately predicts the outcome.
model$set_effects("treatment=0.5, baseline_score=0.3")

# 3. Variable types — treatment is binary (0=control, 1=therapy).
model$set_variable_type("treatment=binary")

# 4. Short form (printed automatically). One row per effect with Power,
#    95% CI, and a check/cross marker against the target power.
cat(">>> model$find_power(sample_size = 120, target_test = 'treatment')\n")
invisible(model$find_power(sample_size = 120, target_test = "treatment"))

# 5. Long form via summary(). verbose=FALSE suppresses the auto short form;
#    target_test="all" adds the omnibus "Overall" row and the joint-
#    significance distribution.
cat("\n>>> result <- model$find_power(sample_size = 120, target_test = 'all', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(sample_size = 120, target_test = "all", verbose = FALSE)
print(summary(result))

# 6. Robustness — rerun the analysis under three assumption scenarios:
#      optimistic — all assumptions hold (what you computed above)
#      realistic  — how your data might actually look
#      doomer     — a pessimistic (but still plausible) version of your data
#    delta = power drop vs the optimistic baseline.
cat("\n>>> model$find_power(sample_size = 120, target_test = 'all', scenarios = TRUE)\n")
invisible(model$find_power(sample_size = 120, target_test = "all", scenarios = TRUE))

# 7. Programmatic access — as_tibble() gives a tidy (test x scenario) frame;
#    plot() renders the power-at-N chart as a Vega-Lite widget. Both optional
#    deps are in Suggests, so guard on availability.
if (requireNamespace("tibble", quietly = TRUE)) {
  cat("\n>>> tibble::as_tibble(result)\n")
  print(tibble::as_tibble(result))
}
if (requireNamespace("vegawidget", quietly = TRUE)) {
  cat("\n>>> plot(result)  # power-at-N Vega-Lite widget\n")
}
