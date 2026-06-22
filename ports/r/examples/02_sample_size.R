# Sample Size Calculation Example
# ===============================
#
# Find the minimum sample size that reaches your target power, and read the
# result object: short form (Required N per effect), long form (summary()),
# and the power-vs-N curve via plot().
#
# Run with:  Rscript 02_sample_size.R

suppressMessages(library(mcpower))

# 1. Define the model.
model <- MCPower$new("test_score ~ intervention + prior_knowledge + motivation")

# 2. Expected effects from literature / pilot data.
model$set_effects("intervention=0.4, prior_knowledge=0.35, motivation=0.3")

# 3. Variable types — intervention is binary (0=control, 1=intervention).
model$set_variable_type("intervention=binary")

# 4. Short form (printed automatically). find_sample_size() sweeps a grid and
#    reports the Required N column. The headline is the model-based fitted N
#    (isotonic crossing of the power curve, atom-ceiled to the cluster size)
#    when the fit succeeded; otherwise the grid's first_achieved value.
#    The long-form summary() adds a "Required N & 95% CI" table with Wilson
#    band-inversion bounds, rounded outward to integers.
cat(">>> model$find_sample_size(target_test = 'intervention', from_size = 30, to_size = 300, by = 10)\n")
invisible(model$find_sample_size(
  target_test = "intervention", from_size = 30, to_size = 300, by = 10
))

# 5. Long form for all effects, plus the power-vs-N curve.
cat("\n>>> result <- model$find_sample_size(target_test = 'all', from_size = 30, to_size = 300, by = 10, verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_sample_size(
  target_test = "all", from_size = 30, to_size = 300, by = 10, verbose = FALSE
)
print(summary(result))
if (requireNamespace("vegawidget", quietly = TRUE)) {
  cat("\n>>> plot(result)  # power-vs-N curves, Vega-Lite widget\n")
}

# 6. Robustness — the optimistic / realistic / doomer sweep (see 01) applied to
#    the sample-size search: the required N under each scenario.
cat("\n>>> model$find_sample_size(target_test = 'intervention', from_size = 30, to_size = 400, by = 20, scenarios = TRUE)\n")
invisible(model$find_sample_size(
  target_test = "intervention", from_size = 30, to_size = 400, by = 20,
  scenarios = TRUE
))

# 7. Higher power requirement (90%).
cat("\n>>> model$set_power(90)\n")
model$set_power(90)  # default is 80
cat(">>> model$find_sample_size(target_test = 'intervention', from_size = 50, to_size = 400, by = 20)\n")
invisible(model$find_sample_size(
  target_test = "intervention", from_size = 50, to_size = 400, by = 20
))
