# Custom Scenario Configurations
# ==============================
#
# Tuning the scenario sweep. set_scenario_configs() overrides the built-in
# optimistic / realistic / doomer presets and defines custom assumption sets
# that find_power() then runs side by side.
#
# Run with:  Rscript 10_custom_scenarios.R

suppressMessages(library(mcpower))

# Example: Field experiment whose real-world conditions are messier than the
# clean simulation defaults. Scenarios stress-test a power estimate against
# assumption violations.

# 1. A simple model with a binary treatment and a continuous covariate.
model <- MCPower$new("outcome ~ treatment + experience")
model$set_effects("treatment=0.4, experience=0.3")
model$set_variable_type("treatment=binary")

# 2. The built-in sweep (toured in example 01): optimistic / realistic / doomer.
cat(">>> model$find_power(sample_size = 150, target_test = 'all', scenarios = TRUE)\n")
invisible(model$find_power(sample_size = 150, target_test = "all", scenarios = TRUE))

# 3. Customise the sweep. Each scenario is a bundle of assumption-violation
#    knobs:
#      heterogeneity            — variation in true effects across units
#      heteroskedasticity_ratio — residual-variance ratio (1.0 = none)
#      correlation_noise_sd     — jitter applied to the correlation structure
#      distribution_change_prob — chance a predictor is redrawn non-normal
#    Naming an existing preset updates only the keys you pass; a brand-new name
#    inherits every key from "optimistic" and applies your overrides on top.
model$set_scenario_configs(list(
  # Make "realistic" a little harsher than its default.
  realistic = list(heteroskedasticity_ratio = 3.0, correlation_noise_sd = 0.20),
  # A bespoke worst-case bundle for this study.
  stress_test = list(
    heterogeneity = 0.5,
    heteroskedasticity_ratio = 5.0,
    correlation_noise_sd = 0.35,
    distribution_change_prob = 0.9
  )
))

# 4. Run a chosen set of scenarios by name (including the custom one). Pass a
#    character vector to scenarios= to control exactly which appear, and in what
#    order. Long form via summary() adds the "Robustness" table — each scenario's
#    delta power against the optimistic baseline — and the joint distribution.
cat("\n>>> result <- model$find_power(sample_size = 150, target_test = 'all',\n")
cat("...   scenarios = c('optimistic', 'realistic', 'stress_test'), verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(
  sample_size = 150, target_test = "all",
  scenarios = c("optimistic", "realistic", "stress_test"), verbose = FALSE
)
print(summary(result))
