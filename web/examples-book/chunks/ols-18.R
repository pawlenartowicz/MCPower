suppressMessages(library(mcpower))

# Linear dose-response: one continuous outcome regressed on an ordered dose level.
# The dose_level values (0, 1, 2, 3, ...) are read as a single continuous predictor,
# so the test is one slope -- a linear-by-level trend across the dose.
model <- MCPower$new("tumor_shrinkage ~ dose_level")

# Expected effect on the standardised benchmark scale:
#   dose_level=0.25 -> a medium linear trend of the outcome across the dose levels.
model$set_effects("dose_level=0.25")

# Power at N=150 with the OLS defaults (1600 sims, alpha=0.05, seed=2137).
invisible(model$find_power(sample_size = 150, target_test = "dose_level"))
