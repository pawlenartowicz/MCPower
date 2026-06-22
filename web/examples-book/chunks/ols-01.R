suppressMessages(library(mcpower))

# Simple linear regression: one continuous outcome, one continuous predictor.
model <- MCPower$new("wage ~ years_education")

# Expected effect on the standardised benchmark scale:
#   years_education=0.25 -> a medium association between years_education and wage.
model$set_effects("years_education=0.25")

# Power at N=150 with the OLS defaults (1600 sims, alpha=0.05, seed=2137).
invisible(model$find_power(sample_size = 150, target_test = "years_education"))
