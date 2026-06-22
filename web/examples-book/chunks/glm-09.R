suppressMessages(library(mcpower))

# Covariate-adjusted logistic regression (parallel slopes on the log-odds).
# Research question: does years of work experience shift the probability of
# employment once we account for which region the respondent lives in?
# family = "logit" makes employed a binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("employed ~ experience_years + region", family = "logit")

# region is a categorical control with 3 levels -> 2 dummy contrasts.
model$set_variable_type("region=(factor,3)")

# Expected effects on the standardised benchmark scales.
#   experience_years=0.25   -> a medium continuous association with the log-odds.
#   region[2]/[3]           -> a medium factor effect for each non-reference region
#                              (effects are set per dummy contrast, not on the bare factor).
model$set_effects("experience_years=0.25, region[2]=0.50, region[3]=0.50")

# Logistic GLMs need a baseline event rate to anchor the intercept: at the
# reference region and average experience, 30% of respondents are employed.
model$set_baseline_probability(0.30)

# Power at N=250, targeting the adjusted experience effect (region held constant).
invisible(model$find_power(sample_size = 250, target_test = "experience_years"))
