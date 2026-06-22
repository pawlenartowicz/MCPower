suppressMessages(library(mcpower))

# Covariate-adjusted logistic association: does `years_education` predict whether a
# person is employed (yes/no), holding age and gender constant?
# family="logit" makes the outcome binary — the engine fits a GLM (logit link)
# on every Monte Carlo iteration.
model <- MCPower$new("employed ~ years_education + age + gender", family = "logit")

# years_education and age are continuous (default); gender is a binary 0/1 predictor.
model$set_variable_type("gender=binary")

# Baseline probability — the employment rate when every predictor is at its
# reference value. Required for logit; it sets the model intercept via
# log(p / (1 - p)). Here 25% of reference individuals are employed.
model$set_baseline_probability(0.25)

# Standardised effects (log-odds shifts).
#   years_education=0.25 -> the key predictor, a medium adjusted association (continuous benchmark).
#   age=0.20             -> a continuous nuisance covariate carried along for adjustment.
#   gender=0.50          -> a binary covariate, medium on the binary benchmark scale.
model$set_effects("years_education=0.25, age=0.20, gender=0.50")

# Age correlates with years of education — that confounding is the reason we adjust,
# and it changes the power for the years_education coefficient. Correlations are only
# defined between continuous variables, so gender (binary) cannot be correlated
# here; it enters from its own marginal.
model$set_correlations("corr(years_education, age)=0.3")

model$find_power(sample_size = 400, target_test = "years_education")
