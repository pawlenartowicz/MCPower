suppressMessages(library(mcpower))

# Logistic regression with a treatment-by-covariate interaction: does the
# treatment's effect on the odds of remission depend on how elevated the patient's
# biomarker was at baseline? '*' expands treatment * biomarker_level to
# treatment + biomarker_level + treatment:biomarker_level, so the
# moderation term is fitted explicitly. family = "logit" makes remission a
# binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("remission ~ treatment * biomarker_level", family = "logit")

# treatment is a two-level arm (0=control, 1=treatment); biomarker_level is a
# continuous covariate.
model$set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50                     -> medium arm shift (binary benchmark).
#   biomarker_level=0.40               -> strong covariate-outcome association
#                                         on the log-odds (continuous benchmark).
#   treatment:biomarker_level=0.25     -> moderate moderation (the interaction).
model$set_effects("treatment=0.50, biomarker_level=0.40, treatment:biomarker_level=0.25")

# A logistic GLM needs a baseline event rate to anchor the intercept: 30% of
# control patients reach remission when every predictor is at its reference
# value. family = "logit" requires this before find_power.
model$set_baseline_probability(0.30)

model$set_seed(2137)

# Power for the moderation test (the interaction) at N=300.
invisible(model$find_power(sample_size = 300, target_test = "treatment:biomarker_level"))
