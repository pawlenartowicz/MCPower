suppressMessages(library(mcpower))

# Two-group comparison on a binary (remission / no-remission) outcome: logistic regression.
model <- MCPower$new("remission ~ treatment", family = "logit")

# treatment is a binary two-level predictor (0 = control, 1 = active treatment).
model$set_variable_type("treatment=binary")

# Expected group effect on the binary benchmark scale: 0.50 = a medium gap.
model$set_effects("treatment=0.50")

# Remission rate in the control group (logit family needs a baseline probability).
model$set_baseline_probability(0.20)

invisible(model$find_power(sample_size = 200, target_test = "treatment"))
