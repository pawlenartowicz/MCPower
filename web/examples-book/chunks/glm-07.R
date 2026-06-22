suppressMessages(library(mcpower))

# 2x2 factorial on a yes/no outcome: did the respondent vote? The question is
# whether the effect of `gender` on voting depends on `urban` residence.
# '*' expands gender * urban to gender + urban + gender:urban, so the
# interaction (the difference in cell differences, on the log-odds scale) is
# fitted explicitly. family="logit" makes voted binary (0/1) and fits a GLM.
model <- MCPower$new("voted ~ gender * urban", family = "logit")

# Both predictors are two-level factors: gender (e.g. women vs men) and
# urban (e.g. urban vs rural).
model$set_variable_type("gender=binary, urban=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80), each a shift
# in the log-odds of voting:
#   gender=0.50        -> medium main effect of gender.
#   urban=0.50         -> medium main effect of urban residence.
#   gender:urban=0.50  -> medium interaction (the moderation effect).
model$set_effects("gender=0.50, urban=0.50, gender:urban=0.50")

# Baseline voting rate in the reference cell (logit family needs a baseline probability).
model$set_baseline_probability(0.50)

model$set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=400.
invisible(model$find_power(sample_size = 400, target_test = "gender:urban"))
