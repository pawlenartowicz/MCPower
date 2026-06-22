suppressMessages(library(mcpower))

# 2x2 factorial: does the effect of `gender` depend on `sector`? '*' expands
# gender * sector to gender + sector + gender:sector, so the interaction
# (the difference in cell differences) is fitted explicitly.
model <- MCPower$new("job_satisfaction ~ gender * sector")

# Both predictors are two-level factors: gender and sector.
model$set_variable_type("gender=binary, sector=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   gender=0.50          -> medium main effect of gender.
#   sector=0.50          -> medium main effect of sector.
#   gender:sector=0.50   -> medium interaction (the moderation effect).
model$set_effects("gender=0.50, sector=0.50, gender:sector=0.50")

model$set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=200.
invisible(model$find_power(sample_size = 200, target_test = "gender:sector"))
