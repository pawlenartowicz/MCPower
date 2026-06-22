suppressMessages(library(mcpower))

# Continuous outcome monthly_income on a binary group (union_member) plus a
# continuous covariate (experience_years) -- parallel slopes, no interaction.
# This is the ANCOVA/adjusted two-group comparison: the union_member effect is
# the group gap holding experience_years fixed.
model <- MCPower$new("monthly_income ~ union_member + experience_years")

# union_member is the binary grouping variable; experience_years stays
# continuous (the default).
model$set_variable_type("union_member=binary")

# Fabricated-plausible effects on the benchmark scale: a medium binary group
# gap (0.50) and a medium continuous experience_years slope (0.25).
model$set_effects("union_member=0.50, experience_years=0.25")

invisible(model$find_power(sample_size = 120, target_test = "all", verbose = FALSE))
