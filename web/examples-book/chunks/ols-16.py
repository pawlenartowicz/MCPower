from mcpower import MCPower

# Continuous-by-continuous moderation with an additive control variable: does the
# effect of dose on recovery_days depend on age, after adjusting for baseline_severity?
# '*' expands dose * age to dose + age + dose:age; '+ baseline_severity' adds the control
# term only (no interaction with it). The full model is dose + age + dose:age + baseline_severity.
model = MCPower("recovery_days = dose * age + baseline_severity")

# Standardised effects (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   dose=0.30, age=0.25    -> moderate main effects.
#   dose:age=0.15          -> the smaller moderation effect (the test of interest).
#   baseline_severity=0.25 -> a moderate control association we adjust for.
model.set_effects("dose=0.30, age=0.25, dose:age=0.15, baseline_severity=0.25")

# Power for the interaction term at N=220.
model.find_power(sample_size=220, target_test="dose:age")
