from mcpower import MCPower

# Two-group mean comparison: does pain_score differ between the two treatment groups?
model = MCPower("pain_score = treatment")

# treatment is a binary two-level predictor (0 = control, 1 = treatment).
model.set_variable_type("treatment=binary")

# Expected effect on the binary benchmark scale: 0.50 = a medium group gap.
model.set_effects("treatment=0.50")

model.find_power(sample_size=120, target_test="treatment")
