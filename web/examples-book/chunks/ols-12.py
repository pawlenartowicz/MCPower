from mcpower import MCPower

# Three habitat types: one continuous outcome, one 3-level grouping factor.
# Research question: do the habitat types differ on species abundance,
# tested as the two dummy contrasts against the reference habitat?
model = MCPower("abundance = habitat")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
model.set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts the outcome by a medium amount vs reference.
# Factors expand to per-level dummies (habitat[2], habitat[3]); effects
# and targets are addressed by those dummy names, not the bare factor.
model.set_effects("habitat[2]=0.5, habitat[3]=0.5")

# Power at N=150 for both dummy contrasts (each non-reference level vs reference).
model.find_power(sample_size=150, target_test="habitat[2], habitat[3]")
