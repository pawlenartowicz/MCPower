from mcpower import MCPower

# Three-way factorial ANOVA: seed yield crossed by three binary growth factors.
# Research question: does the watering-by-nitrogen interaction itself shift
# across light levels? -> the three-way watering:nitrogen:light interaction.
# '*' expands watering * nitrogen * light to all main effects, all pairwise
# interactions, and the three-way term, so every cell of the design is fitted.
model = MCPower("seed_yield = watering * nitrogen * light")

# All three predictors are two-level factors (a 2x2x2 design).
model.set_variable_type("watering=binary, nitrogen=binary, light=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   main effects 0.50                 -> medium shift per factor.
#   two-way interactions 0.50         -> medium moderation between each pair.
#   watering:nitrogen:light=0.50      -> medium three-way interaction (the target).
model.set_effects(
    "watering=0.50, nitrogen=0.50, light=0.50, "
    "watering:nitrogen=0.50, watering:light=0.50, nitrogen:light=0.50, "
    "watering:nitrogen:light=0.50"
)
model.set_simulations(1600)
model.set_seed(2137)

# Power at N=320 for the three-way interaction (the highest-order term).
model.find_power(sample_size=320, target_test="watering:nitrogen:light")
