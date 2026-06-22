from mcpower import MCPower

# Three-arm survey: life satisfaction measured across residents of three regions.
# Research question: you have ONE pre-specified comparison in mind (a planned
# contrast) — e.g. region 2 vs region 3 — decided before seeing the data.
model = MCPower("life_satisfaction = region")

# region is a categorical predictor with 3 levels.
model.set_variable_type("region=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (region[2], region[3]) — the bare factor name is not an effect.
model.set_effects("region[2]=0.50, region[3]=0.50")

# Power for the one planned contrast — level 2 vs level 3 of region.
# A single pre-specified comparison needs no multiplicity correction.
model.find_power(sample_size=150, target_test="region[2] vs region[3]")
