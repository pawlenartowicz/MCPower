from mcpower import MCPower

# Two continuous predictors of a continuous outcome (OLS).
# rainfall is the predictor of interest; soil_nitrogen is a second continuous covariate
# whose association we also want enough power to detect.
model = MCPower("plant_biomass = rainfall + soil_nitrogen")

# Standardised effect sizes (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   rainfall=0.25 -> a medium association.
#   soil_nitrogen=0.10 -> a small association.
model.set_effects("rainfall=0.25, soil_nitrogen=0.10")

# Both predictors are continuous, so no set_variable_type() is needed.
# OLS defaults apply: 1600 simulations, alpha=0.05, seed=2137.
model.find_power(sample_size=200, target_test="rainfall, soil_nitrogen")
