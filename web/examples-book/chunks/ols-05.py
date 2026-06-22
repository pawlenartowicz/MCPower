from mcpower import MCPower

model = MCPower("growth_rate ~ temperature * moisture * soil_ph", family="ols")
model.set_effects(
    "temperature=0.25, moisture=0.25, soil_ph=0.25, "
    "temperature:moisture=0.10, temperature:soil_ph=0.10, moisture:soil_ph=0.10, "
    "temperature:moisture:soil_ph=0.10"
)
model.set_correlations("corr(temperature,moisture)=0.2")
model.set_simulations(1600)
model.set_seed(2137)

model.find_power(sample_size=300, target_test="temperature:moisture:soil_ph")
