from mcpower import MCPower

# Three-way interaction on a yes/no outcome: does the way one environmental
# factor moderates another itself depend on a third, when the response is binary?
# family="logit" makes `germinated` a binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("germinated ~ light * moisture * temperature", family="logit")

# `*` expands to all three main effects, all three two-way interactions, and the
# single three-way term light:moisture:temperature -- the coefficient this page actually powers.
# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40):
# main effects at medium (0.25), every interaction at small (0.10) on the log-odds.
model.set_effects(
    "light=0.25, moisture=0.25, temperature=0.25, "
    "light:moisture=0.10, light:temperature=0.10, moisture:temperature=0.10, "
    "light:moisture:temperature=0.10"
)
model.set_simulations(1600)
model.set_seed(2137)

# Logistic GLMs need a baseline event rate: it pins the intercept so the
# log-odds effects above land on a concrete probability scale. Required for
# family="logit" -- find_power errors without it.
model.set_baseline_probability(0.3)

# Power at N=600 for the three-way term itself (the thinnest slice of the design).
model.find_power(sample_size=600, target_test="light:moisture:temperature")
