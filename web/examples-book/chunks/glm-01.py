from mcpower import MCPower

# Simple logistic regression: a binary (yes/no) outcome on one continuous
# predictor. family="logit" makes the outcome binary and fits a GLM.
model = MCPower("relapse = biomarker_level", family="logit")

# family="logit" requires a baseline event rate: the probability of relapse=1
# at the predictor's reference level. This pins the intercept (log-odds).
model.set_baseline_probability(0.3)

# Expected effect on the standardised benchmark scale (continuous predictor):
#   biomarker_level=0.25 -> a medium association with the log-odds of relapse.
model.set_effects("biomarker_level=0.25")

# Power at N=200 with the GLM defaults (1600 sims, alpha=0.05, seed=2137).
model.find_power(sample_size=200, target_test="biomarker_level")
