from mcpower import MCPower

# Repeated measures: systolic blood pressure recorded at several clinic visits on
# the same patients. The (1|patient) term adds a random intercept per patient;
# family="lme" fits it by maximum likelihood (MLE estimator).
model = MCPower("systolic_bp = phase + (1|patient)", family="lme")

# Expected effect on the standardised benchmark scale (continuous predictor):
#   phase=0.25 -> a medium change in systolic BP per unit of phase.
model.set_effects("phase=0.25")

# Describe the clustering: ICC=0.30 (30% of variance is between patients)
# across 40 patients. At N=200 that is 5 visits per patient.
model.set_cluster("patient", ICC=0.30, n_clusters=40)

# Power at N=200 for the fixed effect of phase (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
model.find_power(sample_size=200, target_test="phase")
