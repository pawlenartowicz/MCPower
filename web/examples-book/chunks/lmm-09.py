from mcpower import MCPower

# Repeated-measures biochemistry experiment: every sample is assayed under each
# of several conditions, so the condition factor varies *within* sample. The
# (1|sample) term adds a random intercept per sample to soak up the
# between-sample correlation; family="lme" fits it by maximum likelihood
# (MLE estimator).
model = MCPower("enzyme_activity = condition + (1|sample)", family="lme")

# condition is a categorical predictor with 3 levels -> 2 dummy contrasts, each
# comparing a non-reference condition against the reference.
model.set_variable_type("condition=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference condition shifts enzyme activity by a medium amount vs reference.
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (condition[2], condition[3]) — the bare factor name is not an effect.
model.set_effects("condition[2]=0.5, condition[3]=0.5")

# Describe the clustering: ICC=0.30 (30% of variance is between samples)
# across 40 samples. At N=200 that is 5 measurements per sample — the
# minimum the engine requires for reliable mixed-model estimation.
model.set_cluster("sample", ICC=0.30, n_clusters=40)

# Power at N=200 for both dummy contrasts (mixed defaults: 800 sims, alpha=0.05,
# seed=2137). The omnibus test is not reported for mixed models; target the
# condition dummy coefficients directly (the bare factor name is not a test).
model.find_power(sample_size=200, target_test="condition[2], condition[3]")
