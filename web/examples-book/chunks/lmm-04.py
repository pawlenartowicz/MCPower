from mcpower import MCPower

# A randomised longitudinal plant trial: each seedling is measured repeatedly
# over `week`, half under `fertilizer` treatment and half under control, and the
# question is whether the two groups diverge in growth rate as weeks go on —
# the fertilizer-by-week interaction. '*' expands fertilizer * week to
# fertilizer + week + fertilizer:week, so the divergence term is fitted
# explicitly. (1 + week | seedling) gives every seedling its own intercept AND
# its own week slope, so individual growth trajectories are allowed to vary.
# family="lme" makes this a linear mixed model; the default MLE estimator fits
# the variance components.
model = MCPower("seedling_height = fertilizer * week + (1 + week | seedling)", family="lme")

# fertilizer is a two-arm 0/1 factor; week is a continuous within-seedling measure.
model.set_variable_type("fertilizer=binary")

# Effect sizes on the benchmark scale.
#   fertilizer=0.50          -> medium baseline group gap (binary benchmark).
#   week=0.25                -> medium average within-seedling growth trend (continuous benchmark).
#   fertilizer:week=0.25     -> medium divergence: the fertilized group's growth rate
#                               exceeds control's (the test of interest).
model.set_effects("fertilizer=0.50, week=0.25, fertilizer:week=0.25")

# Clustering by seedling: a conditional ICC of 0.3 (moderate within-seedling
# correlation), 60 seedlings, and a random slope on week whose own variance
# is modest and is mildly positively correlated with the intercept (taller
# seedlings at baseline tend to grow a little faster).
model.set_cluster(
    "seedling",
    ICC=0.3,
    n_clusters=60,
    random_slopes=["week"],
    slope_variance=0.05,
    slope_intercept_corr=0.3,
)

model.set_seed(2137)

# Power for the fertilizer-by-week divergence at 8 measurements per seedling.
model.find_power(sample_size=480, target_test="fertilizer:week")
