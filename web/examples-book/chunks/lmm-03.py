from mcpower import MCPower

# Repeated-measures growth curve: each seedling is measured at several weekly
# time points and we ask whether height changes with `week`. `(week | seedling)`
# gives every seedling its own random intercept (baseline height) AND its own
# random slope of week (personal growth rate), so the test of the average
# `week` slope is judged against how much those individual growth rates scatter.
# family="lme" with the default MLE estimator fits the mixed model.
model = MCPower("seedling_height ~ week + (week | seedling)", family="lme")

# `week` is a continuous (normally distributed) within-seedling predictor.
model.set_variable_type("week=normal")

# Average weekly growth slope on the continuous benchmark scale:
#   week=0.25 -> a medium average rate of change across seedlings.
model.set_effects("week=0.25")

# Clustering by seedling: 40 seedlings, conditional ICC 0.3 (baseline
# heights correlate within a seedling after `week` is accounted for), and a
# random slope of `week` whose variance 0.05 sets how much individual growth
# rates differ; slope_intercept_corr ties taller seedlings to faster growth.
model.set_cluster(
    "seedling",
    ICC=0.3,
    n_clusters=40,
    random_slopes=["week"],
    slope_variance=0.05,
    slope_intercept_corr=0.2,
)

model.set_seed(2137)

# Power for the average `week` slope. sample_size is the total number of
# observations (seedlings x measurements per seedling).
model.find_power(sample_size=200, target_test="week")
