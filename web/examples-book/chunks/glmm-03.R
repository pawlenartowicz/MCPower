suppressMessages(library(mcpower))

# Difference-in-differences on a binary employment outcome: each individual is
# observed at several periods, and we record whether they are `employed` (yes/no).
# A policy `policy_group` is crossed with `period` — the question is whether the
# change over periods differs between groups (the DiD interaction). '*' expands
# policy_group * period to policy_group + period + policy_group:period, so the
# interaction is fitted explicitly, on the log-odds scale. family = "logit" makes
# employed binary (0/1); the (1|individual) random intercept makes it a logistic
# GLMM, fitted by the GLM estimator.
model <- MCPower$new("employed ~ policy_group * period + (1|individual)", family = "logit")

# policy_group is a two-level factor (policy vs control); period is the
# continuous measurement occasion.
model$set_variable_type("policy_group=binary")

# Effect sizes on the relevant benchmark scales, each a shift in the log-odds of
# being employed:
#   policy_group=0.50             -> medium baseline arm difference (factor benchmark).
#   period=0.25                   -> medium average change over time (continuous benchmark).
#   policy_group:period=0.50      -> medium DiD interaction: the policy group's
#                                    trajectory diverges from the control group's
#                                    (factor benchmark).
model$set_effects("policy_group=0.50, period=0.25, policy_group:period=0.50")

# Baseline employment rate of 30% when all predictors are at their reference.
model$set_baseline_probability(0.30)

# Clustering: ICC=0.20 (20% of the latent variance is between individuals)
# across 40 individuals. At N=240 that is 6 observations per individual.
model$set_cluster("individual", ICC = 0.20, n_clusters = 40)

# Power at N=240 for the DiD interaction (mixed defaults: 800 sims, alpha=0.05,
# seed=2137). The omnibus test is not reported for mixed models; target the
# interaction coefficient directly.
invisible(model$find_power(sample_size = 240, target_test = "policy_group:period"))
