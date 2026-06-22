suppressMessages(library(mcpower))

# Multisite (cluster-randomised) clinical trial: whole sites are assigned to
# treatment or control, one continuous recovery outcome per patient. The
# (treatment | site) term gives each site its own intercept AND its own
# treatment effect — the benefit is allowed to vary from site to site.
# family = "lme" fits it by maximum likelihood (MLE estimator).
model <- MCPower$new("recovery_days ~ treatment + (treatment | site)", family = "lme")

# Treatment is randomised at the site level, so it is a binary 0/1 predictor.
model$set_variable_type("treatment=binary")

# Expected effect on the standardised benchmark scale (binary predictor):
#   treatment=0.50 -> a medium average reduction in recovery days due to treatment.
model$set_effects("treatment=0.50")

# Clustering: ICC=0.10 (10% of residual variance is between sites) across 30
# sites. random_slopes carries one structured spec per slope: variance sizes how
# much the treatment effect swings across sites and corr_with_intercept links a
# site's baseline to its treatment response. treatment is constant within a site,
# so it is also a cluster-level predictor.
model$set_cluster(
  "site",
  ICC = 0.10,
  n_clusters = 30,
  random_slopes = list(
    list(predictor = "treatment", variance = 0.05, corr_with_intercept = 0.0)
  ),
  cluster_level_vars = "treatment"
)

# Power at N=300 for the fixed effect of treatment (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
invisible(model$find_power(sample_size = 300, target_test = "treatment"))
