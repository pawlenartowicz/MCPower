from mcpower import MCPower

# Multisite (cluster-randomised) clinical trial: whole sites are assigned to
# treatment or control, one continuous recovery outcome per patient. The
# (treatment | site) term gives each site its own intercept AND its own
# treatment effect — the benefit is allowed to vary from site to site.
# family="lme" fits it by maximum likelihood (MLE estimator).
model = MCPower("recovery_days = treatment + (treatment | site)", family="lme")

# Treatment is randomised at the site level, so it is a binary 0/1 predictor.
model.set_variable_type("treatment=binary")

# Expected effect on the standardised benchmark scale (binary predictor):
#   treatment=0.50 -> a medium average reduction in recovery days due to treatment.
model.set_effects("treatment=0.50")

# Clustering: ICC=0.10 (10% of residual variance is between sites) across 30
# sites. random_slopes=["treatment"] turns on the random treatment slope, with
# slope_variance sizing how much the treatment effect swings across sites and
# slope_intercept_corr linking a site's baseline to its treatment response.
# treatment is constant within a site, so it is a cluster-level predictor.
model.set_cluster(
    "site",
    ICC=0.10,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.05,
    slope_intercept_corr=0.0,
    cluster_level_vars=["treatment"],
)

# Power at N=300 for the fixed effect of treatment (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
model.find_power(sample_size=300, target_test="treatment")
