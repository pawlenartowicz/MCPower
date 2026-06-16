"""Power for clustered designs with a cluster-level predictor; set_cluster's
cluster_level_vars argument flags a fixed effect as measured once per group
(treatment assigned at site level, not individual level), giving it the
correct 1/sqrt(n_clusters) SE instead of 1/sqrt(N)."""

from mcpower import MCPower

# Example: Multi-site clinical trial where a training intervention (treat) is
# assigned at the SITE level (30 sites) while a continuous patient covariate
# (baseline_score) varies within each site.
# Research question: does the site-level treatment raise outcomes, accounting
# for between-site clustering?

# 1. Declare the mixed model. The (1|site) term adds a random intercept
#    for each of the 30 sites; family="lme" fits by MLE.
model = MCPower(
    "outcome = treat + baseline_score + (1|site)", family="lme"
)
model.set_variable_type("treat=binary")
model.set_effects("treat=0.40, baseline_score=0.25")

# 2. Describe the clustering. ICC=0.20 means 20% of outcome variance is
#    between sites. cluster_level_vars=["treat"] tells the engine that
#    'treat' is assigned once per site — not per patient — so its
#    effective sample size is n_clusters=30, not total N.
model.set_cluster("site", ICC=0.20, n_clusters=30,
                  cluster_level_vars=["treat"])

# 3. Power at N = 600 (30 sites × 20 patients each).
print(">>> model.find_power(sample_size=600, target_test='treat, baseline_score')")
model.find_power(sample_size=600, target_test="treat, baseline_score")

# 4. Contrast: same total N, half the sites (15 × 40). The patient-level
#    predictor (baseline_score) loses little power; the site-level predictor
#    (treat) loses substantially — it has only 15 information units.
print("\n>>> Fewer sites (15 × 40, same N):")
model_15 = MCPower(
    "outcome = treat + baseline_score + (1|site)", family="lme"
)
model_15.set_variable_type("treat=binary")
model_15.set_effects("treat=0.40, baseline_score=0.25")
model_15.set_cluster("site", ICC=0.20, n_clusters=15,
                     cluster_level_vars=["treat"])
model_15.find_power(sample_size=600, target_test="treat, baseline_score")

# 5. Sample-size search sized to the site-limited predictor.
print("\n>>> model.find_sample_size(target_test='treat', from_size=300, to_size=900, by=60)")
model.find_sample_size(
    target_test="treat", from_size=300, to_size=900, by=60
)
