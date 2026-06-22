suppressMessages(library(mcpower))

# Cluster-randomised trial on a binary (infection / no-infection) outcome:
# hospitals are randomised to control vs treatment, and every patient in a
# hospital shares its assignment. The (1|hospital) term adds a random intercept
# per hospital; family = "logit" makes this a clustered logistic model (binary GLMM).
model <- MCPower$new("infection ~ treatment + (1|hospital)", family = "logit")

# treatment is a binary two-level predictor (0 = control, 1 = treatment),
# assigned at the hospital level rather than per patient.
model$set_variable_type("treatment=binary")

# Expected treatment effect on the binary benchmark scale: 0.50 = a medium gap.
model$set_effects("treatment=0.50")

# Infection rate in the control arm (logit family needs a baseline probability).
model$set_baseline_probability(0.20)

# Describe the clustering: ICC=0.05 (5% of variance is between hospitals) across
# 30 hospitals. At N=300 that is 10 patients per hospital.
model$set_cluster("hospital", ICC = 0.05, n_clusters = 30)

# Power at N=300 for the fixed effect of treatment (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
invisible(model$find_power(sample_size = 300, target_test = "treatment"))
