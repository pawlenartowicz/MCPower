suppressMessages(library(mcpower))

# Longitudinal binary outcome: symptom presence (yes/no) recorded at several
# months on the same patients, in two arms (control vs. treatment). The
# (1|patient) term adds a random intercept per patient; family = "logit" with a
# cluster makes this a binary GLMM (GLM estimator).
model <- MCPower$new("symptom_present ~ month + treatment + (1|patient)", family = "logit")

# treatment is a binary two-level predictor (0 = control, 1 = treatment); month
# is continuous (measurement occasion on the standardised benchmark scale).
model$set_variable_type("treatment=binary")

# Expected effects on the binary benchmark scale:
#   treatment=0.50 -> a medium gap between arms,
#   month=0.25     -> a medium change in symptom odds per month.
model$set_effects("treatment=0.50, month=0.25")

# Symptom rate in the control group at baseline (logit family needs one).
model$set_baseline_probability(0.20)

# Describe the clustering: ICC=0.10 (10% of the latent variance is between
# patients) across 50 patients. At N=300 that is 6 measurements each.
model$set_cluster("patient", ICC = 0.10, n_clusters = 50)

# Power at N=300 for both fixed effects (mixed defaults: 800 sims, alpha=0.05,
# seed=2137). The omnibus test is not reported for mixed models; target the
# coefficients directly.
invisible(model$find_power(sample_size = 300, target_test = "month, treatment"))
