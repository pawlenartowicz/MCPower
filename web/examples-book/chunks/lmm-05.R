suppressMessages(library(mcpower))

# Cluster-randomized trial: does the treatment lower cholesterol, once we
# account for patients in the same clinic being correlated? A random intercept
# per clinic soaks up that between-clinic variation.
model <- MCPower$new("cholesterol ~ treatment + (1|clinic)", family = "lme")

# treatment is a binary two-level predictor (0 = control, 1 = treatment).
model$set_variable_type("treatment=binary")

# Expected effect on the binary benchmark scale: 0.50 = a medium treatment gap.
model$set_effects("treatment=0.50")

# Clustering: ICC=0.10 (10% of variance is between-clinic) across 40 clinics.
# At N=400 that is 10 patients per clinic.
model$set_cluster("clinic", ICC = 0.10, n_clusters = 40)

invisible(model$find_power(sample_size = 400, target_test = "treatment"))
