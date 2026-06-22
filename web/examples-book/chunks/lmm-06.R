suppressMessages(library(mcpower))

# Cluster-randomised trial, baseline-adjusted: clinics are assigned to
# treatment or control, patients are measured within clinics, and each patient
# has a baseline blood pressure reading. The (1|clinic) term adds a random
# intercept per clinic; family = "lme" fits it by maximum likelihood (MLE estimator).
model <- MCPower$new("blood_pressure ~ treatment + baseline_bp + (1|clinic)", family = "lme")

# Expected effect sizes (standardised benchmark scale):
#   treatment=0.50   -> a medium binary (between-arm) effect on blood pressure.
#   baseline_bp=0.40 -> the baseline reading is a strong continuous covariate.
model$set_effects("treatment=0.50, baseline_bp=0.40")

# Treatment is randomised at the clinic level (0=control, 1=treatment);
# baseline_bp stays continuous by default.
model$set_variable_type("treatment=binary")

# Describe the clustering: ICC=0.10 (10% of outcome variance is between clinics)
# across 30 clinics. At N=300 that is 10 patients per clinic.
model$set_cluster("clinic", ICC = 0.10, n_clusters = 30)

# Power at N=300 for the baseline-adjusted treatment effect (mixed defaults:
# 800 sims, alpha=0.05, seed=2137). The omnibus test is not reported for mixed
# models; target the coefficient directly.
invisible(model$find_power(sample_size = 300, target_test = "treatment"))
