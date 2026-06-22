suppressMessages(library(mcpower))

# ANCOVA with a homogeneity-of-slopes test: does the treatment effect on
# blood_pressure depend on each patient's baseline_bp? '*' expands treatment * baseline_bp
# to treatment + baseline_bp + treatment:baseline_bp, so the interaction is fitted explicitly.
model <- MCPower$new("blood_pressure ~ treatment * baseline_bp")

# treatment is a two-level treatment factor (0=control, 1=treatment); baseline_bp is a
# continuous covariate.
model$set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50             -> medium treatment shift (binary benchmark).
#   baseline_bp=0.40           -> strong baseline-outcome association (continuous benchmark).
#   treatment:baseline_bp=0.25 -> moderate slope difference (the moderation effect).
model$set_effects("treatment=0.50, baseline_bp=0.40, treatment:baseline_bp=0.25")

model$set_seed(2137)

# Power for the homogeneity-of-slopes test (the interaction) at N=180.
invisible(model$find_power(sample_size = 180, target_test = "treatment:baseline_bp"))
