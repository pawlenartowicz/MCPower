suppressMessages(library(mcpower))

# Covariate-adjusted two-group comparison (ANCOVA-style, parallel slopes).
# Research question: does the treatment shift blood_pressure once we adjust for
# each patient's baseline_bp measurement?
model <- MCPower$new("blood_pressure ~ treatment + baseline_bp")

# Expected effect sizes (standardised).
#   treatment=0.50   -> treatment shifts blood_pressure by a medium binary effect.
#   baseline_bp=0.40 -> baseline_bp is a strong continuous covariate.
model$set_effects("treatment=0.50, baseline_bp=0.40")

# Variable types — treatment is binary (0=control, 1=treatment); baseline_bp stays
# continuous by default.
model$set_variable_type("treatment=binary")

# Power at N=120, targeting the adjusted treatment effect.
invisible(model$find_power(sample_size = 120, target_test = "treatment"))
