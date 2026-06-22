suppressMessages(library(mcpower))

# Two-way factorial ANCOVA: blood pressure crossed by treatment and sex, adjusted
# for a continuous baseline_bp covariate. '*' expands treatment * sex to
# treatment + sex + treatment:sex, so both main effects and the interaction are
# fitted, each read net of baseline_bp (parallel slopes, common slope across all cells).
model <- MCPower$new("blood_pressure ~ treatment * sex + baseline_bp")

# treatment and sex are two-level factors; baseline_bp stays continuous by default.
model$set_variable_type("treatment=binary, sex=binary")

# Effect sizes on the benchmark scales.
#   treatment=0.50          -> medium main effect (factor benchmark 0.20/0.50/0.80).
#   sex=0.50                -> medium main effect.
#   treatment:sex=0.50      -> medium interaction (the cell-difference-of-differences).
#   baseline_bp=0.40        -> strong continuous covariate (benchmark 0.10/0.25/0.40).
model$set_effects("treatment=0.50, sex=0.50, treatment:sex=0.50, baseline_bp=0.40")

model$set_seed(2137)

# Power for both main effects at N=160 (the adjusted factorial F-tests).
invisible(model$find_power(sample_size = 160, target_test = "treatment, sex"))
