# One-Way ANOVA with Post-Hoc Contrasts
# ======================================
#
# The omnibus F asks "do the group means differ at all?"; the post-hoc table
# asks "which specific pairs differ?". target_test = "overall, all-contrasts"
# reports both. correction = "tukey" applies Tukey HSD across the pairwise
# family; running it once without and once with correction shows the
# family-wise power cost.
#
# Run with:  Rscript 09_anova_posthoc.R

suppressMessages(library(mcpower))

# Three-arm trial: does pain reduction differ across placebo / low / high dose?
model <- MCPower$new("pain_reduction ~ dose_group")

# 3 roughly equal arms. No uploaded data -> integer-labelled levels 1, 2, 3
# (level 1 = reference). With uploaded data the labels are the data values.
model$set_variable_type("dose_group=(factor,0.34,0.33,0.33)")

# Per-arm effects vs the reference (placebo). The 2-vs-3 pairwise contrast is the
# 0.3 gap between these - the comparison only a full post-hoc view exposes.
model$set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

# 1. Omnibus F + every pairwise contrast, UNCORRECTED.
cat(">>> model$find_power(sample_size = 130, target_test = 'overall, all-contrasts')\n")
invisible(model$find_power(sample_size = 130, target_test = "overall, all-contrasts"))

# 2. Same design, Tukey HSD across the pairwise family. Compare the post-hoc
#    table's 'corrected' column to step 1 to see the family-wise power cost.
cat("\n>>> model$find_power(sample_size = 130, target_test = 'overall, all-contrasts', correction = 'tukey')\n")
invisible(model$find_power(
  sample_size = 130, target_test = "overall, all-contrasts", correction = "tukey"
))

# 3. Long form adds CIs and the joint-significance distribution.
cat("\n>>> result <- model$find_power(sample_size = 130, target_test = 'overall, all-contrasts', correction = 'tukey', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(
  sample_size = 130, target_test = "overall, all-contrasts",
  correction = "tukey", verbose = FALSE
)
print(summary(result))
