# Crossed Random Effects Example
# ==============================
#
# Power for designs with crossed random effects, e.g. a psycholinguistics
# experiment where participants (subjects) and stimuli (items) are both random
# sources of variation. Crossed groupings are declared in the formula —
# (1|subject) + (1|item) — and sized with one set_cluster call per factor.
#
# Run with:  Rscript 13_crossed_random_effects.R

suppressMessages(library(mcpower))

# Example: Psycholinguistics word-recognition experiment.
# Each participant sees each item, so subjects and items are CROSSED (not nested).
# Research question: does word frequency predict reaction time after accounting
# for both by-subject and by-item variability?

# 1. Declare the mixed model. Both grouping factors live in the formula:
#    (1|subject) and (1|item) are CROSSED random intercepts.
model <- MCPower$new(
  "rt ~ frequency + (1|subject) + (1|item)", family = "lme"
)
model$set_effects("frequency=-0.12")

# 2. Describe the double-grouping structure with one set_cluster call per
#    grouping factor (the formula already names both).
#    subject: 20 subjects, ICC=0.20 (by-subject baseline variability).
#    item:    12 items,    ICC=0.13 (by-item baseline variability).
#    For crossed groupings N must be a multiple of the atom = 20 × 12 = 240
#    (one full subject × item block); the engine snaps N otherwise.
model$set_cluster("subject", ICC = 0.20, n_clusters = 20)
model$set_cluster("item", ICC = 0.13, n_clusters = 12)

# 3. Power at n = 480 = 2 × 240 (each subject sees each item exactly twice).
cat(">>> model$find_power(sample_size = 480, target_test = 'frequency')\n")
invisible(model$find_power(sample_size = 480, target_test = "frequency"))

# 4. Long form with diagnostics.
cat("\n>>> result <- model$find_power(sample_size = 480, target_test = 'frequency', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(
  sample_size = 480, target_test = "frequency", verbose = FALSE
)
print(summary(result))

# 5. Sample-size search. The grid steps by the 240 atom so every point is a
#    balanced crossed design (240, 480, 720, 960 — one to four full blocks).
cat("\n>>> model$find_sample_size(target_test = 'frequency', from_size = 240, to_size = 960, by = 240)\n")
invisible(model$find_sample_size(
  target_test = "frequency", from_size = 240, to_size = 960, by = 240
))
