# NOTE: Reframed from a 2-occasion (pre/post) design to a 5-occasion repeated-
# measures design. The original 60 patients x 2 occasions gave only 2
# observations per cluster, which the mixed-model validator rejects (it requires
# at least 5 observations per cluster for reliable estimation). Raising
# sample_size to 300 yields 5 occasions per patient, clearing the floor, and
# `week` is now a continuous (normal) measurement-occasion covariate instead of
# binary.
suppressMessages(library(mcpower))

# Two-arm longitudinal pain trial (difference-in-differences): every patient is
# assessed repeatedly over weeks, in either the treatment or the control arm.
# Research question: does the treatment reduce pain MORE than control does over
# time? -> the treatment:week interaction. '*' expands
# treatment * week to treatment + week + treatment:week, so both main effects
# and the diff-in-diff interaction are fitted. family="lme" adds the
# (1|patient) random intercept and fits by maximum likelihood (MLE estimator).
model <- MCPower$new("pain_score ~ treatment * week + (1|patient)", family = "lme")

# treatment is the two arms; week is the continuous measurement occasion.
model$set_variable_type("treatment=binary, week=normal")

# Effect sizes on the benchmark scale:
#   treatment=0.20 (factor)     -> small baseline arm gap (groups nearly balanced).
#   week=0.25 (continuous)      -> medium overall drift over weeks (both arms move).
#   treatment:week=0.25         -> medium diff-in-diff: the treatment arm's slope
#                                  over weeks exceeds the control arm's (the target).
model$set_effects("treatment=0.20, week=0.25, treatment:week=0.25")

# Repeated measures: ICC=0.50 of the variance is between-patient (the
# occasions per person are strongly correlated) across 60 patients.
model$set_cluster("patient", ICC = 0.50, n_clusters = 60)

model$set_simulations(800)
model$set_seed(2137)

# Power at N=300 (60 patients x 5 occasions) for the diff-in-diff interaction.
invisible(model$find_power(sample_size = 300, target_test = "treatment:week"))
