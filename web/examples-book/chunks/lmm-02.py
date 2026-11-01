from mcpower import MCPower

# Two-arm longitudinal pain trial (difference-in-differences): every patient is
# assessed twice, at baseline and at follow-up, in either the treatment or the
# control arm.
# Research question: does the treatment reduce pain MORE than control does over
# time? -> the treatment:week interaction. '*' expands
# treatment * week to treatment + week + treatment:week, so both main effects
# and the diff-in-diff interaction are fitted. family="lme" adds the
# (1|patient) random intercept and fits by maximum likelihood (MLE estimator).
model = MCPower("pain_score = treatment * week + (1|patient)", family="lme")

# treatment is the two arms; week is the occasion (baseline vs follow-up).
model.set_variable_type("treatment=binary, week=binary")

# Effect sizes on the benchmark scale:
#   treatment=0.20 (factor)     -> small baseline arm gap (groups nearly balanced).
#   week=0.50 (binary)          -> medium overall change from baseline (both arms move).
#   treatment:week=0.50         -> medium diff-in-diff: the treatment arm changes more
#                                  from baseline than the control arm (the target).
model.set_effects("treatment=0.20, week=0.50, treatment:week=0.50")

# Repeated measures: ICC=0.50 of the variance is between-patient (the two
# occasions per person are strongly correlated) across 60 patients.
model.set_cluster("patient", ICC=0.50, n_clusters=60)

model.set_simulations(800)
model.set_seed(2137)

# Power at N=120 (60 patients x 2 occasions) for the diff-in-diff interaction.
model.find_power(sample_size=120, target_test="treatment:week")
