---
title: "Power for a treatment x time mixed model"
description: "Power & sample-size analysis by Monte Carlo simulation for a treatment-by-time interaction in longitudinal data, random intercept. Free, Python & R."
---
# Two-arm longitudinal pain trial: a treatment-by-week difference-in-differences with a random intercept

You ran a pre/post pain intervention study: every patient was assessed twice, at
baseline and at follow-up, and each was assigned to either the treatment arm or the
control arm. You want the power to detect that the treatment arm's pain score
changed *more* from baseline than the control arm did — the difference-in-differences —
while accounting for the fact that a patient's repeated measurements are correlated.

As an MCPower model this is `pain_score = treatment * week + (1|patient)` with
`family="lme"`, where `treatment` is a binary arm indicator and `week` is a
binary occasion indicator (baseline vs follow-up). The `(1|patient)` random intercept
lets each patient have their own baseline pain level. The `*` expands to the
`treatment` and `week` main effects plus the `treatment:week` interaction; that
interaction *is* the diff-in-diff, and it is what you read power off.
`family="lme"` fits the model by maximum likelihood (the MLE estimator), so the
within-patient correlation widens the standard errors the way it would in a real
mixed-model fit.

## Variations

- **More than two timepoints.** A finer follow-up schedule simply increases
  `sample_size` while holding `n_clusters` constant; with several visits, declare
  `week=normal` so it is a measurement-occasion covariate rather than a two-level
  indicator.
- **Stronger or weaker correlation between occasions.** The `ICC` is how much of
  the variance sits between patients; bump it toward `0.70` for tightly tracking
  repeated measures or down toward `0.20` for loosely related ones, and watch how
  the within-patient design either gains or loses its advantage.
- **More patients vs more visits.** `n_clusters` is the patient count and the
  sample size is patients x occasions. Adding patients is usually the more
  efficient lever for the interaction than adding visits per patient.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="treatment:week", from_size=120, to_size=480, by=60)`
  to get the smallest N that reaches target power on the diff-in-diff term.
- **Search for the number of visits instead.** With the 60 patients fixed, ask
  how many weekly assessments each needs:
  `find_sample_size(target_test="treatment:week", unit="cluster_size", from_size=2, to_size=20, by=1)`.
  The range reads as visits per patient (2 is the floor), one step is 60 more
  measurements, and the result reports the total N alongside it. The
  diff-in-diff is a within-patient slope contrast, so it is one of the terms
  that extra visits actually buy.
- **Same design, other fields:**
  - `seedling_height ~ fertilizer * week + (1|seedling)` — one plant per fertilizer arm, measured weekly; test whether growth rates diverge (ecology).
  - `well_being ~ intervention * wave + (1|individual)` — one person per arm, assessed at several waves; test whether well-being trajectories differ (social science).

## Not this setup?

- [[lmm/lmm-04|Same treatment-by-week design, but let each patient have their own week slope]]
- [[lmm/lmm-01|Drop the treatment arm: simple repeated measures with just a phase effect]]
- [[glmm/glmm-03|The same group-by-time difference-in-differences, but for a binary outcome]]

## If you'd rather have…

- [[lmm/lmm-04]] — Same treatment x week interaction but lets each patient
  have their own week slope (`week | patient`) — the conditional
  growth-curve version when individual trajectories vary.
- [[lmm/lmm-01]] — Drop the treatment arm: simple repeated measures with
  just a phase effect and a random intercept per patient.
- [[lmm/lmm-03]] — Single-arm linear growth curve — week as a continuous slope
  with a random intercept and slope, no treatment factor.
- [[glmm/glmm-03]] — Same group x time difference-in-differences design but for a
  binary outcome (logistic GLMM).
- [[lmm/lmm-05]] — Two-arm trial with a random intercept but clustered
  (cluster-randomised) rather than longitudinal repeated measures — treatment
  main effect only, no week interaction.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:lmm-02 -->
```python
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
```
<!-- /chunk:py:lmm-02 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:lmm-02 -->
```r
suppressMessages(library(mcpower))

# Two-arm longitudinal pain trial (difference-in-differences): every patient is
# assessed twice, at baseline and at follow-up, in either the treatment or the
# control arm.
# Research question: does the treatment reduce pain MORE than control does over
# time? -> the treatment:week interaction. '*' expands
# treatment * week to treatment + week + treatment:week, so both main effects
# and the diff-in-diff interaction are fitted. family="lme" adds the
# (1|patient) random intercept and fits by maximum likelihood (MLE estimator).
model <- MCPower$new("pain_score ~ treatment * week + (1|patient)", family = "lme")

# treatment is the two arms; week is the occasion (baseline vs follow-up).
model$set_variable_type("treatment=binary, week=binary")

# Effect sizes on the benchmark scale:
#   treatment=0.20 (factor)     -> small baseline arm gap (groups nearly balanced).
#   week=0.50 (binary)          -> medium overall change from baseline (both arms move).
#   treatment:week=0.50         -> medium diff-in-diff: the treatment arm changes more
#                                  from baseline than the control arm (the target).
model$set_effects("treatment=0.20, week=0.50, treatment:week=0.50")

# Repeated measures: ICC=0.50 of the variance is between-patient (the two
# occasions per person are strongly correlated) across 60 patients.
model$set_cluster("patient", ICC = 0.50, n_clusters = 60)

model$set_simulations(800)
model$set_seed(2137)

# Power at N=120 (60 patients x 2 occasions) for the diff-in-diff interaction.
invisible(model$find_power(sample_size = 120, target_test = "treatment:week"))
```
<!-- /chunk:r:lmm-02 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/lmm-02-setup.png|600|theme-light]]
![[assets/lmm-02-setup-dark.png|600|theme-dark]]

</details>
