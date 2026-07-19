---
title: "Power for repeated-measures mixed models"
description: "Power & sample-size analysis by Monte Carlo simulation for repeated-measures data with a random intercept per subject (mixed model). Free, Python & R."
---
# Repeated measures over time (random-intercept growth)

You measured systolic blood pressure at several clinic visits on the same
patients, and you want to know whether it changes with phase. Because the
repeated measurements on one patient are correlated, you fit a linear mixed
model with a random intercept per patient:
`systolic_bp ~ phase + (1|patient)`.

## Variations

- **Stronger or weaker clustering.** The `ICC=0.30` in `set_cluster` says 30% of
  the variance sits between patients. Push it toward `ICC=0.10` (patients are
  nearly interchangeable) or `ICC=0.50` (patients differ a lot) — higher ICC leaves
  less independent information per visit and costs power for the same N.
- **More patients vs more visits.** Hold N=200 but change `n_clusters` — 20
  patients means 10 visits each, 50 means 4 each. Adding patients usually buys
  more power than adding visits to the same people.
- **Smaller or larger phase effect.** `phase=0.25` is a medium change on the
  continuous benchmark scale; swap it for `phase=0.10` (small) or `phase=0.40`
  (large) to see how the expected slope moves power.
- **Solve for N instead.** Replace `find_power(sample_size=200, …)` with
  `find_sample_size(target_test="phase", from_size=120, to_size=420, by=40)` to
  get the minimum total sample that reaches 80% power.
- **Same design, other fields:**
  - `seedling_height ~ week + (1|seedling)` — one plant measured weekly; test whether height increases on average (ecology).
  - `well_being ~ week + (1|individual)` — one person assessed at several wave points; test whether well-being trends over the study (social science).

## Not this setup?

- [[lmm/lmm-09|Three-level nesting (measurements in patients in sites)]]
- [[lmm/lmm-05|Cluster-randomised trial (treatment varies between clusters)]]
- [[lmm/lmm-02|Two arms compared over time (group x time interaction)]]

## If you'd rather have…

- [[lmm/lmm-03|Random intercept and slope]] — let each patient have their own
  slope of phase too (random intercept and slope), modelling individual
  trajectories rather than a single shared phase effect.
- [[lmm/lmm-02|Two arms compared over time]] — compare two arms over time: add a
  treatment x phase interaction to test whether change differs between groups.
- [[glmm/glmm-02|Longitudinal binary outcome]] — same longitudinal
  random-intercept structure but for a binary outcome measured over time.
- [[ols/ols-07|Two-group comparison without repeated measures]] — drop the
  repeated-measures structure and treat it as a simple two-group comparison if
  each patient is measured once.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:lmm-01 -->
```python
from mcpower import MCPower

# Repeated measures: systolic blood pressure recorded at several clinic visits on
# the same patients. The (1|patient) term adds a random intercept per patient;
# family="lme" fits it by maximum likelihood (MLE estimator).
model = MCPower("systolic_bp = phase + (1|patient)", family="lme")

# Expected effect on the standardised benchmark scale (continuous predictor):
#   phase=0.25 -> a medium change in systolic BP per unit of phase.
model.set_effects("phase=0.25")

# Describe the clustering: ICC=0.30 (30% of variance is between patients)
# across 40 patients. At N=200 that is 5 visits per patient.
model.set_cluster("patient", ICC=0.30, n_clusters=40)

# Power at N=200 for the fixed effect of phase (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
model.find_power(sample_size=200, target_test="phase")
```
<!-- /chunk:py:lmm-01 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:lmm-01 -->
```r
suppressMessages(library(mcpower))

# Repeated measures: systolic blood pressure recorded at several clinic visits on
# the same patients. The (1|patient) term adds a random intercept per patient;
# family = "lme" fits it by maximum likelihood (MLE estimator).
model <- MCPower$new("systolic_bp ~ phase + (1|patient)", family = "lme")

# Expected effect on the standardised benchmark scale (continuous predictor):
#   phase=0.25 -> a medium change in systolic BP per unit of phase.
model$set_effects("phase=0.25")

# Describe the clustering: ICC=0.30 (30% of variance is between patients)
# across 40 patients. At N=200 that is 5 visits per patient.
model$set_cluster("patient", ICC = 0.30, n_clusters = 40)

# Power at N=200 for the fixed effect of phase (mixed defaults: 800 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
invisible(model$find_power(sample_size = 200, target_test = "phase"))
```
<!-- /chunk:r:lmm-01 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/lmm-01-setup.png|600|theme-light]]
![[assets/lmm-01-setup-dark.png|600|theme-dark]]

</details>
