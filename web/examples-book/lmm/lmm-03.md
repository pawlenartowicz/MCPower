---
title: "Power for a random-slope growth curve model"
description: "Power & sample-size analysis by Monte Carlo simulation for a linear growth curve with random intercept and slope of time (mixed model). Free, Python & R."
---
# Growth curve: does seedling height change over weeks, with each seedling on its own trajectory?

Every seedling is measured at several weekly time points, and the question is
whether height moves with `week` on average. Because seedlings start at different
heights *and* grow at different rates, each seedling gets its own intercept and its
own slope of week. As an MCPower formula:
`seedling_height ~ week + (week | seedling)`, fit as a linear mixed model (the
default MLE estimator). The test of interest is the **average** `week` slope,
judged against how widely the individual growth rates scatter.

## Variations

- **Drop the random slope.** If individual growth rates are not your concern,
  use `seedling_height ~ week + (1 | seedling)` — random intercepts only. Every
  seedling shares one average slope, which usually raises power for `week`
  because the slope no longer competes with between-seedling slope variance.
- **More seedlings vs. more measurements.** Power for a fixed total `sample_size`
  shifts with the seedling/measurement split; raise `n_clusters` (more seedlings,
  fewer measurements each) or lower it (fewer seedlings, denser follow-up) to see
  which your design can afford.
- **Tighter or looser individual trajectories.** Raise `slope_variance` to model
  seedlings whose growth rates differ a lot (harder to detect the average
  slope), or lower it toward zero to approach a common-slope design.
- **Swap the trend strength.** Replace `week=0.25` with `week=0.10` (small) or
  `week=0.40` (large) to bracket a plausible average growth rate.
- **Same design, other fields:**
  - `systolic_bp ~ phase + (phase|patient)` — one patient measured at several clinical phases; individual BP trajectories vary in slope (clinical).
  - `life_satisfaction ~ wave + (wave|individual)` — one person surveyed repeatedly; individual well-being slopes vary (social science).

## Not this setup?

- [[lmm/lmm-01|Random intercept only — one shared week slope]]
- [[lmm/lmm-04|A fertilizer treatment that moderates the individual slopes]]
- [[lmm/lmm-07|Random slope that varies across sites, not seedlings]]

## If you'd rather have…

- [[lmm/lmm-01|Random intercept only]] — same week-trend design but the simpler
  form: random intercept only, no random slope of week.
- [[lmm/lmm-04|Conditional growth with a fertilizer treatment]] — adds a treatment
  that moderates the individual slopes, layering conditional growth on top of this
  same random-slope structure.
- [[lmm/lmm-02|Treatment x week, random intercept only]] — a treatment x week
  interaction but with random intercept only — a between-arm trajectory difference
  without random slopes.
- [[lmm/lmm-07|Random slope over sites]] — another random-slope design, but the
  slope varies over clusters (sites) for a treatment effect rather than over time
  within seedlings.
- [[glmm/glmm-04|Random slope on a binary outcome]] — the same random-slope idea
  on a binary outcome: a continuous predictor with a random slope per subject
  (logistic GLMM).

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:lmm-03 -->
```python
from mcpower import MCPower

# Repeated-measures growth curve: each seedling is measured at several weekly
# time points and we ask whether height changes with `week`. `(week | seedling)`
# gives every seedling its own random intercept (baseline height) AND its own
# random slope of week (personal growth rate), so the test of the average
# `week` slope is judged against how much those individual growth rates scatter.
# family="lme" with the default MLE estimator fits the mixed model.
model = MCPower("seedling_height ~ week + (week | seedling)", family="lme")

# `week` is a continuous (normally distributed) within-seedling predictor.
model.set_variable_type("week=normal")

# Average weekly growth slope on the continuous benchmark scale:
#   week=0.25 -> a medium average rate of change across seedlings.
model.set_effects("week=0.25")

# Clustering by seedling: 40 seedlings, conditional ICC 0.3 (baseline
# heights correlate within a seedling after `week` is accounted for), and a
# random slope of `week` whose variance 0.05 sets how much individual growth
# rates differ; slope_intercept_corr ties taller seedlings to faster growth.
model.set_cluster(
    "seedling",
    ICC=0.3,
    n_clusters=40,
    random_slopes=["week"],
    slope_variance=0.05,
    slope_intercept_corr=0.2,
)

model.set_seed(2137)

# Power for the average `week` slope. sample_size is the total number of
# observations (seedlings x measurements per seedling).
model.find_power(sample_size=200, target_test="week")
```
<!-- /chunk:py:lmm-03 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:lmm-03 -->
```r
suppressMessages(library(mcpower))

# Repeated-measures growth curve: each seedling is measured at several weekly
# time points and we ask whether height changes with `week`. `(week | seedling)`
# gives every seedling its own random intercept (baseline height) AND its own
# random slope of week (personal growth rate), so the test of the average
# `week` slope is judged against how much those individual growth rates scatter.
# family="lme" with the default MLE estimator fits the mixed model.
model <- MCPower$new("seedling_height ~ week + (week | seedling)", family = "lme")

# `week` is a continuous (normally distributed) within-seedling predictor.
model$set_variable_type("week=normal")

# Average weekly growth slope on the continuous benchmark scale:
#   week=0.25 -> a medium average rate of change across seedlings.
model$set_effects("week=0.25")

# Clustering by seedling: 40 seedlings, conditional ICC 0.3 (baseline
# heights correlate within a seedling after `week` is accounted for), and a
# random slope of `week` whose variance 0.05 sets how much individual growth
# rates differ; corr_with_intercept ties taller seedlings to faster growth.
model$set_cluster(
  "seedling",
  ICC = 0.3,
  n_clusters = 40L,
  random_slopes = list(
    list(predictor = "week", variance = 0.05, corr_with_intercept = 0.2)
  )
)

model$set_seed(2137)

# Power for the average `week` slope. sample_size is the total number of
# observations (seedlings x measurements per seedling).
invisible(model$find_power(sample_size = 200, target_test = "week"))
```
<!-- /chunk:r:lmm-03 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/lmm-03-setup.png|600|theme-light]]
![[assets/lmm-03-setup-dark.png|600|theme-dark]]

</details>
