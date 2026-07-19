---
title: "Power analysis for simple logistic regression"
description: "Power & sample-size analysis by Monte Carlo simulation for logistic regression of a binary outcome on one continuous predictor. Free, Python & R."
---
# Simple logistic regression: biomarker level predicting relapse

Does a patient's `biomarker_level` shift the odds of `relapse` (yes/no)? That is
the starting point for a binary outcome: one continuous predictor, no covariates,
no interactions. As an MCPower formula this is `relapse ~ biomarker_level` with
`family="logit"`: a binary outcome fitted by a logistic GLM, one continuous
predictor, nothing else held constant.

## Variations

- Dial the expected association up or down by changing the effect: `biomarker_level=0.10`
  for a small relationship, `biomarker_level=0.40` for a large one (the medium benchmark
  for a continuous predictor is `0.25`).
- Searching for the sample size that reaches 80% power instead of scoring a
  fixed N? Swap `find_power(sample_size=200, ...)` for
  `find_sample_size(target_test="biomarker_level", from_size=50, to_size=500, by=10)`.
- **Same design, other fields:**
  - `germination_rate ~ soil_nitrogen` — does soil nitrogen predict whether a seed germinates? (ecology)
  - `voted ~ social_support` — does social support shift the odds of turning out to vote? (social science)

## Not this setup?

- [[glm/glm-02|Two-group / binary predictor (chi-square recast)]]
- [[glm/glm-03|Multi-level categorical predictor (outcome ~ group)]]
- [[glm/glm-04|Adjusted binary-outcome model with covariates]]

## If you'd rather have…

- [[glm/glm-02|Two-group comparison of proportions]] — the predictor is a
  two-group/binary variable instead of continuous (chi-square recast).
- [[glm/glm-03|Categorical predictor]] — the predictor is a multi-level
  categorical instead of a single continuous one.
- [[glm/glm-04|Adjusted association with covariates]] — add covariates for an
  adjusted binary-outcome model (age, gender).
- [[glm/glm-05|Moderation between two predictors]] — add a second continuous
  predictor with their interaction.
- [[glm/glm-09|Continuous predictor with a categorical control]] — keep the
  continuous predictor but add a categorical control.
- [[ols/ols-01|Same design on a continuous outcome]] — the same
  one-continuous-predictor design but on a continuous outcome (linear instead of
  logistic).

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:glm-01 -->
```python
from mcpower import MCPower

# Simple logistic regression: a binary (yes/no) outcome on one continuous
# predictor. family="logit" makes the outcome binary and fits a GLM.
model = MCPower("relapse = biomarker_level", family="logit")

# family="logit" requires a baseline event rate: the probability of relapse=1
# at the predictor's reference level. This pins the intercept (log-odds).
model.set_baseline_probability(0.3)

# Expected effect on the standardised benchmark scale (continuous predictor):
#   biomarker_level=0.25 -> a medium association with the log-odds of relapse.
model.set_effects("biomarker_level=0.25")

# Power at N=200 with the GLM defaults (1600 sims, alpha=0.05, seed=2137).
model.find_power(sample_size=200, target_test="biomarker_level")
```
<!-- /chunk:py:glm-01 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:glm-01 -->
```r
suppressMessages(library(mcpower))

# Simple logistic regression: a binary (yes/no) outcome on one continuous
# predictor. family = "logit" makes the outcome binary and fits a GLM.
model <- MCPower$new("relapse ~ biomarker_level", family = "logit")

# family = "logit" requires a baseline event rate: the probability of relapse=1
# at the predictor's reference level. This pins the intercept (log-odds).
model$set_baseline_probability(0.3)

# Expected effect on the standardised benchmark scale (continuous predictor):
#   biomarker_level=0.25 -> a medium association with the log-odds of relapse.
model$set_effects("biomarker_level=0.25")

# Power at N=200 with the GLM defaults (1600 sims, alpha=0.05, seed=2137).
invisible(model$find_power(sample_size = 200, target_test = "biomarker_level"))
```
<!-- /chunk:r:glm-01 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/glm-01-setup.png|600|theme-light]]
![[assets/glm-01-setup-dark.png|600|theme-dark]]

</details>
