---
title: "Power for logistic regression with a categorical predictor"
description: "Power & sample-size analysis by Monte Carlo simulation for logistic regression of a binary outcome on a multi-level categorical predictor. Free, Python & R."
---
# Logistic one-way design: survival across three habitat types

You sampled seedlings across three habitat types — a reference habitat and two
alternatives — and recorded a binary outcome for each seedling: did it survive
(yes/no)? You want the power to detect that the habitat types differ in survival
probability, read off the individual habitat coefficients rather than a single
omnibus verdict.

As an MCPower model this is `survived ~ habitat` with `family="logit"`,
where `habitat` is a 3-level factor and `survived` is the binary outcome.
The logistic GLM expands the factor into two dummy contrasts, each comparing a
non-reference habitat against the reference habitat on the log-odds scale, and
reports power for each contrast separately.

## Variations

- **More than three habitats.** Bump the factor to its real level count —
  `habitat=(factor,4)` for a four-habitat study. Each extra level adds one
  more dummy contrast (and one more row of power) against the same reference habitat.
- **Unequal expected effects per habitat.** A single `habitat=0.5` puts the
  same log-odds shift on every non-reference habitat. To say the second habitat
  differs more strongly, set the level effects apart rather than sharing one
  number across the factor.
- **A covariate to soak up noise.** Add a continuous control, e.g.
  `survived ~ habitat + soil_nitrogen`, with `soil_nitrogen` on the
  continuous benchmark scale (`soil_nitrogen=0.25`) — adjusting the binary
  outcome for a prognostic covariate.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="habitat", from_size=100, to_size=600, by=20)`
  to get the smallest N that reaches target power on both contrasts.
- **Same design, other fields:**
  - `relapse ~ treatment_arm` — does treatment arm predict relapse in a three-arm clinical trial? (clinical)
  - `employed ~ sector` — do employment rates differ across three industry sectors? (social science)

## Not this setup?

- [[glm/glm-02|A binary predictor instead of a 3-level factor]]
- [[glm/glm-01|A single continuous predictor on the binary outcome]]
- [[ols/ols-12|The same three-arm design but on a continuous outcome]]

## If you'd rather have…

- [[ols/ols-12]] — Same multi-level (3-level) categorical predictor, but on a
  continuous outcome instead of a binary one.
- [[glm/glm-09]] — Logistic outcome where a categorical predictor enters as a
  control alongside a continuous predictor.
- [[anova/anova-01]] — Omnibus test of differences across 3+ groups — the
  linear-model analogue of comparing habitat types.
- [[glm/glm-04]] — Multiple logistic regression adjusting the binary outcome for
  additional covariates.

## Copy-paste setup

<!-- chunk:py:glm-03 -->
```python
from mcpower import MCPower

# Three-habitat ecology study with a yes/no outcome: did the seedling survive?
# Research question: do the habitat types differ in seedling survival probability,
# tested as the two dummy contrasts against the reference habitat.
# family="logit" makes survived a binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("survived = habitat", family="logit")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
# A 3-level factor expands into per-level dummies habitat[2] and
# habitat[3] (level 1 is the reference); effects and tests address those
# dummy names directly, not the bare factor name.
model.set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference habitat shifts the log-odds of survival by a medium amount
# relative to the reference habitat.
model.set_effects("habitat[2]=0.5, habitat[3]=0.5")

# family="logit" requires a baseline event rate: the survival probability in the
# reference habitat when all predictors are at their reference. It sets the model
# intercept (log-odds) for every Monte Carlo iteration; without it find_power
# raises. 0.30 = a 30% reference-habitat survival rate.
model.set_baseline_probability(0.30)

# Power at N=300 for both dummy contrasts (each habitat vs the reference habitat).
model.find_power(sample_size=300, target_test="habitat[2], habitat[3]")
```
<!-- /chunk:py:glm-03 -->

<!-- chunk:r:glm-03 -->
```r
suppressMessages(library(mcpower))

# Three-habitat ecology study with a yes/no outcome: did the seedling survive?
# Research question: do the habitat types differ in seedling survival probability,
# tested as the two dummy contrasts against the reference habitat.
# family="logit" makes survived a binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("survived ~ habitat", family = "logit")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
# A 3-level factor expands into per-level dummies habitat[2] and
# habitat[3] (level 1 is the reference); effects and tests address those
# dummy names directly, not the bare factor name.
model$set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference habitat shifts the log-odds of survival by a medium amount
# relative to the reference habitat.
model$set_effects("habitat[2]=0.5, habitat[3]=0.5")

# family="logit" requires a baseline event rate: the survival probability in the
# reference habitat when all predictors are at their reference. It sets the model
# intercept (log-odds) for every Monte Carlo iteration; without it find_power
# raises. 0.30 = a 30% reference-habitat survival rate.
model$set_baseline_probability(0.30)

# Power at N=300 for both dummy contrasts (each habitat vs the reference habitat).
invisible(model$find_power(sample_size = 300, target_test = "habitat[2], habitat[3]"))
```
<!-- /chunk:r:glm-03 -->

![[assets/glm-03-setup.png|600|theme-light]]
![[assets/glm-03-setup-dark.png|600|theme-dark]]
