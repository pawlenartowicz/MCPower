---
title: "Power for a categorical (factor) predictor"
description: "Power & sample-size analysis by Monte Carlo simulation for a 3-level categorical OLS predictor (abundance ~ habitat). Free, Python & R."
---
# One-way design: species abundance across three habitat types

You surveyed species abundance across three habitat types (forest, grassland,
and wetland) and want the power to detect that some habitats differ from the
reference in abundance, read off the individual dummy coefficients rather than a
single omnibus verdict.

As an MCPower model this is `abundance ~ habitat`, where `habitat` is a 3-level
factor. OLS expands it into two dummy contrasts, each comparing a non-reference
habitat against the reference habitat, and reports power for each contrast
separately.

## Variations

- **More than three habitats.** Bump the factor to its real level count —
  `habitat=(factor,4)` for a four-habitat design. Each extra level adds one more
  dummy contrast (and one more row of power) against the same reference.
- **Unequal expected effects per habitat.** Setting both to `0.5` puts the same
  shift on every non-reference level. To say one habitat differs more strongly,
  set the level effects apart: `habitat[2]=0.3, habitat[3]=0.7`.
- **A covariate to soak up noise.** Add a continuous control, e.g.
  `abundance ~ habitat + rainfall`, with `rainfall` on the continuous benchmark
  scale (`rainfall=0.25`) — an analysis-of-covariance flavour that usually lifts
  power on the `habitat` contrasts.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="habitat[2], habitat[3]", from_size=50, to_size=400, by=10)`
  to get the smallest N that reaches target power on both contrasts.
- **Same design, other fields:**
  - Clinical: `pain_score ~ dose_level` — pain score compared across three dose
    levels (placebo, low, and high), with `dose_level=(factor,3)`.
  - Social science: `job_satisfaction ~ sector` — job satisfaction compared
    across three employment sectors (public, private, and nonprofit), with
    `sector=(factor,3)`.

## Not this setup?

- [[ols/ols-07|Two groups instead of three]]
- [[anova/anova-01|The omnibus F-test of overall group differences]]
- [[ols/ols-14|A group factor interacting with a continuous predictor]]

## If you'd rather have…

- [[ols/ols-07]] — Only two groups instead of three: an independent t-test
  recast as regression with a single dummy.
- [[anova/anova-01]] — Same 3+ group structure, but you want the omnibus
  F-test of overall group differences rather than per-dummy coefficients.
- [[anova/anova-02]] — Three-level group factor where you also want all
  pairwise post-hoc comparisons, not just the dummy contrasts vs the reference.
- [[ols/ols-14]] — Same dummy-coded factor but interacting with a continuous
  predictor (moderation).
- [[ols/ols-15]] — Two categorical predictors interacting: a factorial design
  as regression.
- [[glm/glm-03]] — Same multi-level categorical predictor but on a binary
  outcome (logistic).

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-12 -->
```python
from mcpower import MCPower

# Three habitat types: one continuous outcome, one 3-level grouping factor.
# Research question: do the habitat types differ on species abundance,
# tested as the two dummy contrasts against the reference habitat?
model = MCPower("abundance = habitat")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
model.set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts the outcome by a medium amount vs reference.
# Factors expand to per-level dummies (habitat[2], habitat[3]); effects
# and targets are addressed by those dummy names, not the bare factor.
model.set_effects("habitat[2]=0.5, habitat[3]=0.5")

# Power at N=150 for both dummy contrasts (each non-reference level vs reference).
model.find_power(sample_size=150, target_test="habitat[2], habitat[3]")
```
<!-- /chunk:py:ols-12 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-12 -->
```r
suppressMessages(library(mcpower))

# Three habitat types: one continuous outcome, one 3-level grouping factor.
# Research question: do the habitat types differ on species abundance,
# tested as the two dummy contrasts against the reference habitat?
model <- MCPower$new("abundance ~ habitat")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
model$set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts the outcome by a medium amount vs reference.
# Factors expand to per-level dummies (habitat[2], habitat[3]); effects
# and targets are addressed by those dummy names, not the bare factor.
model$set_effects("habitat[2]=0.5, habitat[3]=0.5")

# Power at N=150 for both dummy contrasts (each non-reference level vs reference).
invisible(model$find_power(sample_size = 150, target_test = "habitat[2], habitat[3]"))
```
<!-- /chunk:r:ols-12 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-12-setup.png|600|theme-light]]
![[assets/ols-12-setup-dark.png|600|theme-dark]]

</details>
