---
title: "Power for two-predictor multiple regression"
description: "Power & sample-size analysis by Monte Carlo simulation for two-predictor OLS regression (plant_biomass ~ rainfall + soil_nitrogen). Free, Python & R."
---
# Multiple regression with two continuous predictors

Does annual rainfall predict plant biomass once soil nitrogen is held constant?
Two continuous predictors of a continuous outcome — the classic two-predictor
multiple regression. As an MCPower formula this is
`plant_biomass ~ rainfall + soil_nitrogen`, fit by ordinary least squares:
`rainfall` is the predictor of interest, `soil_nitrogen` a second continuous
variable whose own association you also want enough power to detect.

## Variations

- **One predictor matters, the other is just a control.** Keep
  `target_test="rainfall"` to power only the predictor of interest;
  `soil_nitrogen` then acts as an adjustment covariate you don't need to detect.
- **Different effect-size guesses.** Continuous predictors run on the
  0.10 / 0.25 / 0.40 small / medium / large scale — dial each predictor up or
  down to match your literature estimate.
- **Make a predictor categorical.** Swap continuous `soil_nitrogen` for a binary
  group (`set_variable_type("soil_nitrogen=binary")`) and rescale its effect to
  the 0.20 / 0.50 / 0.80 binary benchmarks.
- **Solve for sample size instead of power.** Replace `find_power(...)` with
  `find_sample_size(target_test="rainfall, soil_nitrogen", from_size=50, to_size=400)`
  to sweep a grid and report the smallest N that reaches 80% power.
- **Correlated predictors.** If `rainfall` and `soil_nitrogen` are themselves
  related, add `set_correlations("corr(rainfall, soil_nitrogen)=0.3")` so the
  simulation reflects the collinearity you expect in real data.
- **Same design, other fields:**
  - Clinical: `cholesterol ~ dose + age` — adjusted effect of dose on
    cholesterol holding age constant; same two-predictor additive OLS structure.
  - Social: `wage ~ years_education + experience_years` — adjusted wage return
    to education controlling for experience; same formula shape.

## Not this setup?

- [[ols/ols-01|plant_biomass ~ rainfall — a single continuous predictor]]
- [[ols/ols-03|plant_biomass ~ rainfall + soil_nitrogen + temperature — three continuous predictors]]
- [[ols/ols-04|plant_biomass ~ rainfall * soil_nitrogen — two predictors with an interaction]]
- [[ols/ols-10|y ~ group — a single categorical predictor]]

## If you'd rather have…

- [[ols/ols-13|Several continuous controls]] — adds more continuous covariates
  when you need to adjust an association for several covariates rather than just
  two predictors.
- [[ols/ols-16|Two predictors that interact, plus a covariate]] — keeps two
  continuous predictors but lets them interact and adds a covariate; choose this
  if a moderation effect plus adjustment is what you actually want.
- [[ols/ols-08|ANCOVA: a group effect adjusted for a baseline]] — a group effect
  adjusted for one continuous baseline covariate, the two-predictor structure
  where one predictor is categorical.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-02 -->
```python
from mcpower import MCPower

# Two continuous predictors of a continuous outcome (OLS).
# rainfall is the predictor of interest; soil_nitrogen is a second continuous covariate
# whose association we also want enough power to detect.
model = MCPower("plant_biomass = rainfall + soil_nitrogen")

# Standardised effect sizes (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   rainfall=0.25 -> a medium association.
#   soil_nitrogen=0.10 -> a small association.
model.set_effects("rainfall=0.25, soil_nitrogen=0.10")

# Both predictors are continuous, so no set_variable_type() is needed.
# OLS defaults apply: 1600 simulations, alpha=0.05, seed=2137.
model.find_power(sample_size=200, target_test="rainfall, soil_nitrogen")
```
<!-- /chunk:py:ols-02 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-02 -->
```r
suppressMessages(library(mcpower))

# Two continuous predictors of a continuous outcome (OLS).
# rainfall is the predictor of interest; soil_nitrogen is a second continuous covariate
# whose association we also want enough power to detect.
model <- MCPower$new("plant_biomass ~ rainfall + soil_nitrogen")

# Standardised effect sizes (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   rainfall=0.25 -> a medium association.
#   soil_nitrogen=0.10 -> a small association.
model$set_effects("rainfall=0.25, soil_nitrogen=0.10")

# Both predictors are continuous, so no set_variable_type() is needed.
# OLS defaults apply: 1600 simulations, alpha=0.05, seed=2137.
invisible(model$find_power(sample_size = 200, target_test = "rainfall, soil_nitrogen"))
```
<!-- /chunk:r:ols-02 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-02-setup.png|600|theme-light]]
![[assets/ols-02-setup-dark.png|600|theme-dark]]

</details>
