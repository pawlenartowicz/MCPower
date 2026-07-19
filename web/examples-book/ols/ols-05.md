---
title: "Power for a three-way continuous interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a three-way OLS interaction (growth_rate ~ temperature * moisture * soil_ph). Free, Python & R."
---
# Three-way continuous interaction (does temperature's moderation of growth rate itself depend on soil pH?)

You measure a continuous outcome `growth_rate` and suspect that the way `temperature` moderates
the effect of `moisture` on growth rate is *itself* conditional on `soil_ph` — a full
three-way moderation. The classic case: an effect of `temperature` that grows with `moisture`,
but where that growth is steeper at some levels of `soil_ph` than others. In MCPower
this is the model `growth_rate ~ temperature * moisture * soil_ph`, where `*` expands to all three main
effects, all three two-way interactions, and the single three-way term
`temperature:moisture:soil_ph` — the coefficient you actually care about here.

Three-way interactions are notoriously underpowered: the highest-order term sits
on the thinnest slice of the design, so plausible effects need large samples.
This page powers the three-way term itself at a medium-ish sample, with main
effects at the medium benchmark (0.25) and every interaction at the small
benchmark (0.10).

## Variations

- **Power every term, not just the three-way.** Drop `target_test="temperature:moisture:soil_ph"`
  and the default reports the omnibus F plus every coefficient — useful to see
  how much more sample the three-way needs than the main effects.
- **Probe the two-way interactions too.** Use `target_test="temperature:moisture, temperature:soil_ph, moisture:soil_ph"`
  to report the three lower-order interactions side by side.
- **Bigger three-way effect.** If theory says the highest-order term is
  substantial, raise `temperature:moisture:soil_ph` to the medium continuous benchmark (0.25) — the
  required sample drops sharply.
- **Search for the sample, don't guess it.** Swap `find_power(sample_size=300, …)`
  for `find_sample_size(target_test="temperature:moisture:soil_ph")` to get the smallest N that hits
  80% power for the three-way term.
- **Correlated predictors.** Bump `corr(temperature,moisture)=0.2` higher, or add
  `corr(temperature,soil_ph)` / `corr(moisture,soil_ph)`, to reflect predictors that travel together —
  collinearity erodes interaction power fast.
- **Same design, other fields:**
  - Clinical: `recovery_days ~ dose * age * baseline_severity` — does a dose×age moderation of recovery time itself depend on baseline disease severity?
  - Social: `well_being ~ income * social_support * years_education` — does the income×social-support moderation of well-being further depend on education level?

## Not this setup?

- [[ols/ols-04]] — two continuous predictors interacting (`income * social_support`), the
  simpler two-way moderation without a third variable.
- [[ols/ols-06]] — a neighbouring three-predictor design with a different
  interaction structure.

## If you'd rather have…

- [[ols/ols-04|Two continuous predictors interacting]] — just `temperature * moisture` instead
  of three, the simpler moderation case.
- [[ols/ols-16|A two-way interaction plus an additive covariate]] — moderation
  while adjusting for another predictor.
- [[ols/ols-03|Three continuous predictors, additive only]] — main effects of
  `temperature`, `moisture`, `soil_ph` with no interactions.
- [[ols/ols-09|An interaction with a categorical term]] — `group * baseline`,
  where one side of the interaction is a factor rather than continuous.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-05 -->
```python
from mcpower import MCPower

model = MCPower("growth_rate ~ temperature * moisture * soil_ph", family="ols")
model.set_effects(
    "temperature=0.25, moisture=0.25, soil_ph=0.25, "
    "temperature:moisture=0.10, temperature:soil_ph=0.10, moisture:soil_ph=0.10, "
    "temperature:moisture:soil_ph=0.10"
)
model.set_correlations("corr(temperature,moisture)=0.2")
model.set_simulations(1600)
model.set_seed(2137)

model.find_power(sample_size=300, target_test="temperature:moisture:soil_ph")
```
<!-- /chunk:py:ols-05 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-05 -->
```r
suppressMessages(library(mcpower))

model <- MCPower$new("growth_rate ~ temperature * moisture * soil_ph", family = "ols")
model$set_effects(paste0(
  "temperature=0.25, moisture=0.25, soil_ph=0.25, ",
  "temperature:moisture=0.10, temperature:soil_ph=0.10, moisture:soil_ph=0.10, ",
  "temperature:moisture:soil_ph=0.10"
))
model$set_correlations("corr(temperature,moisture)=0.2")
model$set_simulations(1600)
model$set_seed(2137)

model$find_power(sample_size = 300, target_test = "temperature:moisture:soil_ph")
```
<!-- /chunk:r:ols-05 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-05-setup.png|600|theme-light]]
![[assets/ols-05-setup-dark.png|600|theme-dark]]

</details>
