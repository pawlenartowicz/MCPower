---
title: "Power for an interaction-only regression term"
description: "Power & sample-size analysis by Monte Carlo simulation for an interaction-only OLS term (yield ~ nitrogen + nitrogen:water). Free, Python & R."
---
# Moderation without the moderator's main effect (nitrogen predicts yield; water modulates the slope)

You expect `nitrogen` to drive a continuous outcome `yield`, and `water` to *change how strong
that nitrogen effect is* — but `water` has no average effect on yield of its own. This is the
deliberately reduced moderation model `yield ~ nitrogen + nitrogen:water`: the `nitrogen` main
slope plus the `nitrogen:water` product term, with `water`'s own main effect left out on purpose
(`:` adds the interaction only, never the lower-order term).

Both predictors are continuous, so the interaction column is the product of two
uncorrelated standard normals — it has unit variance, and the continuous
effect-size benchmarks (small 0.10 / medium 0.25 / large 0.40) apply to it
directly.

## Variations

- **Put `water`'s main effect back in.** Switch the formula to `yield ~ nitrogen * water`, which
  expands to `nitrogen + water + nitrogen:water`, and add a `water=` effect — the standard full
  moderation model when you don't want to assume `water` is inert on average.
- **Make the moderator a group.** Swap continuous `water` for a binary `water` (via
  `set_variable_type("water=binary")`); the interaction now asks whether `nitrogen`'s
  slope differs between two groups, and the binary benchmarks (0.20 / 0.50 /
  0.80) apply to it.
- **Let the predictors correlate.** Add `set_correlations("corr(nitrogen, water)=0.3")`;
  once the components are correlated the interaction column is no longer
  unit-variance, so treat its benchmark effect size as approximate.
- **Find the N instead of the power.** Replace the `find_power` call with
  `find_sample_size(target_test="nitrogen:water", from_size=100, to_size=600, by=25)` to
  sweep for the smallest sample that detects the moderation.
- **Same design, other fields:**
  - Clinical: `recovery_days ~ treatment + treatment:baseline_severity` — treatment predicts recovery, baseline severity modulates the treatment slope, but baseline severity has no direct average effect.
  - Social: `wage ~ years_education + years_education:urban` — education predicts wage, urban/rural status modulates the education slope, but urban itself has no direct main effect.

## Not this setup?

- [[ols/ols-04|well_being ~ income * social_support — full continuous moderation]]
- [[ols/ols-02|plant_biomass ~ rainfall + soil_nitrogen — additive two-predictor regression]]
- [[ols/ols-16|recovery_days ~ dose * age + baseline_severity — moderation with control]]

## If you'd rather have…

- [[ols/ols-04|Full continuous-by-continuous moderation (well_being ~ income * social_support)]] — keeps
  both lower-order main effects plus the interaction, the standard form when you
  don't want to omit `water`'s main effect.
- [[ols/ols-02|Plain additive two-predictor regression (plant_biomass ~ rainfall + soil_nitrogen)]] — the
  no-interaction baseline if the product term isn't of interest.
- [[ols/ols-16|Continuous moderation with an added covariate (recovery_days ~ dose * age + baseline_severity)]]
  — interaction plus statistical control.
- [[ols/ols-05|Three-way continuous interaction (growth_rate ~ temperature * moisture * soil_ph)]] — scales the
  interaction structure up to a third predictor.
- [[ols/ols-11|Binary-by-continuous moderation (wage ~ gender * experience_years)]] — same
  interaction idea when the moderator is a two-group factor instead of
  continuous.

## Copy-paste setup

<!-- chunk:py:ols-06 -->
```python
from mcpower import MCPower

# Moderation without water's own main effect: nitrogen predicts yield, and that slope is
# modulated by water (nitrogen:water), but water has no average effect of its own.
# `:` is interaction-only — it adds the product term without water's main effect.
model = MCPower("yield = nitrogen + nitrogen:water")

# Both predictors are continuous (the default), so the interaction column is the
# product of two uncorrelated standard normals — unit variance, and the
# continuous benchmarks (0.10 / 0.25 / 0.40) apply to it directly.
#   nitrogen=0.40       -> large main slope.
#   nitrogen:water=0.25 -> medium moderation of that slope.
model.set_effects("nitrogen=0.40, nitrogen:water=0.25")

# Power for the interaction term — the hard-to-detect quantity here.
model.find_power(sample_size=200, target_test="nitrogen:water")
```
<!-- /chunk:py:ols-06 -->

<!-- chunk:r:ols-06 -->
```r
suppressMessages(library(mcpower))

# Moderation without water's own main effect: nitrogen predicts yield, and that slope is
# modulated by water (nitrogen:water), but water has no average effect of its own.
# `:` is interaction-only — it adds the product term without water's main effect.
model <- MCPower$new("yield ~ nitrogen + nitrogen:water")

# Both predictors are continuous (the default), so the interaction column is the
# product of two uncorrelated standard normals — unit variance, and the
# continuous benchmarks (0.10 / 0.25 / 0.40) apply to it directly.
#   nitrogen=0.40       -> large main slope.
#   nitrogen:water=0.25 -> medium moderation of that slope.
model$set_effects("nitrogen=0.40, nitrogen:water=0.25")

# Power for the interaction term — the hard-to-detect quantity here.
model$find_power(sample_size = 200, target_test = "nitrogen:water")
```
<!-- /chunk:r:ols-06 -->

![[assets/ols-06-setup.png|600|theme-light]]
![[assets/ols-06-setup-dark.png|600|theme-dark]]
