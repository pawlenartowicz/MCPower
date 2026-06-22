---
title: "Power for an adjusted two-group regression"
description: "Power & sample-size analysis by Monte Carlo simulation for an adjusted two-group OLS (monthly_income ~ union_member + experience_years). Free, Python & R."
---
# Adjusted two-group comparison: union membership predicting monthly income controlling for experience

You are comparing monthly income between union members and non-members, but
you want the comparison to hold years of experience fixed. The question: does
`monthly_income` differ by `union_member` once you adjust for `experience_years`?
This is the ANCOVA / parallel-slopes regression, and as an MCPower formula it
is `monthly_income ~ union_member + experience_years` — a binary predictor
(`union_member`) and a continuous predictor (`experience_years`) entered as
separate main effects, with the two groups sharing a single experience slope.
The power you care about is usually the `union_member` effect: the group gap
that survives adjustment for `experience_years`.

## Variations

- **Swap the covariate's role.** If `experience_years` is the predictor of
  interest and `union_member` is the nuisance control, nothing in the setup
  changes — just read power off `experience_years` instead of `union_member` by
  setting `target_test="experience_years"`.
- **Make the group difference larger or smaller.** The `union_member=0.50`
  benchmark is a medium binary effect; drop it to `0.20` for a small group gap
  or raise it to `0.80` for a large one and re-run to see how the required
  sample size moves.
- **Add a second continuous control.** Append another covariate, e.g.
  `monthly_income ~ union_member + experience_years + tenure`, and give it its
  own effect — the `union_member` power then reflects adjustment for both
  controls.
- **Let the covariate be a factor instead.** Swap the continuous `experience_years`
  for a 3-level factor such as `sector`; the slope term becomes a set of dummy
  contrasts rather than one coefficient.
- **Search for N instead of fixing it.** Replace `find_power` with
  `find_sample_size(target_test="union_member", from_size=50, to_size=400, by=10)`
  to find the smallest N that reaches 80% power on the union gap.
- **Same design, other fields:**
  - Clinical: `blood_pressure ~ treatment + baseline_bp` — treatment effect on
    blood pressure adjusting for the patient's baseline measurement.
  - Ecology: `plant_biomass ~ habitat + rainfall` — habitat effect on biomass
    adjusting for local rainfall.

## Not this setup?

- [[ols/ols-11|Group difference where the slope itself differs by group]] — when
  the two groups should *not* share one experience slope.
- [[ols/ols-08|A different mix of binary and continuous predictors]].
- [[ols/ols-02|Sample-size search for a continuous-predictor model]].

## If you'd rather have…

- [[ols/ols-11|Let gender and experience_years interact (gender moderates the experience slope)]] —
  instead of parallel slopes, fit `wage ~ gender * experience_years` so the
  experience effect is allowed to differ between the two groups.
- [[ols/ols-07|Compare the two groups alone (independent t-test)]] — drop the
  continuous covariate entirely and just test the group difference.
- [[ols/ols-13|Adjust for several continuous controls]] — when one covariate is
  not enough and you need to net out a whole block of continuous variables.
- [[ols/ols-01|A single continuous predictor, no grouping]] — when there is no
  group variable at all and you only have one continuous predictor.

## Copy-paste setup

<!-- chunk:py:ols-10 -->
```python
from mcpower import MCPower

# Continuous outcome monthly_income on a binary group (union_member) plus a
# continuous covariate (experience_years) — parallel slopes, no interaction.
# This is the ANCOVA/adjusted two-group comparison: the union_member effect is
# the group gap holding experience_years fixed.
model = MCPower("monthly_income = union_member + experience_years")

# union_member is the binary grouping variable; experience_years stays
# continuous (the default).
model.set_variable_type("union_member=binary")

# Fabricated-plausible effects on the benchmark scale: a medium binary group
# gap (0.50) and a medium continuous experience_years slope (0.25).
model.set_effects("union_member=0.50, experience_years=0.25")

model.find_power(sample_size=120, target_test="all", verbose=False)
```
<!-- /chunk:py:ols-10 -->

<!-- chunk:r:ols-10 -->
```r
suppressMessages(library(mcpower))

# Continuous outcome monthly_income on a binary group (union_member) plus a
# continuous covariate (experience_years) -- parallel slopes, no interaction.
# This is the ANCOVA/adjusted two-group comparison: the union_member effect is
# the group gap holding experience_years fixed.
model <- MCPower$new("monthly_income ~ union_member + experience_years")

# union_member is the binary grouping variable; experience_years stays
# continuous (the default).
model$set_variable_type("union_member=binary")

# Fabricated-plausible effects on the benchmark scale: a medium binary group
# gap (0.50) and a medium continuous experience_years slope (0.25).
model$set_effects("union_member=0.50, experience_years=0.25")

invisible(model$find_power(sample_size = 120, target_test = "all", verbose = FALSE))
```
<!-- /chunk:r:ols-10 -->

![[assets/ols-10-setup.png|600|theme-light]]
![[assets/ols-10-setup-dark.png|600|theme-dark]]
