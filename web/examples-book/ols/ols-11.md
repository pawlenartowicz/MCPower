---
title: "Power for binary-by-continuous moderation"
description: "Power & sample-size analysis by Monte Carlo simulation for binary-by-continuous OLS moderation (wage ~ gender * experience_years). Free, Python & R."
---
# Does the experience slope on wage differ between gender groups?

You measured wage as a continuous outcome and want to know whether the slope of
`experience_years` on `wage` is *different* for the two `gender` groups — a
binary-by-continuous moderation (a "does the experience effect depend on
gender?" test). In MCPower terms the model is `wage ~ gender * experience_years`,
where `*` expands to `gender + experience_years + gender:experience_years`; the
`gender:experience_years` term is the moderation you actually care about.

## Variations

- Swap the binary `gender` for a 3-level factor (e.g. `sector` with public /
  private / nonprofit): write `wage ~ sector * experience_years` and set
  `sector=factor`. The interaction then spans one slope-difference per
  non-reference level.
- Make `experience_years` the moderator of a binary union membership instead —
  same `gender * experience_years` shape, just relabelled; the interpretation
  flips to "does the union wage premium grow with experience?".
- Probe a stronger or weaker moderation by moving the `gender:experience_years`
  effect between the small / medium / large benchmarks (0.10 / 0.25 / 0.40) to
  bracket the sample size you'd need.
- Add a nuisance covariate you must adjust for (e.g. `+ tenure`) without letting
  it interact — keep `gender * experience_years` and append `+ tenure` as a main
  effect only.
- **Same design, other fields:**
  - Clinical: `blood_pressure ~ treatment * baseline_bp` — does the baseline
    slope on blood pressure differ between treatment arms?
  - Ecology: `plant_biomass ~ habitat * rainfall` — does habitat moderate the
    rainfall slope on plant biomass?

## Not this setup?

- [[ols/ols-10|Both main effects, no interaction (parallel slopes)]]
- [[ols/ols-04|Two continuous moderators (income * social_support)]]
- [[ols/ols-09|ANCOVA homogeneity-of-slopes (treatment * baseline_bp)]]

## If you'd rather have…

- [[ols/ols-10|Same predictors, parallel slopes]] — same `union_member` +
  `experience_years`, but no interaction. Use this if you only need both main
  effects and don't expect the experience slope to differ by group.
- [[ols/ols-04|Both moderators continuous]] — same moderation structure with
  `income * social_support` instead of binary-by-continuous.
- [[ols/ols-09|Group-by-covariate as ANCOVA]] — the same binary-by-continuous
  moderation recast as a homogeneity-of-slopes test (`treatment * baseline_bp`).
- [[ols/ols-14|Multi-level factor × continuous]] — `habitat * rainfall`,
  the categorical-with-3+-levels generalisation of this binary moderator.
- [[ols/ols-16|Moderation plus a covariate]] — extend this design when you must
  adjust the interaction for a third predictor.

## Copy-paste setup

<!-- chunk:py:ols-11 -->
```python
from mcpower import MCPower

# Does the slope of `experience_years` on the outcome differ between the two
# `gender` groups? `gender * experience_years` expands to gender +
# experience_years + gender:experience_years, so the interaction term carries
# the moderation. gender is binary; experience_years is continuous.
model = MCPower("wage = gender * experience_years")
model.set_effects("gender=0.5, experience_years=0.25, gender:experience_years=0.2")
model.set_variable_type("gender=binary")

# Power for the interaction term — moderation is the question, so target it.
model.find_power(sample_size=300, target_test="gender:experience_years")
```
<!-- /chunk:py:ols-11 -->

<!-- chunk:r:ols-11 -->
```r
suppressMessages(library(mcpower))

# Does the slope of `experience_years` on the outcome differ between the two
# `gender` groups? `gender * experience_years` expands to gender +
# experience_years + gender:experience_years, so the interaction term carries
# the moderation. gender is binary; experience_years is continuous.
model <- MCPower$new("wage ~ gender * experience_years")
model$set_effects("gender=0.5, experience_years=0.25, gender:experience_years=0.2")
model$set_variable_type("gender=binary")

# Power for the interaction term — moderation is the question, so target it.
invisible(model$find_power(sample_size = 300, target_test = "gender:experience_years"))
```
<!-- /chunk:r:ols-11 -->

The headline number is the power for `gender:experience_years`. Interaction
terms are expensive — detecting a moderation needs a markedly larger sample than
either main effect, so expect the interaction row to trail `gender` and
`experience_years`. Drop `target_test` to `"all"` to see all three terms side
by side, or switch to `find_sample_size(...)` to find the N that lands the
interaction at 80% power.

![[assets/ols-11-setup.png|600|theme-light]]
![[assets/ols-11-setup-dark.png|600|theme-dark]]
