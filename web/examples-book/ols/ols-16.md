---
title: "Power for OLS moderation plus a covariate"
description: "Power & sample-size analysis by Monte Carlo simulation for OLS moderation plus a covariate (recovery_days ~ dose * age + baseline_severity). Free, Python & R."
---
# Drug dose moderation by age on recovery, adjusting for baseline severity

You have a continuous outcome `recovery_days`, two continuous predictors `dose`
and `age` whose interaction is the hypothesis, and a third continuous variable
`baseline_severity` you want to hold constant. The moderation question — does
the slope of `dose` change across levels of `age`? — is the same
continuous-by-continuous interaction as before, but here you also adjust for
`baseline_severity` so the interaction is estimated net of that control.

As an MCPower formula this is `recovery_days ~ dose * age + baseline_severity`,
where `*` expands to the two main effects plus their product
(`dose + age + dose:age`) and `+ baseline_severity` adds the control term
additively — it has no interaction with anything. The test that carries the
moderation hypothesis is the interaction coefficient `dose:age`.

## Variations

- **Score the whole model, not just the interaction.** Swap
  `target_test="dose:age"` for `target_test="all"` to get power for each main
  effect, the covariate, the interaction, and the omnibus test in one run.
- **Weaker or stronger moderation.** The interaction is the uncertain term —
  re-run with `dose:age=0.10` (small) or `dose:age=0.25` (medium) to watch how
  fast the required sample size moves once the product term shrinks.
- **A correlated covariate.** Controls are rarely orthogonal to the predictors.
  Add `set_correlations("corr(dose, baseline_severity)=0.3")` to see how shared
  variance with the control erodes power for the interaction.
- **Find the N instead of the power.** Replace the `find_power` call with
  `find_sample_size(target_test="dose:age", from_size=120, to_size=600, by=25)`
  to sweep for the smallest sample that reaches 80% power on the interaction.
- **Same design, other fields:**
  - Ecology: `growth_rate ~ rainfall * temperature + soil_nitrogen` — does the
    rainfall effect on growth rate depend on temperature, adjusting for soil
    nitrogen?
  - Social science: `wage ~ years_education * experience_years + tenure` — does
    the education wage return depend on experience level, adjusting for tenure?

## Not this setup?

- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]]
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]]
- [[ols/ols-05|Three-way continuous interaction]]

## If you'd rather have…

- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]] — Same continuous-by-continuous moderation without the
  extra additive covariate — the direct parent of this setup.
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]] — Covariate adjustment combined with an interaction, but
  the interacting predictor is a group factor (ANCOVA with heterogeneous slopes).
- [[ols/ols-08|ANCOVA: group effect adjusting for a baseline covariate]] — Adjust a group effect for a baseline covariate with no
  interaction — the additive ANCOVA analog of adding a covariate.
- [[ols/ols-02|Two-predictor multiple regression]] — Plain additive multiple regression if you want
  covariate-style controls without any moderation term.
- [[ols/ols-05|Three-way continuous interaction]] — Step up to a full three-way continuous interaction
  instead of one interaction plus an additive control.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-16 -->
```python
from mcpower import MCPower

# Continuous-by-continuous moderation with an additive control variable: does the
# effect of dose on recovery_days depend on age, after adjusting for baseline_severity?
# '*' expands dose * age to dose + age + dose:age; '+ baseline_severity' adds the control
# term only (no interaction with it). The full model is dose + age + dose:age + baseline_severity.
model = MCPower("recovery_days = dose * age + baseline_severity")

# Standardised effects (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   dose=0.30, age=0.25    -> moderate main effects.
#   dose:age=0.15          -> the smaller moderation effect (the test of interest).
#   baseline_severity=0.25 -> a moderate control association we adjust for.
model.set_effects("dose=0.30, age=0.25, dose:age=0.15, baseline_severity=0.25")

# Power for the interaction term at N=220.
model.find_power(sample_size=220, target_test="dose:age")
```
<!-- /chunk:py:ols-16 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-16 -->
```r
suppressMessages(library(mcpower))

# Continuous-by-continuous moderation with an additive control variable: does the
# effect of dose on recovery_days depend on age, after adjusting for baseline_severity?
# '*' expands dose * age to dose + age + dose:age; '+ baseline_severity' adds the control
# term only (no interaction with it). The full model is dose + age + dose:age + baseline_severity.
model <- MCPower$new("recovery_days ~ dose * age + baseline_severity")

# Standardised effects (continuous benchmarks: 0.10 / 0.25 / 0.40).
#   dose=0.30, age=0.25    -> moderate main effects.
#   dose:age=0.15          -> the smaller moderation effect (the test of interest).
#   baseline_severity=0.25 -> a moderate control association we adjust for.
model$set_effects("dose=0.30, age=0.25, dose:age=0.15, baseline_severity=0.25")

# Power for the interaction term at N=220.
invisible(model$find_power(sample_size = 220, target_test = "dose:age"))
```
<!-- /chunk:r:ols-16 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-16-setup.png|600|theme-light]]
![[assets/ols-16-setup-dark.png|600|theme-dark]]

</details>
