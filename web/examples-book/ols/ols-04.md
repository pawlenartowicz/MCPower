---
title: "Power for a continuous interaction (moderation)"
description: "Power & sample-size analysis by Monte Carlo simulation for a continuous OLS interaction (well_being ~ income * social_support). Free, Python & R."
---
# Does income's effect on well-being depend on social support? (continuous moderation)

You measured a continuous outcome `well_being` and two continuous predictors, `income` and
`social_support`. The question is not just whether each one matters on its own, but whether
the slope of `income` *changes* across levels of `social_support` — a continuous-by-continuous
interaction. As an MCPower formula that is `well_being ~ income * social_support`, where `*`
expands to both main effects plus their product term (`income + social_support + income:social_support`).
The test that carries the moderation hypothesis is the interaction coefficient `income:social_support`.

## Variations

- **Test the whole model, not just the interaction.** Swap `target_test="income:social_support"`
  for `target_test="all"` to get power for each main effect, the interaction,
  and the omnibus test in one run.
- **Weaker or stronger moderation.** The interaction effect is the uncertain
  one — re-run with `income:social_support=0.10` (small) or `income:social_support=0.25` (medium)
  to see how quickly the required sample size moves.
- **Correlated predictors.** Real moderators are rarely independent of the
  variable they moderate. Add `set_correlations("corr(income, social_support)=0.3")` to see how
  collinearity erodes power for the product term.
- **Find the N instead of the power.** Replace the `find_power` call with
  `find_sample_size(target_test="income:social_support", from_size=100, to_size=600, by=25)`
  to sweep for the smallest sample that reaches 80% power on the interaction.
- **Same design, other fields:**
  - Clinical: `blood_pressure ~ dose * baseline_bp` — does a drug's blood-pressure effect depend on the patient's baseline level?
  - Ecology: `growth_rate ~ temperature * moisture` — does the temperature effect on growth rate depend on soil moisture?

## Not this setup?

- [[ols/ols-02|Two-predictor multiple regression]]
- [[ols/ols-05|Three-way continuous interaction]]
- [[ols/ols-16|Continuous moderation with a covariate]]
- [[ols/ols-06|Interaction-only term (no lower-order main of one predictor)]]

## If you'd rather have…

- [[ols/ols-16|Continuous moderation with a covariate]] — same continuous `income*social_support` moderation but with an added
  covariate to adjust for.
- [[ols/ols-05|Three-way continuous interaction]] — extend to a three-way continuous interaction
  (`income * social_support * years_education`).
- [[ols/ols-11|Binary-by-continuous moderation]] — moderation where the moderator is binary instead of
  continuous (`sex * age`).
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]] — moderation by a continuous covariate in a group
  design (ANCOVA, homogeneity of slopes).
- [[glm/glm-05|Logistic continuous-by-continuous moderation]] — the same continuous-by-continuous moderation on a
  binary outcome (logistic).

## Copy-paste setup

<!-- chunk:py:ols-04 -->
```python
from mcpower import MCPower

# Continuous-by-continuous moderation: does the effect of income on well_being depend on social_support?
# '*' expands to the two main effects plus their interaction (income + social_support + income:social_support).
model = MCPower("well_being = income * social_support")

# Standardised effects. Main effects are moderate; the interaction (the test of
# interest) is smaller, as moderation effects usually are.
model.set_effects("income=0.30, social_support=0.25, income:social_support=0.15")

# Power for the interaction term at N=200.
model.find_power(sample_size=200, target_test="income:social_support")
```
<!-- /chunk:py:ols-04 -->

<!-- chunk:r:ols-04 -->
```r
suppressMessages(library(mcpower))

# Continuous-by-continuous moderation: does the effect of income on well_being depend on social_support?
# '*' expands to the two main effects plus their interaction (income + social_support + income:social_support).
model <- MCPower$new("well_being ~ income * social_support")

# Standardised effects. Main effects are moderate; the interaction (the test of
# interest) is smaller, as moderation effects usually are.
model$set_effects("income=0.30, social_support=0.25, income:social_support=0.15")

# Power for the interaction term at N=200.
invisible(model$find_power(sample_size = 200, target_test = "income:social_support"))
```
<!-- /chunk:r:ols-04 -->

![[assets/ols-04-setup.png|600|theme-light]]
![[assets/ols-04-setup-dark.png|600|theme-dark]]
