---
title: "Power for a logistic interaction (moderation)"
description: "Power & sample-size analysis by Monte Carlo simulation for a continuous-by-continuous interaction (moderation) on a binary logistic outcome. Free, Python & R."
---
# Does a biomarker's effect on relapse depend on the patient's age? (logistic moderation)

You recorded a binary outcome `relapse` (did the patient relapse, yes/no) and
two continuous predictors, `biomarker_level` and `age`. The question is not just
whether each one shifts the odds of relapse on its own, but whether the effect
of `biomarker_level` *changes* across patient ages — a continuous-by-continuous
interaction on a binary outcome. As an MCPower formula that is
`relapse ~ biomarker_level * age` with `family="logit"`, where `*` expands to
both main effects plus their product term
(`biomarker_level + age + biomarker_level:age`) and the outcome is fitted by a
logistic GLM. The test that carries the moderation hypothesis is the interaction
coefficient `biomarker_level:age`.

## Variations

- **Test the whole model, not just the interaction.** Swap
  `target_test="biomarker_level:age"` for `target_test="all"` to get power for
  each main effect, the interaction, and the omnibus test in one run.
- **Weaker or stronger moderation.** The interaction effect is the uncertain
  one — re-run with `biomarker_level:age=0.10` (small) or
  `biomarker_level:age=0.25` (medium) to see how quickly the required sample size moves.
- **Correlated predictors.** Biomarker level and age are rarely independent.
  Add `set_correlations("corr(biomarker_level, age)=0.3")` to see how
  collinearity erodes power for the product term.
- **Find the N instead of the power.** Replace the `find_power` call with
  `find_sample_size(target_test="biomarker_level:age", from_size=150, to_size=800, by=25)`
  to sweep for the smallest sample that reaches 80% power on the interaction.
- **Same design, other fields:**
  - `germinated ~ soil_nitrogen * moisture` — does the effect of soil nitrogen on germination depend on moisture level? (ecology)
  - `employed ~ experience_years * tenure` — does the effect of experience on employment depend on job tenure? (social science)

## Not this setup?

- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]]
- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]]
- [[glm/glm-04|Multiple logistic regression: covariate-adjusted binary outcome]]

## If you'd rather have…

- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]] — logistic interaction, but the moderator is binary ×
  continuous (treatment × biomarker) instead of two continuous predictors.
- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]] — the exact same continuous-by-continuous moderation, on
  a continuous (normal) outcome instead of a binary one.
- [[glm/glm-07|Logistic factor-by-factor interaction (2x2)]] — logistic interaction where both moderators are
  categorical factors (2×2) rather than continuous.
- [[glm/glm-08|Logistic three-way interaction on a binary outcome]] — step up to a logistic three-way interaction on a
  binary outcome.
- [[glm/glm-04|Multiple logistic regression: covariate-adjusted binary outcome]] — drop the interaction: a plain multiple logistic
  regression with adjustment covariates, no moderation.

## Copy-paste setup

<!-- chunk:py:glm-05 -->
```python
from mcpower import MCPower

# Continuous-by-continuous moderation on a binary outcome: does the effect of
# biomarker_level on whether a patient relapses (yes/no) depend on their age?
# '*' expands to both main effects plus the product term
# (biomarker_level + age + biomarker_level:age). family="logit" makes relapse a
# binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("relapse = biomarker_level * age", family="logit")

# family="logit" needs the event rate at the reference level (all predictors at
# their mean): here 30% of patients relapse. This fixes the logistic
# intercept, so it must be set before find_power.
model.set_baseline_probability(0.3)

# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40).
# Both main effects are moderate; the interaction (the test of interest) is
# smaller, as moderation effects usually are.
model.set_effects("biomarker_level=0.30, age=0.25, biomarker_level:age=0.15")

# Power for the interaction term at N=300.
model.find_power(sample_size=300, target_test="biomarker_level:age")
```
<!-- /chunk:py:glm-05 -->

<!-- chunk:r:glm-05 -->
```r
suppressMessages(library(mcpower))

# Continuous-by-continuous moderation on a binary outcome: does the effect of
# biomarker_level on whether a patient relapses (yes/no) depend on their age?
# '*' expands to both main effects plus the product term
# (biomarker_level + age + biomarker_level:age). family="logit" makes relapse a
# binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("relapse ~ biomarker_level * age", family = "logit")

# family="logit" needs the event rate at the reference level (all predictors at
# their mean): here 30% of patients relapse. This fixes the logistic
# intercept, so it must be set before find_power.
model$set_baseline_probability(0.3)

# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40).
# Both main effects are moderate; the interaction (the test of interest) is
# smaller, as moderation effects usually are.
model$set_effects("biomarker_level=0.30, age=0.25, biomarker_level:age=0.15")

# Power for the interaction term at N=300.
invisible(model$find_power(sample_size = 300, target_test = "biomarker_level:age"))
```
<!-- /chunk:r:glm-05 -->

![[assets/glm-05-setup.png|600|theme-light]]
![[assets/glm-05-setup-dark.png|600|theme-dark]]
