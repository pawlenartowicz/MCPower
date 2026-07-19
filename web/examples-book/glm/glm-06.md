---
title: "Power for a logistic treatment-by-covariate interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a binary-by-continuous interaction (treatment x covariate) on a logistic outcome. Free, Python & R."
---
# Logistic regression with a treatment-by-biomarker interaction: remission ~ treatment * biomarker_level

You ran a two-arm trial with a yes/no outcome — did the patient reach remission?
— and measured each patient's biomarker level at baseline. Before trusting a
single average treatment effect on the odds of remission, you want to know
whether the treatment works differently depending on the patient's biomarker
profile: does the log-odds slope on `biomarker_level` differ between the
treatment and control arms? That is a **treatment-by-covariate interaction** on a
binary outcome — the logistic analogue of an ANCOVA homogeneity-of-slopes test.

As an MCPower formula this is `remission ~ treatment * biomarker_level` with
`family="logit"`, where `*` expands to
`treatment + biomarker_level + treatment:biomarker_level`. `treatment` is a
two-level arm and `biomarker_level` is a continuous covariate; the test of
interest is the interaction term `treatment:biomarker_level`.

## Variations

- **Search for the N you need** instead of scoring one design: swap
  `find_power(sample_size=300, …)` for
  `find_sample_size(target_test="treatment:biomarker_level", from_size=150,
  to_size=800, by=25)`. Interactions on a binary outcome need markedly more N
  than main effects, so widen the upper bound rather than guess.
- **Test the adjusted main effect too** by setting `target_test="treatment"` (or
  `target_test="all"` for every term plus the omnibus), if you also care about
  the average treatment effect on the odds of remission, not only the slope
  difference.
- **Stronger or weaker moderation:** move `treatment:biomarker_level` between
  0.10 (a subtle difference in how the arms respond to biomarker level) and
  0.40 (the two arms respond very differently) to see how fast power for the
  moderation test collapses as the effect shrinks.
- **A graded treatment intensity** instead of two arms: drop
  `set_variable_type("treatment=binary")` and keep `treatment` continuous — that
  turns the design into continuous-by-continuous moderation on the log-odds.
- **Same design, other fields:**
  - `germinated ~ rainfall * soil_nitrogen` — does the effect of rainfall on germination depend on soil nitrogen? (ecology)
  - `voted ~ urban * social_support` — does the effect of urban residence on voting depend on social support levels? (social science)

## Not this setup?

- [[glm/glm-05|Logistic continuous-by-continuous moderation]]
- [[glm/glm-07|Logistic factor-by-factor interaction (2x2)]]
- [[ols/ols-11|Binary-by-continuous moderation]]

## If you'd rather have…

- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]] — Same treatment-by-covariate interaction structure but
  on a continuous outcome (ANCOVA homogeneity-of-slopes) instead of a binary one.
- [[glm/glm-04|Multiple logistic regression: covariate-adjusted binary outcome]] — Logistic with the treatment effect adjusted for
  covariates but no interaction term, if the moderation is not of interest.
- [[glm/glm-08|Logistic three-way interaction on a binary outcome]] — Extends the logistic interaction to a three-way
  moderation if a second moderator is in play.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:glm-06 -->
```python
from mcpower import MCPower

# Logistic regression with a treatment-by-covariate interaction: does the
# treatment's effect on the odds of remission depend on how elevated the patient's
# biomarker was at baseline? '*' expands treatment * biomarker_level to
# treatment + biomarker_level + treatment:biomarker_level, so the
# moderation term is fitted explicitly. family="logit" makes remission a binary
# (0/1) outcome fitted by a logistic GLM.
model = MCPower("remission = treatment * biomarker_level", family="logit")

# treatment is a two-level arm (0=control, 1=treatment); biomarker_level is a
# continuous covariate.
model.set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50                     -> medium arm shift (binary benchmark).
#   biomarker_level=0.40               -> strong covariate-outcome association
#                                         on the log-odds (continuous benchmark).
#   treatment:biomarker_level=0.25     -> moderate moderation (the interaction).
model.set_effects("treatment=0.50, biomarker_level=0.40, treatment:biomarker_level=0.25")

# A logistic GLM needs a baseline event rate to anchor the intercept: 30% of
# control patients reach remission when every predictor is at its reference
# value. family="logit" requires this before find_power.
model.set_baseline_probability(0.30)

model.set_seed(2137)

# Power for the moderation test (the interaction) at N=300.
model.find_power(sample_size=300, target_test="treatment:biomarker_level")
```
<!-- /chunk:py:glm-06 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:glm-06 -->
```r
suppressMessages(library(mcpower))

# Logistic regression with a treatment-by-covariate interaction: does the
# treatment's effect on the odds of remission depend on how elevated the patient's
# biomarker was at baseline? '*' expands treatment * biomarker_level to
# treatment + biomarker_level + treatment:biomarker_level, so the
# moderation term is fitted explicitly. family = "logit" makes remission a
# binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("remission ~ treatment * biomarker_level", family = "logit")

# treatment is a two-level arm (0=control, 1=treatment); biomarker_level is a
# continuous covariate.
model$set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50                     -> medium arm shift (binary benchmark).
#   biomarker_level=0.40               -> strong covariate-outcome association
#                                         on the log-odds (continuous benchmark).
#   treatment:biomarker_level=0.25     -> moderate moderation (the interaction).
model$set_effects("treatment=0.50, biomarker_level=0.40, treatment:biomarker_level=0.25")

# A logistic GLM needs a baseline event rate to anchor the intercept: 30% of
# control patients reach remission when every predictor is at its reference
# value. family = "logit" requires this before find_power.
model$set_baseline_probability(0.30)

model$set_seed(2137)

# Power for the moderation test (the interaction) at N=300.
invisible(model$find_power(sample_size = 300, target_test = "treatment:biomarker_level"))
```
<!-- /chunk:r:glm-06 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/glm-06-setup.png|600|theme-light]]
![[assets/glm-06-setup-dark.png|600|theme-dark]]

</details>
