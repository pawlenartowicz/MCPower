---
title: "Power for a 2x2 logistic factorial interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a 2x2 factor-by-factor interaction on a binary logistic outcome. Free, Python & R."
---
# A 2x2 factorial interaction on a binary outcome: voted ~ gender * urban

You ran a survey with a yes/no outcome — did the respondent `vote`? — and
crossed two two-level factors: `gender` (women vs men) fully crossed with `urban`
residence (urban vs rural). The question that motivated the design is not either
main effect but whether they *interact*: does the gender gap in voter turnout
differ between urban and rural respondents? That is the classic **2x2 factorial
interaction**, the difference in cell differences — here on the log-odds scale,
because the outcome is binary.

As an MCPower formula this is `voted ~ gender * urban` with `family="logit"`,
where `*` expands to `gender + urban + gender:urban`. Both `gender` and `urban`
are two-level factors; the test of interest is the interaction term
`gender:urban`, fitted by a logistic GLM.

## Variations

- **Search for the N you need** instead of scoring one design: swap
  `find_power(sample_size=400, …)` for `find_sample_size(target_test="gender:urban",
  from_size=200, to_size=1200, by=50)`. A logistic interaction needs markedly
  more N than either main effect — and binary outcomes carry less information
  than continuous ones — so set the upper bound generously.
- **Test the main effects too** by setting `target_test="gender"` (or
  `target_test="all"` for every term plus the omnibus), if you also care about
  the average effects, not only the interaction.
- **Stronger or weaker moderation:** move `gender:urban` across the factor
  benchmarks — 0.20 (subtle), 0.50 (medium), 0.80 (the two residence groups
  show very different gender gaps) — to watch how fast power for the interaction
  collapses as the effect shrinks.
- **A wider design:** give one factor three or more levels with
  `set_variable_type("urban=(factor,3)")` — the interaction then spans several
  contrasts and demands still more N.
- **Same design, other fields:**
  - `infection ~ treatment * dose_level` — does the effect of treatment on infection depend on dose level? (clinical)
  - `germinated ~ light * moisture` — does the effect of light on germination depend on moisture availability? (ecology)

## Not this setup?

- [[glm/glm-08|Logistic three-way interaction on a binary outcome]]
- [[glm/glm-05|Logistic continuous-by-continuous moderation]]
- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]]
- [[glm/glm-03|Logistic regression with a multi-level categorical predictor]]

## If you'd rather have…

- [[glm/glm-08|Logistic three-way interaction on a binary outcome]] — Add a third factor: logistic three-way interaction on
  a binary outcome (a*b*c).
- [[glm/glm-05|Logistic continuous-by-continuous moderation]] — Same logistic interaction but with two continuous
  moderators instead of two factors.
- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]] — Mixed interaction: a binary treatment crossed with a
  continuous moderator on a binary outcome.
- [[glm/glm-03|Logistic regression with a multi-level categorical predictor]] — Drop the interaction: a single multi-level categorical
  predictor on a binary outcome.
- [[ols/ols-15|Two interacting categorical predictors (factorial as regression)]] — Same two interacting categorical predictors but on a
  continuous outcome (factorial as OLS regression).
- [[anova/anova-04|Two-way factorial ANOVA with interaction (2x2 / 2x3 / 3x3 / 2x4)]] — The 2x2 factorial framed as a two-way ANOVA with
  interaction on a continuous outcome.

## Copy-paste setup

<!-- chunk:py:glm-07 -->
```python
from mcpower import MCPower

# 2x2 factorial on a yes/no outcome: did the respondent vote? The question is
# whether the effect of `gender` on voting depends on `urban` residence.
# '*' expands gender * urban to gender + urban + gender:urban, so the
# interaction (the difference in cell differences, on the log-odds scale) is
# fitted explicitly. family="logit" makes voted binary (0/1) and fits a GLM.
model = MCPower("voted = gender * urban", family="logit")

# Both predictors are two-level factors: gender (e.g. women vs men) and
# urban (e.g. urban vs rural).
model.set_variable_type("gender=binary, urban=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80), each a shift
# in the log-odds of voting:
#   gender=0.50        -> medium main effect of gender.
#   urban=0.50         -> medium main effect of urban residence.
#   gender:urban=0.50  -> medium interaction (the moderation effect).
model.set_effects("gender=0.50, urban=0.50, gender:urban=0.50")

# Baseline voting rate in the reference cell (logit family needs a baseline probability).
model.set_baseline_probability(0.50)

model.set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=400.
model.find_power(sample_size=400, target_test="gender:urban")
```
<!-- /chunk:py:glm-07 -->

<!-- chunk:r:glm-07 -->
```r
suppressMessages(library(mcpower))

# 2x2 factorial on a yes/no outcome: did the respondent vote? The question is
# whether the effect of `gender` on voting depends on `urban` residence.
# '*' expands gender * urban to gender + urban + gender:urban, so the
# interaction (the difference in cell differences, on the log-odds scale) is
# fitted explicitly. family="logit" makes voted binary (0/1) and fits a GLM.
model <- MCPower$new("voted ~ gender * urban", family = "logit")

# Both predictors are two-level factors: gender (e.g. women vs men) and
# urban (e.g. urban vs rural).
model$set_variable_type("gender=binary, urban=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80), each a shift
# in the log-odds of voting:
#   gender=0.50        -> medium main effect of gender.
#   urban=0.50         -> medium main effect of urban residence.
#   gender:urban=0.50  -> medium interaction (the moderation effect).
model$set_effects("gender=0.50, urban=0.50, gender:urban=0.50")

# Baseline voting rate in the reference cell (logit family needs a baseline probability).
model$set_baseline_probability(0.50)

model$set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=400.
invisible(model$find_power(sample_size = 400, target_test = "gender:urban"))
```
<!-- /chunk:r:glm-07 -->

![[assets/glm-07-setup.png|600|theme-light]]
![[assets/glm-07-setup-dark.png|600|theme-dark]]
