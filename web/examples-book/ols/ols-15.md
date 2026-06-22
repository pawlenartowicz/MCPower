---
title: "Power for a 2x2 factorial interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a 2x2 factorial OLS interaction (job_satisfaction ~ gender * sector). Free, Python & R."
---
# A 2x2 factorial interaction

You ran a study that crosses two two-level factors — `gender` fully crossed with
`sector` — and the question that motivated the design is not either main effect
but whether they *interact*: does the effect of gender on job satisfaction differ
by sector, or equivalently does the effect of sector differ by gender? That is the
classic **2x2 factorial interaction**, the difference in cell differences.

As an MCPower formula this is `job_satisfaction ~ gender * sector`, where `*`
expands to `gender + sector + gender:sector`. Both `gender` and `sector` are
two-level factors; the test of interest is the interaction term `gender:sector`.

## Variations

- **Search for the N you need** instead of scoring one design: swap
  `find_power(sample_size=200, …)` for `find_sample_size(target_test="gender:sector",
  from_size=100, to_size=600, by=25)`. A 2x2 interaction needs markedly more N
  than either main effect, so set the upper bound generously.
- **Test the main effects too** by setting `target_test="gender"` (or
  `target_test="all"` for every term plus the omnibus), if you also care about
  the average effects, not only the interaction.
- **Stronger or weaker moderation:** move `gender:sector` across the factor
  benchmarks — 0.20 (subtle), 0.50 (medium), 0.80 (the two groups respond very
  differently) — to see how fast power for the interaction collapses as the
  effect shrinks.
- **A wider design:** give one factor three or more levels with
  `set_variable_type("sector=(factor,3)")` — the interaction then spans several
  contrasts and demands still more N.
- **Same design, other fields:**
  - Clinical: `pain_score ~ treatment * site` — does treatment effectiveness differ across clinical sites?
  - Ecology: `abundance ~ habitat * nitrogen` — does the nitrogen effect on species abundance differ between habitat types?

## Not this setup?

- [[ols/ols-14|Dummy-coded factor interacting with a continuous predictor]]
- [[anova/anova-04|Two-way factorial ANOVA with interaction (2x2 / 2x3 / 3x3 / 2x4)]]
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]]

## If you'd rather have…

- [[ols/ols-14|Dummy-coded factor interacting with a continuous predictor]] — One of the two factors becomes a continuous
  moderator: gender (factor) interacting with a continuous predictor instead
  of a second factor.
- [[anova/anova-04|Two-way factorial ANOVA with interaction (2x2 / 2x3 / 3x3 / 2x4)]] — Same 2-factor factorial design analyzed as a
  two-way ANOVA with omnibus F-tests for the interaction rather than dummy-coded
  regression coefficients.
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]] — Replace the second categorical factor with a
  continuous covariate: gender x baseline (ANCOVA, homogeneity-of-slopes) keeps
  the factor-by-X interaction shape.
- [[glm/glm-07|Logistic factor-by-factor interaction (2x2)]] — Same factor-by-factor (2x2) interaction but on a
  binary outcome via logistic regression.
- [[ols/ols-12|Three-level dummy-coded categorical predictor]] — Drop to a single multi-level categorical predictor (no
  interaction) if you only need one factor's main effect.

## Copy-paste setup

<!-- chunk:py:ols-15 -->
```python
from mcpower import MCPower

# 2x2 factorial: does the effect of `gender` depend on `sector`? '*' expands
# gender * sector to gender + sector + gender:sector, so the interaction
# (the difference in cell differences) is fitted explicitly.
model = MCPower("job_satisfaction = gender * sector")

# Both predictors are two-level factors: gender and sector.
model.set_variable_type("gender=binary, sector=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   gender=0.50         -> medium main effect of gender.
#   sector=0.50         -> medium main effect of sector.
#   gender:sector=0.50  -> medium interaction (the moderation effect).
model.set_effects("gender=0.50, sector=0.50, gender:sector=0.50")

model.set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=200.
model.find_power(sample_size=200, target_test="gender:sector")
```
<!-- /chunk:py:ols-15 -->

<!-- chunk:r:ols-15 -->
```r
suppressMessages(library(mcpower))

# 2x2 factorial: does the effect of `gender` depend on `sector`? '*' expands
# gender * sector to gender + sector + gender:sector, so the interaction
# (the difference in cell differences) is fitted explicitly.
model <- MCPower$new("job_satisfaction ~ gender * sector")

# Both predictors are two-level factors: gender and sector.
model$set_variable_type("gender=binary, sector=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   gender=0.50          -> medium main effect of gender.
#   sector=0.50          -> medium main effect of sector.
#   gender:sector=0.50   -> medium interaction (the moderation effect).
model$set_effects("gender=0.50, sector=0.50, gender:sector=0.50")

model$set_seed(2137)

# Power for the interaction (the 2x2 cell-difference test) at N=200.
invisible(model$find_power(sample_size = 200, target_test = "gender:sector"))
```
<!-- /chunk:r:ols-15 -->

![[assets/ols-15-setup.png|600|theme-light]]
![[assets/ols-15-setup-dark.png|600|theme-dark]]
