---
title: "Power for a three-way logistic interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a three-way interaction on a binary logistic outcome. Free, Python & R."
---
# Three-way logistic interaction: does germination's response to light and moisture depend on temperature?

Your outcome is binary — germinated or not — and you suspect the way `light`
moderates the effect of `moisture` on germination is *itself* conditional on
`temperature`. That is a full three-way moderation on the log-odds: an effect of
`light` that grows with `moisture`, where that growth is steeper at some
temperatures than others. In MCPower this is the logistic model
`germinated ~ light * moisture * temperature`, where `*` expands to all three
main effects, all three two-way interactions, and the single three-way term
`light:moisture:temperature` — the coefficient you actually care about here.
`family="logit"` makes `germinated` a binary (0/1) outcome fitted by a logistic
GLM.

Three-way interactions are notoriously underpowered, and a binary outcome only
sharpens the problem: the highest-order term sits on the thinnest slice of the
design and carries the least information per observation, so plausible effects
need large samples. This page powers the three-way term itself, with main effects
at the medium continuous benchmark (0.25) and every interaction at the small
benchmark (0.10) on the log-odds scale.

## Variations

- **Power every term, not just the three-way.** Drop `target_test="light:moisture:temperature"` and
  the default reports every coefficient — useful to see how much more sample the
  three-way needs than the main effects.
- **Probe the two-way interactions too.** Use
  `target_test="light:moisture, light:temperature, moisture:temperature"` to
  report the three lower-order interactions side by side.
- **Bigger three-way effect.** If theory says the highest-order term is
  substantial, raise `light:moisture:temperature` to the medium continuous
  benchmark (0.25) — the required sample drops sharply.
- **A factor instead of a continuous predictor.** Swap any of `light`, `moisture`,
  `temperature` for a categorical via `set_variable_type("temperature=(factor,2)")`; the
  interaction then uses the factor benchmark scale (0.20 / 0.50 / 0.80).
- **Search for the sample, don't guess it.** Swap
  `find_power(sample_size=600, …)` for
  `find_sample_size(target_test="light:moisture:temperature")` to
  get the smallest N that hits 80% power for the three-way term.
- **Same design, other fields:**
  - `relapse ~ biomarker_level * dose * age` — does the joint effect of biomarker and dose on relapse depend on patient age? (clinical)
  - `voted ~ gender * urban * social_support` — does the gender-by-urbanicity interaction on voter turnout vary with social support? (social science)

## Not this setup?

- [[glm/glm-07]] — a two-way logistic interaction between two factors (2×2),
  without the third interacting predictor.
- [[glm/glm-05]] — a neighbouring logistic design with a different predictor and
  interaction structure.
- [[ols/ols-05]] — the same three-way interaction structure on a *continuous*
  outcome (OLS) rather than a binary one.

## If you'd rather have…

- [[glm/glm-07|A two-way logistic interaction (2×2 factor-by-factor)]] — drop to
  two interacting factors if a three-way is more than your design needs.
- [[glm/glm-06|A single logistic treatment-by-moderator interaction]] — binary ×
  continuous, instead of three interacting predictors.
- [[ols/ols-05|The same three-way interaction on a continuous outcome]] — OLS
  rather than a binary outcome.
- [[anova/anova-05|A three-way 2×2×2 factorial as ANOVA omnibus tests]] — framed
  as factorial omnibus tests instead of logistic regression coefficients.
- [[glm/glm-04|A covariate-adjusted binary outcome, main effects only]] — no
  interaction terms at all.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:glm-08 -->
```python
from mcpower import MCPower

# Three-way interaction on a yes/no outcome: does the way one environmental
# factor moderates another itself depend on a third, when the response is binary?
# family="logit" makes `germinated` a binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("germinated ~ light * moisture * temperature", family="logit")

# `*` expands to all three main effects, all three two-way interactions, and the
# single three-way term light:moisture:temperature -- the coefficient this page actually powers.
# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40):
# main effects at medium (0.25), every interaction at small (0.10) on the log-odds.
model.set_effects(
    "light=0.25, moisture=0.25, temperature=0.25, "
    "light:moisture=0.10, light:temperature=0.10, moisture:temperature=0.10, "
    "light:moisture:temperature=0.10"
)
model.set_simulations(1600)
model.set_seed(2137)

# Logistic GLMs need a baseline event rate: it pins the intercept so the
# log-odds effects above land on a concrete probability scale. Required for
# family="logit" -- find_power errors without it.
model.set_baseline_probability(0.3)

# Power at N=600 for the three-way term itself (the thinnest slice of the design).
model.find_power(sample_size=600, target_test="light:moisture:temperature")
```
<!-- /chunk:py:glm-08 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:glm-08 -->
```r
suppressMessages(library(mcpower))

# Three-way interaction on a yes/no outcome: does the way one environmental
# factor moderates another itself depend on a third, when the response is binary?
# family="logit" makes `germinated` a binary (0/1) outcome fitted by a logistic GLM.
model <- MCPower$new("germinated ~ light * moisture * temperature", family = "logit")

# `*` expands to all three main effects, all three two-way interactions, and the
# single three-way term light:moisture:temperature -- the coefficient this page actually powers.
# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40):
# main effects at medium (0.25), every interaction at small (0.10) on the log-odds.
model$set_effects(paste0(
  "light=0.25, moisture=0.25, temperature=0.25, ",
  "light:moisture=0.10, light:temperature=0.10, moisture:temperature=0.10, ",
  "light:moisture:temperature=0.10"
))
model$set_simulations(1600)
model$set_seed(2137)

# Logistic GLMs need a baseline event rate: it pins the intercept so the
# log-odds effects above land on a concrete probability scale. Required for
# family="logit" -- find_power errors without it.
model$set_baseline_probability(0.3)

# Power at N=600 for the three-way term itself (the thinnest slice of the design).
invisible(model$find_power(sample_size = 600, target_test = "light:moisture:temperature"))
```
<!-- /chunk:r:glm-08 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/glm-08-setup.png|600|theme-light]]
![[assets/glm-08-setup-dark.png|600|theme-dark]]

</details>
