---
title: "Power for a three-way factorial ANOVA (2x2x2)"
description: "Power & sample-size analysis by Monte Carlo simulation for a three-way 2x2x2 factorial ANOVA and its three-way interaction. Free, Python & R."
---
# Three-way factorial ANOVA: seed yield across watering, nitrogen, and light

You ran a fully crossed greenhouse experiment with three binary growth factors —
watering level, nitrogen supplementation, and light intensity — so each plot
lands in one cell of a 2x2x2 design, and you measured seed yield per plot.
Beyond each factor's own effect and each pair's interaction, you want the power
to detect the *three-way* interaction: the way the watering-by-nitrogen pattern
itself changes across light levels.

As an MCPower model this is `seed_yield = watering * nitrogen * light`, all
three two-level binary factors. The `*` expands to every main effect, all three
pairwise interactions, and the top-level `watering:nitrogen:light` term, so the
whole factorial is fitted at once; here you read power off that highest-order
interaction.

## Variations

- **More levels per factor.** A factor need not be two-level — set, say,
  `nitrogen=(factor,3)` for a 2x2x3 design. Each extra level multiplies the
  number of cells and the dummy contrasts the three-way term spans, so power for
  the same effect size usually drops.
- **Test a different term.** Swap the `target_test` to a pairwise interaction
  (`watering:nitrogen`) or a main effect (`watering`) if that, not the three-way
  term, is the hypothesis you're powering for.
- **Unequal expected effects.** A single `0.50` puts the same shift on every
  term. To say the three-way interaction is weaker than the main effects — the
  usual reality — set it smaller (e.g. `watering:nitrogen:light=0.20`) and watch
  how much more N the higher-order term demands.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="watering:nitrogen:light", from_size=100, to_size=600, by=20)`
  to get the smallest N that reaches target power on the three-way interaction.
- **Same design, other fields:**
  - `blood_pressure = treatment * sex * age_group` — clinical: three-way interaction of treatment arm, sex, and age category
  - `job_satisfaction = sector * gender * union_member` — social science: three-way interaction of sector, gender, and union membership

## Not this setup?

- [[anova/anova-04|Drop one factor: a two-way factorial ANOVA]]
- [[anova/anova-06|Factorial ANOVA that also adjusts for a covariate]]
- [[anova/anova-01|A single one-way omnibus ANOVA across 3+ groups]]

## If you'd rather have…

- [[anova/anova-04]] — Drop to a two-way factorial ANOVA (2x2 / 2x3 / 3x3 / 2x4)
  if a three-way design is more than you need.
- [[anova/anova-06]] — Two-way factorial ANOVA that also adjusts for a
  continuous covariate (factorial ANCOVA).
- [[anova/anova-01]] — A single one-way omnibus ANOVA across 3+ groups, the
  simplest factor design.
- [[ols/ols-05]] — The same three-way interaction structure but with continuous
  predictors instead of factors (OLS).
- [[glm/glm-08]] — Three-way interaction recast for a binary outcome via
  logistic GLM.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:anova-05 -->
```python
from mcpower import MCPower

# Three-way factorial ANOVA: seed yield crossed by three binary growth factors.
# Research question: does the watering-by-nitrogen interaction itself shift
# across light levels? -> the three-way watering:nitrogen:light interaction.
# '*' expands watering * nitrogen * light to all main effects, all pairwise
# interactions, and the three-way term, so every cell of the design is fitted.
model = MCPower("seed_yield = watering * nitrogen * light")

# All three predictors are two-level factors (a 2x2x2 design).
model.set_variable_type("watering=binary, nitrogen=binary, light=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   main effects 0.50                 -> medium shift per factor.
#   two-way interactions 0.50         -> medium moderation between each pair.
#   watering:nitrogen:light=0.50      -> medium three-way interaction (the target).
model.set_effects(
    "watering=0.50, nitrogen=0.50, light=0.50, "
    "watering:nitrogen=0.50, watering:light=0.50, nitrogen:light=0.50, "
    "watering:nitrogen:light=0.50"
)
model.set_simulations(1600)
model.set_seed(2137)

# Power at N=320 for the three-way interaction (the highest-order term).
model.find_power(sample_size=320, target_test="watering:nitrogen:light")
```
<!-- /chunk:py:anova-05 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:anova-05 -->
```r
suppressMessages(library(mcpower))

# Three-way factorial ANOVA: seed yield crossed by three binary growth factors.
# Research question: does the watering-by-nitrogen interaction itself shift
# across light levels? -> the three-way watering:nitrogen:light interaction.
# '*' expands watering * nitrogen * light to all main effects, all pairwise
# interactions, and the three-way term, so every cell of the design is fitted.
model <- MCPower$new("seed_yield ~ watering * nitrogen * light")

# All three predictors are two-level factors (a 2x2x2 design).
model$set_variable_type("watering=binary, nitrogen=binary, light=binary")

# Effect sizes on the factor benchmark scale (0.20 / 0.50 / 0.80):
#   main effects 0.50                 -> medium shift per factor.
#   two-way interactions 0.50         -> medium moderation between each pair.
#   watering:nitrogen:light=0.50      -> medium three-way interaction (the target).
model$set_effects(paste0(
  "watering=0.50, nitrogen=0.50, light=0.50, ",
  "watering:nitrogen=0.50, watering:light=0.50, nitrogen:light=0.50, ",
  "watering:nitrogen:light=0.50"
))
model$set_simulations(1600)
model$set_seed(2137)

# Power at N=320 for the three-way interaction (the highest-order term).
invisible(model$find_power(sample_size = 320, target_test = "watering:nitrogen:light"))
```
<!-- /chunk:r:anova-05 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/anova-05-setup.png|600|theme-light]]
![[assets/anova-05-setup-dark.png|600|theme-dark]]

</details>
