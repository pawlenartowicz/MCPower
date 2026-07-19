---
title: "Power for a two-way factorial ANOVA interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a two-way factorial ANOVA with two crossed factors and their interaction. Free, Python & R."
---
# Two-way factorial ANOVA: hourly wage crossed by sector and gender

You collected wages from workers across three employment sectors (e.g. public,
private, non-profit) and two gender categories, with every combination of sector
and gender represented in your sample. Beyond each factor's own wage gap, the
headline question is the interaction: does the sector wage premium depend on
gender?

As an MCPower model this is `hourly_wage = sector * gender`. The `*` expands to
`sector + gender + sector:gender`, so OLS fits both main effects and the
interaction. With a 3-level sector and a 2-level gender factor the design has six
cells; the interaction is carried by the cell-difference contrasts, and power is
reported for each.

## Variations

- **A balanced 2x2 design.** Drop `sector` to two levels
  (`sector=(factor,2)`) so the interaction collapses to a single
  cell-difference contrast — the classic 2x2 factorial.
- **Larger grids.** Bump either factor to its real level count (e.g.
  `sector=(factor,4)`); each extra level adds dummy contrasts to that main
  effect and to the interaction.
- **Test a main effect instead.** Point `target_test` at `sector` (or
  `gender`) to read power for a main-effect's contrasts rather than the
  interaction.
- **Unequal effects across levels.** A single `sector=0.50` puts the same
  shift on every non-reference level; set the level effects apart when one
  sector is expected to move wages more than another.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="sector:gender", from_size=60, to_size=400, by=10)`
  to get the smallest N that reaches target power on the interaction contrasts.
- **Same design, other fields:**
  - `blood_pressure = treatment * sex` — clinical: interaction between treatment arm (3 levels) and patient sex
  - `biomass = fertilizer * habitat` — ecology: interaction between fertilizer (3 levels) and habitat type

## Not this setup?

- [[anova/anova-05|A third factor: three-way (2x2x2) factorial ANOVA]]
- [[anova/anova-06|The same two factors plus a continuous covariate (ANCOVA)]]
- [[ols/ols-15|The same two interacting factors framed as regression]]

## If you'd rather have…

- [[anova/anova-05]] — Add a third factor: three-way factorial ANOVA (2x2x2)
  with all main effects and interactions.
- [[anova/anova-06]] — Keep the two-way factorial but adjust for a continuous
  covariate (factorial ANCOVA).
- [[anova/anova-01]] — Drop to a single factor: one-way ANOVA omnibus across
  3+ groups.
- [[ols/ols-15]] — Same two interacting categorical predictors framed as a
  regression rather than an ANOVA.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:anova-04 -->
```python
from mcpower import MCPower

# Two-way factorial ANOVA: hourly wage crossed by employment sector and gender.
# Research question -- does the sector wage gap depend on gender? -> the interaction.
# '*' expands sector * gender to sector + gender + sector:gender, so
# both main effects and the interaction are fitted explicitly.
model = MCPower("hourly_wage = sector * gender")

# sector has 3 levels -> 2 dummy contrasts; gender has 2 levels -> 1 dummy.
# The interaction therefore contributes 2 cell-difference contrasts.
model.set_variable_type("sector=(factor,3), gender=(factor,2)")

# Effects are assigned per dummy contrast, not per base factor name: with no
# uploaded data the levels are integer-labelled, level 1 is the reference, and
# the expansion produces exactly these five terms. All on the factor benchmark
# scale (0.20 / 0.50 / 0.80) at medium = 0.50:
#   sector[2], sector[3]                -> sector's two main contrasts.
#   gender[2]                           -> gender's main contrast.
#   sector[2]:gender[2], sector[3]:gender[2] -> the two interaction cells.
model.set_effects(
    "sector[2]=0.50, sector[3]=0.50, gender[2]=0.50, "
    "sector[2]:gender[2]=0.50, sector[3]:gender[2]=0.50"
)

model.set_seed(2137)

# Power for one interaction cell -- the factorial design's focal test -- at
# N=240. (target_test names a single expanded effect; there is no bare
# 'sector:gender' term after dummy expansion.)
model.find_power(sample_size=240, target_test="sector[2]:gender[2]")
```
<!-- /chunk:py:anova-04 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:anova-04 -->
```r
suppressMessages(library(mcpower))

# Two-way factorial ANOVA: hourly wage crossed by employment sector and gender.
# Research question -- does the sector wage gap depend on gender? -> the interaction.
# '*' expands sector * gender to sector + gender + sector:gender, so
# both main effects and the interaction are fitted explicitly.
model <- MCPower$new("hourly_wage ~ sector * gender")

# sector has 3 levels -> 2 dummy contrasts; gender has 2 levels -> 1 dummy.
# The interaction therefore contributes 2 cell-difference contrasts.
model$set_variable_type("sector=(factor,3), gender=(factor,2)")

# Effects are assigned per dummy contrast, not per base factor name: with no
# uploaded data the levels are integer-labelled, level 1 is the reference, and
# the expansion produces exactly these five terms. All on the factor benchmark
# scale (0.20 / 0.50 / 0.80) at medium = 0.50:
#   sector[2], sector[3]                -> sector's two main contrasts.
#   gender[2]                           -> gender's main contrast.
#   sector[2]:gender[2], sector[3]:gender[2] -> the two interaction cells.
model$set_effects(
  paste0(
    "sector[2]=0.50, sector[3]=0.50, gender[2]=0.50, ",
    "sector[2]:gender[2]=0.50, sector[3]:gender[2]=0.50"
  )
)

model$set_seed(2137)

# Power for one interaction cell -- the factorial design's focal test -- at
# N=240. (target_test names a single expanded effect; there is no bare
# 'sector:gender' term after dummy expansion.)
invisible(model$find_power(sample_size = 240, target_test = "sector[2]:gender[2]"))
```
<!-- /chunk:r:anova-04 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/anova-04-setup.png|600|theme-light]]
![[assets/anova-04-setup-dark.png|600|theme-dark]]

</details>
