---
title: "Power for a factor-by-continuous interaction"
description: "Power & sample-size analysis by Monte Carlo simulation for a factor-by-continuous OLS interaction (biomass ~ habitat * rainfall). Free, Python & R."
---
# Does the rainfall effect on biomass differ across habitats? Moderation by habitat

Three habitat types where you suspect the slope of rainfall on biomass is not
the same in every habitat: rainfall acts on biomass, but you expect that effect
to be *stronger or weaker depending on which habitat* the sample comes from.
That is a moderation, and in MCPower it is the model `biomass ~ habitat * rainfall`
— the `*` expands to the two habitat contrasts, the `rainfall` slope, and the
`habitat:rainfall` interaction that carries the moderation. The interaction is the
focal test; with a 3-level `habitat` it is two dummy-by-`rainfall` terms (each
non-reference habitat's slope vs the reference habitat's slope).

## Variations

- **More than three habitat types.** Bump `habitat=(factor,4)` (or higher) — the
  interaction grows to one dummy-by-`rainfall` term per non-reference level, all
  tested together when you target `habitat:rainfall`.
- **Swap the continuous moderator for a second factor.** Replace `rainfall` with a
  categorical predictor and declare it as a factor; `habitat * group` becomes a
  factorial design (a two-way ANOVA written as regression).
- **Unequal group sizes.** If the habitats are not balanced, set the factor
  level proportions so the design reflects the sampling you actually expect,
  rather than equal thirds.
- **Different moderation strength.** The interaction effect (`habitat:rainfall`) is
  the quantity you are usually least sure about — re-run with it at the small
  (0.10) and large (0.40) continuous benchmarks to bracket how much power hinges
  on that guess.
- **Find the N instead of the power.** Swap `find_power(sample_size=...)` for
  `find_sample_size(target_test="habitat:rainfall", from_size=..., to_size=...)` to
  get the smallest N that powers the moderation.
- **Same design, other fields:**
  - Clinical: `recovery_days ~ dose_level * baseline_severity` — does baseline severity moderate the dose-level effect on recovery time across three dose groups?
  - Social: `wage ~ sector * experience_years` — does the experience-years wage slope differ across three employment sectors?

## Not this setup?

- [[ols/ols-11|Binary-by-continuous moderation]]
- [[ols/ols-15|Two interacting categorical predictors (factorial as regression)]]
- [[ols/ols-09|ANCOVA with treatment-by-covariate interaction (homogeneity of slopes)]]
- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]]

## If you'd rather have…

- [[ols/ols-12|Three-level dummy-coded categorical predictor]] — same three-level dummy-coded `habitat` factor but as
  a main effect only, no interaction with a continuous predictor.
- [[ols/ols-16|Continuous moderation with a covariate]] — a moderation (interaction) that also adjusts for an
  additional covariate alongside the interacting predictors.
- [[ols/ols-15|Two interacting categorical predictors (factorial as regression)]] — keep the categorical `habitat` predictor but moderate
  it by another categorical factor instead of a continuous one (factorial as
  regression).

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-14 -->
```python
from mcpower import MCPower

# Three-habitat study with a continuous moderator.
# Research question: does the effect of rainfall on biomass differ across the
# three habitat types -- i.e. does habitat moderate the rainfall slope?
# '*' expands habitat * rainfall to habitat + rainfall + habitat:rainfall.
model = MCPower("biomass = habitat * rainfall")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts;
# rainfall is continuous (left at its default standardised distribution).
model.set_variable_type("habitat=(factor,3)")

# A 3-level factor expands to per-level dummies, so its effects and the
# interaction are named per level: habitat[2], habitat[3], and
# habitat[2]:rainfall, habitat[3]:rainfall (the bare names habitat / habitat:rainfall
# do not exist after expansion).
# Standardised effects:
#   habitat[2]=habitat[3]=0.5 -> factor benchmark (0.20/0.50/0.80), each
#                                 non-reference level shifts the outcome by
#                                 a medium amount.
#   rainfall=0.25              -> continuous benchmark (0.10/0.25/0.40), a
#                                 medium slope.
#   habitat[*]:rainfall=0.4   -> the moderation: how much each habitat's
#                                 rainfall slope departs from the reference slope.
model.set_effects(
    "habitat[2]=0.5, habitat[3]=0.5, rainfall=0.25, "
    "habitat[2]:rainfall=0.4, habitat[3]:rainfall=0.4"
)

# Power at N=200 for the interaction dummies -- the moderation is the focal test.
model.find_power(sample_size=200, target_test="habitat[2]:rainfall, habitat[3]:rainfall")
```
<!-- /chunk:py:ols-14 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-14 -->
```r
suppressMessages(library(mcpower))

# Three-habitat study with a continuous moderator.
# Research question: does the effect of rainfall on biomass differ across the
# three habitat types -- i.e. does habitat moderate the rainfall slope?
# '*' expands habitat * rainfall to habitat + rainfall + habitat:rainfall.
model <- MCPower$new("biomass ~ habitat * rainfall")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts;
# rainfall is continuous (left at its default standardised distribution).
model$set_variable_type("habitat=(factor,3)")

# A 3-level factor expands to per-level dummies, so its effects and the
# interaction are named per level: habitat[2], habitat[3], and
# habitat[2]:rainfall, habitat[3]:rainfall (the bare names habitat / habitat:rainfall
# do not exist after expansion).
# Standardised effects:
#   habitat[2]=habitat[3]=0.5 -> factor benchmark (0.20/0.50/0.80), each
#                                 non-reference level shifts the outcome by
#                                 a medium amount.
#   rainfall=0.25              -> continuous benchmark (0.10/0.25/0.40), a
#                                 medium slope.
#   habitat[*]:rainfall=0.4   -> the moderation: how much each habitat's
#                                 rainfall slope departs from the reference slope.
model$set_effects(paste0(
  "habitat[2]=0.5, habitat[3]=0.5, rainfall=0.25, ",
  "habitat[2]:rainfall=0.4, habitat[3]:rainfall=0.4"
))

# Power at N=200 for the interaction dummies -- the moderation is the focal test.
invisible(model$find_power(
  sample_size = 200,
  target_test = "habitat[2]:rainfall, habitat[3]:rainfall"
))
```
<!-- /chunk:r:ols-14 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-14-setup.png|600|theme-light]]
![[assets/ols-14-setup-dark.png|600|theme-dark]]

</details>
