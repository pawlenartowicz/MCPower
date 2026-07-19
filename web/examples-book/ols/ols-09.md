---
title: "Power for ANCOVA homogeneity-of-slopes test"
description: "Power & sample-size analysis by Monte Carlo simulation for ANCOVA homogeneity of slopes (blood_pressure ~ treatment * baseline_bp). Free, Python & R."
---
# ANCOVA with a homogeneity-of-slopes test

You ran a two-group trial and measured each patient's blood pressure at baseline.
Before trusting a single adjusted treatment effect, you want to know whether the
treatment works differently depending on where a patient started — does the slope
of `blood_pressure` on `baseline_bp` differ between the treatment and control
arms? That is a **group-by-covariate interaction**, the homogeneity-of-slopes
test that an ordinary ANCOVA assumes away.

As an MCPower formula this is `blood_pressure = treatment * baseline_bp`, where
`*` expands to `treatment + baseline_bp + treatment:baseline_bp`. `treatment` is
a two-level treatment factor and `baseline_bp` is a continuous covariate; the
test of interest is the interaction term `treatment:baseline_bp`.

## Variations

- **Search for the N you need** instead of scoring one design: swap
  `find_power(sample_size=180, …)` for `find_sample_size(target_test="treatment:baseline_bp",
  from_size=80, to_size=400, by=20)`. Interactions need markedly more N than
  main effects, so widen the upper bound rather than guess.
- **Test the adjusted main effect too** by setting `target_test="treatment"` (or
  `target_test="all"` for every term plus the omnibus), if you also care about
  the average treatment effect, not only the slope difference.
- **Stronger or weaker moderation:** move `treatment:baseline_bp` between 0.10
  (subtle slope difference) and 0.40 (the two arms respond very differently) to
  see how fast power for the homogeneity test collapses as the effect shrinks.
- **A continuous treatment intensity** instead of two arms: drop
  `set_variable_type("treatment=binary")` and keep `treatment` continuous — that
  turns the design into continuous-by-continuous moderation.
- **Same design, other fields:**
  - Ecology: `plant_biomass ~ habitat * soil_nitrogen` — does habitat moderate the soil-nitrogen slope on plant biomass?
  - Social science: `wage ~ gender * experience_years` — does gender moderate the experience-years wage slope?

## Not this setup?

- [[ols/ols-08|ANCOVA: group effect adjusting for a baseline covariate]]
- [[ols/ols-11|Binary-by-continuous moderation]]
- [[ols/ols-14|Dummy-coded factor interacting with a continuous predictor]]

## If you'd rather have…

- [[ols/ols-08|ANCOVA: group effect adjusting for a baseline covariate]] — Same ANCOVA design but parallel slopes — group effect
  adjusted for baseline, no group-by-covariate interaction (drop the
  homogeneity-of-slopes test).
- [[ols/ols-11|Binary-by-continuous moderation]] — Same binary-by-continuous moderation structure, framed
  as a predictor interaction (`gender * experience_years`) rather than a
  treatment-vs-covariate ANCOVA.
- [[ols/ols-14|Dummy-coded factor interacting with a continuous predictor]] — Interaction of a categorical predictor with a
  continuous one, but the factor has 3+ levels (`habitat * rainfall`) instead of a
  two-level treatment group.
- [[ols/ols-04|Continuous-by-continuous moderation (interaction)]] — Continuous-by-continuous moderation if your 'treatment' is
  actually a continuous predictor rather than a treatment factor.
- [[ols/ols-16|Continuous moderation with a covariate]] — Continuous moderation plus an additive covariate —
  useful if you want an interaction and a separate adjustment term.

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-09 -->
```python
from mcpower import MCPower

# ANCOVA with a homogeneity-of-slopes test: does the treatment effect on
# blood_pressure depend on each patient's baseline_bp? '*' expands treatment * baseline_bp
# to treatment + baseline_bp + treatment:baseline_bp, so the interaction is fitted explicitly.
model = MCPower("blood_pressure = treatment * baseline_bp")

# treatment is a two-level treatment factor (0=control, 1=treatment); baseline_bp is a
# continuous covariate.
model.set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50             → medium treatment shift (binary benchmark).
#   baseline_bp=0.40           → strong baseline-outcome association (continuous benchmark).
#   treatment:baseline_bp=0.25 → moderate slope difference (the moderation effect).
model.set_effects("treatment=0.50, baseline_bp=0.40, treatment:baseline_bp=0.25")

model.set_seed(2137)

# Power for the homogeneity-of-slopes test (the interaction) at N=180.
model.find_power(sample_size=180, target_test="treatment:baseline_bp")
```
<!-- /chunk:py:ols-09 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-09 -->
```r
suppressMessages(library(mcpower))

# ANCOVA with a homogeneity-of-slopes test: does the treatment effect on
# blood_pressure depend on each patient's baseline_bp? '*' expands treatment * baseline_bp
# to treatment + baseline_bp + treatment:baseline_bp, so the interaction is fitted explicitly.
model <- MCPower$new("blood_pressure ~ treatment * baseline_bp")

# treatment is a two-level treatment factor (0=control, 1=treatment); baseline_bp is a
# continuous covariate.
model$set_variable_type("treatment=binary")

# Effect sizes on the benchmark scale.
#   treatment=0.50             -> medium treatment shift (binary benchmark).
#   baseline_bp=0.40           -> strong baseline-outcome association (continuous benchmark).
#   treatment:baseline_bp=0.25 -> moderate slope difference (the moderation effect).
model$set_effects("treatment=0.50, baseline_bp=0.40, treatment:baseline_bp=0.25")

model$set_seed(2137)

# Power for the homogeneity-of-slopes test (the interaction) at N=180.
invisible(model$find_power(sample_size = 180, target_test = "treatment:baseline_bp"))
```
<!-- /chunk:r:ols-09 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-09-setup.png|600|theme-light]]
![[assets/ols-09-setup-dark.png|600|theme-dark]]

</details>
