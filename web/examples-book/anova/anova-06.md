---
title: "Power for a two-way factorial ANCOVA"
description: "Power & sample-size analysis by Monte Carlo simulation for a two-way factorial ANCOVA adjusting group effects for a covariate. Free, Python & R."
---
# Two-way factorial ANCOVA: blood pressure by treatment and sex, adjusted for baseline

You have a classic factorial experiment — two crossed treatment factors (treatment
arm and patient sex), each applied at two levels, in every combination — but you
also measured each patient's systolic blood pressure before the manipulation. Running
the plain two-way ANOVA on the post-treatment outcome leaves that baseline's
variability in the error term, shrinking every F-test. Folding it in as a
covariate soaks up between-subject noise and sharpens both main-effect and
interaction tests: this is the two-way factorial ANCOVA, with all cells sharing
one common slope on the covariate (parallel slopes).

The model is `blood_pressure = treatment * sex + baseline_bp` — `*` expands to
`treatment + sex + treatment:sex`, giving you both main effects plus the
interaction, all read *net of* the continuous baseline and fit by OLS.

## Variations

- **Power the interaction, not the main effects.** Point `target_test` at
  `treatment:sex` when the cell-difference-of-differences is the hypothesis —
  interactions usually need a noticeably larger N than main effects to reach the
  same power.
- **More than two treatment levels.** Swap treatment for a 3-level factor (e.g.
  `treatment=(factor,3)` for placebo, dose A, dose B); the page stays a two-way
  factorial ANCOVA, now with a multi-level omnibus test on treatment.
- **A weaker baseline covariate.** Dial `baseline_bp` down to the 0.25 medium
  benchmark (or 0.10 small) when the baseline only loosely predicts the outcome
  — the adjustment buys you less, so the factor tests need more N to clear power.
- **Let the slopes diverge.** Add a `treatment:baseline_bp` term to test whether
  the baseline adjustment differs across treatment arms — that turns the
  parallel-slopes assumption into something you test rather than assume.
- **Same design, other fields:**
  - `seed_yield = watering * habitat + soil_nitrogen` — ecology: two binary crop factors adjusted for soil nitrogen baseline
  - `hourly_wage = sector * gender + experience_years` — social science: sector-by-gender wage model adjusted for experience

## Not this setup?

- [[anova/anova-04|Plain two-way factorial ANOVA]] — the same `treatment * sex` design with no covariate adjustment.
- [[ols/ols-08|Single-factor ANCOVA]] — one group factor adjusted for a baseline covariate, instead of a two-way factorial.
- [[anova/anova-05|Three-way factorial ANOVA]] — add a third crossed factor instead of a continuous covariate.

## If you'd rather have…

- [[anova/anova-04|Drop the covariate]] — plain two-way factorial ANOVA with interaction (`blood_pressure = treatment * sex`), no covariate adjustment.
- [[ols/ols-08|Single-factor ANCOVA]] — one group factor adjusted for a baseline covariate (`blood_pressure = treatment + baseline_bp`), instead of a two-way factorial.
- [[anova/anova-05|Add a third factor instead of a covariate]] — three-way factorial ANOVA (`seed_yield = watering * nitrogen * light`).
- [[ols/ols-15|The same two interacting predictors as regression coefficients/contrasts]] (`blood_pressure = treatment * sex`) rather than ANOVA omnibus tests.
- [[ols/ols-09|ANCOVA where the covariate interacts with the factor]] — homogeneity-of-slopes test (`blood_pressure = treatment * baseline_bp`), rather than a parallel-slopes covariate adjustment.

## Copy-paste setup

<!-- chunk:py:anova-06 -->
```python
from mcpower import MCPower

# Two-way factorial ANCOVA: blood pressure crossed by treatment and sex, adjusted
# for a continuous baseline_bp covariate. '*' expands treatment * sex to
# treatment + sex + treatment:sex, so both main effects and the interaction are
# fitted, each read net of baseline_bp (parallel slopes, common slope across all cells).
model = MCPower("blood_pressure = treatment * sex + baseline_bp")

# treatment and sex are two-level factors; baseline_bp stays continuous by default.
model.set_variable_type("treatment=binary, sex=binary")

# Effect sizes on the benchmark scales.
#   treatment=0.50          → medium main effect (factor benchmark 0.20/0.50/0.80).
#   sex=0.50                → medium main effect.
#   treatment:sex=0.50      → medium interaction (the cell-difference-of-differences).
#   baseline_bp=0.40        → strong continuous covariate (benchmark 0.10/0.25/0.40).
model.set_effects("treatment=0.50, sex=0.50, treatment:sex=0.50, baseline_bp=0.40")

model.set_seed(2137)

# Power for both main effects at N=160 (the adjusted factorial F-tests).
model.find_power(sample_size=160, target_test="treatment, sex")
```
<!-- /chunk:py:anova-06 -->

<!-- chunk:r:anova-06 -->
```r
suppressMessages(library(mcpower))

# Two-way factorial ANCOVA: blood pressure crossed by treatment and sex, adjusted
# for a continuous baseline_bp covariate. '*' expands treatment * sex to
# treatment + sex + treatment:sex, so both main effects and the interaction are
# fitted, each read net of baseline_bp (parallel slopes, common slope across all cells).
model <- MCPower$new("blood_pressure ~ treatment * sex + baseline_bp")

# treatment and sex are two-level factors; baseline_bp stays continuous by default.
model$set_variable_type("treatment=binary, sex=binary")

# Effect sizes on the benchmark scales.
#   treatment=0.50          -> medium main effect (factor benchmark 0.20/0.50/0.80).
#   sex=0.50                -> medium main effect.
#   treatment:sex=0.50      -> medium interaction (the cell-difference-of-differences).
#   baseline_bp=0.40        -> strong continuous covariate (benchmark 0.10/0.25/0.40).
model$set_effects("treatment=0.50, sex=0.50, treatment:sex=0.50, baseline_bp=0.40")

model$set_seed(2137)

# Power for both main effects at N=160 (the adjusted factorial F-tests).
invisible(model$find_power(sample_size = 160, target_test = "treatment, sex"))
```
<!-- /chunk:r:anova-06 -->

![[assets/anova-06-setup.png|600|theme-light]]
![[assets/anova-06-setup-dark.png|600|theme-dark]]
