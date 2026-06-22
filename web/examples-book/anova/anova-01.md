---
title: "Power for a one-way ANOVA omnibus F-test"
description: "Power & sample-size analysis by Monte Carlo simulation for a one-way ANOVA omnibus F-test of group mean differences. Free, Python & R."
---
# One-way ANOVA: the omnibus F-test of fertilizer group differences

You planted seedlings into three fertilizer regimes — a control, a low-dose
treatment, and a high-dose treatment — and measured plant biomass at harvest.
Before comparing any single pair of regimes you want the headline verdict: do
the fertilizer groups differ *at all*? That is the classic one-way ANOVA omnibus
F-test.

As an MCPower model this is `biomass = fertilizer`, where `fertilizer` is a
3-level factor. OLS expands it into dummy contrasts internally, but the omnibus
test pools them into a single F-test of "any group difference" — so power here
is the chance that F-test reaches significance, not the power of any one contrast.

## Variations

- **More than three groups.** Bump the factor to its real arm count —
  `fertilizer=(factor,4)` for a four-regime design. The omnibus F-test absorbs
  the extra levels automatically; you still request it the same way with
  `target_test="all"`.
- **Unequal expected effects per arm.** A single `fertilizer=0.5` puts the same
  shift on every non-reference level. To say the high-dose arm differs more from
  control than the low-dose arm, set the level effects apart rather than sharing
  one number across the factor — the omnibus power reflects the combined spread of
  group means.
- **Read the per-group coefficients instead of the omnibus.** Swap
  `target_test="all"` for `target_test="fertilizer"` to get power for each dummy
  contrast against the reference rather than the pooled F-test plus coefficients.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="all", from_size=50, to_size=400, by=10)` to
  get the smallest N that takes the omnibus F-test to target power.
- **Same design, other fields:**
  - `pain_reduction = treatment` (placebo / drug_a / drug_b) — three-arm clinical trial omnibus F-test
  - `life_satisfaction = region` — survey comparing three geographic regions

## Not this setup?

- [[anova/anova-02|All pairwise post-hoc comparisons, not just the omnibus]]
- [[anova/anova-03|A single planned pairwise contrast instead of the F-test]]
- [[ols/ols-12|The same factor read as per-dummy regression coefficients]]

## If you'd rather have…

- [[anova/anova-02]] — Same one-way design and `biomass = fertilizer` formula,
  but adds all pairwise post-hoc comparisons instead of just the omnibus test.
- [[anova/anova-03]] — Same one-way design, but tests a single planned pairwise
  contrast rather than the omnibus F-test.
- [[ols/ols-12]] — The regression recast: a three-level dummy-coded categorical
  predictor (`biomass = condition`) — same data structure analysed as OLS.
- [[ols/ols-07]] — Drop to two groups: independent t-test as regression
  (`biomass = group`), the simplest group-comparison case.
- [[anova/anova-04]] — Add a second factor: two-way factorial ANOVA with
  interaction when you have a crossed second grouping variable.
- [[ols/ols-08]] — Add a continuous baseline covariate to the group effect
  (ANCOVA, `biomass = fertilizer + baseline`).

## Copy-paste setup

<!-- chunk:py:anova-01 -->
```python
# NOTE: the omnibus F-test cannot be requested on its own — the engine requires
# at least one marginal/contrast/post-hoc target alongside it. target_test="all"
# is the canonical way to get the omnibus: it reports the overall F PLUS every
# per-dummy coefficient. (Bare target_test="overall" is inexpressible and was
# replaced with "all".)
from mcpower import MCPower

# One-way ANOVA: plant biomass measured across three fertilizer regimes.
# Research question: do the fertilizer groups differ overall? -> the omnibus F-test.
model = MCPower("biomass = fertilizer")

# fertilizer is a categorical predictor with 3 levels (control / low / high).
model.set_variable_type("fertilizer=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts biomass by a medium amount vs the control.
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (fertilizer[2], fertilizer[3]) — the bare factor name is not an effect.
model.set_effects("fertilizer[2]=0.5, fertilizer[3]=0.5")

# Power at N=120 for the omnibus F-test of overall fertilizer differences, reported
# alongside each per-dummy coefficient ("all" = overall F + every β).
model.find_power(sample_size=120, target_test="all")
```
<!-- /chunk:py:anova-01 -->

<!-- chunk:r:anova-01 -->
```r
# NOTE: the omnibus F-test cannot be requested on its own — the engine requires
# at least one marginal/contrast/post-hoc target alongside it. target_test="all"
# is the canonical way to get the omnibus: it reports the overall F PLUS every
# per-dummy coefficient. (Bare target_test="overall" is inexpressible and was
# replaced with "all".)
suppressMessages(library(mcpower))

# One-way ANOVA: plant biomass measured across three fertilizer regimes.
# Research question: do the fertilizer groups differ overall? -> the omnibus F-test.
model <- MCPower$new("biomass ~ fertilizer")

# fertilizer is a categorical predictor with 3 levels (control / low / high).
model$set_variable_type("fertilizer=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts biomass by a medium amount vs the control.
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (fertilizer[2], fertilizer[3]) — the bare factor name is not an effect.
model$set_effects("fertilizer[2]=0.5, fertilizer[3]=0.5")

# Power at N=120 for the omnibus F-test of overall fertilizer differences, reported
# alongside each per-dummy coefficient ("all" = overall F + every β).
invisible(model$find_power(sample_size = 120, target_test = "all"))
```
<!-- /chunk:r:anova-01 -->

![[assets/anova-01-setup.png|600|theme-light]]
![[assets/anova-01-setup-dark.png|600|theme-dark]]
