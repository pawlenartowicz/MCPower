---
title: "Power for ANOVA pairwise post-hoc comparisons"
description: "Power & sample-size analysis by Monte Carlo simulation for one-way ANOVA with all pairwise post-hoc comparisons and correction. Free, Python & R."
---
# One-way design: which treatment arms differ, with all pairwise post-hoc tests

You ran a three-arm clinical trial — placebo, drug A, and drug B — and measured
pain reduction in each participant. A single omnibus "the arms differ" verdict
isn't enough: you want to know *which* treatments differ from which, so you plan
all pairwise post-hoc comparisons and need the power for each of them once a
multiple-comparison correction is applied.

As an MCPower model this is `pain_reduction = treatment`, where `treatment` is a
3-level factor. OLS fits the group means and the all-pairwise post-hoc request
turns them into three direct comparisons (placebo vs drug_a, placebo vs drug_b,
drug_a vs drug_b), with a family-wise correction shared across the set.

## Variations

- **More groups, more pairs.** Bump the factor to its real arm count —
  `treatment=(factor,4)` for four arms. The pair count grows fast: four levels
  give six pairwise comparisons, so the correction has more tests to share and
  each one needs a larger N for the same power.
- **A different correction.** Swap `correction="holm"` for `"bonferroni"`
  (more conservative), `"bh"` (false-discovery-rate, more lenient), or
  `"none"` (uncorrected per-comparison power) to see how much the correction
  costs you.
- **Unequal group separations.** A single `treatment=0.5` puts the same spacing
  on every level. To make drug B further from placebo than drug A, set the level
  effects apart rather than sharing one number across the factor — the pairwise
  comparisons then have visibly different power.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="all-contrasts", correction="holm", from_size=80, to_size=400, by=20)`
  to get the smallest N that reaches target power on every pairwise comparison.
- **Same design, other fields:**
  - `biomass = fertilizer` (control / low / high) — ecology: comparing three fertilizer regimes
  - `life_satisfaction = region` — social science: comparing residents of three geographic regions

## Not this setup?

- [[anova/anova-01|Just the omnibus F — any arm differs, no pairwise breakdown]]
- [[anova/anova-03|One planned pairwise contrast instead of all pairs]]
- [[ols/ols-12|The same factor as dummy contrasts vs a reference level]]

## If you'd rather have…

- [[anova/anova-01]] — Same one-way `pain_reduction ~ treatment` design but
  testing only the omnibus F (any-arm-differs), without the pairwise breakdown.
- [[anova/anova-03]] — Same arms, but one planned pairwise contrast instead
  of all pairwise comparisons — fewer tests, less correction.
- [[anova/anova-04]] — Step up to a two-way factorial (`sector * gender`)
  when a second grouping factor and its interaction are of interest.
- [[ols/ols-12]] — The identical three-level categorical predictor framed as
  dummy-coded regression rather than an ANOVA omnibus/post-hoc.
- [[ols/ols-07]] — Drop to two arms — the pairwise comparison reduces to a
  single independent t-test as regression.

## Copy-paste setup

<!-- chunk:py:anova-02 -->
```python
from mcpower import MCPower

# One-way design with all pairwise post-hoc comparisons: pain reduction
# across three treatment arms. Research question goes past "do the arms differ?"
# to "which arms differ from which?" — every treatment-vs-treatment pair, tested directly.
model = MCPower("pain_reduction = treatment")

# treatment is a categorical predictor with 3 levels -> 3 pairwise comparisons
# (placebo-drug_a, placebo-drug_b, drug_a-drug_b).
model.set_variable_type("treatment=(factor,3)")

# Standardised effect on the factor benchmark scale (0.20 / 0.50 / 0.80):
# a medium separation between the treatment means. A 3-level factor expands to two
# reference-coded dummies (treatment[2], treatment[3]) — set each one explicitly.
model.set_effects("treatment[2]=0.5, treatment[3]=0.5")

# Power at N=180 for every pairwise post-hoc comparison, with Holm correction
# to control the family-wise error rate across the three tests.
model.find_power(
    sample_size=180,
    target_test="all-contrasts",
    correction="holm",
)
```
<!-- /chunk:py:anova-02 -->

<!-- chunk:r:anova-02 -->
```r
suppressMessages(library(mcpower))

# One-way design with all pairwise post-hoc comparisons: pain reduction
# across three treatment arms. Research question goes past "do the arms differ?"
# to "which arms differ from which?" — every treatment-vs-treatment pair, tested directly.
model <- MCPower$new("pain_reduction ~ treatment")

# treatment is a categorical predictor with 3 levels -> 3 pairwise comparisons
# (placebo-drug_a, placebo-drug_b, drug_a-drug_b).
model$set_variable_type("treatment=(factor,3)")

# Standardised effect on the factor benchmark scale (0.20 / 0.50 / 0.80):
# a medium separation between the treatment means. A 3-level factor expands to two
# reference-coded dummies (treatment[2], treatment[3]) — set each one explicitly.
model$set_effects("treatment[2]=0.5, treatment[3]=0.5")

# Power at N=180 for every pairwise post-hoc comparison, with Holm correction
# to control the family-wise error rate across the three tests.
invisible(model$find_power(
  sample_size = 180,
  target_test = "all-contrasts",
  correction = "holm"
))
```
<!-- /chunk:r:anova-02 -->

![[assets/anova-02-setup.png|600|theme-light]]
![[assets/anova-02-setup-dark.png|600|theme-dark]]
