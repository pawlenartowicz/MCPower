---
title: "ANOVA power analysis examples (Python & R)"
description: "Power & sample-size analysis by Monte Carlo simulation for ANOVA designs - one-way, factorial, ANCOVA, with F-tests and post-hoc contrasts. Free, Python & R."
---
# ANOVA

Factor designs analysed as OLS: one-way, factorial, and mixed factorial layouts,
with omnibus F tests and pairwise post-hoc contrasts. Reach here when your
predictors are categorical groups and you care about differences between levels.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is binary, you want
> [[glm/index|GLM]]; if your data is grouped or repeated-measures, see
> [[lmm/index|mixed models]].

## Examples

<!-- examples-index -->
### One-way

- [[anova/anova-01|One-way ANOVA, omnibus F-test across 3+ groups]]
  `biomass ~ fertilizer` — power for the pooled F-test that any group mean differs.
- [[anova/anova-02|One-way ANOVA, all pairwise post-hoc comparisons]]
  `pain_reduction ~ treatment` — power for every group-vs-group pair, with correction.
- [[anova/anova-03|One-way ANOVA, one planned pairwise contrast]]
  `life_satisfaction ~ region` — power for a single pre-specified pair, no multiplicity penalty.

### Factorial

- [[anova/anova-04|Two-way factorial ANOVA with interaction]]
  `hourly_wage ~ sector * gender` — two crossed factors plus their interaction.
- [[anova/anova-05|Three-way factorial ANOVA (2x2x2)]]
  `seed_yield ~ watering * nitrogen * light` — three crossed factors and the three-way interaction.
- [[anova/anova-06|Two-way factorial ANCOVA, covariate-adjusted]]
  `blood_pressure ~ treatment * sex + baseline_bp` — factorial group effects adjusted for a baseline covariate.
<!-- /examples-index -->
