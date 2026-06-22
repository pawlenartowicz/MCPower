---
title: "Power for a planned ANOVA pairwise contrast"
description: "Power & sample-size analysis by Monte Carlo simulation for a single planned pairwise contrast in one-way ANOVA, no correction. Free, Python & R."
---
# One planned contrast between two of three regions

You surveyed residents of three geographic regions and measured life satisfaction.
You do not care about every regional comparison: before collecting data you
committed to **one** comparison, say region 2 against region 3. You want the
power for that single planned contrast, not an omnibus verdict and not every
pairwise test.

As an MCPower model this is `life_satisfaction = region`, where `region` is a
3-level factor. You target one specific pairwise comparison — `region[2] vs
region[3]` — and because it is a single pre-specified contrast it needs no
multiplicity correction.

## Variations

- **A different pair.** Point the contrast at whichever two regions you planned
  to compare — `region[1] vs region[2]` for region 2 against the reference
  region, for example. The reference level is just one of the three; any pair
  is fair game.
- **One region clearly different.** A single `region=0.50` puts the same shift
  on every non-reference level, so the planned contrast lands on the difference
  of two equal effects. To make the regions differ, set the per-level effects
  apart rather than sharing one number across the factor.
- **More than three regions.** Bump the factor to its real count —
  `region=(factor,4)` — and the planned-contrast syntax still names exactly the
  two levels you decided on in advance.
- **Search for the N instead of fixing it.** Swap the `find_power` call for
  `find_sample_size(target_test="region[2] vs region[3]", from_size=50,
  to_size=400, by=10)` to get the smallest N that reaches target power on that
  one contrast.
- **Same design, other fields:**
  - `biomass = fertilizer` (control / low / high) — ecology: one planned fertilizer comparison
  - `pain_reduction = treatment` (placebo / drug_a / drug_b) — clinical: one planned treatment comparison

## Not this setup?

- [[anova/anova-02|Test every pairwise difference, not just one]]
- [[anova/anova-01|Just the omnibus F-test of overall group differences]]
- [[ols/ols-07|Drop to two groups: a plain independent t-test]]

## If you'd rather have…

- [[anova/anova-02]] — Test every pairwise difference instead of one contrast —
  same `life_satisfaction ~ region` design, all post-hoc comparisons with
  multiplicity correction.
- [[anova/anova-01]] — Just the omnibus "are the groups different at all"
  F-test, without any planned or post-hoc pairwise comparison.
- [[ols/ols-12]] — The same 3-level region as a dummy-coded regression — gives
  each level's coefficient vs the reference rather than ANOVA-style contrasts.
- [[ols/ols-07]] — Drop to two groups — a planned single comparison becomes a
  plain independent t-test recast as regression.
- [[anova/anova-04]] — Add a second factor for a two-way factorial design with
  an interaction, if your grouping has crossed structure.

## Copy-paste setup

<!-- chunk:py:anova-03 -->
```python
from mcpower import MCPower

# Three-arm survey: life satisfaction measured across residents of three regions.
# Research question: you have ONE pre-specified comparison in mind (a planned
# contrast) — e.g. region 2 vs region 3 — decided before seeing the data.
model = MCPower("life_satisfaction = region")

# region is a categorical predictor with 3 levels.
model.set_variable_type("region=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (region[2], region[3]) — the bare factor name is not an effect.
model.set_effects("region[2]=0.50, region[3]=0.50")

# Power for the one planned contrast — level 2 vs level 3 of region.
# A single pre-specified comparison needs no multiplicity correction.
model.find_power(sample_size=150, target_test="region[2] vs region[3]")
```
<!-- /chunk:py:anova-03 -->

<!-- chunk:r:anova-03 -->
```r
suppressMessages(library(mcpower))

# Three-arm survey: life satisfaction measured across residents of three regions.
# Research question: you have ONE pre-specified comparison in mind (a planned
# contrast) — e.g. region 2 vs region 3 — decided before seeing the data.
model <- MCPower$new("life_satisfaction ~ region")

# region is a categorical predictor with 3 levels.
model$set_variable_type("region=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (region[2], region[3]) — the bare factor name is not an effect.
model$set_effects("region[2]=0.50, region[3]=0.50")

# Power for the one planned contrast — level 2 vs level 3 of region.
# A single pre-specified comparison needs no multiplicity correction.
invisible(model$find_power(sample_size = 150, target_test = "region[2] vs region[3]"))
```
<!-- /chunk:r:anova-03 -->

![[assets/anova-03-setup.png|600|theme-light]]
![[assets/anova-03-setup-dark.png|600|theme-dark]]
