---
title: "Power Analysis for Interaction Effects - Python"
description: "Power analysis for interaction effects in Python with MCPower: model treatment-by-covariate interactions and find the sample size needed to detect them."
---
# Interactions

An interaction lets one predictor's effect depend on the level of another. This rung builds on
the `satisfaction` model from rung 01 by asking a sharper question: does the treatment effect
vary with age?

## The model

The formula `satisfaction = treatment * age` uses the `*` shorthand, which the engine expands
to `treatment + age + treatment:age` — both main effects plus their product term. If you want
the product alone, use `:` directly; see [[concepts/model-specification|formula syntax]] for
details.

Three calls are all it takes:

- `set_variable_type` marks `treatment` as binary, same as before;
- `set_effects` now includes a third entry, `treatment:age=0.35`, the standardised
  interaction effect to detect;
- `target_test="treatment:age"` restricts the power report to the term we care about — the
  main effects are modelled but not the focus of the budget calculation.

<!-- example:02-interaction -->
```python
from mcpower import MCPower

# `*` expands to treatment + age + treatment:age
model = MCPower("satisfaction = treatment * age")
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, age=0.3, treatment:age=0.35")

result = model.find_power(sample_size=200, target_test="treatment:age", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: satisfaction = treatment * age
estimator: OLS  N=200  sims=1600  α=0.05  target=80%
effects: treatment=0.50, age=0.30, treatment:age=0.35

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
treatment:age        68.7%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
treatment:age        68.7%   [66.4%, 70.9%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0       31.3%       100%
1       68.7%      68.7%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

## Read the numbers

`treatment:age` lands at **68.7%** at N = 200 — well short of the 80% target, even though the
individual main effects would be comfortably powered at that size. This is the central lesson of
interactions: the product term absorbs variance from both predictors, which means the effective
signal-to-noise ratio is lower. Detecting it reliably typically demands a noticeably larger
sample than detecting either constituent main effect alone.

If 80% is the floor for your study, plug the interaction label into `find_sample_size` with the
same grid approach from rung 01 to find where the curve crosses the threshold.

> [!tip]
> Use `target_test="all"` to see main-effect power alongside the interaction in one table.
> The joint significance distribution then tells you the probability of catching *everything*
> in the same study — which is lower still.

next → [[03_correlations|Correlations]]
