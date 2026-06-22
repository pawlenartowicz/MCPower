---
title: "Power Analysis for Logistic Regression - Python"
description: "Power and sample size for logistic regression (binary outcomes) by Monte Carlo simulation in Python with MCPower: set baseline probability and log-odds effects."
---
# Logistic regression

So far every outcome has been continuous — a satisfaction score, a wellbeing index.
When the outcome is binary (recovered or not, churned or stayed, clicked or
ignored), the right model is logistic regression. The only change from the previous
rungs is a single constructor argument.

## The model

We study whether a treatment improves recovery, controlling for age. The outcome
`recovered` is 0 or 1, so we need a GLM with a logit link. Pass `family="logit"`
directly to the constructor — it is **not** a setter method, and it is **not** added
via `set_*` after construction. Logistic regression requires one additional piece of information that linear
regression does not: the baseline event rate. `set_baseline_probability(0.3)`
tells the engine that 30% of the reference group is expected to recover without
treatment. Without this call the model cannot generate realistic binary outcomes,
so the call is **required** for `family="logit"`.

Effects are still standardised and act as shifts on the log-odds scale. The
benchmarks for binary or factor predictors differ from continuous ones: small
**0.20**, medium **0.50**, large **0.80** — see [[concepts/effect-sizes|effect
sizes]] for the reasoning behind those values.

> [!note]
> Logistic regression typically needs considerably more participants than the
> equivalent linear model for the same standardised effect. This is a structural
> property of binary outcomes, not a quirk of the engine.

## How much power at n = 300?

<!-- example:04-logistic -->
```python
from mcpower import MCPower

model = MCPower("recovered = treatment + age", family="logit")
model.set_variable_type("treatment=binary")
model.set_baseline_probability(0.3)
model.set_effects("treatment=0.5, age=0.3")

result = model.find_power(sample_size=300, target_test="all", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: recovered = treatment + age
estimator: GLM  N=300  sims=1600  α=0.05  target=80%
effects: treatment=0.50, age=0.30

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
LR χ²                82.3%      80%
treatment            54.4%      80%
age                  69.4%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
LR χ²                82.3%   [80.4%, 84.1%]
treatment            54.4%   [51.9%, 56.8%]
age                  69.4%   [67.1%, 71.6%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0       14.6%       100%
1       47.0%      85.4%
2       38.4%      38.4%
────────────────────────

Estimator details
  baseline_prob_realized: nan
  singular_fit_rate: 0
  tau_squared_hat_mean: nan

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

`treatment` lands at **54.4%** and `age` at **69.4%** — both well below the 80%
target despite a sample size that would be more than adequate for OLS. The
likelihood-ratio omnibus test (`LR χ²`) reaches 82.3%, just above the target on
its own.

The joint significance distribution tells the fuller story: only **38.4%** of
simulations see both effects detected in the same study, so jointly powering
both tests will require a substantially larger N. Use `find_sample_size` to find
the crossing point.

> [!tip]
> The compact table prints automatically when you omit `verbose=False`. Use
> `.summary()` when you want the confidence intervals and the joint distribution.

next → [[05_mixed-models|Mixed models]]
