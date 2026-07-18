---
title: "Power Analysis for Mixed Models - Python"
description: "Sample size and power for mixed-effects models with clustered data in Python with MCPower: set ICC, number of clusters, and random intercepts via simulation."
---
# Mixed models

Everything so far has assumed observations are independent. That breaks down the
moment your data has a natural grouping: students nested in classrooms, patients
nested in clinics, measurements nested in subjects. Ignoring that structure inflates
your effective sample size and overstates power. A [[concepts/mixed-effects|mixed
effects]] model fixes this by adding a random intercept that absorbs between-group
variation.

## What's new on this rung

- `family="lme"` in the constructor selects the mixed-effects estimator — fit by
  maximum likelihood, so the summary header reports `estimator: MLE`;
- the random-intercept term goes directly in the formula as `(1|classroom)`;
- `set_cluster` tells the engine about the grouping structure — the cluster
  variable name, its ICC, and the number of groups.

Everything else — `find_power`, `target_test`, `verbose=False`, `.summary()` —
works exactly as before.

## The model

We study a classroom experiment: does a new `teaching_method` lift `score`,
controlling for `prior_gpa`? Students are nested in 30 classrooms.

- `family="lme"` selects the mixed estimator;
- `(1|classroom)` in the formula adds a random intercept for classroom;
- `set_cluster("classroom", ICC=0.15, n_clusters=30)` supplies the two numbers
  that define the grouping: the **ICC** (intraclass correlation — the fraction of
  total variance that lies *between* classrooms; 0.15 means 15% of score variation
  is between classrooms rather than between students) and **n_clusters** (how many
  distinct classrooms exist);
- effects follow the same [[concepts/effect-sizes|effect-size]] benchmarks as OLS —
  0.10 / 0.25 / 0.40 for small / medium / large.

> [!note]
> Mixed models default to **800 simulations** rather than the 1,600 used for OLS,
> because each mixed-model fit is more expensive. Confidence intervals are
> correspondingly a little wider; increase `n_sims` explicitly if you need tighter
> Monte-Carlo uncertainty.

## How much power at n = 300?

<!-- example:05-mixed -->
```python
from mcpower import MCPower

model = MCPower("score = teaching_method + prior_gpa + (1|classroom)", family="lme")
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.4, prior_gpa=0.18")
model.set_cluster("classroom", ICC=0.15, n_clusters=30)

result = model.find_power(sample_size=300, target_test="teaching_method, prior_gpa", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: score = teaching_method + prior_gpa + (1|classroom)
estimator: MLE  N=300  sims=800  α=0.05  target=80%
effects: teaching_method=0.40, prior_gpa=0.18

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
teaching_method      90.9%      80%
prior_gpa            85.8%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
teaching_method      90.9%   [88.7%, 92.7%]
prior_gpa            85.8%   [83.2%, 88.0%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=800.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        1.6%       100%
1       20.1%      98.4%
2       78.2%      78.2%
────────────────────────

Estimator details
  tau_estimate: nan
  boundary_hits: 0
  joint_uncorrected_rate: 0.9812
  joint_corrected_rate: 0.9812
  singular_fit_rate: 0
  boundary_rate_per_component: [0.0]

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

Both targets are comfortably above 80%: `teaching_method` lands at **90.9%** and
`prior_gpa` at **85.8%**. The joint significance distribution shows a 78.2% chance
of detecting *both* effects in the same study — close to the 80% target, so N = 300
is a reasonable budget when both effects matter.

The estimator details block reports `boundary_hits: 0` — no simulation hit a
variance boundary during fitting. Boundary hits show up more often in smaller
samples, where mixed-model fitting is numerically demanding, so seeing none here is
reassuring at this sample size.

> [!tip]
> ICC values are rarely known precisely in advance. Run a few scenarios with
> `ICC=0.05`, `ICC=0.15`, and `ICC=0.30` to see how sensitive your power estimate
> is to that assumption — the same way you would stress-test effect sizes.

## Going further

### Repeated measures

A repeated-measures (within-subjects) design — each participant measured across many trials — is a clustered design with the **participant as the cluster**. The catch is the sample size: `n` counts **measurements, not participants**, so `n = participants × trials-per-participant`. Sixty participants on 100 trials each is `sample_size=6000` with `n_clusters=60` — passing `sample_size=60` would model one measurement per person:

```python
# 60 participants × 100 trials = 6000 measurements
model = MCPower("rt = condition + (1|participant)", family="lme")
model.set_cluster("participant", ICC=0.30, n_clusters=60)
result = model.find_power(sample_size=6000, target_test="condition", verbose=False)
```

Extra trials sharpen a **within-participant** predictor (one rotated trial by trial, estimated against the within-person residual) but do nothing for a **between-participant** one (group, age — constant within a person). Mark between-participant predictors as cluster-level so the engine uses the participant count as their effective sample size:

```python
# `group` is between-participant; `condition` stays within-participant
model = MCPower("rt = condition + group + (1|participant)", family="lme")
model.set_cluster("participant", ICC=0.30, n_clusters=60,
                  cluster_level_vars=["group"])
```

See [[concepts/mixed-effects#repeated-measures|Repeated measures]] for the within- vs between-participant power discussion.

### Cluster-level predictors

If your primary predictor is assigned at the group level — a treatment applied to all students in a classroom, a policy rolled out site-wide — declare it in `set_cluster` so the engine uses `n_clusters` (not total N) as the effective sample size:

```python
model.set_cluster("site", ICC=0.15, n_clusters=30,
                  cluster_level_vars=["treatment"])
```

Halving `n_clusters` at fixed total N roughly doubles the SE of a cluster-level predictor while barely affecting within-cluster predictors. See [[concepts/mixed-effects#cluster-level-predictors|Cluster-level predictors]] for the design-effect discussion.

### Crossed and nested groupings

For crossed designs (e.g. participants × stimuli) or nested designs (classrooms within schools), name every grouping factor in the formula and call `set_cluster` once per factor:

```python
# Crossed: 24 subjects × 12 items — formula "rt ~ frequency + (1|subject) + (1|item)"
model.set_cluster("subject", ICC=0.20, n_clusters=24)
model.set_cluster("item", ICC=0.15, n_clusters=12)
```

For crossed groupings, N must be a multiple of the design atom (`n_subjects × n_items`); for nested, call `set_cluster` once for the outer factor and once for the `"school:classroom"` child with `n_per_parent`. See [[concepts/mixed-effects#multiple-groupings-crossed-and-nested|Multiple groupings]].

### Random slopes

A random intercept lets each group start at a different baseline; a **random slope** lets the effect itself vary across groups. Declare it in the formula as `(1 + x|group)` and give it a variance in `set_cluster` — the extra variability widens the SE of the *average* effect, so treating a varying slope as fixed overstates power:

```python
model = MCPower("rt = dose + (1 + dose|subject)", family="lme")
model.set_cluster("subject", ICC=0.20, n_clusters=30,
                  random_slopes=["dose"], slope_variance=0.15)
```

Random-slope fits hit variance boundaries more often than intercept-only ones, so watch the boundary-hit rate in `.summary()`. See [[concepts/mixed-effects#random-slopes|Random slopes]].

### Clustered logistic (GLMM)

A binary outcome *with* clustering is a logistic GLMM — combine `family="logit"` with a `(1|group)` term, where the ICC is read on the log-odds (latent) scale. This composes with the cluster-level trick above: a treatment randomised at the clinic level in a binary-outcome trial is just `family="logit"` plus `cluster_level_vars`:

```python
model = MCPower("recovered = treatment + baseline_severity + (1|clinic)", family="logit")
model.set_baseline_probability(0.3)
model.set_cluster("clinic", ICC=0.10, n_clusters=40, cluster_level_vars=["treatment"])
```

See [[concepts/mixed-effects#clustered-logistic-glmm|Clustered logistic (GLMM)]].

### Clustered probit and Poisson

The same GLMM path covers `family="probit"` and `family="poisson"` — see
[[concepts/supported-families|supported families]] for the full picture.
Clustered Poisson has no latent-scale ICC, so its random intercept is sized by
**raw variance** with `tau_squared` instead of `ICC`:

```python
# Clustered probit — same set_cluster call as logit
model = MCPower("recovered = treatment + (1|clinic)", family="probit")
model.set_baseline_probability(0.3)
model.set_cluster("clinic", ICC=0.15, n_clusters=30)

# Clustered Poisson — tau_squared in place of ICC
model = MCPower("visits = treatment + (1|clinic)", family="poisson")
model.set_baseline_rate(2.0)
model.set_cluster("clinic", tau_squared=0.10, n_clusters=30)
```

### Estimation mode for clustered GLMMs

A clustered binary/count GLMM (logit, probit, or Poisson) accepts two more
`find_power`/`find_sample_size` arguments: `wald_se` (default `"rx"`, a fast
Wald-SE shortcut; `"hessian"` is slower and slightly more conservative) and
`agq` (default `1`, Laplace; an odd value up to 25 switches to adaptive
quadrature for a single-grouping, ≤3-random-effects design):

```python
result = model.find_power(sample_size=300, target_test="treatment",
                          wald_se="hessian", agq=5, verbose=False)
```

See [[concepts/simulation-settings#estimation-mode|Estimation mode]] for the full
explanation of both controls, including AGQ's eligibility rules and what happens
when a design falls outside them.

### Scenario stress-testing for mixed models

Mixed models support three scenario knobs: `random_effect_dist` (normal / heavy_tailed / right_skewed), `random_effect_df`, and `icc_noise_sd`. Use them to probe how sensitive power is to non-Gaussian random effects or an uncertain ICC:

```python
model.set_scenario_configs({
    "re_stress": {"random_effect_dist": "heavy_tailed",
                  "random_effect_df": 5, "icc_noise_sd": 0.07}
})
model.find_power(sample_size=600, target_test="teaching_method",
                 scenarios=["optimistic", "re_stress"])
```

See [[concepts/scenario-analysis#mixed-model-knobs|Mixed-model scenario knobs]].

next → [[06_anova-posthoc|ANOVA & post-hoc]]
