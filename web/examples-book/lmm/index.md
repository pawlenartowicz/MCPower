---
title: "Power analysis for linear mixed models"
description: "Power & sample-size analysis by Monte Carlo simulation for linear mixed models: repeated measures, growth curves, cluster trials. Free, Python & R."
---
# Linear mixed models

Linear mixed models for a continuous outcome with grouped or clustered data —
repeated measures, students within classrooms, sites in a trial. The grouping
soaks up correlation that plain OLS would ignore.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your data is not grouped, you
> want [[ols/index|OLS]]; if your outcome is binary, see [[glmm/index|GLMM]].

## Examples

<!-- examples-index -->
- [[lmm/lmm-01|Repeated measures: random intercept per subject]]
  `systolic_bp ~ phase + (1|patient)` — does a continuous outcome change over repeated visits, each subject as its own control.
- [[lmm/lmm-02|Treatment x time interaction (random intercept)]]
  `pain_score ~ treatment * week + (1|patient)` — does the trajectory over time differ between two arms (difference-in-differences).
- [[lmm/lmm-03|Growth curve: random intercept and slope of time]]
  `seedling_height ~ week + (week | seedling)` — the average time slope judged against scattering individual growth rates.
- [[lmm/lmm-04|Conditional growth: treatment moderating individual slopes]]
  `seedling_height ~ fertilizer * week + (1 + week | seedling)` — whether two arms diverge in growth rate, with random intercepts and slopes.
- [[lmm/lmm-05|Two-level cluster-randomized trial (random intercept)]]
  `cholesterol ~ treatment + (1|clinic)` — a trial randomising whole clusters, one continuous outcome per member.
- [[lmm/lmm-06|Cluster RCT with a baseline covariate (adjusted)]]
  `blood_pressure ~ treatment + baseline_bp + (1|clinic)` — a cluster trial adjusted for a baseline covariate.
- [[lmm/lmm-07|Multisite trial: treatment effect varying across sites]]
  `recovery_days ~ treatment + (treatment | site)` — a treatment effect allowed to vary across sites (random slope).
- [[lmm/lmm-09|Within-subjects one-way design: repeated condition factor]]
  `enzyme_activity ~ condition + (1|sample)` — a repeated 3-level condition factor compared within each subject.
<!-- /examples-index -->
