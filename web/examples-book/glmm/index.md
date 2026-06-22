---
title: "GLMM power analysis: clustered binary outcomes"
description: "Power & sample-size analysis by Monte Carlo simulation for generalised linear mixed models: grouped or clustered binary outcomes. Free, Python & R."
---
# Generalised linear mixed models

Generalised linear mixed models: a binary outcome with grouped or clustered
data. The mixed-model counterpart of GLM.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is continuous, you
> want [[lmm/index|mixed models]]; if your data is not grouped, see
> [[glm/index|GLM]].

## Examples

<!-- examples-index -->
- [[glmm/glmm-01|Cluster-randomised binary trial (random intercept per cluster)]]
  `infection ~ treatment + (1|hospital)` — one between-cluster treatment effect on a yes/no outcome, clusters randomised whole.
- [[glmm/glmm-02|Longitudinal binary outcome over time (random intercept per subject)]]
  `symptom_present ~ month + treatment + (1|patient)` — a yes/no outcome tracked over time within subjects across two arms.
- [[glmm/glmm-03|Difference-in-differences on a binary outcome (group x time GLMM)]]
  `employed ~ policy_group * period + (1|individual)` — the group-by-time interaction (DiD) on a clustered binary outcome.
- [[glmm/glmm-04|Logistic GLMM with a continuous predictor and random slope]]
  `species_present ~ temperature + (1 + temperature|site)` — a continuous predictor whose slope varies across groups.
<!-- /examples-index -->
