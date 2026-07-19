---
title: "GLMM power analysis: clustered binary and count outcomes"
description: "Power & sample-size analysis by Monte Carlo simulation for generalised linear mixed models: grouped or clustered logistic, probit, and Poisson outcomes. Free, Python & R."
---
# Generalised linear mixed models

Generalised linear mixed models: a binary (logistic or probit) or count
(Poisson) outcome with grouped or clustered data, including crossed and nested
grouping structures. The mixed-model counterpart of GLM.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is continuous, you
> want [[lmm/index|mixed models]]; if your data is not grouped, see
> [[glm/index|GLM]].

## Examples

<!-- examples-index -->
### Binary (logistic)

- [[glmm/glmm-01|Cluster-randomised binary trial (random intercept per cluster)]]
  `infection ~ treatment + (1|hospital)` — one between-cluster treatment effect on a yes/no outcome, clusters randomised whole.
- [[glmm/glmm-02|Longitudinal binary outcome over time (random intercept per subject)]]
  `symptom_present ~ month + treatment + (1|patient)` — a yes/no outcome tracked over time within subjects across two arms.
- [[glmm/glmm-03|Difference-in-differences on a binary outcome (group x time GLMM)]]
  `employed ~ policy_group * period + (1|individual)` — the group-by-time interaction (DiD) on a clustered binary outcome.
- [[glmm/glmm-04|Logistic GLMM with a continuous predictor and random slope]]
  `species_present ~ temperature + (1 + temperature|site)` — a continuous predictor whose slope varies across groups.

### Binary (probit)

- [[glmm/glmm-08|Dose-response trial on an adverse response (longitudinal probit)]]
  `adverse_response ~ dose + (1|subject)` — repeated dose-level measurements within subject, binary outcome via the probit link.

### Count (Poisson)

- [[glmm/glmm-05|Cluster-randomised trial on adverse-event counts (clustered Poisson)]]
  `adverse_events ~ treatment + (1|clinic)` — a count outcome, whole clinics randomised to control vs treatment.
- [[glmm/glmm-06|Two arms followed over months on a clinic-visit count]]
  `visits ~ month + treatment + (1|patient)` — a longitudinal count outcome tracked over time within subjects across two arms.
- [[glmm/glmm-07|Pollutant-exposure effect on species counts with site-varying slopes]]
  `count ~ exposure + (1 + exposure|site)` — a continuous predictor on a count outcome whose slope varies across groups.

### Advanced structure

- [[glmm/glmm-09|Survey agreement across respondents and items: crossed random effects (logistic)]]
  `agree ~ condition + (1|respondent) + (1|item)` — two independent, fully-crossed grouping factors on a binary outcome.
- [[glmm/glmm-10|Nested cluster trial on a binary pass/fail outcome: classrooms inside schools]]
  `passed ~ treatment + (1|school/classroom)` — a nested 3-level grouping structure on a binary outcome.
<!-- /examples-index -->
