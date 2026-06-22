---
title: "Power analysis for logistic regression (GLM)"
description: "Power & sample-size analysis by Monte Carlo simulation for logistic regression and binary-outcome GLM designs, from t-test to interactions. Free, Python & R."
---
# GLM

Generalised linear models for a binary outcome (logistic regression, logit
link). Use these when your response is yes/no, success/failure, or
event/no-event.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is continuous, you
> want [[ols/index|OLS]]; if your data is grouped or clustered, see
> [[glmm/index|GLMM]].

## Examples

<!-- examples-index -->
- [[glm/glm-01|Simple logistic regression: one continuous predictor]]
  `relapse ~ biomarker_level` — one continuous predictor on a yes/no outcome, no covariates.
- [[glm/glm-02|Logistic two-group comparison (binary predictor)]]
  `remission ~ treatment` — compare event rates between two groups (chi-square recast).
- [[glm/glm-03|Logistic regression with a categorical predictor]]
  `survived ~ habitat` — a multi-level categorical predictor on a binary outcome.
- [[glm/glm-04|Multiple logistic regression: covariate-adjusted]]
  `employed ~ years_education + age + gender` — a focal predictor on a binary outcome, covariate-adjusted.
- [[glm/glm-05|Logistic continuous-by-continuous moderation]]
  `relapse ~ biomarker_level * age` — do two continuous predictors interact on a binary outcome?
- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]]
  `remission ~ treatment * biomarker_level` — a binary-by-continuous interaction on a yes/no outcome.
- [[glm/glm-07|Logistic 2x2 factor-by-factor interaction]]
  `voted ~ gender * urban` — two binary factors interacting on a binary outcome.
- [[glm/glm-08|Logistic three-way interaction]]
  `germinated ~ light * moisture * temperature` — a three-way interaction on a binary outcome.
- [[glm/glm-09|Logistic regression: continuous predictor plus categorical control]]
  `employed ~ experience_years + region` — a continuous predictor plus a categorical control (parallel slopes).
<!-- /examples-index -->
