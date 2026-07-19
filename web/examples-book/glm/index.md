---
title: "Power analysis for logistic, probit, and Poisson regression (GLM)"
description: "Power & sample-size analysis by Monte Carlo simulation for GLM designs: logistic and probit binary outcomes, Poisson count outcomes. Free, Python & R."
---
# GLM

Generalised linear models: logistic and probit regression for a binary outcome
(yes/no, success/failure, event/no-event), and Poisson regression for a count
outcome (number of events, visits, occurrences).

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is continuous, you
> want [[ols/index|OLS]]; if your data is grouped or clustered, see
> [[glmm/index|GLMM]].

## Examples

<!-- examples-index -->
### Logistic, single and two-group

- [[glm/glm-01|Simple logistic regression: one continuous predictor]]
  `relapse ~ biomarker_level` — one continuous predictor on a yes/no outcome, no covariates.
- [[glm/glm-02|Logistic two-group comparison (binary predictor)]]
  `remission ~ treatment` — compare event rates between two groups (chi-square recast).

### Logistic, adjusted and categorical

- [[glm/glm-03|Logistic regression with a categorical predictor]]
  `survived ~ habitat` — a multi-level categorical predictor on a binary outcome.
- [[glm/glm-04|Multiple logistic regression: covariate-adjusted]]
  `employed ~ years_education + age + gender` — a focal predictor on a binary outcome, covariate-adjusted.
- [[glm/glm-09|Logistic regression: continuous predictor plus categorical control]]
  `employed ~ experience_years + region` — a continuous predictor plus a categorical control (parallel slopes).

### Logistic interactions

- [[glm/glm-05|Logistic continuous-by-continuous moderation]]
  `relapse ~ biomarker_level * age` — do two continuous predictors interact on a binary outcome?
- [[glm/glm-06|Logistic treatment-by-moderator interaction (binary x continuous)]]
  `remission ~ treatment * biomarker_level` — a binary-by-continuous interaction on a yes/no outcome.
- [[glm/glm-07|Logistic 2x2 factor-by-factor interaction]]
  `voted ~ gender * urban` — two binary factors interacting on a binary outcome.
- [[glm/glm-08|Logistic three-way interaction]]
  `germinated ~ light * moisture * temperature` — a three-way interaction on a binary outcome.

### Count and probit outcomes

- [[glm/glm-10|Poisson count regression: clinic visits predicted by treatment and age]]
  `clinic_visits ~ treatment + age` — a count outcome, log-link Poisson regression with a binary and a continuous predictor.
- [[glm/glm-11|Probit regression: household income predicting voter turnout, controlling for region]]
  `voted ~ income + region` — a binary outcome via the probit link, one continuous predictor plus a categorical control.
<!-- /examples-index -->
