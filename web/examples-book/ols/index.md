---
title: "OLS / linear regression power analysis examples"
description: "Power & sample-size analysis by Monte Carlo simulation for OLS / linear regression - t-tests, ANCOVA, interactions, factorial designs. Free, Python & R."
---
# OLS / linear regression

Ordinary least squares for a continuous outcome: the workhorse family. Use these
when your response is a measured quantity (a score, a time, a concentration) and
your predictors are any mix of continuous variables, binary groups, and factors.

> [!note]
> These pages are a *recognition* index — organised by the shape of the analysis,
> not by which MCPower feature they show off. If your outcome is binary, you want
> [[glm/index|GLM]]; if your data is grouped or repeated-measures, see
> [[lmm/index|mixed models]].

## Examples

<!-- examples-index -->
- [[ols/ols-01|Simple linear regression: one continuous predictor]]
  `wage ~ years_education` — one continuous predictor's effect on a continuous outcome, nothing held constant.
- [[ols/ols-02|Two-predictor multiple regression]]
  `plant_biomass ~ rainfall + soil_nitrogen` — two continuous predictors, each adjusted for the other.
- [[ols/ols-03|Three continuous predictors, side by side]]
  `cholesterol ~ age + bmi + exercise_hours` — three additive continuous slopes, mutually adjusted.
- [[ols/ols-04|Continuous-by-continuous moderation]]
  `well_being ~ income * social_support` — does income's slope change across levels of social support?
- [[ols/ols-05|Three-way continuous interaction]]
  `growth_rate ~ temperature * moisture * soil_ph` — a full three-way interaction among continuous predictors.
- [[ols/ols-06|Interaction-only term (no moderator main effect)]]
  `yield ~ nitrogen + nitrogen:water` — water bends the nitrogen slope but has no main effect of its own.
- [[ols/ols-07|Two groups as regression (independent t-test)]]
  `pain_score ~ treatment` — compare two group means as a single-binary-predictor regression.
- [[ols/ols-08|ANCOVA: group effect adjusting for a baseline]]
  `blood_pressure ~ treatment + baseline_bp` — group effect net of a continuous baseline (parallel slopes).
- [[ols/ols-09|ANCOVA homogeneity-of-slopes test]]
  `blood_pressure ~ treatment * baseline_bp` — does the treatment effect depend on the baseline covariate?
- [[ols/ols-10|Adjusted two-group comparison (parallel slopes)]]
  `monthly_income ~ union_member + experience_years` — a binary group gap holding a continuous control fixed.
- [[ols/ols-11|Binary-by-continuous moderation]]
  `wage ~ gender * experience_years` — does the experience slope differ between two groups?
- [[ols/ols-12|Three-level categorical predictor]]
  `abundance ~ habitat` — a single 3-level factor, read as its two dummy contrasts vs the reference.
- [[ols/ols-13|One focal predictor adjusted for covariates]]
  `hourly_wage ~ years_education + age + experience_years + tenure` — one focal predictor net of correlated controls.
- [[ols/ols-14|Factor interacting with a continuous predictor]]
  `biomass ~ habitat * rainfall` — does the rainfall slope differ across the levels of a factor?
- [[ols/ols-15|Two interacting categorical predictors (2x2 factorial)]]
  `job_satisfaction ~ gender * sector` — two two-level factors and their interaction (factorial as regression).
- [[ols/ols-16|Continuous moderation with a covariate]]
  `recovery_days ~ dose * age + baseline_severity` — a continuous interaction adjusted for an additive covariate.
- [[ols/ols-18|Ordinal predictor as a linear trend]]
  `tumor_shrinkage ~ dose_level` — an ordered dose (0–3) read as one slope.
<!-- /examples-index -->
