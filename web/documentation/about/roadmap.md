---
title: "MCPower Roadmap - Planned Features & Scope"
description: "MCPower's power analysis roadmap: planned outcome families, new tests, and features under consideration - plus what is deliberately out of scope."
---
# Roadmap

MCPower is under active development. This page is an honest list of where it is
likely headed — the features with a settled design, the ideas still being
weighed, and the things deliberately left out. It is a wishlist, not a release
schedule: nothing here has a ship date, ordering can change, and an idea can be
dropped if it turns out not to earn its place.

For what MCPower covers **today** — OLS regression, logistic regression (GLM),
mixed-effects models, and factorial ANOVA — see [[about/index|About MCPower]]
and [[about/comparison|how it compares]].

## More outcome families

The simulation engine already fits a model to each synthetic dataset; adding an
outcome family is mostly teaching it a new model to fit. The next candidates:

- **Count outcomes (Poisson)** — for outcomes that are counts of events, in
  both the regression (GLM) and mixed-effects (GLMM) forms. One caveat carries
  into the planning interface: a count model on the log scale has no standard
  *ICC* the way a binary model does, so the random-effect size would be set as a
  raw variance rather than an intraclass correlation.
- **Probit link for binary outcomes** — an alternative to the logistic link
  already supported, for both GLM and mixed models, for studies whose analysis
  uses probit.
- **Ordinal outcomes (proportional odds)** — for ordered categorical responses
  such as Likert-scale items, slotting in alongside the binary and count
  families above.

## More tests and decision rules

Same fitted models, new ways to ask the question:

- **A guided two-group comparison in the app** — the most common design, a
  difference between two groups, made into a short dedicated flow rather than a
  full regression setup.
- **Robust and non-parametric alternatives** — rank-based tests and robust
  estimators as alternatives to ordinary regression and the standard *t*-tests,
  for designs where the usual normal-theory assumptions are shaky.
- **Observation weights** — weighted regression, so studies that weight
  observations (survey or sampling weights, inverse-variance weights, unequal
  precision) can plan power under the same weighting they will analyse with.
- **Equivalence and non-inferiority (TOST)** — power for showing an effect is
  *small enough* to be negligible within stated bounds, rather than power to
  show it is non-zero. Same fitted models, a different decision rule.
- **Minimum detectable effect (sensitivity)** — given a fixed sample size and a
  target power, the smallest effect you could reliably catch. The third leg of
  the classic power triad, reusing the sample-size search machinery.
- **Precision-based planning** — size a study for a target confidence-interval
  *width* instead of a power target, for when the goal is a tight estimate
  rather than a significance call.

## Being weighed

Real candidates, but with an open question that has to clear before they ship:

- **Bayesian power (Bayes factors)** — under consideration only if it can run
  fast enough to stay interactive, since the appeal of MCPower is getting an
  answer in seconds, not minutes.
- **Satterthwaite degrees of freedom for mixed models** — a finer-grained
  small-sample correction than between-within, for designs with few clusters,
  where the Wald z test is mildly anticonservative. A between-within
  degrees-of-freedom correction (`df_method="between_within"`) already
  shipped and is the default; on linear mixed models it matched lmerTest
  Satterthwaite in the [[validation/validation_df_calibration|calibration]],
  on logit GLMMs with 5–10 clusters it still over-rejects. Linear mixed
  models with a crossed or nested extra grouping already use Satterthwaite df
  under `"between_within"`; single-grouping designs and GLMMs do not.

## Deliberately out of scope

To keep MCPower a focused planning tool, some things are not on the roadmap at
all:

- It does **not** analyse your collected data — it plans the study before you
  run it. (Pilot data can be uploaded to *inform* the plan; see
  [[concepts/upload-data|using empirical data]].)
- It is **not** a framework for building new statistical methods, and there is
  no general scripting language for arbitrary models — the supported families
  above are the surface.

Have a design MCPower can't yet express, or a strong vote for one of the
candidates above? That feedback is what moves an item from this page into the
build.
