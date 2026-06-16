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
- **Small-sample significance reference for mixed models** — finer-grained
  critical values (e.g. a between-within or Satterthwaite degrees-of-freedom
  correction) for designs with few clusters, where the standard test is mildly
  anticonservative. The shipped test is the honest power of the test most people
  actually run, so this would be built **on request** rather than by default.
- **Random slopes on crossed and nested groupings** — today a mixed model can
  carry random slopes only on its primary grouping; additional crossed or nested
  factors enter as random intercepts only (see
  [[concepts/limitations|limitations]]). Lifting this to match what `lme4`
  expresses (varying slopes on a second grouping) needs a different solver path,
  so it would be added **on request** rather than by default.

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
