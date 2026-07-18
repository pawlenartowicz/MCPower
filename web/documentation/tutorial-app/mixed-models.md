---
title: "Mixed-model power analysis - MCPower app"
description: "Power & sample-size analysis for clustered and repeated-measures designs in the MCPower app - linear and logistic mixed models by Monte Carlo simulation."
---
# Mixed-model power (app)

Mixed models add **random effects** for clustered or repeated-measures data — observations grouped within schools, clinics, subjects, and so on. Top to bottom:

## 1. Formula with a random effect

Write the fixed part as in regression, then add a grouping term in parentheses — a random intercept for `cluster` is `(1 | cluster)`, e.g. `y = x1 + x2 + (1 | cluster)`. Fixed-effect operators match regression: `:` is the interaction term on its own and `*` expands to `a + b + a:b`. See [[concepts/model-specification|formula syntax]] and [[concepts/mixed-effects|mixed effects]].

## 2. Outcome type — Linear, Logit, Probit, or Poisson

The **Outcome type** dropdown at the top of the Model section switches the mixed model's family — the same four choices as plain regression, see [[concepts/supported-families|supported families]]. **Linear** (the default) is a Gaussian linear mixed model; **Logit** and **Probit** are a clustered GLMM for a binary outcome (a logistic or probit regression with a cluster-level random intercept); **Poisson** is a clustered GLMM for a count outcome.

Under **Logit** or **Probit** a **baseline probability** input appears — the event probability when every predictor sits at its reference level, which fixes the model intercept. Predictor effects are then read on the **log-odds** (Logit) or **probit** scale; see [[concepts/effect-sizes#Logistic regression|logistic effect sizes]] for what a log-odds beta means in probability terms. Under **Poisson** a **baseline rate** input appears instead — the expected count at the reference level — and effects read as log-rate shifts. Switching back to Linear restores the Gaussian fit.

> [!note] Poisson clustering uses raw variance, not ICC
> The other three families size their random intercept with an **ICC** (0–1). Poisson has no natural latent-scale ICC, so its cluster panel asks for **τ² (tau-squared)** — the random intercept's raw variance — instead. See [[concepts/supported-families#clustered-counts-and-probits|clustered counts and probits]].

## 3. Cluster configuration

The grouping term unlocks the cluster panel: the **cluster name** (mirrored from the `(1 | …)` term), the **ICC** — the between-cluster variance share in [0, 1); a higher ICC makes within-cluster observations more alike, lowering the effective sample size — and the **number of clusters** (or cluster size; fix one and the other follows from n).

> [!important] Repeated measures: N is measurements, not participants
> For a repeated-measures design the cluster is the **participant** and the sample size counts **measurements** — `N = participants × trials-per-participant`. Sixty participants on 100 trials each is `n = 6000` with **number of clusters = 60**, not `n = 60`. See [[concepts/mixed-effects#repeated-measures|Repeated measures]].

## 4. Predictors

Each fixed-effect predictor is one card: pick its type next to the name (continuous, binary, or factor), then set its standardised effect size in the same card — continuous **0.10 / 0.25 / 0.40**, binary or factor **0.20 / 0.50 / 0.80**. See [[concepts/variable-types|variable types]] and [[concepts/effect-sizes|effect sizes]].

## 5. Robustness scenarios

The **Robustness scenarios** toggle in the status bar repeats every run under three perturbation sets — **Optimistic** (your exact settings, no perturbations), **Realistic** (moderate assumption violations), and **Doomer** (severe violations, a worst case) — so you get a *range* of power instead of one optimistic number. Mixed models additionally perturb the random effect (its distribution, df, and ICC noise). If even Doomer clears your target, the design is robust; if only Optimistic reaches it, increase the sample size. Each set's knobs are editable under **Settings → Scenarios**. See [[concepts/scenario-analysis|scenario analysis]].

## 6. Optional settings

The **Advanced** section exposes the number of simulations (mixed models default to 800, since each fit is heavier), α (0.05), seed (2137), and the failed-simulation tolerance.

For a clustered **Logit**, **Probit**, or **Poisson** model, Advanced also shows the estimation-mode switch — **Fast**, **Accurate**, or **AGQ** — see [[concepts/simulation-settings#estimation-mode|estimation mode]] for what each does and when AGQ is available.
