---
title: "Regression power analysis - MCPower app"
description: "Power & sample-size analysis for OLS and logistic regression in the MCPower app - formula, predictors, effect sizes, and robustness scenarios, step by step."
---
# Regression power (app)

Regression covers four outcome types in one panel — **Linear** (continuous, OLS), **Logit** and **Probit** (binary), and **Poisson** (counts). Pick the outcome with the dropdown at the top; the controls below adapt. See [[concepts/supported-families|supported families]] for the link function and baseline setter behind each choice. Top to bottom:

## 1. Formula

Enter the right-hand side of your model — e.g. `y = x1 + x2 + x1:x2`. Operators follow R: `+` adds a predictor, `:` is the interaction term on its own, and `*` expands to the main effects **and** their interaction (`a * b` = `a + b + a:b`, so don't also write `a*b` alongside `a + b`). See [[concepts/model-specification|formula syntax]].

## 2. Predictors

Each predictor is one card. Pick its type next to the name — **continuous**, **binary**, or **factor** (the type sets how it is simulated and how its effect size is read) — then set the standardised effect size to detect in the same card (larger effects need fewer observations). A factor with k levels expands into k − 1 indicator rows, one per non-reference level; each interaction term gets its own card. Benchmarks: continuous **0.10 / 0.25 / 0.40**, binary or factor **0.20 / 0.50 / 0.80** (small / medium / large) — prefer a value justified by prior evidence. See [[concepts/variable-types|variable types]] and [[concepts/effect-sizes|effect sizes]].

## 3. Robustness scenarios

The **Robustness scenarios** toggle in the status bar repeats every run under three perturbation sets — **Optimistic** (your exact settings, no perturbations), **Realistic** (moderate assumption violations: effects vary between studies, correlations fluctuate, distributions drift from normal), and **Doomer** (severe violations, a worst case) — so you get a *range* of power instead of one optimistic number. If even Doomer clears your target, the design is robust; if only Optimistic reaches it, increase the sample size. Each set's knobs are editable under **Settings → Scenarios**. See [[concepts/scenario-analysis|scenario analysis]].

## 4. Optional settings

- **Correlations** — a collapsed, optional sub-section: set pairwise correlations if predictors are not independent; correlated predictors share information and usually lower power. Leave it collapsed (all zero) for independence. Only **continuous** predictors appear in the triangle — binary and factor predictors are excluded. [[concepts/correlations|predictor correlations]]
- **Baseline probability** (Logit / Probit) — the outcome probability when every predictor is at its reference level. It fixes the model intercept and strongly affects power (outcomes near 0 or 1 are harder). Hidden for Linear and Poisson outcomes.
- **Baseline rate** (Poisson only) — the expected count when every predictor is at its reference level; fixes the log-link intercept the same way baseline probability does for the binary families.
- **Tests & corrections** — test all coefficients, the first only, or a custom subset; correcting several (Bonferroni, Holm, Benjamini–Hochberg) controls the error rate at the cost of power. [[concepts/multiple-testing|multiple testing]]
- **Advanced** — number of simulations (continuous defaults to 1,600), α (0.05), seed (2137), and the failed-simulation tolerance.

## Find power at a sample size

The **Find power** card takes a fixed sample size and reports the power your design has at that *n* — type the planned number of observations into the single `n` field and run. It is the complement of **Find sample size**, which instead searches a range of *n* for the sample size that hits a target power; here *n* is the input and power is the result. Use it to check "how much power does the study I can afford actually have?"

Both modes share the same model, predictors, and effect sizes — only the run card differs. **Find power** answers one *n* with one power number per tested effect; **Find sample size** sweeps a from/to grid and reads the required *n* off the fitted power curve. See [[concepts/required-sample-size|how required N is estimated]].
