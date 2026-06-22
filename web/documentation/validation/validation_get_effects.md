---
title: "MCPower — Validating get_effects_from_data"
description: "Validation of MCPower get_effects_from_data: round-trip recovery of standardized effect sizes for OLS, GLM logistic, and MLE mixed-effects models."
right_sidebar: body
---
# MCPower validation — get_effects_from_data round-trip

# What this report validates

`get_effects_from_data` recovers standardized effect sizes from uploaded
pilot data by fitting the model the data came from. This report is the
**acceptance gate for that recovery convention**: for each estimator we
*specify* an effect of size `s`, let MCPower *simulate* a dataset from
it, then *recover* the effect from that dataset — and check the recovery
lands where the convention says it should.

The estimator is dispatched off the model family, and the recovery scale
differs by estimator:

- **OLS (continuous outcome)** z-scores the outcome, so it recovers the
  *standardized regression coefficient* — `s / sqrt(Σs² + 1)` for
  independent continuous predictors with the engine’s default N(0, 1)
  residual. The shrinkage is the residual variance the standardization
  divides through; it is small for the small/medium effects benchmarked
  here and is the same approximation the shipped OLS recovery has always
  carried.
- **GLM (logistic)** and **MLE (mixed)** fit the **native** outcome (raw
  0/1 for logit, the raw response for mixed), so they recover the
  coefficient **directly**: expected recovery is `s` itself. MLE
  recovery is fixed-effects-only and reads the grouping column from the
  uploaded data.

Scope: **continuous main effects only** — the settled convention.
Factor-dummy and interaction scaling under non-continuous outcomes is a
separate, still-open item and is intentionally not gated here.

# How the check works

1.  **Simulate.** Generate a dataset from each case’s data-generating
    process at n = 4000 (LME scales the cluster count to match), via the
    engine’s own `create_data()` — the same DGP the power simulation
    uses.
2.  **Recover.** Rebuild the raw predictor frame (plus the outcome, plus
    the grouping column for LME), `upload_data()` it, and call
    `get_effects_from_data()`.
3.  **Average.** Repeat over 20 draws (seeds 2137…2156) and take the
    mean — single-draw noise is ~1/sqrt(n), so averaging isolates the
    convention from sampling scatter (the A↔B harness pattern).
4.  **Gate.** The mean recovered effect must land within 0.02 (absolute)
    of the convention-predicted value.

# The threshold

| Quantity | Allowed difference | Why |
|----|----|----|
| Mean recovered − expected | 0.02 absolute | K-draw mean vs the convention-predicted value; worst measured margin 0.007 → ~2.8× headroom |

This is the `GETEFFECTS_TOL` gate from `tolerances.R`. A wrong
estimator, a sign flip, a missing/inverted standardization, or a 2×
scale would move the mean by 0.1–0.5 — far outside this band.

| Case         | Estimator        | Formula                |    n |   K | Verdict |
|:-------------|:-----------------|:-----------------------|-----:|----:|:--------|
| ols_simple_a | OLS (continuous) | y ~ x1                 | 4000 |  20 | PASS    |
| ols_two_a    | OLS (continuous) | y ~ x1 + x2            | 4000 |  20 | PASS    |
| ols_corr_a   | OLS (continuous) | y ~ x1 + x2            | 4000 |  20 | PASS    |
| glm_simple_a | GLM (logistic)   | y ~ x1                 | 4000 |  20 | PASS    |
| glm_two_b    | GLM (logistic)   | y ~ x1 + x2            | 4000 |  20 | PASS    |
| lme_simple_a | MLE (mixed)      | y ~ x1 + (1\|grp)      | 3990 |  20 | PASS    |
| lme_two_a    | MLE (mixed)      | y ~ x1 + x2 + (1\|grp) | 3990 |  20 | PASS    |

> **All estimators round-trip:** every specified continuous main effect
> is recovered to within the gate of the convention-predicted value —
> OLS at the shrunk standardized scale, GLM and MLE at the native
> coefficient.

png 2 ![Mean recovered effect vs the convention-predicted value for
every term, coloured by estimator.](figures/get_effects_roundtrip.png)

*Solid line = exact recovery (identity); dotted lines = the ±0.02
absolute gate. Every term lands on the identity line, well inside the
band.*

## ols_simple_a · OLS (continuous)

R formula `y ~ x1` · n=4000 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |          0.25 |  0.24254 |        0.24393 |  0.0014 | PASS    |

## ols_two_a · OLS (continuous)

R formula `y ~ x1 + x2` · n=4000 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |          0.25 |  0.24140 |        0.24286 | 0.00145 | PASS    |
| x2   |          0.10 |  0.09656 |        0.09490 | 0.00166 | PASS    |

## ols_corr_a · OLS (continuous)

R formula `y ~ x1 + x2` · n=4000 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |          0.25 |  0.23864 |        0.24102 | 0.00238 | PASS    |
| x2   |          0.10 |  0.09545 |        0.09343 | 0.00202 | PASS    |

## glm_simple_a · GLM (logistic)

R formula `y ~ x1` · n=4000 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |           0.5 |      0.5 |        0.48882 | 0.01118 | PASS    |

## glm_two_b · GLM (logistic)

R formula `y ~ x1 + x2` · n=4000 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |           0.8 |      0.8 |        0.79534 | 0.00467 | PASS    |
| x2   |           0.5 |      0.5 |        0.50462 | 0.00462 | PASS    |

## lme_simple_a · MLE (mixed)

R formula `y ~ x1 + (1|grp)` · n=3990 · K=20 draws (seeds 2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |           0.5 |      0.5 |        0.50154 | 0.00154 | PASS    |

## lme_two_a · MLE (mixed)

R formula `y ~ x1 + x2 + (1|grp)` · n=3990 · K=20 draws (seeds
2137–2156).

| Term | Specified (s) | Expected | Mean recovered | \|err\| | Verdict |
|:-----|--------------:|---------:|---------------:|--------:|:--------|
| x1   |           0.5 |      0.5 |        0.50152 | 0.00152 | PASS    |
| x2   |           0.3 |      0.3 |        0.29892 | 0.00107 | PASS    |

| item                       | value                        |
|:---------------------------|:-----------------------------|
| generated                  | 2026-06-21                   |
| R                          | R version 4.5.3 (2026-03-11) |
| mcpower                    | 1.0.0                        |
| round-trip gate (mean abs) | 0.02 abs                     |
| n / K / seed0              | 4000 / 20 / 2137             |

Reproduce:
`rmarkdown::render("mcpower/validation/validation_get_effects.rmd", output_dir = "mcpower/web/documentation/validation")`.
