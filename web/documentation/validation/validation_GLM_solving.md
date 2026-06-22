---
title: "MCPower — Validating the GLM (Logistic) Solver"
description: "MCPower GLM solver validation: byte-for-byte comparison of logistic IRLS coefficients, Wald z-statistics, and critical values against R stats::glm."
right_sidebar: body
---
# MCPower validation — GLM solver (data → results)

# What this report shows

MCPower generates a binary dataset from a formula, then **fits it back**
to recover the coefficients. This report checks the *fitting* half:
given one fixed dataset, does MCPower’s own solver return the same
numbers a trusted, independent R fit returns?

For each formula we take the exact bytes MCPower saved (the same design
matrix and 0/1 outcome, byte-for-byte), fit them two ways —

- **R (the reference):** `stats::glm(family = binomial())` through the
  origin on the design matrix, the textbook maximum-likelihood logistic
  fit.
- **MCPower:** the engine’s own `load_data()` path, which runs the
  *same* iterative IRLS solver the power simulation uses.

— and compare the fitted coefficients, the test statistics, and the
decision threshold. Because both fit the identical bytes, sampling noise
cancels: any difference is a solver discrepancy, and the allowed
differences are small.

# How the check works

1.  **Provenance.** Re-generate the dataset from the committed seed and
    confirm its content hash matches the saved file — proof the bytes
    haven’t drifted.
2.  **B (R) vs C (MCPower)** on those bytes: coefficients, the Wald *z*
    statistics (β̂/se), and the critical value `qnorm(1 − α/2)` ≈ 1.96.
3.  **C vs A** (a readable sanity overlay): MCPower’s recovered
    coefficients next to the formula’s true values (on the log-odds
    scale).

# The thresholds

| Quantity | Allowed difference | Why |
|----|----|----|
| Coefficient (β) | 10^{-4} relative | iterative IRLS fit — agrees with R well inside this band |
| Statistic (*z*) | 10^{-4} relative | derived from the same β and variance |
| Critical value | 10^{-8} absolute | engine’s own normal/χ² quantile vs R’s `qnorm` ≈1.6e-9 |

These are the B↔C gates from `tolerances.R`. GLM is an **iterative
IRLS** fit, so its agreement with R is looser than OLS’s closed-form
solution: it gets the `estimate_rel_iter` band (typically ~1e-6 to 1e-9
for well-converged IRLS), not the tight OLS `estimate_rel_ols`. The
per-formula tables below show each formula’s actual margin.

| Formula     |    n | B↔C  | Converged | Reproduces |
|:------------|-----:|:-----|:----------|:-----------|
| y ~ x1      |  600 | PASS | yes       | yes        |
| y ~ x1      |  600 | PASS | yes       | yes        |
| y ~ x1 + x2 |  800 | PASS | yes       | yes        |
| y ~ x1 + x2 |  800 | PASS | yes       | yes        |
| y ~ x1 + g  | 1000 | PASS | yes       | yes        |
| y ~ x1 + g  | 1000 | PASS | yes       | yes        |
| y ~ x1\*x2  | 1000 | PASS | yes       | yes        |
| y ~ x1\*x2  | 1000 | PASS | yes       | yes        |

> **All GLM formulas pass:** MCPower’s IRLS solver matches R’s `glm`
> well inside the iterative band on every saved dataset, and every file
> reproduces from its seed.

png 2 ![B↔C relative agreement of every coefficient and statistic across
all GLM formulas, against the tolerance
gate.](figures/glm_agreement.png)

*Each point is one fitted quantity on one saved dataset; the red line is
the gate. The iterative IRLS fit agrees with R’s `glm` comfortably
inside the band on every quantity.*

## log-odds(y = 1) = logit(0.30) + 0.5·x1

R formula `y ~ x1` · logistic regression (binary outcome) · n=600,
seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | -0.847298 | -0.821395 | NA | NA | 8.1e-16 | — | PASS |
| x1 | 0.500000 | 0.706388 | 7.1761 | 1.959964 | 1.7e-08 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.50) + 0.8·x1

R formula `y ~ x1` · logistic regression (binary outcome) · n=600,
seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.013884 | NA | NA | 9.3e-12 | — | PASS |
| x1 | 0.8 | 0.710581 | 7.496297 | 1.959964 | 1.1e-06 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.30) + 0.5·x1 + 0.3·x2

R formula `y ~ x1 + x2` · logistic regression (binary outcome) · n=800,
seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | -0.847298 | -0.919551 | NA | NA | 1.4e-13 | — | PASS |
| x1 | 0.500000 | 0.710861 | 8.020069 | 1.959964 | 2.5e-07 | 1.6e-09 | PASS |
| x2 | 0.300000 | 0.213309 | 2.510503 | 1.959964 | 1.1e-07 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.50) + 0.8·x1 + 0.5·x2

R formula `y ~ x1 + x2` · logistic regression (binary outcome) · n=800,
seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.008705 | NA | NA | 1.8e-13 | — | PASS |
| x1 | 0.8 | 0.651796 | 7.961513 | 1.959964 | 1.2e-07 | 1.6e-09 | PASS |
| x2 | 0.5 | 0.539642 | 6.718397 | 1.959964 | 1.1e-07 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.30) + 0.5·x1 + 0.4·g\[2\] + 0.8·g\[3\]

R formula `y ~ x1 + g` · logistic regression (binary outcome) · n=1000,
seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | -0.847298 | -0.788464 | NA | NA | 4.1e-15 | — | PASS |
| x1 | 0.500000 | 0.650709 | 8.945066 | 1.959964 | 3.1e-08 | 1.6e-09 | PASS |
| g\[2\] | 0.400000 | 0.273404 | 1.705934 | 1.959964 | 1.5e-08 | 1.6e-09 | PASS |
| g\[3\] | 0.800000 | 0.517325 | 2.872484 | 1.959964 | 1.3e-08 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.50) + 0.8·x1 + 0.5·g\[2\] + 0.8·g\[3\]

R formula `y ~ x1 + g` · logistic regression (binary outcome) · n=1000,
seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.052089 | NA | NA | 1.5e-14 | — | PASS |
| x1 | 0.8 | 0.686227 | 9.293969 | 1.959964 | 7.6e-08 | 1.6e-09 | PASS |
| g\[2\] | 0.5 | 0.665670 | 4.224519 | 1.959964 | 3.7e-08 | 1.6e-09 | PASS |
| g\[3\] | 0.8 | 0.846361 | 4.776921 | 1.959964 | 4.3e-08 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.30) + 0.5·x1 + 0.3·x2 + 0.3·x1:x2

R formula `y ~ x1*x2` · logistic regression (binary outcome) · n=1000,
seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | -0.847298 | -0.858136 | NA | NA | 6.7e-10 | — | PASS |
| x1 | 0.500000 | 0.661026 | 8.333015 | 1.959964 | 1.8e-05 | 1.6e-09 | PASS |
| x2 | 0.300000 | 0.183059 | 2.360110 | 1.959964 | 9.2e-06 | 1.6e-09 | PASS |
| x1:x2 | 0.300000 | 0.487888 | 5.586386 | 1.959964 | 2.0e-05 | 1.6e-09 | PASS |

## log-odds(y = 1) = logit(0.50) + 0.8·x1 + 0.5·x2 + 0.4·x1:x2

R formula `y ~ x1*x2` · logistic regression (binary outcome) · n=1000,
seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.044399 | NA | NA | 1.0e-09 | — | PASS |
| x1 | 0.8 | 0.775735 | 9.832031 | 1.959964 | 5.3e-06 | 1.6e-09 | PASS |
| x2 | 0.5 | 0.624758 | 8.093964 | 1.959964 | 4.8e-06 | 1.6e-09 | PASS |
| x1:x2 | 0.4 | 0.428403 | 5.308531 | 1.959964 | 6.9e-06 | 1.6e-09 | PASS |

| item | value |
|:---|:---|
| generated | 2026-06-21 |
| R | R version 4.5.3 (2026-03-11) |
| mcpower | 1.0.0 |
| GLM solving bands (beta/stat rel · crit abs) | 0.0001 rel / 0.0001 rel / 1e-08 abs |

Reproduce: `Rscript mcpower/validation/data_generation.r` then
`rmarkdown::render("mcpower/validation/validation_GLM_solving.rmd", output_dir = "mcpower/web/documentation/validation")`.
