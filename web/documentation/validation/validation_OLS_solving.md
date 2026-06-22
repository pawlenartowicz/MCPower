---
title: "MCPower — Validating the OLS Solver"
description: "MCPower OLS solver validation: byte-for-byte comparison of fitted coefficients, t-statistics, and critical values against R stats::lm."
right_sidebar: body
---
# MCPower validation — OLS solver (data → results)

# What this report shows

MCPower generates a dataset from a formula, then **fits it back** to
recover the coefficients. This report checks the *fitting* half: given
one fixed dataset, does MCPower’s own solver return the same numbers a
trusted, independent R fit returns?

For each formula we take the exact bytes MCPower saved (the same design
matrix and outcome, byte-for-byte), fit them two ways —

- **R (the reference):** `stats::lm` through the origin on the design
  matrix, the textbook least-squares solution.
- **MCPower:** the engine’s own `load_data()` path, which runs the
  *same* solver the power simulation uses.

— and compare the fitted coefficients, the test statistics, and the
decision threshold. Because both fit the identical bytes, sampling noise
cancels: any difference is a solver discrepancy, and the allowed
differences are tiny.

# How the check works

1.  **Provenance.** Re-generate the dataset from the committed seed and
    confirm its content hash matches the saved file — proof the bytes
    haven’t drifted.
2.  **B (R) vs C (MCPower)** on those bytes: coefficients, the marginal
    *t* statistics, and the critical value.
3.  **C vs A** (a readable sanity overlay): MCPower’s recovered
    coefficients next to the formula’s true values.

# The thresholds

| Quantity | Allowed difference | Why |
|----|----|----|
| Coefficient (β) | 10^{-11} relative | exact normal equations; observed worst agreement ≈1.2e-12 |
| Statistic (*t*) | 10^{-11} relative | derived from the same β and variance |
| Critical value | 10^{-9} absolute | `qt` quantile agreement ~1e-15 |

These are the B↔C gates from `tolerances.R`. OLS is closed-form, so it
gets the **tight** `estimate_rel_ols` band — set just above the measured
worst agreement (≈1.2e-12) so a real solver regression would trip it.
The iterative GLM/MLE fits use the looser `estimate_rel_iter` band in
their own reports; the per-formula tables below show OLS’s actual
margin.

| Formula     |   n | B↔C  | Converged | Reproduces |
|:------------|----:|:-----|:----------|:-----------|
| y ~ x1      | 400 | PASS | yes       | yes        |
| y ~ x1      | 400 | PASS | yes       | yes        |
| y ~ x1 + x2 | 400 | PASS | yes       | yes        |
| y ~ x1 + x2 | 400 | PASS | yes       | yes        |
| y ~ x1 + x2 | 400 | PASS | yes       | yes        |
| y ~ x1 + x2 | 400 | PASS | yes       | yes        |
| y ~ x1 + x2 | 400 | PASS | yes       | yes        |
| y ~ x1\*x2  | 600 | PASS | yes       | yes        |
| y ~ x1\*x2  | 600 | PASS | yes       | yes        |
| y ~ x1 + g  | 600 | PASS | yes       | yes        |
| y ~ x1 + g  | 600 | PASS | yes       | yes        |
| y ~ x1\*g   | 800 | PASS | yes       | yes        |
| y ~ x1\*g   | 800 | PASS | yes       | yes        |
| y ~ g1\*g2  | 800 | PASS | yes       | yes        |
| y ~ g1\*g2  | 800 | PASS | yes       | yes        |

> **All OLS formulas pass:** MCPower’s solver matches R to machine
> precision on every saved dataset, and every file reproduces from its
> seed.

png 2 ![B↔C relative agreement of every coefficient and statistic across
all OLS formulas, against the tolerance
gate.](figures/ols_agreement.png)

*Each point is one fitted quantity on one saved dataset; the red line is
the gate. Every β and statistic agrees with R orders of magnitude inside
tolerance — exact (bit-identical) matches pile up at the left floor.*

## y = 0.25·x1 + noise

R formula `y ~ x1` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.024493 | NA | NA | 5.7e-16 | — | PASS |
| x1 | 0.25 | 0.165365 | 3.335427 | 1.965942 | 1.1e-15 | 9.2e-14 | PASS |

## y = 0.40·x1 + noise

R formula `y ~ x1` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.022639 | NA | NA | 1.1e-15 | — | PASS |
| x1 | 0.4 | 0.459178 | 9.266394 | 1.965942 | 1.2e-15 | 9.2e-14 | PASS |

## y = 0.25·x1 + 0.10·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.025209 | NA | NA | 1.4e-16 | — | PASS |
| x1 | 0.25 | 0.165589 | 3.335281 | 1.965957 | 1.3e-15 | 1.9e-14 | PASS |
| x2 | 0.10 | 0.088433 | 1.707747 | 1.965957 | 9.1e-16 | 1.9e-14 | PASS |

## y = 0.40·x1 + 0.25·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | -0.021662 | NA | NA | 3.2e-16 | — | PASS |
| x1 | 0.40 | 0.460753 | 9.286902 | 1.965957 | 1.1e-15 | 1.9e-14 | PASS |
| x2 | 0.25 | 0.289420 | 5.923721 | 1.965957 | 1.3e-15 | 1.9e-14 | PASS |

## y = 0.25·x1 + 0.00·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.025209 | NA | NA | 2.8e-16 | — | PASS |
| x1 | 0.25 | 0.165589 | 3.335281 | 1.965957 | 1.1e-15 | 1.9e-14 | PASS |
| x2 | 0.00 | -0.011567 | 0.223381 | 1.965957 | 2.5e-15 | 1.9e-14 | PASS |

## y = 0.25·x1 + 0.10·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.025209 | NA | NA | 5.5e-16 | — | PASS |
| x1 | 0.25 | 0.172268 | 2.946235 | 1.965957 | 6.4e-16 | 1.9e-14 | PASS |
| x2 | 0.10 | 0.086643 | 1.449025 | 1.965957 | 1.5e-15 | 1.9e-14 | PASS |

## y = 0.40·x1 + 0.25·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | -0.021662 | NA | NA | 1.6e-16 | — | PASS |
| x1 | 0.40 | 0.448355 | 8.730127 | 1.965957 | 8.1e-16 | 1.9e-14 | PASS |
| x2 | 0.25 | 0.291324 | 5.688034 | 1.965957 | 1.6e-15 | 1.9e-14 | PASS |

## y = 0.25·x1 + 0.10·x2 + -0.20·x1:x2 + noise

R formula `y ~ x1*x2` · ordinary least squares (continuous outcome) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.034747 | NA | NA | 6.0e-16 | — | PASS |
| x1 | 0.25 | 0.150662 | 3.670234 | 1.963952 | 1.7e-15 | 1.2e-13 | PASS |
| x2 | 0.10 | 0.150648 | 3.495398 | 1.963952 | 1.9e-15 | 1.2e-13 | PASS |
| x1:x2 | -0.20 | -0.255044 | 6.163746 | 1.963952 | 1.4e-15 | 1.2e-13 | PASS |

## y = 0.40·x1 + 0.25·x2 + 0.15·x1:x2 + noise

R formula `y ~ x1*x2` · ordinary least squares (continuous outcome) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.021636 | NA | NA | 1.6e-16 | — | PASS |
| x1 | 0.40 | 0.449526 | 11.112832 | 1.963952 | 1.6e-15 | 1.2e-13 | PASS |
| x2 | 0.25 | 0.270807 | 6.775460 | 1.963952 | 2.0e-15 | 1.2e-13 | PASS |
| x1:x2 | 0.15 | 0.231824 | 5.912351 | 1.963952 | 1.7e-15 | 1.2e-13 | PASS |

## y = 0.25·x1 + 0.50·g\[2\] + 0.80·g\[3\] + noise

R formula `y ~ x1 + g` · ordinary least squares (continuous outcome) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.020076 | NA | NA | 1.8e-14 | — | PASS |
| x1 | 0.25 | 0.162527 | 4.178903 | 1.963952 | 1.9e-15 | 1.2e-13 | PASS |
| g\[2\] | 0.50 | 0.484884 | 5.205693 | 1.963952 | 1.9e-15 | 1.2e-13 | PASS |
| g\[3\] | 0.80 | 0.816131 | 7.643534 | 1.963952 | 1.9e-15 | 1.2e-13 | PASS |

## y = 0.40·x1 + 0.20·g\[2\] + 0.50·g\[3\] + noise

R formula `y ~ x1 + g` · ordinary least squares (continuous outcome) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.030843 | NA | NA | 6.9e-15 | — | PASS |
| x1 | 0.4 | 0.452522 | 11.212639 | 1.963952 | 1.1e-15 | 1.2e-13 | PASS |
| g\[2\] | 0.2 | 0.118823 | 1.258325 | 1.963952 | 1.6e-15 | 1.2e-13 | PASS |
| g\[3\] | 0.5 | 0.541393 | 5.207824 | 1.963952 | 1.7e-15 | 1.2e-13 | PASS |

## y = 0.30·x1 + 0.40·g\[2\] + 0.60·g\[3\] + 0.20·x1:g\[2\] + 0.30·x1:g\[3\] + noise

R formula `y ~ x1*g` · ordinary least squares (continuous outcome) ·
n=800, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.009292 | NA | NA | 2.1e-15 | — | PASS |
| x1 | 0.3 | 0.213214 | 4.502722 | 1.962956 | 2.6e-15 | 5.4e-14 | PASS |
| g\[2\] | 0.4 | 0.426134 | 5.291139 | 1.962956 | 5.0e-16 | 5.4e-14 | PASS |
| g\[3\] | 0.6 | 0.676633 | 7.366583 | 1.962956 | 9.6e-16 | 5.4e-14 | PASS |
| x1:g\[2\] | 0.2 | 0.236672 | 3.075062 | 1.962956 | 1.4e-15 | 5.4e-14 | PASS |
| x1:g\[3\] | 0.3 | 0.374305 | 4.303693 | 1.962956 | 1.0e-15 | 5.4e-14 | PASS |

## y = 0.40·x1 + 0.50·g\[2\] + 0.80·g\[3\] + 0.25·x1:g\[2\] + 0.40·x1:g\[3\] + noise

R formula `y ~ x1*g` · ordinary least squares (continuous outcome) ·
n=800, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.00 | 0.043679 | NA | NA | 3.3e-15 | — | PASS |
| x1 | 0.40 | 0.454781 | 7.990442 | 1.962956 | 2.2e-15 | 5.4e-14 | PASS |
| g\[2\] | 0.50 | 0.408558 | 5.062804 | 1.962956 | 2.7e-16 | 5.4e-14 | PASS |
| g\[3\] | 0.80 | 0.781313 | 8.789444 | 1.962956 | 4.3e-16 | 5.4e-14 | PASS |
| x1:g\[2\] | 0.25 | 0.257545 | 3.204705 | 1.962956 | 3.0e-15 | 5.4e-14 | PASS |
| x1:g\[3\] | 0.40 | 0.351679 | 3.995772 | 1.962956 | 1.7e-15 | 5.4e-14 | PASS |

## y = 0.50·g1\[2\] + 0.40·g2\[2\] + 0.30·g1\[2\]:g2\[2\] + noise

R formula `y ~ g1*g2` · ordinary least squares (continuous outcome) ·
n=800, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.014224 | NA | NA | 5.9e-15 | — | PASS |
| g1\[2\] | 0.5 | 0.662579 | 7.406768 | 1.962949 | 6.0e-16 | 3.4e-14 | PASS |
| g2\[2\] | 0.4 | 0.433519 | 4.334558 | 1.962949 | 8.2e-16 | 3.4e-14 | PASS |
| g1\[2\]:g2\[2\] | 0.3 | 0.067685 | 0.478532 | 1.962949 | 1.4e-15 | 3.4e-14 | PASS |

## y = 0.20·g1\[2\] + 0.80·g2\[2\] + 0.50·g1\[2\]:g2\[2\] + noise

R formula `y ~ g1*g2` · ordinary least squares (continuous outcome) ·
n=800, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.088768 | NA | NA | 3.3e-15 | — | PASS |
| g1\[2\] | 0.2 | 0.186150 | 1.977150 | 1.962949 | 3.9e-15 | 3.4e-14 | PASS |
| g2\[2\] | 0.8 | 0.614136 | 6.245221 | 1.962949 | 1.8e-15 | 3.4e-14 | PASS |
| g1\[2\]:g2\[2\] | 0.5 | 0.543064 | 3.864930 | 1.962949 | 3.0e-15 | 3.4e-14 | PASS |

| item | value |
|:---|:---|
| generated | 2026-06-21 |
| R | R version 4.5.3 (2026-03-11) |
| mcpower | 1.0.0 |
| OLS solving bands (beta/stat rel · crit abs) | 1e-11 rel / 1e-11 rel / 1e-09 abs |

Reproduce: `Rscript mcpower/validation/data_generation.r` then
`rmarkdown::render("mcpower/validation/validation_OLS_solving.rmd", output_dir = "mcpower/web/documentation/validation")`.
