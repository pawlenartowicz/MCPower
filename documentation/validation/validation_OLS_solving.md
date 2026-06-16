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

## y = 0.25·x1 + noise

R formula `y ~ x1` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.024493 | 0.024493 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.165365 | 0.165365 | 3.335427 | 3.335427 | 1.965942 | 1.965942 | PASS |

## y = 0.40·x1 + noise

R formula `y ~ x1` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | -0.022639 | -0.022639 | NA | NA | NA | NA | PASS |
| x1 | 0.4 | 0.459178 | 0.459178 | 9.266394 | 9.266394 | 1.965942 | 1.965942 | PASS |

## y = 0.25·x1 + 0.10·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.025209 | 0.025209 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.165589 | 0.165589 | 3.335281 | 3.335281 | 1.965957 | 1.965957 | PASS |
| x2 | 0.10 | 0.088433 | 0.088433 | 1.707747 | 1.707747 | 1.965957 | 1.965957 | PASS |

## y = 0.40·x1 + 0.25·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | -0.021662 | -0.021662 | NA | NA | NA | NA | PASS |
| x1 | 0.40 | 0.460753 | 0.460753 | 9.286902 | 9.286902 | 1.965957 | 1.965957 | PASS |
| x2 | 0.25 | 0.289420 | 0.289420 | 5.923721 | 5.923721 | 1.965957 | 1.965957 | PASS |

## y = 0.25·x1 + 0.00·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.025209 | 0.025209 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.165589 | 0.165589 | 3.335281 | 3.335281 | 1.965957 | 1.965957 | PASS |
| x2 | 0.00 | -0.011567 | -0.011567 | 0.223381 | 0.223381 | 1.965957 | 1.965957 | PASS |

## y = 0.25·x1 + 0.10·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.025209 | 0.025209 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.172268 | 0.172268 | 2.946235 | 2.946235 | 1.965957 | 1.965957 | PASS |
| x2 | 0.10 | 0.086643 | 0.086643 | 1.449025 | 1.449025 | 1.965957 | 1.965957 | PASS |

## y = 0.40·x1 + 0.25·x2 + noise

R formula `y ~ x1 + x2` · ordinary least squares (continuous outcome) ·
n=400, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | -0.021662 | -0.021662 | NA | NA | NA | NA | PASS |
| x1 | 0.40 | 0.448355 | 0.448355 | 8.730127 | 8.730127 | 1.965957 | 1.965957 | PASS |
| x2 | 0.25 | 0.291324 | 0.291324 | 5.688034 | 5.688034 | 1.965957 | 1.965957 | PASS |

## y = 0.25·x1 + 0.10·x2 + -0.20·x1:x2 + noise

R formula `y ~ x1*x2` · ordinary least squares (continuous outcome) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.034747 | 0.034747 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.150662 | 0.150662 | 3.670234 | 3.670234 | 1.963952 | 1.963952 | PASS |
| x2 | 0.10 | 0.150648 | 0.150648 | 3.495398 | 3.495398 | 1.963952 | 1.963952 | PASS |
| x1:x2 | -0.20 | -0.255044 | -0.255044 | 6.163746 | 6.163746 | 1.963952 | 1.963952 | PASS |

## y = 0.40·x1 + 0.25·x2 + 0.15·x1:x2 + noise

R formula `y ~ x1*x2` · ordinary least squares (continuous outcome) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.021636 | 0.021636 | NA | NA | NA | NA | PASS |
| x1 | 0.40 | 0.449526 | 0.449526 | 11.112832 | 11.112832 | 1.963952 | 1.963952 | PASS |
| x2 | 0.25 | 0.270807 | 0.270807 | 6.775460 | 6.775460 | 1.963952 | 1.963952 | PASS |
| x1:x2 | 0.15 | 0.231824 | 0.231824 | 5.912351 | 5.912351 | 1.963952 | 1.963952 | PASS |

## y = 0.25·x1 + 0.50·g\[2\] + 0.80·g\[3\] + noise

R formula `y ~ x1 + g` · ordinary least squares (continuous outcome) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.020076 | 0.020076 | NA | NA | NA | NA | PASS |
| x1 | 0.25 | 0.162527 | 0.162527 | 4.178903 | 4.178903 | 1.963952 | 1.963952 | PASS |
| g\[2\] | 0.50 | 0.484884 | 0.484884 | 5.205693 | 5.205693 | 1.963952 | 1.963952 | PASS |
| g\[3\] | 0.80 | 0.816131 | 0.816131 | 7.643534 | 7.643534 | 1.963952 | 1.963952 | PASS |

## y = 0.40·x1 + 0.20·g\[2\] + 0.50·g\[3\] + noise

R formula `y ~ x1 + g` · ordinary least squares (continuous outcome) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.030843 | 0.030843 | NA | NA | NA | NA | PASS |
| x1 | 0.4 | 0.452522 | 0.452522 | 11.212639 | 11.212639 | 1.963952 | 1.963952 | PASS |
| g\[2\] | 0.2 | 0.118823 | 0.118823 | 1.258325 | 1.258325 | 1.963952 | 1.963952 | PASS |
| g\[3\] | 0.5 | 0.541393 | 0.541393 | 5.207824 | 5.207824 | 1.963952 | 1.963952 | PASS |

## y = 0.30·x1 + 0.40·g\[2\] + 0.60·g\[3\] + 0.20·x1:g\[2\] + 0.30·x1:g\[3\] + noise

R formula `y ~ x1*g` · ordinary least squares (continuous outcome) ·
n=800, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.009292 | 0.009292 | NA | NA | NA | NA | PASS |
| x1 | 0.3 | 0.213214 | 0.213214 | 4.502722 | 4.502722 | 1.962956 | 1.962956 | PASS |
| g\[2\] | 0.4 | 0.426134 | 0.426134 | 5.291139 | 5.291139 | 1.962956 | 1.962956 | PASS |
| g\[3\] | 0.6 | 0.676633 | 0.676633 | 7.366583 | 7.366583 | 1.962956 | 1.962956 | PASS |
| x1:g\[2\] | 0.2 | 0.236672 | 0.236672 | 3.075062 | 3.075062 | 1.962956 | 1.962956 | PASS |
| x1:g\[3\] | 0.3 | 0.374305 | 0.374305 | 4.303693 | 4.303693 | 1.962956 | 1.962956 | PASS |

## y = 0.40·x1 + 0.50·g\[2\] + 0.80·g\[3\] + 0.25·x1:g\[2\] + 0.40·x1:g\[3\] + noise

R formula `y ~ x1*g` · ordinary least squares (continuous outcome) ·
n=800, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.00 | 0.043679 | 0.043679 | NA | NA | NA | NA | PASS |
| x1 | 0.40 | 0.454781 | 0.454781 | 7.990442 | 7.990442 | 1.962956 | 1.962956 | PASS |
| g\[2\] | 0.50 | 0.408558 | 0.408558 | 5.062804 | 5.062804 | 1.962956 | 1.962956 | PASS |
| g\[3\] | 0.80 | 0.781313 | 0.781313 | 8.789444 | 8.789444 | 1.962956 | 1.962956 | PASS |
| x1:g\[2\] | 0.25 | 0.257545 | 0.257545 | 3.204705 | 3.204705 | 1.962956 | 1.962956 | PASS |
| x1:g\[3\] | 0.40 | 0.351679 | 0.351679 | 3.995772 | 3.995772 | 1.962956 | 1.962956 | PASS |

## y = 0.50·g1 + 0.40·g2\[2\] + 0.30·g1:g2\[2\] + noise

R formula `y ~ g1*g2` · ordinary least squares (continuous outcome) ·
n=800, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.065392 | 0.065392 | NA | NA | NA | NA | PASS |
| g1 | 0.5 | 0.467398 | 0.467398 | 11.061554 | 11.061554 | 1.962949 | 1.962949 | PASS |
| g2\[2\] | 0.4 | 0.315998 | 0.315998 | 4.467255 | 4.467255 | 1.962949 | 1.962949 | PASS |
| g1:g2\[2\] | 0.3 | 0.227428 | 0.227428 | 3.344119 | 3.344119 | 1.962949 | 1.962949 | PASS |

## y = 0.20·g1 + 0.80·g2\[2\] + 0.50·g1:g2\[2\] + noise

R formula `y ~ g1*g2` · ordinary least squares (continuous outcome) ·
n=800, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.080018 | 0.080018 | NA | NA | NA | NA | PASS |
| g1 | 0.2 | 0.261406 | 0.261406 | 5.545823 | 5.545823 | 1.962949 | 1.962949 | PASS |
| g2\[2\] | 0.8 | 0.636686 | 0.636686 | 9.115533 | 9.115533 | 1.962949 | 1.962949 | PASS |
| g1:g2\[2\] | 0.5 | 0.458038 | 0.458038 | 6.652102 | 6.652102 | 1.962949 | 1.962949 | PASS |

| item | value |
|:---|:---|
| generated | 2026-06-14 |
| R | R version 4.5.3 (2026-03-11) |
| mcpower | 0.0.0.9000 |
| OLS solving bands (beta/stat rel · crit abs) | 1e-11 rel / 1e-11 rel / 1e-09 abs |

Reproduce: `Rscript mcpower/validation/data_generation.r` then
`rmarkdown::render("mcpower/validation/validation_OLS_solving.rmd", output_dir = "mcpower/documentation/validation")`.
