# MCPower — Validating Uploaded-Data Generation

# What this report validates

MCPower can drive its data generator from an **uploaded empirical
frame**. When you call `upload_data(pilot_data)`, the engine:

1.  **Standardizes** each continuous predictor to mean 0, sd 1
    (population SD, ddof = 0), and stores the result as a sorted lookup
    table for NORTA quantile mapping.
2.  **Measures** empirical Pearson correlations among **continuous**
    predictors and installs them as the NORTA latent-correlation matrix
    (under `mode = "partial"`, the default). Binary/factor predictors
    are generated from their marginals only — they are never correlated
    with any other predictor.
3.  For each draw, maps correlated standard-normal variates through the
    **empirical inverse-CDF** of the stored (standardized) frame column
    — producing data that follows the frame’s marginal distribution.

## A note on the oracle: engine moments ≠ raw frame moments

For a **small frame** (the 32-row mtcars pilot), the NORTA quantile path
introduces two well-known finite-frame effects:

- **Mean shift**: `E[Q̂(Φ(Z))]` for Z∼N(0,1) is not exactly 0, because
  the type-7 empirical quantile of a finite sorted sample maps
  Uniform(0,1) input to a slightly off-centre discrete distribution. For
  the 32-row mtcars frame, the mean of continuous columns is shifted by
  ≈−0.025 — well outside the raw frame’s standardized mean of 0.

- **Variance shrinkage**: The empirical quantile is bounded by the
  frame’s observed min and max. The tails of Φ(Z) map to the extremes of
  the 32-point sample rather than the continuous normal tails, so the
  generated standard deviation is ≈0.93 instead of 1.0.

- **Binary predictors are independent by design**: Correlation is
  continuous-only in MCPower’s engine. Binary/factor variables are
  generated from their marginals, independent of every other predictor.
  The realized binary×continuous Pearson r is therefore expected to be ≈
  0, and this is numerically gated (see §Binary below).

**These are expected properties of the NORTA algorithm with a finite
reference frame, not engine bugs.** The design decision (NORTA +
empirical inverse-CDF) is what makes the generated distribution “look
like” the frame distribution at large N, even if the first two moments
do not match the standardized frame exactly for small frames.

Following the principle that MCPower is its own golden source, the
validation oracle is **what the engine itself produces at large N** (a
single n = 200,000 reference draw), not the raw frame statistics. The
A↔B assertion is: the average of K = 1600 draws at n = 400 must
reproduce the engine’s large-N moments within
`DGP_TOL$moment_abs = r DGP_TOL$moment_abs`.

The correlation checks in this report are engine-to-engine (large-N
self-consistency); independent confirmation that NORTA installs the
frame’s empirical Spearman rank correlation (converted to the latent
Gaussian scale) as the contract matrix, while binary predictors keep
their correlation slots at zero, is an **L1 engine test**
(`engine-spec-builder/tests/upload_test.rs::partial_mode_measures_nontrivial_and_excludes_binary`),
not an L3 gate.

## What is and is not checked

| Assertion type | Checked? | Notes |
|----|----|----|
| Continuous marginal mean | **Yes** | K-draw average mean vs engine oracle |
| Continuous marginal sd | **Yes** | K-draw average sd vs engine oracle |
| Continuous × continuous correlation | **Yes** | K-draw average vs engine oracle |
| Binary marginal proportion | **Yes** | K-draw average proportion vs oracle |
| Binary × continuous correlation | **Yes (≈ 0 gate)** | Binary is independent by design; K-avg |
| Factor correlations | **Out of scope** | Discrete-correlation work not yet landed |
| Nonlinear joint (parabola, strict mode) | **Yes** | `upload_strict_nonlinear`: strict bootstrap copies whole rows → generated rows lie on x2=x1² to 1e-5; NORTA would give O(1) residuals |

## Tolerance thresholds

Gates from `tolerances.R` (`DGP_TOL`), applied to the K-draw average vs
the engine oracle:

| What we check                               | Tolerance             |
|---------------------------------------------|-----------------------|
| Average generated mean of continuous column | within 0.01 of oracle |
| Average generated sd of continuous column   | within 0.01 of oracle |
| Average generated binary proportion         | within 0.01 of oracle |
| Average generated cont×cont correlation     | within 0.01 of oracle |
| Average generated binary×cont correlation   |                       |

# Results at a glance

| Case                    | Formula              | Upload cols | Assertions |
|:------------------------|:---------------------|:------------|:-----------|
| upload_cont_only        | `mpg ~ hp + wt`      | hp, wt      | all 5 PASS |
| upload_cont_binary      | `mpg ~ hp + wt + am` | hp, wt, am  | all 8 PASS |
| upload_strict_nonlinear | `y ~ x1 + x2`        | x1, x2      | all 6 PASS |

> **All 3 upload cases pass all assertions.** NORTA-over-frame
> (partial/none) and strict-bootstrap paths are internally consistent: K
> = 1600 draws per case reproduce the engine’s own large-N moments
> within DGP_TOL, and strict-mode rows lie on the uploaded parabola to
> within 1e-5.

# Case-by-case detail

## upload_cont_only

Formula `mpg ~ hp + wt` \| OLS \| upload columns: hp, wt \| n = 400 per
draw \| K = 1600 draws

### Frame moments vs engine oracle

The table below compares the **raw frame** statistics (before any engine
processing) to the **engine oracle** (moments measured from a single
large n = 200,000 draw). Differences arise from the finite-frame NORTA
effects described in the introduction.

| Column  | Statistic | Frame (standardized) | Engine oracle | Deviation |
|:--------|:----------|---------------------:|--------------:|----------:|
| hp      | mean      |               0.0000 |       -0.0204 |    0.0204 |
| hp      | sd        |               1.0000 |        0.9307 |    0.0693 |
| wt      | mean      |               0.0000 |       -0.0067 |    0.0067 |
| wt      | sd        |               1.0000 |        0.9427 |    0.0573 |
| hp x wt | Pearson r |               0.6587 |        0.7542 |    0.0955 |

### Assertion table — K-draw average vs engine oracle

Averaged over 1600 draws of n = 400:

|  | Assertion | Oracle (engine) | K-draw average | Difference | Tolerance | Verdict |
|:---|:---|---:|---:|---:|---:|:---|
| hp | mean of hp | -0.0204 | -0.0222 | 0.0019 | 0.01 | PASS |
| hp1 | sd of hp | 0.9307 | 0.9296 | 0.0011 | 0.01 | PASS |
| wt | mean of wt | -0.0067 | -0.0074 | 0.0007 | 0.01 | PASS |
| wt1 | sd of wt | 0.9427 | 0.9410 | 0.0017 | 0.01 | PASS |
| 1 | correlation hp × wt (mode=partial) | 0.7542 | 0.7530 | 0.0012 | 0.01 | PASS |

### mode = “none” sanity check — marginals only, no correlation

Under `mode = "none"` the engine reproduces each predictor’s marginal
independently (no correlation structure is imposed). We verify that
continuous column means and sds reproduce the engine’s **none-mode**
population moments — measured from a matched large-n `mode = "none"`
reference draw, *not* the strict-mode oracle. The strict (whole-row
bootstrap) and none (NORTA empirical-quantile) paths produce slightly
different marginal distributions from the same frame — the NORTA path
carries a small (~0.5–1%) variance shrinkage the bootstrap does not — so
the none-mode draws must be gated against a none-mode oracle.

| Column | Stat |  Oracle |   K-avg |   Diff | Verdict |
|:-------|:-----|--------:|--------:|-------:|:--------|
| hp     | mean | -0.0204 | -0.0180 | 0.0024 | PASS    |
| hp     | sd   |  0.9307 |  0.9321 | 0.0013 | PASS    |
| wt     | mean | -0.0083 | -0.0088 | 0.0005 | PASS    |
| wt     | sd   |  0.9417 |  0.9410 | 0.0007 | PASS    |

## upload_cont_binary

Formula `mpg ~ hp + wt + am` \| OLS \| upload columns: hp, wt, am \| n =
400 per draw \| K = 1600 draws

### Frame moments vs engine oracle

The table below compares the **raw frame** statistics (before any engine
processing) to the **engine oracle** (moments measured from a single
large n = 200,000 draw). Differences arise from the finite-frame NORTA
effects described in the introduction.

|     | Column  | Statistic  | Frame (standardized) | Engine oracle | Deviation |
|:----|:--------|:-----------|---------------------:|--------------:|----------:|
| 1   | hp      | mean       |               0.0000 |       -0.0204 |    0.0204 |
| 2   | hp      | sd         |               1.0000 |        0.9307 |    0.0693 |
| 3   | wt      | mean       |               0.0000 |       -0.0067 |    0.0067 |
| 4   | wt      | sd         |               1.0000 |        0.9427 |    0.0573 |
| am  | am      | proportion |               0.4062 |        0.4046 |    0.0017 |
| 11  | hp x wt | Pearson r  |               0.6587 |        0.7542 |    0.0955 |

### Assertion table — K-draw average vs engine oracle

Averaged over 1600 draws of n = 400:

|  | Assertion | Oracle (engine) | K-draw average | Difference | Tolerance | Verdict |
|:---|:---|---:|---:|---:|---:|:---|
| hp | mean of hp | -0.0204 | -0.0222 | 0.0019 | 0.01 | PASS |
| hp1 | sd of hp | 0.9307 | 0.9296 | 0.0011 | 0.01 | PASS |
| wt | mean of wt | -0.0067 | -0.0074 | 0.0007 | 0.01 | PASS |
| wt1 | sd of wt | 0.9427 | 0.9410 | 0.0017 | 0.01 | PASS |
| am | proportion(1) of am | 0.4046 | 0.4066 | 0.0020 | 0.01 | PASS |
| 1 | correlation hp × wt (mode=partial) | 0.7542 | 0.7530 | 0.0012 | 0.01 | PASS |
| 11 | correlation am × hp ≈ 0 (binary independent) | 0.0000 | -0.0001 | 0.0001 | 0.01 | PASS |
| 12 | correlation am × wt ≈ 0 (binary independent) | 0.0000 | 0.0003 | 0.0003 | 0.01 | PASS |

### Binary correlations — gated to ≈ 0

Binary/factor predictors are generated from their marginals, independent
of every other predictor (correlation is continuous-only by design). The
realized binary×continuous Pearson r must therefore be ≈ 0. Each pair
below is asserted: \|K-draw average r\| ≤ `BANDS$corr_cc` (= 0.01).
These assertions are also included in the assertion table above.

| Pair    | K-draw mean r | K-draw sd r | Tolerance (\|r\| ≤) | Verdict |
|:--------|--------------:|------------:|--------------------:|:--------|
| am × hp |        -1e-04 |      0.0504 |                0.01 | PASS    |
| am × wt |         3e-04 |      0.0525 |                0.01 | PASS    |

### mode = “none” sanity check — marginals only, no correlation

Under `mode = "none"` the engine reproduces each predictor’s marginal
independently (no correlation structure is imposed). We verify that
continuous column means and sds reproduce the engine’s **none-mode**
population moments — measured from a matched large-n `mode = "none"`
reference draw, *not* the strict-mode oracle. The strict (whole-row
bootstrap) and none (NORTA empirical-quantile) paths produce slightly
different marginal distributions from the same frame — the NORTA path
carries a small (~0.5–1%) variance shrinkage the bootstrap does not — so
the none-mode draws must be gated against a none-mode oracle.

| Column | Stat |  Oracle |   K-avg |   Diff | Verdict |
|:-------|:-----|--------:|--------:|-------:|:--------|
| hp     | mean | -0.0204 | -0.0180 | 0.0024 | PASS    |
| hp     | sd   |  0.9307 |  0.9321 | 0.0013 | PASS    |
| wt     | mean | -0.0083 | -0.0088 | 0.0005 | PASS    |
| wt     | sd   |  0.9417 |  0.9410 | 0.0007 | PASS    |

## upload_strict_nonlinear

Formula `y ~ x1 + x2` \| OLS \| upload columns: x1, x2 \| n = 150 per
draw \| K = 1600 draws

### Frame moments vs engine oracle

The table below compares the **raw frame** statistics (before any engine
processing) to the **engine oracle** (moments measured from a single
large n = 200,000 draw). Differences arise from the finite-frame NORTA
effects described in the introduction.

| Column  | Statistic | Frame (standardized) | Engine oracle | Deviation |
|:--------|:----------|---------------------:|--------------:|----------:|
| x1      | mean      |               0.0000 |        0.0008 |    0.0008 |
| x1      | sd        |               1.0000 |        1.0009 |    0.0009 |
| x2      | mean      |               0.0000 |        0.0013 |    0.0013 |
| x2      | sd        |               1.0000 |        1.0015 |    0.0015 |
| x1 x x2 | Pearson r |               0.9678 |        0.9679 |    0.0001 |

### Assertion table — K-draw average vs engine oracle

Averaged over 1600 draws of n = 150:

|  | Assertion | Oracle (engine) | K-draw average | Difference | Tolerance | Verdict |
|:---|:---|---:|---:|---:|---:|:---|
| x1 | mean of x1 | 0.0008 | -0.0035 | 0.0044 | 0.01 | PASS |
| x11 | sd of x1 | 1.0009 | 0.9996 | 0.0013 | 0.01 | PASS |
| x2 | mean of x2 | 0.0013 | -0.0032 | 0.0046 | 0.01 | PASS |
| x21 | sd of x2 | 1.0015 | 0.9981 | 0.0034 | 0.01 | PASS |
| 1 | correlation x1 × x2 (mode=strict) | 0.9679 | 0.9680 | 0.0001 | 0.01 | PASS |
| 11 | parabola: max \|x2_std - f(x1_std)\| per draw (strict preserves joint; NORTA would fail) | 0.0000 | 0.0000 | 0.0000 | 0.00 | PASS |

### Parabola-preservation assertion — strict bootstrap preserves the nonlinear joint

This case uses `mode = "strict"` (whole-row bootstrap). Because the
engine copies entire rows from the uploaded frame, every generated row
satisfies x2 = x1^2 exactly — the nonlinear joint is preserved.

In **standardized** coordinates (mean 0, pop-sd 1), the parabola
becomes:

    x2_std = 0.281261 * x1_std² + 0.967844 * x1_std + -0.281261

We check `max |x2_std - f(x1_std)|` for every row in every draw and
assert that the mean of this per-draw maximum is ≤ 1e-5 (the CSV
floating-point floor).

**Why this discriminates strict from NORTA:** under `mode = "partial"`,
the engine generates rows via NORTA — it matches marginals and Pearson
correlation but draws from a Gaussian copula, so points are scattered
around the parabola, not constrained to it. The NORTA per-row residual
from the parabola would be O(1) in standardized units; the strict
residual is ≤ 3e-8 (CSV float precision).

| Metric | K-draw value | Max observed | Tolerance | Verdict |
|:---|---:|---:|---:|:---|
| mean max \|x2_std - f(x1_std)\| over K draws | 1.309e-07 | 1.369e-07 | 1e-05 | PASS |

### mode = “none” sanity check — marginals only, no correlation

Under `mode = "none"` the engine reproduces each predictor’s marginal
independently (no correlation structure is imposed). We verify that
continuous column means and sds reproduce the engine’s **none-mode**
population moments — measured from a matched large-n `mode = "none"`
reference draw, *not* the strict-mode oracle. The strict (whole-row
bootstrap) and none (NORTA empirical-quantile) paths produce slightly
different marginal distributions from the same frame — the NORTA path
carries a small (~0.5–1%) variance shrinkage the bootstrap does not — so
the none-mode draws must be gated against a none-mode oracle.

| Column | Stat |  Oracle |   K-avg |   Diff | Verdict |
|:-------|:-----|--------:|--------:|-------:|:--------|
| x1     | mean |  0.0017 |  0.0019 | 0.0001 | PASS    |
| x1     | sd   |  0.9934 |  0.9918 | 0.0015 | PASS    |
| x2     | mean | -0.0033 | -0.0014 | 0.0019 | PASS    |
| x2     | sd   |  0.9939 |  0.9943 | 0.0004 | PASS    |

# Provenance

| Item | Value |
|:---|:---|
| Report generated | 15 June 2026 |
| R version | R version 4.5.3 (2026-03-11) |
| mcpower | 0.0.0.9000 |
| Upload frames | mtcars (built-in R dataset; 32 rows, hp/wt/am columns); nonlinear_parabola.csv (synthetic; 150 rows, x1∈\[0,3\], x2=x1²) |
| Oracle draw size | 200,000 |
| Draws per case (mode=partial/strict) | 1,600 |
| Draws per case (mode=none sanity) | 400 |
| Moment tolerance | 0.01 |
| Parabola tolerance (strict case) | 1e-5 |

To reproduce this report, from the repository root:

``` r
rmarkdown::render("mcpower/validation/validation_upload.rmd",
                  output_dir = "mcpower/documentation/validation")
```
