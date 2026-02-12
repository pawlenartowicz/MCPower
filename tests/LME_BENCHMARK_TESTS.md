# LME Benchmark Tests Summary

## Overview

Proper benchmark tests for mixed model (LME) power analysis, following the same
pattern as the gold-standard OLS tests: **analytical formula -> MC margin -> tight assertion**.

Replaces the previous 20pp-tolerance "validation" test with statistically-derived
margins (~7-9pp at mid-range power with 500 simulations).

## Data Generating Process (DGP)

```
y_ij = X*beta + b_i + eps

X ~ N(0, Sigma)    iid across ALL observations (no cluster structure in X)
b_i ~ N(0, tau^2)  random intercept per cluster
eps ~ N(0, 1)       residual noise
tau^2 = ICC / (1 - ICC)
```

## Analytical Formulas

### Design Effect for Within-Cluster (iid) Predictors

```
Deff_within = (1 + (m-1)*ICC) / (1 + (m-2)*ICC)    where m = n_total / K
```

This is **much milder** than the between-cluster design effect `1 + (m-1)*ICC`.
For m=50: Deff_within ~ 1.02-1.06 across all ICC values, vs Deff_between ~ 6-25.

### z-test Power (Individual Fixed Effects)

statsmodels MixedLM uses Wald z-tests for individual fixed effects:

```
NCP = |beta| * sqrt(n_eff / (VIF_j * Deff_within)) / sigma_eps
Power = 1 - Phi(z_{alpha/2} - NCP) + Phi(-z_{alpha/2} - NCP)
```

where `n_eff = max(n_total - p - 1, p + 2)` (finite-sample correction).

Implementation: `analytical_z_power_lme()` in `tests/helpers/analytical.py`.

### Likelihood Ratio Test Power (Overall Model)

```
NCP = n_total * beta' * Sigma * beta / (sigma_eps^2 * Deff_within)
Power = 1 - F_{nc_chi2}(chi2_{alpha,p}; p, NCP)
```

Implementation: `analytical_lr_power_lme()` in `tests/helpers/analytical.py`.

## Assertion Criteria

All tests use the same MC margin formula as the OLS benchmarks:

```python
margin = MC_Z * sqrt(p*(1-p) / n_sims) * 100 + ALLOWED_BIAS
```

where:
- `MC_Z = 3.5` (Bonferroni-safe z-score for 100+ tests)
- `ALLOWED_BIAS = 1` pp (accounts for finite-sample approximation)
- `n_sims = 500` (LME_N_SIMS_BENCHMARK)
- `p = true_power / 100` (proportion scale)

Typical margins at mid-range power (~50-80%): **~8-9 pp**.

## Configuration

| Constant              | Value | Source          |
|-----------------------|-------|-----------------|
| LME_N_SIMS_BENCHMARK | 500   | `tests/config.py` |
| MC_Z                  | 3.5   | `tests/config.py` |
| ALLOWED_BIAS          | 1 pp  | `tests/config.py` |
| SEED                  | 2137  | `tests/config.py` |
| max_failed_simulations| 10%   | per test        |

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `tests/helpers/analytical.py` | Modified | Added `_deff_within()`, `analytical_z_power_lme()`, `analytical_lr_power_lme()` |
| `tests/config.py` | Modified | Added `LME_N_SIMS_BENCHMARK = 500` |
| `tests/specs/test_power_accuracy_lme.py` | Created | 14 benchmark accuracy tests |
| `tests/specs/test_type1_error_lme.py` | Created | 7 Type I error control tests |
| `tests/specs/test_monotonicity_lme.py` | Created | 4 monotonicity property tests |

## Test Inventory

### test_power_accuracy_lme.py (14 tests) — `@pytest.mark.slow`

**Class: TestLMEAccuracyVsAnalytical**

| Test | Cases | What it checks |
|------|-------|----------------|
| `test_single_predictor_z_test` | 7 parametrized | MC z-test power vs `analytical_z_power_lme` |
| `test_single_predictor_lr_test` | 3 parametrized | MC LR test power vs `analytical_lr_power_lme` |
| `test_two_predictors_uncorrelated` | 2 parametrized | Both z-tests + LR test with Sigma = I |
| `test_two_predictors_correlated` | 2 parametrized | z-tests with VIF from rho = 0.3, 0.5 |

**Parametrized designs (all have m = n/K = 50 obs/cluster):**

| beta | n_total | K  | ICC | Deff_within |
|------|---------|----|-----|-------------|
| 0.3  | 1000    | 20 | 0.1 | 1.019       |
| 0.5  | 1000    | 20 | 0.1 | 1.019       |
| 0.3  | 1500    | 30 | 0.2 | 1.038       |
| 0.5  | 1000    | 20 | 0.2 | 1.038       |
| 0.3  | 1000    | 20 | 0.3 | 1.060       |
| 0.5  | 1500    | 30 | 0.3 | 1.060       |
| 0.2  | 2500    | 50 | 0.2 | 1.038       |

### test_type1_error_lme.py (7 tests) — `@pytest.mark.slow`

**Class: TestLMETypeIErrorControl**

| Test | Design | What it checks |
|------|--------|----------------|
| `test_single_predictor_null_overall` | K=20, ICC=0.2, n=1000 | LR test rejection ~ alpha |
| `test_single_predictor_null_individual` | K=20, ICC=0.2, n=1000 | z-test rejection ~ alpha |
| `test_two_predictors_null_each` | K=20, ICC=0.2, n=1000 | Both z-tests ~ alpha |
| `test_large_sample_null_no_inflation` | K=50, ICC=0.2, n=2500 | No Type I error inflation with N |

**Class: TestLMEAlphaCalibration**

| Test | Parametrized over | What it checks |
|------|-------------------|----------------|
| `test_null_rejection_matches_alpha` | alpha in {0.01, 0.05, 0.10} | Rejection rate tracks nominal alpha |

Criterion: `|observed_rejection - alpha*100| < mc_margin(alpha, 500)`

### test_monotonicity_lme.py (4 tests) — `@pytest.mark.slow`

**Class: TestLMEPowerMonotonicity**

| Test | Varied parameter | Values | Fixed design |
|------|-----------------|--------|--------------|
| `test_power_increases_with_effect_size` | beta | 0.05, 0.10, 0.15 | K=20, ICC=0.2, n=1000 |
| `test_power_increases_with_sample_size` | n_total | 1000, 1500, 2000 | beta=0.05, ICC=0.2, m=50 |
| `test_power_decreases_with_correlation` | rho | 0.0, 0.3, 0.6 | beta=0.10, K=20, ICC=0.2, n=1000 |
| `test_power_increases_with_alpha` | alpha | 0.01, 0.05, 0.10 | beta=0.10, K=20, ICC=0.2, n=1000 |

Criterion: strict monotonicity (`powers[i] < powers[i+1]` or `>` as appropriate).

## Design Decisions

### Why no ICC monotonicity test?

For MCPower's DGP where X is iid (no cluster structure in predictors), the
within-cluster design effect Deff_within is nearly identical across all ICC values
when m is large:

- ICC=0.1, m=50: Deff = 1.0172
- ICC=0.3, m=50: Deff = 1.0195
- ICC=0.5, m=50: Deff = 1.0200

The power difference is < 0.1pp — undetectable with 500 sims. This is **correct
behavior**, not a bug. Replaced with correlation (VIF) monotonicity test which
has a strong, detectable effect.

### Why n_total >= 1000?

LME complexity validation requires >= 10 observations per parameter per cluster.
With 5 parameters (intercept + 1 fixed + random variance + residual + ...) and
K=20 clusters, minimum cluster_size = 50, so n_total >= 1000.

### Why small effects (beta = 0.05-0.15) in monotonicity tests?

With n=1000 and Deff ~ 1.02, even beta=0.2 gives ~99% power. Small effects
produce mid-range power where monotonicity is testable.

### No finite-sample calibration needed

The analytical formulas matched MC results without any additional correction
beyond `n_eff = n - p - 1`. The ALLOWED_BIAS = 1pp buffer was sufficient.

## Running the Tests

```bash
# Individual test files
python -m pytest MCPower/tests/specs/test_power_accuracy_lme.py -v
python -m pytest MCPower/tests/specs/test_type1_error_lme.py -v
python -m pytest MCPower/tests/specs/test_monotonicity_lme.py -v

# All LME benchmark tests
python -m pytest MCPower/tests/specs/test_power_accuracy_lme.py \
                 MCPower/tests/specs/test_type1_error_lme.py \
                 MCPower/tests/specs/test_monotonicity_lme.py -v

# All slow tests
python -m pytest MCPower/tests/specs/ -m slow -v

# Full suite (including OLS tests, no regressions)
python -m pytest MCPower/tests/ -v
```

## Test Results (Initial Run)

| Test file | Tests | Passed | Time |
|-----------|-------|--------|------|
| test_power_accuracy_lme.py | 14 | 14/14 | ~224s |
| test_type1_error_lme.py | 7 | 7/7 | ~56s |
| test_monotonicity_lme.py | 4 | 4/4 | ~82s |
| **Total LME benchmark** | **25** | **25/25** | **~362s** |
| Existing test suite | 380 | 380/380 | ~43s (no regressions) |
