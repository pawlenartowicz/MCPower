# How It Is Validated

MCPower's statistical engine is validated through two complementary systems: an **internal test suite** covering OLS and mixed-effects accuracy, and an **external cross-validation framework** comparing MCPower's LME solver against R's lme4 package.

## Internal Test Suite

MCPower includes ~11,000 lines of tests organized into specs (accuracy/validation), integration, unit, and mixed-model tests. The specs tests are the core statistical validation.

### Power Accuracy Tests

Monte Carlo power estimates are compared against **exact analytical power** from non-central t and F distributions.

**OLS models** (`test_power_accuracy.py`):
- Single predictor: 5 parametrized cases varying β and N
- Two uncorrelated predictors: 3 cases with Σ = I
- Two correlated predictors: 4 cases with VIF correction (ρ = 0.3, 0.5, 0.7)

**Acceptance criterion:** MC estimate within `3.5 × √[p(1−p)/5000] × 100 + 1pp` of the analytical value. This is a Bonferroni-safe margin (~2-3 percentage points at typical power levels) using 5,000 simulations.

**LME models** (`test_power_accuracy_lme.py`):
- Single predictor z-test: 7 parametrized cases
- Single predictor likelihood-ratio test: 3 cases
- Two uncorrelated predictors: 2 cases
- Two correlated predictors: 2 cases (ρ = 0.3, 0.5)

All LME accuracy tests use m = 50 observations per cluster, where the within-cluster design effect is small (~1.02–1.06), allowing comparison against analytical formulas.

### Type I Error Control

Under the null hypothesis (all effects = 0), the rejection rate must equal the nominal α.

**OLS** (`test_type1_error.py`):
- Single predictor null (F-test and t-test)
- Two predictors null (each rejects at ~α)
- Large-sample null (catches bugs where power inflates with N)
- Alpha calibration at α ∈ {0.01, 0.05, 0.10}

**LME** (`test_type1_error_lme.py`):
- Same structure with K = 20–50 clusters, ICC = 0.2
- Alpha calibration across standard levels

**Criterion:** |observed rejection rate − α × 100| < MC margin

### Monotonicity Tests

Power must **strictly increase** with:
- Effect size (larger β → more power)
- Sample size (larger N → more power)
- Significance level (larger α → more power)

Tested for both OLS (`test_monotonicity.py`) and LME (`test_monotonicity_lme.py`) models. These tests catch subtle implementation bugs that wouldn't violate accuracy bounds but would produce nonsensical results.

### Multiple Comparison Corrections

**Correction conservativeness** (`test_corrections.py`):
- Corrected power ≤ uncorrected power under H₀
- Bonferroni more conservative than FDR
- FWER ≤ α for Bonferroni and Holm

**Extended alpha validation** (`test_alpha_levels.py`):
- 9 tests validating Bonferroni/Holm/FDR at non-default α ∈ {0.01, 0.10}
- Multi-predictor null calibration with corrections

### LME Accuracy Tests

The analytical formulas used as benchmarks for LME tests:

**Design effect (within-cluster):**
$$D_{\text{eff}} = \frac{1 + (m-1) \times \text{ICC}}{1 + (m-2) \times \text{ICC}}$$

This is much milder than the between-cluster design effect for iid predictors — typically 1.02–1.06 for m = 50.

**z-test non-centrality parameter:**
$$\text{NCP} = \frac{\beta \sqrt{n_{\text{eff}}}}{\sigma \sqrt{\text{VIF} \times D_{\text{eff}}}}$$

**Likelihood-ratio test NCP:**
$$\text{NCP} = \frac{n \cdot \boldsymbol{\beta}' \Sigma \boldsymbol{\beta}}{\sigma^2 \times D_{\text{eff}}}$$

---

## External Cross-Validation (LME4)

MCPower's C++ LME solver is cross-validated against R's lme4 package using the **[MCPower-LME4-validation](https://github.com/pawlenartowicz/MCPower-LME4-validation)** framework. This is a separate repository with its own test harness.

### Four Validation Strategies

| Strategy | What It Tests | How |
|----------|--------------|-----|
| **1. External Data Agreement** | Do MCPower and lme4 reach the same significance decision on identical data? | Generate data with numpy, fit both solvers, compare significance decisions. Target: ≥95% agreement rate. |
| **2. MCPower Pipeline Validation** | Does MCPower's full pipeline (data generation → fitting) produce results consistent with lme4? | Extract raw data from MCPower's simulations, re-fit with lme4, compare significance decisions. |
| **3. Parallel Power Simulation** | Do independent power simulations produce the same power estimate? | Both MCPower and R independently generate data and estimate power. Target: |difference| ≤ 5 percentage points. |
| **4. Statistical z-Test** | Is the power difference statistically significant? | Two-proportion z-test on the power estimates from Strategy 3, with Benjamini-Hochberg FDR correction across all scenarios. |

Strategy 1 validates the **solver** in isolation. Strategy 2 validates the **full pipeline** (including data generation). Strategy 3 validates **end-to-end power estimates**. Strategy 4 provides **statistical rigor** for the power comparison.

### Scenario Coverage

**95 unique scenarios** across three model types:

| Model Type | Core | Sensitivity | Total |
|------------|------|-------------|-------|
| Random intercepts (1 predictor) | 36 | 24 | 60 |
| Random intercepts (2 predictors) | 2 | 8 | 10 |
| Random slopes | 4 | 10 | 14 |
| Nested effects | 3 | 8 | 11 |
| **Total** | **45** | **50** | **95** |

Core scenarios run all 4 strategies (45 × 4 = 180 tests). Sensitivity scenarios run Strategy 4 only (50 × 1 = 50 tests). **Total: 230 scenario-strategy combinations.**

Core scenarios vary: ICC ∈ {0.1, 0.2, 0.3}, clusters ∈ {10, 20, 50}, N ∈ {500, 1000}, effects ∈ {small, medium}.

Sensitivity scenarios systematically sweep one parameter while holding others fixed, producing power curves for visual and statistical comparison.

### Pass/Fail Thresholds

| Metric | Threshold | Strategy |
|--------|-----------|----------|
| Significance agreement rate | ≥ 95% | 1, 2 |
| Beta estimate correlation | ≥ 0.98 | 1, 2 |
| SE estimate correlation | ≥ 0.95 | 1, 2 |
| τ² estimate correlation | ≥ 0.95 | 1, 2 |
| Power difference (absolute) | ≤ 5 pp | 3 |
| Type I error rate | 3%–7% (at α = 0.05) | 3 |
| z-test (FDR-corrected) | p > 0.05 | 4 |

### Latest Results

**Result:** **230/230 PASS**

The validation report is published at: https://freestylerscientist.pl/reports/lme4-validation-report.html

---

## How to Run

### Internal test suite

```bash
# OLS tests only (fast, ~30s)
python -m pytest MCPower/tests/ -v -m "not lme"

# All tests including LME (~6 min)
python -m pytest MCPower/tests/ -v

# Accuracy tests only
python -m pytest MCPower/tests/specs/ -v
```

### External LME4 validation

The LME4 cross-validation lives in a **separate repository**: [MCPower-LME4-validation](https://github.com/pawlenartowicz/MCPower-LME4-validation). Clone it and follow the instructions in its README.

See the [MCPower-LME4-validation repository](https://github.com/pawlenartowicz/MCPower-LME4-validation) README for setup instructions and usage.

---

## Validation Methodology

### Why Monte Carlo margins?

Monte Carlo power estimates are inherently noisy — each estimate is a binomial proportion (fraction of simulations where p < α). The standard error is `√[p(1−p)/n_sims]`. MCPower uses 5,000 simulations for accuracy tests, giving SE ≈ 1% at typical power levels.

The acceptance margin `3.5 × SE + 1pp` uses z = 3.5 (Bonferroni correction for ~100 simultaneous tests) plus 1 percentage point for finite-sample approximation bias.

### Why cross-validate against lme4?

For OLS models, exact analytical power formulas exist (non-central t and F distributions), so MCPower can be validated against theory. For mixed-effects models, no closed-form power formulas exist in general. The gold standard is R's lme4 package (Bates et al., 2015), which MCPower's C++ solver reimplements using the same profiled-deviance algorithm.

Cross-validation against lme4 verifies that:
1. MCPower's C++ solver produces the same parameter estimates
2. MCPower's data generation produces valid clustered data
3. MCPower's power estimates match R's independent estimates

### Reproducibility

All tests use fixed random seeds (default: 2137 for MCPower tests, 42 for LME4 validation). Results are deterministic given the same seed and platform.

---

## Learn More

- **[LME Validation Details](lme-validation.md)** — detailed strategy descriptions and scenario configuration
- **[Performance & Backends](concepts/performance.md)** — C++ backend details and simulation precision
- **[MCPower-LME4-validation](https://github.com/pawlenartowicz/MCPower-LME4-validation)** — external validation repository
