# LME Validation (vs R's lme4)

## What It Is

MCPower's mixed-effects solver uses a custom C++ implementation (profiled-deviance optimization with Brent's method for random intercepts, L-BFGS-B for random slopes and nested models). To ensure this solver produces correct results, MCPower includes a comprehensive validation framework that compares it against R's **lme4** — the gold-standard mixed-effects package used across statistics, psychology, ecology, and medicine.

The validation covers **95 scenarios** across three model types (random intercepts, random slopes, nested effects) using **four independent strategies**. The latest run (February 22, 2026) shows **230/230 scenario-strategy combinations PASS**.

---

## How It Works

The validation framework (`LME4-validation/`) uses four complementary strategies. Each tests a different aspect of correctness, so a scenario must pass all applicable strategies to be considered valid.

### Strategy 1: External Data Agreement

Generates datasets using pure NumPy (independent of MCPower), then fits the same data with both MCPower's C++ solver and R's lme4. Compares significance decisions (reject/fail-to-reject) across 100 datasets per scenario.

**What it proves:** MCPower's solver reaches the same conclusions as lme4 on identical data.

### Strategy 2: MCPower DGP Validation

Uses MCPower's internal data generation pipeline, exports the raw data, and fits it with both solvers. Compares parameter estimates (betas, standard errors, variance components) and significance decisions.

**What it proves:** MCPower generates realistic clustered data and fits it correctly.

### Strategy 3: Parallel Power Simulation

Both MCPower and R independently generate data and run power simulations using the same design parameters but independent RNG streams. Compares the resulting power estimates.

**What it proves:** MCPower's full pipeline (data generation + model fitting + power calculation) produces equivalent power estimates to an independent R implementation.

### Strategy 4: Statistical Z-test with FDR Correction

Runs a two-proportion z-test comparing MCPower and R power estimates, with Benjamini-Hochberg FDR correction across all scenarios. This is the most stringent test — it detects statistically significant differences even when absolute differences are small.

**What it proves:** Power estimates are statistically indistinguishable after controlling for multiple comparisons.

---

## Guidelines

### Pass/Fail Thresholds

| Metric | Threshold | Strategy |
|---|---|---|
| Significance agreement rate | ≥ 95% | 1, 2 |
| Beta correlation (Pearson r) | ≥ 0.98 | 1, 2 |
| SE correlation (Pearson r) | ≥ 0.95 | 1, 2 |
| Tau² correlation (Pearson r) | ≥ 0.95 | 1, 2 |
| Power difference | ≤ 0.05 | 3 |
| Z-test (FDR-corrected) | p > 0.05 | 4 |
| Type I error rate | 0.03–0.07 | 1, 2 |

### Scenario Coverage

| Model Type | Core Scenarios | Sensitivity Sweep | Total |
|---|---|---|---|
| Random intercepts (1 predictor) | 36 | 24 | 60 |
| Random intercepts (2 predictors) | 2 | 8 | 10 |
| Random slopes | 4 | 10 | 14 |
| Nested effects | 3 | 8 | 11 |
| **Total** | **45** | **50** | **95 unique scenarios** |

Core scenarios run all four strategies; sensitivity scenarios run Strategy 4 only — yielding 230 total scenario-strategy combinations.

### What the Scenarios Vary

- **ICC**: 0.1 to 0.5
- **Number of clusters**: 5 to 50
- **Sample size**: 50 to 2,400
- **Effect sizes**: 0.05 to 0.50
- **Model complexity**: 1–2 predictors, intercept-only through nested

---

## Common Patterns

### Running Validation Yourself

The validation suite lives in a separate repository: [MCPower-LME4-validation](https://github.com/pawlenartowicz/MCPower-LME4-validation). See its README for setup instructions, usage, and available options.

### Reading the Report

The HTML report contains:

1. **Summary matrix** — pass/fail grid across all scenarios and strategies
2. **Strategy detail tables** — agreement rates, correlations, power differences
3. **Parameter recovery** — scatter plots of MCPower vs R estimates
4. **FDR-corrected p-values** — Strategy 4 statistical test results

---

## Learn More

- **[Latest Validation Report](https://freestylerscientist.pl/reports/lme4-validation-report.html)** — full HTML report with pass/fail matrix, parameter recovery, and FDR-corrected p-values
- **[Validation Process](https://freestylerscientist.pl/reports/lme4-validation-process.html)** — detailed description of the validation methodology and strategies
- **[Mixed-Effects Models](concepts/mixed-effects.md)** — model types, ICC guidelines, and design recommendations
- **[Performance](concepts/performance.md)** — C++ solver details and runtime benchmarks
