# Limitations

MCPower is designed for practical power analysis of linear and mixed-effects models. This page describes scenarios where it may not be the right tool or where results require extra caution.

---

## Very Small Effect Sizes

Monte Carlo power analysis estimates power by counting how often simulations detect an effect. At very small effect sizes (beta < 0.05), the detection rate becomes noisy -- small changes in the simulation count can swing power estimates by several percentage points. For reliable estimates at tiny effects, you may need 5,000--10,000+ simulations, which increases runtime substantially.

**Recommendation:** If your expected effect is very small, increase simulations with `model.set_simulations(5000)` or higher and verify stability by running the analysis twice with different seeds.

---

## Very Large Models

Models with many predictors (dozens or more) combined with multiple-testing corrections can produce extremely conservative power estimates. The computational cost also scales with the number of predictors. Bonferroni correction over 50+ tests may make it nearly impossible to achieve adequate power for individual effects, even with large samples.

**Recommendation:** Focus `target_test` on the effects you actually care about rather than testing "all". Consider FDR correction instead of Bonferroni for large test sets.

---

## Numerical Instability (Mixed Models)

Mixed-effects model estimation can fail to converge under certain conditions:

- **Low observations per cluster** -- fewer than 10 observations per cluster increases failure rates significantly.
- **Extreme ICC values** -- MCPower restricts ICC to 0 or 0.1--0.9 for this reason.
- **Complex random structures** -- random slopes, nested effects, and multiple variance components are harder to estimate.
- **Small number of clusters** -- fewer than 10 clusters provides insufficient information for variance component estimation.

When convergence failures exceed the allowed threshold (default 3%), MCPower raises an error. Use `model.set_max_failed_simulations(0.10)` to relax the threshold if needed, but high failure rates may indicate an inadequate study design rather than a software limitation.

---

## GWAS / Genomics

MCPower is not designed for genome-wide association studies or other analyses involving millions of simultaneous tests. The per-simulation overhead (data generation, model fitting, p-value extraction) makes it impractical at genomic scale. Specialized GWAS power tools exist for this purpose.

---

## Non-linear / Non-standard Models

MCPower currently supports:

- **OLS linear regression** (fully supported)
- **Linear mixed-effects models** (fully supported)

The following model types are **not** currently supported:

- Logistic regression (binary outcomes)
- Poisson / negative binomial regression (count outcomes)
- Ordinal regression
- Structural equation models (SEM)
- Time series models
- Survival / Cox regression

---

## Mitigating Data Generation Limitations

MCPower generates synthetic data from parametric distributions (normal, binary, factor, skewed, etc.). If your real data has complex distributional properties -- multimodal distributions, ceiling/floor effects, or unusual correlation structures -- the synthetic data may not capture these features.

**Mitigation:** Upload your own empirical data via `upload_data()` with `preserve_correlation="strict"`. This bootstraps whole rows from your dataset, preserving the exact multivariate relationships and distributional shapes. This is especially valuable when you have pilot data or a related dataset.

---

## What's Next

Planned model types for future releases:

- **Logistic regression** -- binary outcome models (coming soon)
- **Robust regression models** -- methods for handling outliers and heteroskedasticity
- **Alternatives to t-tests** -- handling unmet assumptions in group comparisons
