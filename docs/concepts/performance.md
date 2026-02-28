# Performance

## What It Is

MCPower uses a **native C++ backend** for all computation. The C++ extension is built automatically during installation via scikit-build-core and CMake, and is required for MCPower to run. There is no pure-Python fallback.

The backend handles OLS regression (QR decomposition), data generation (Cholesky decomposition for correlations, probability integral transforms for distributions), all three mixed-model solvers (Brent's method for random intercepts, L-BFGS-B for random slopes and nested models), and statistical distribution functions (via Boost.Math). For mixed models, the C++ solver provides roughly a 200x speedup over equivalent Python implementations, making large-scale simulations practical.

Parallelization is available through joblib. By default, MCPower parallelizes only mixed-model analyses (where the per-simulation cost justifies the overhead). Standard OLS models run sequentially because the C++ backend is already fast enough that parallelization overhead outweighs the benefit at default simulation counts.

---

## How It Works in MCPower

Configure simulation count with [`set_simulations()`](../api/configuration.md):

```python
model.set_simulations(5000)  # higher precision
model.set_simulations(800, model_type="mixed")  # mixed models only
```

Configure parallelization with [`set_parallel()`](../api/configuration.md):

```python
model.set_parallel(True)            # parallel for all analyses
model.set_parallel(False)           # single core
model.set_parallel("mixedmodels")   # default: parallel only for mixed models
```

---

## Guidelines

### Simulation Count vs. Precision

| Simulations | Precision (95% CI) | Use Case |
|---|---|---|
| 500 | +/- 3--4% | Quick exploration, model iteration |
| 1,600 (OLS default) | +/- 1--2% | Standard analysis |
| 800 (mixed default) | +/- 2--3% | Standard mixed-model analysis |
| 5,000 | +/- 0.5--1% | Precise estimates |
| 10,000 | +/- 0.3--0.5% | Publication-quality precision |

The precision follows from the binomial standard error of the power estimate: SE = sqrt(p(1-p)/n_sims). At p = 0.80 and n_sims = 1600, SE is approximately 1%.

### C++ Backend Components

| Component | Library | Purpose |
|---|---|---|
| Linear algebra | Eigen3 | QR decomposition, Cholesky, matrix operations |
| Distributions | Boost.Math | F, t, chi-squared, normal CDFs and PPFs |
| Studentized range | R port | Tukey HSD critical values (Legendre quadrature) |
| Optimization | LBFGSPP | L-BFGS-B for random slopes and nested LME |
| 1D optimization | Brent's method | Random intercept LME (profiled deviance) |
| RNG | std::mt19937 | Data generation |

### Parallelization Modes

| Mode | Value | When It Helps |
|---|---|---|
| Mixed only (default) | `"mixedmodels"` | Best default. Mixed models benefit; OLS stays fast single-core. |
| Always on | `True` | Useful for OLS with 5,000+ simulations. |
| Off | `False` | Debugging, reproducibility, or when overhead exceeds benefit. |

Default core count: `cpu_count // 2`. Override with `set_parallel(True, n_cores=4)`.

### Approximate Runtimes

| Model Type | Simulations | Approximate Time |
|---|---|---|
| OLS | 1,600 | 1--3 seconds |
| OLS | 10,000 | 5--15 seconds |
| Mixed (random intercept) | 800 | 2--10 seconds |
| Mixed (random slopes) | 800 | 5--30 seconds |
| Mixed (nested) | 800 | 5--30 seconds |

*(Times depend on sample size, number of predictors, and hardware.)*

---

## Common Patterns

### Recommended Workflow

| Stage | Simulations | Parallel | Purpose |
|---|---|---|---|
| Model exploration | 500 | Default | Fast iteration on model specification |
| Standard analysis | 1,600 (OLS) / 800 (mixed) | Default | Default precision |
| Final estimates | 5,000--10,000 | `True` | Publication-quality results |
| Batch/scripted runs | Any | `True` | Disable progress for speed |

### Tips

- **Start with defaults.** Increase simulations only for final estimates.
- **Parallel for mixed models is automatic.** The default `"mixedmodels"` mode is optimal for most users.
- **More clusters > larger clusters.** Better power and faster convergence per simulation.
- **Disable progress for batch runs:** `find_power(..., progress_callback=False, print_results=False)`.
- **Scenario analysis triples runtime.** Three full simulation runs (Optimistic, Realistic, Doomer).

---

## Learn More

- **[LME Validation (vs lme4)](../lme-validation.md)** -- proof that the C++ solver matches R's lme4
- **[API Reference: set_simulations](../api/configuration.md)** -- simulation count configuration
- **[API Reference: set_parallel](../api/configuration.md)** -- parallelization configuration
