---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Random Slopes

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Goal

You expect the effect of a predictor to vary across clusters (e.g., a treatment works differently in different schools) and need to account for this heterogeneity in your power analysis.

---

## Table of Contents

- [Full Working Example](#full-working-example)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Statistical Model](#statistical-model)
- [Interpreting slope_variance and slope_intercept_corr](#interpreting-slope_variance-and-slope_intercept_corr)
- [Design Recommendations](#design-recommendations)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model with a random intercept AND random slope for treatment
model = MCPower("outcome ~ treatment + motivation + (1 + treatment|site)")
model.set_simulations(400)

# 2. Configure the clustering with random slope parameters
model.set_cluster(
    "site",
    ICC=0.2,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.1,
    slope_intercept_corr=0.3,
)

# 3. Set the expected effect sizes
model.set_effects("treatment=0.5, motivation=0.3")

# 4. Random slopes need more convergence tolerance
model.set_max_failed_simulations(0.30)

# 5. Run the power analysis
model.find_power(sample_size=1500)
```

Copy this script into a Python file and run it. With 1500 observations across 30 sites (50 per site), this tests whether you have adequate power to detect the treatment effect despite it varying across sites.

---

## Step-by-Step Walkthrough

### Step 1: Define the model formula

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("outcome ~ treatment + motivation + (1 + treatment|site)")
model.set_simulations(400)
```

The key part is `(1 + treatment|site)`. This specifies:
- `1` -- a random intercept per site (some sites have higher outcomes overall)
- `treatment` -- a random slope for treatment per site (the treatment effect differs by site)
- `|site` -- both random effects are grouped by site

This means the model allows each site to have its own baseline outcome level AND its own treatment effect. Site A might show a strong treatment benefit while Site B shows a weaker one.

### Step 2: Configure the clustering with slope parameters

```{code-cell} ipython3
:tags: [remove-output]
model.set_cluster(
    "site",
    ICC=0.2,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.1,
    slope_intercept_corr=0.3,
)
```

- **`"site"`** -- matches the grouping variable in the formula.
- **`ICC=0.2`** -- 20% of outcome variance is due to between-site differences in intercepts.
- **`n_clusters=30`** -- 30 research sites.
- **`random_slopes=["treatment"]`** -- the treatment effect varies across sites. This must be a list of predictor names that appear in the formula.
- **`slope_variance=0.1`** -- the variance of the random slope. A slope variance of 0.1 means the treatment effect has a standard deviation of $\sqrt{0.1} \approx 0.32$ across sites.
- **`slope_intercept_corr=0.3`** -- the correlation between the random intercept and the random slope. A positive value of 0.3 means that sites with higher baseline outcomes tend to show slightly larger treatment effects.

### Step 3: Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("treatment=0.5, motivation=0.3")
```

These are the **average** (fixed) effects across all sites:
- **`treatment=0.5`** -- on average, treatment increases the outcome by 0.5 SD. Individual sites vary around this average with SD $\approx$ 0.32 (from `slope_variance=0.1`).
- **`motivation=0.3`** -- motivation has a fixed effect of 0.3 SD (no random slope, so this effect is the same at every site).

### Step 4: Set a higher failure threshold

```{code-cell} ipython3
:tags: [remove-output]
model.set_max_failed_simulations(0.30)
```

Random slope models are more complex than random-intercept-only models and have higher convergence failure rates. The default threshold of 3% (`0.03`) is usually too strict. A threshold of 30% (`0.30`) is recommended for random slope models. If more than 30% of simulations fail to converge, MCPower raises an error -- this typically signals a design with too few clusters or too few observations per cluster.

### Step 5: Run the analysis

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=1500)
```

MCPower runs 800 simulations. In each simulation, it generates clustered data where both intercepts and slopes vary across sites, fits a linear mixed-effects model using the C++ L-BFGS-B solver, and tests the fixed effects. Power is the proportion of simulations where each fixed effect is significant.

---

## Statistical Model

The random slope model extends the random intercept model by allowing the coefficient of a predictor to vary across clusters:

$$y_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta} + b_{0j} + b_{1j} x_{ij} + e_{ij}$$

The random effects for each cluster are drawn from a multivariate normal distribution:

$$\begin{bmatrix} b_{0j} \\ b_{1j} \end{bmatrix} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{G}\right)$$

where $\mathbf{G}$ is the variance-covariance matrix of the random effects:

$$\mathbf{G} = \begin{bmatrix} \tau^2_{\text{int}} & \rho \cdot \tau_{\text{int}} \cdot \tau_{\text{slope}} \\ \rho \cdot \tau_{\text{int}} \cdot \tau_{\text{slope}} & \tau^2_{\text{slope}} \end{bmatrix}$$

Where:
- $b_{0j}$ is the random intercept for cluster $j$ -- the cluster-level deviation from the grand mean
- $b_{1j}$ is the random slope for cluster $j$ -- how much the treatment effect in cluster $j$ deviates from the average treatment effect $\beta_{\text{treatment}}$
- $\tau^2_{\text{int}} = \frac{\text{ICC}}{1 - \text{ICC}} \cdot \sigma^2_{\text{within}}$ -- the intercept variance, derived from the ICC (adjusted for fixed effects as described in the [random intercepts tutorial](mixed-intercepts.md#statistical-model))
- $\tau^2_{\text{slope}}$ = `slope_variance` -- the slope variance you specify
- $\rho$ = `slope_intercept_corr` -- the correlation between the random intercept and slope
- $e_{ij} \sim \mathcal{N}(0, \sigma^2)$ -- the residual error

The off-diagonal element $\rho \cdot \tau_{\text{int}} \cdot \tau_{\text{slope}}$ is the covariance between the random intercept and slope. This is the element that `slope_intercept_corr` controls.

### Why Random Slopes Reduce Power

Random slopes add another source of variability to the model. With a random intercept only, the treatment effect is the same at every cluster -- you're just estimating it from clustered data. With random slopes, the treatment effect itself varies, making it harder to detect a significant *average* effect. The uncertainty in the average treatment effect increases because you're also estimating how much the effect varies.

---

## Interpreting slope_variance and slope_intercept_corr

### slope_variance

The `slope_variance` parameter ($\tau^2_{\text{slope}}$) controls how much the effect of the predictor varies across clusters. Think of it as the "spread" of individual cluster effects around the average effect.

| slope_variance | SD of slope | Interpretation |
|----------------|-------------|---------------|
| 0.01 | 0.10 | Very little variation -- the effect is nearly the same everywhere |
| 0.05 | 0.22 | Moderate variation -- some clusters differ noticeably |
| 0.10 | 0.32 | Substantial variation -- clusters show meaningfully different effects |
| 0.25 | 0.50 | Large variation -- the effect might be positive in some clusters and near-zero in others |

**Example with slope_variance=0.1 and treatment=0.5:** The average treatment effect is 0.5 SD, but individual sites have effects distributed as $\mathcal{N}(0.5, 0.1)$, so roughly 95% of sites have treatment effects between $0.5 - 2 \times 0.32 = -0.13$ and $0.5 + 2 \times 0.32 = 1.13$. Some sites may show near-zero or even negative treatment effects.

### slope_intercept_corr

The `slope_intercept_corr` parameter ($\rho$) controls the relationship between a cluster's baseline level and its slope:

| slope_intercept_corr | Interpretation |
|---------------------|---------------|
| 0.0 | No relationship -- knowing a site's baseline tells you nothing about its treatment effect |
| 0.3 | Mild positive -- higher-baseline sites tend to show slightly larger treatment effects |
| -0.3 | Mild negative -- higher-baseline sites tend to show slightly smaller treatment effects |
| 0.7 | Strong positive -- high-baseline sites strongly tend to have larger effects |
| -0.7 | Strong negative -- high-baseline sites strongly tend to have smaller effects (compensatory) |

In practice, the correlation is often small (|$\rho$| < 0.3). If you have no prior information, `slope_intercept_corr=0.0` is a reasonable default.

---

## Design Recommendations

### Effect of slope_variance on power

Higher slope variance means greater uncertainty in the average effect, which dramatically reduces power:

| slope_variance | Power (N=900, 30 clusters) | Required N (30 clusters) |
|---------------|---------------------------|--------------------------|
| 0 (intercept only) | 99.4% | 400 |
| 0.01 | 96.1% | -- |
| 0.05 | 77.8% | 1,050 (2.6x intercept) |
| 0.10 | 61.4% | >5,000 |
| 0.15 | 48.0% | >5,000 |
| 0.20 | 39.2% | >5,000 |
| 0.30 | 29.9% | >5,000 |

*Simulation: `treatment=0.15`, ICC=0.15, 30 clusters, 800 simulations, seed=42.*

The impact is much larger than for ICC in random-intercept models. Even a modest `slope_variance=0.05` requires 2.6x more observations than the intercept-only model.

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Try it yourself: slope_variance impact on power
from mcpower import MCPower

for sv in [0.01, 0.05, 0.10, 0.20, 0.30]:
    model = MCPower("score ~ treatment + (1 + treatment|cluster)")
    model.set_simulations(400)
    model.set_cluster("cluster", ICC=0.15, n_clusters=30,
                      random_slopes=["treatment"], slope_variance=sv)
    model.set_effects("treatment=0.15")
    model.set_max_failed_simulations(0.20)
    model.set_seed(42)
    model.find_power(sample_size=900)
```

### More clusters is critical

Random slope models need **more clusters** than random intercept models. You are now estimating not just the average effect but also how much it varies -- this requires data from many clusters:

| n_clusters | Power (slope_var=0.10, N=900) |
|------------|-------------------------------|
| 15 | 38.2% |
| 20 | 46.1% |
| 30 | 61.4% |
| 40 | 68.8% |
| 50 | 73.6% |

*Simulation: `treatment=0.15`, ICC=0.15, slope_variance=0.10, 800 simulations, seed=42.*

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Try it yourself: cluster count impact on random slopes
from mcpower import MCPower

for n_cl in [15, 20, 30, 40, 50]:
    model = MCPower("score ~ treatment + (1 + treatment|cluster)")
    model.set_simulations(400)
    model.set_cluster("cluster", ICC=0.15, n_clusters=n_cl,
                      random_slopes=["treatment"], slope_variance=0.10)
    model.set_effects("treatment=0.15")
    model.set_max_failed_simulations(0.25)
    model.set_seed(42)
    model.find_power(sample_size=900)
```

- **Minimum:** 30 clusters. Below this, power drops sharply.
- **Recommended:** 50+ clusters for adequate power with moderate slope variance.

### Observations per cluster

- **Minimum:** 5 observations per cluster (enforced by MCPower).
- **Recommended:** 20--50 observations per cluster. Random slopes need more within-cluster data to estimate the per-cluster slope.

### Convergence failure threshold

Random slope models have inherently higher convergence failure rates than random intercept models, especially with fewer clusters or smaller cluster sizes.

| Model complexity | Recommended threshold |
|-----------------|----------------------|
| Random intercept only | 0.03 - 0.10 |
| Random slopes, 30+ clusters | 0.10 - 0.20 |
| Random slopes, < 30 clusters | 0.20 - 0.30 |

If failures consistently exceed your threshold, increase `n_clusters` or `sample_size` rather than raising the threshold further.

---

## Common Variations

### Zero correlation between intercept and slope

When you have no reason to expect a relationship between baseline levels and treatment effects:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome ~ treatment + (1 + treatment|site)")
model.set_simulations(400)
model.set_cluster(
    "site",
    ICC=0.2,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.1,
    slope_intercept_corr=0.0,
)
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.30)
model.find_power(sample_size=1500)
```

### Negative slope-intercept correlation

In some contexts, clusters with higher baselines show smaller treatment effects (a ceiling or compensatory pattern):

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome ~ treatment + (1 + treatment|site)")
model.set_simulations(400)
model.set_cluster(
    "site",
    ICC=0.2,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.1,
    slope_intercept_corr=-0.5,
)
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.30)
model.find_power(sample_size=1500)
```

### Small slope variance (nearly fixed effect)

When you expect the effect to be mostly consistent across clusters but want to be conservative:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome ~ treatment + (1 + treatment|site)")
model.set_simulations(400)
model.set_cluster(
    "site",
    ICC=0.15,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.02,
    slope_intercept_corr=0.0,
)
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.20)
model.find_power(sample_size=1000)
```

### Finding the required sample size

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome ~ treatment + motivation + (1 + treatment|site)")
model.set_simulations(400)
model.set_cluster(
    "site",
    ICC=0.2,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.1,
    slope_intercept_corr=0.3,
)
model.set_effects("treatment=0.5, motivation=0.3")
model.set_max_failed_simulations(0.30)
model.find_sample_size(
    from_size=500,
    to_size=3000,
    by=278,
    target_test="treatment",
)
```

### Binary treatment with random slopes

Treatment is often a binary variable (0/1). Combine `set_variable_type` with random slopes:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome ~ treatment + age + (1 + treatment|site)")
model.set_simulations(400)
model.set_cluster(
    "site",
    ICC=0.15,
    n_clusters=30,
    random_slopes=["treatment"],
    slope_variance=0.05,
    slope_intercept_corr=0.0,
)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, age=0.2")
model.set_max_failed_simulations(0.20)
model.find_power(sample_size=1500)
```

---

## Next Steps

- **[Tutorial: Random Intercepts](mixed-intercepts.md)** -- The simpler random-intercept-only model
- **[Tutorial: Nested Random Effects](mixed-nested.md)** -- Students in classrooms in schools
- **[Mixed-Effects Models](../concepts/mixed-effects.md)** -- Overview of all mixed-effects features
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing standardized effect sizes
- **[API Reference](../api/index.md)** -- Full parameter documentation for `set_cluster()` and `set_max_failed_simulations()`
