---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Random Intercepts for Clustered Data

## Goal

You have clustered data (e.g., students in schools) and need to account for the clustering in your power analysis.

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

---

## Table of Contents

- [Full Working Example](#full-working-example)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Statistical Model](#statistical-model)
- [Design Recommendations](#design-recommendations)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model with a random intercept for school
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)

# 2. Configure the clustering structure
model.set_cluster("school", ICC=0.2, n_clusters=20)

# 3. Set the expected effect sizes (standardized coefficients)
model.set_effects("treatment=0.5, motivation=0.3")

# 4. Allow up to 10% convergence failures
model.set_max_failed_simulations(0.10)

# 5. Run the power analysis
model.find_power(sample_size=1000)
```

Copy this script into a Python file and run it. With 1000 observations split across 20 schools (50 students per school), you should see power well above 80% for both effects.

---

## Step-by-Step Walkthrough

### Step 1: Define the model formula

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)
```

The formula uses R-style syntax. The key part is `(1|school)`, which tells MCPower that observations are clustered within schools and each school has its own random intercept. The `1` means "intercept" -- every school gets a baseline shift up or down from the grand mean.

Both `~` and `=` are valid as the formula separator.

### Step 2: Configure the clustering

```{code-cell} ipython3
:tags: [remove-output]
model.set_cluster("school", ICC=0.2, n_clusters=20)
```

- **`"school"`** -- matches the grouping variable name in the formula.
- **`ICC=0.2`** -- the intraclass correlation coefficient. An ICC of 0.2 means that 20% of the total variance in satisfaction is due to between-school differences. This is a typical value for educational research.
- **`n_clusters=20`** -- the number of schools in the study. With `sample_size=1000` and 20 clusters, each school will have 50 students.

You must specify either `n_clusters` or `cluster_size`, but not both:

```{code-cell} ipython3
:tags: [remove-output]
# These two are equivalent when sample_size=1000:
model.set_cluster("school", ICC=0.2, n_clusters=20)    # 1000/20 = 50 per school
model.set_cluster("school", ICC=0.2, cluster_size=50)   # 1000/50 = 20 schools
```

### Step 3: Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("treatment=0.5, motivation=0.3")
```

These are standardized regression coefficients (beta weights):
- **`treatment=0.5`** -- a medium-large effect. Receiving treatment increases satisfaction by 0.5 standard deviations.
- **`motivation=0.3`** -- a small-to-medium effect. Each 1 SD increase in motivation predicts a 0.3 SD increase in satisfaction.

### Step 4: Set the convergence failure threshold

```{code-cell} ipython3
:tags: [remove-output]
model.set_max_failed_simulations(0.10)
```

Mixed-effects models occasionally fail to converge during individual simulations. The default threshold is 3% (`0.03`). For random-intercept models, 10% is a safe tolerance -- most simulations will converge. If the failure rate exceeds this threshold, MCPower raises an error, signaling that your design may have too few clusters or too few observations per cluster.

### Step 5: Run the analysis

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(sample_size=1000)
```

MCPower runs 800 Monte Carlo simulations (the default for mixed models). In each simulation, it:
1. Generates synthetic clustered data with the specified ICC and effects
2. Fits a linear mixed-effects model using the C++ native solver
3. Tests each fixed effect for significance at alpha = 0.05

Power is the proportion of simulations where each effect is significant.

---

## Statistical Model

The random intercept model is:

$$y_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta} + b_j + e_{ij}$$

$$b_j \sim \mathcal{N}(0, \tau^2), \quad e_{ij} \sim \mathcal{N}(0, \sigma^2)$$

Where:
- $y_{ij}$ is the outcome for observation $i$ in cluster $j$
- $\mathbf{x}_{ij}'\boldsymbol{\beta}$ is the fixed-effects component (treatment, motivation)
- $b_j$ is the random intercept for cluster $j$ -- the school-level deviation from the grand mean
- $e_{ij}$ is the residual error for observation $i$ in cluster $j$

The between-cluster variance $\tau^2$ is derived from the ICC:

$$\tau^2 = \frac{\text{ICC}}{1 - \text{ICC}} \cdot \sigma^2_{\text{within}}$$

where $\sigma^2_{\text{within}}$ includes both the residual variance and the variance explained by fixed effects. Since MCPower standardizes continuous predictors to variance 1, this simplifies to:

$$\sigma^2_{\text{within}} = 1 + \sum_k \beta_k^2 \cdot \text{Var}(X_k)$$

MCPower automatically adjusts $\tau^2$ upward so that the actual ICC in the generated data matches the value you specify, even after accounting for the variance explained by fixed effects.

---

## Design Recommendations

MCPower assigns treatment at the individual level within each cluster. Because the treatment contrast is estimated within each cluster, the cluster random intercept largely cancels out. As a result, **ICC and cluster allocation have minimal impact on fixed-effect power** -- power depends primarily on total N.

> **Note:** This applies because MCPower generates treatment at the individual level within each cluster. In cluster-randomized trials where entire clusters receive the same treatment, the design effect (`1 + (m-1) × ICC`) would substantially reduce power.

The number of clusters matters for convergence and estimation stability, not for power:

- **Minimum:** 5 observations per cluster (enforced by MCPower; warning below 10).
- **Recommended:** 10--20 clusters for stable random effect estimation and convergence.

---

## Common Variations

### Using cluster_size instead of n_clusters

If you know the cluster size rather than the number of clusters:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, cluster_size=50)
model.set_effects("treatment=0.5, motivation=0.3")
model.set_max_failed_simulations(0.10)
model.find_power(sample_size=1000)  # 1000/50 = 20 schools
```

### Finding the required sample size

Search for the minimum sample size that achieves 80% power:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5, motivation=0.3")
model.set_max_failed_simulations(0.10)
model.find_sample_size(
    from_size=200,
    to_size=2000,
    by=200,
    target_test="treatment",
)
```

Note: as the total sample size changes, the cluster size changes too (sample_size / n_clusters). MCPower handles this automatically.

### Binary treatment variable

If treatment is a binary variable (0/1):

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3")
model.set_max_failed_simulations(0.10)
model.find_power(sample_size=1000)
```

### Multiple fixed effects

Add as many fixed effects as your design requires:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("satisfaction ~ treatment + motivation + age + gender + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=30)
model.set_variable_type("treatment=binary, gender=binary")
model.set_effects("treatment=0.5, motivation=0.3, age=0.1, gender=0.2")
model.set_max_failed_simulations(0.10)
model.find_power(sample_size=1500)
```

---

## Next Steps

- **[Tutorial: Random Slopes](mixed-slopes.md)** -- When the treatment effect varies across clusters
- **[Tutorial: Nested Random Effects](mixed-nested.md)** -- Students in classrooms in schools
- **[Mixed-Effects Models](../concepts/mixed-effects.md)** -- Overview of all mixed-effects features
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing standardized effect sizes
- **[API Reference](../api/index.md)** -- Full parameter documentation for `set_cluster()` and `find_power()`
