---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Nested Random Effects

## Goal

You have hierarchical data with multiple levels of nesting (e.g., students in classrooms in schools) and need to account for clustering at every level in your power analysis.

---

## Table of Contents

- [Full Working Example](#full-working-example)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
- [Statistical Model](#statistical-model)
- [Understanding the Nesting Structure](#understanding-the-nesting-structure)
- [Design Recommendations](#design-recommendations)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Full Working Example

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model with nested random effects: classrooms within schools
model = MCPower("score ~ treatment + (1|school/classroom)")
model.set_simulations(400)

# 2. Configure the parent level (schools)
model.set_cluster("school", ICC=0.15, n_clusters=10)

# 3. Configure the child level (classrooms within schools)
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)

# 4. Set the expected effect size
model.set_effects("treatment=0.5")

# 5. Allow up to 30% convergence failures for nested models
model.set_max_failed_simulations(0.30)

# 6. Run the power analysis
model.find_power(sample_size=1500)
```

Copy this script into a Python file and run it. With 1500 students across 10 schools and 3 classrooms per school (30 classrooms total, 50 students per classroom), this tests whether you have adequate power for detecting the treatment effect.

---

## Step-by-Step Walkthrough

### Step 1: Define the model formula

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("score ~ treatment + (1|school/classroom)")
model.set_simulations(400)
```

The notation `(1|school/classroom)` means "classrooms are nested within schools." This expands to two random intercept terms:
- `(1|school)` -- a random intercept for each school
- `(1|school:classroom)` -- a random intercept for each classroom within a school

Each student's score is influenced by which school they attend AND which classroom within that school they are in.

### Step 2: Configure the parent level

```{code-cell} ipython3
:tags: [remove-output]
model.set_cluster("school", ICC=0.15, n_clusters=10)
```

- **`"school"`** -- the parent grouping variable, matching the formula.
- **`ICC=0.15`** -- 15% of the total outcome variance is due to between-school differences.
- **`n_clusters=10`** -- 10 schools in the study.

### Step 3: Configure the child level

```{code-cell} ipython3
:tags: [remove-output]
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
```

- **`"classroom"`** -- the child grouping variable, matching the formula.
- **`ICC=0.10`** -- 10% of the total outcome variance is due to between-classroom (within-school) differences.
- **`n_per_parent=3`** -- each school contains 3 classrooms.

The `n_per_parent` parameter is what tells MCPower this is a child level. The total number of classrooms is:

```
total classrooms = n_clusters * n_per_parent = 10 * 3 = 30
```

### Step 4: Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("treatment=0.5")
```

A medium-large treatment effect of 0.5 SD. This is the fixed effect that we are testing for power -- the average effect of treatment across all schools and classrooms.

### Step 5: Set the convergence failure threshold

```{code-cell} ipython3
:tags: [remove-output]
model.set_max_failed_simulations(0.30)
```

Nested models are the most complex mixed-effects structure in MCPower. They have two sets of random effects to estimate simultaneously, leading to higher convergence failure rates than single-level models. A threshold of 30% (`0.30`) is recommended. The default threshold of 3% (`0.03`) is usually too strict for nested designs.

### Step 6: Run the analysis

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=1500)
```

MCPower runs 800 simulations. In each simulation, it:
1. Generates synthetic three-level data (students within classrooms within schools)
2. Generates random intercepts for both schools and classrooms
3. Fits a nested linear mixed-effects model using the C++ L-BFGS-B solver with Woodbury decomposition
4. Tests the fixed effect of treatment for significance

Power is the proportion of simulations where the treatment effect is significant at alpha = 0.05.

---

## Statistical Model

The nested random intercept model has two levels of random effects:

$$y_{ijk} = \mathbf{x}_{ijk}'\boldsymbol{\beta} + b_{\text{school},j} + b_{\text{classroom},jk} + e_{ijk}$$

$$b_{\text{school},j} \sim \mathcal{N}(0, \tau^2_{\text{school}})$$

$$b_{\text{classroom},jk} \sim \mathcal{N}(0, \tau^2_{\text{classroom}})$$

$$e_{ijk} \sim \mathcal{N}(0, \sigma^2)$$

Where:
- $y_{ijk}$ is the outcome for student $i$ in classroom $k$ of school $j$
- $\mathbf{x}_{ijk}'\boldsymbol{\beta}$ is the fixed-effects component (treatment)
- $b_{\text{school},j}$ is the random intercept for school $j$ -- the school-level deviation from the grand mean
- $b_{\text{classroom},jk}$ is the random intercept for classroom $k$ within school $j$ -- the classroom-level deviation from its school's mean
- $e_{ijk}$ is the residual error for student $i$

The variance components are derived from the ICCs:

$$\tau^2_{\text{school}} = \frac{\text{ICC}_{\text{school}}}{1 - \text{ICC}_{\text{school}}} \cdot \sigma^2_{\text{within}}$$

$$\tau^2_{\text{classroom}} = \frac{\text{ICC}_{\text{classroom}}}{1 - \text{ICC}_{\text{classroom}}} \cdot \sigma^2_{\text{within}}$$

> **Note:** Each level's $\tau^2$ is computed independently from its stated ICC. Because the total variance is the sum of all components ($\tau^2_{\text{school}} + \tau^2_{\text{classroom}} + \sigma^2$), the actual variance fractions in the generated data will differ slightly from the stated ICCs. For example, ICC_school = 0.15 and ICC_classroom = 0.10 do not partition the variance into exactly 15% and 10% -- the realized fractions are approximate.

### Variance decomposition

The total variance in the outcome is partitioned into three components:

| Component | Source | Controlled by |
|-----------|--------|---------------|
| $\tau^2_{\text{school}}$ | Between schools | `ICC` of the `"school"` cluster |
| $\tau^2_{\text{classroom}}$ | Between classrooms (within schools) | `ICC` of the `"classroom"` cluster |
| $\sigma^2$ | Between students (within classrooms) | Residual |

With ICC_school = 0.15 and ICC_classroom = 0.10, the combined intraclass correlation at the school level is approximately 0.25 -- a student shares 25% of their outcome variance with schoolmates and 10% specifically with classmates.

---

## Understanding the Nesting Structure

### How observations are distributed

With `n_clusters=10`, `n_per_parent=3`, and `sample_size=1500`:

```
10 schools
  x 3 classrooms per school = 30 classrooms
  1500 students / 30 classrooms = 50 students per classroom
```

The hierarchy:
```
School 1
  ├── Classroom 1  (50 students)
  ├── Classroom 2  (50 students)
  └── Classroom 3  (50 students)
School 2
  ├── Classroom 4  (50 students)
  ├── Classroom 5  (50 students)
  └── Classroom 6  (50 students)
...
School 10
  ├── Classroom 28 (50 students)
  ├── Classroom 29 (50 students)
  └── Classroom 30 (50 students)
```

### Two set_cluster calls -- order matters

The parent cluster must be defined first, then the child:

```{code-cell} ipython3
:tags: [remove-output]
# Parent first
model.set_cluster("school", ICC=0.15, n_clusters=10)
# Child second, using n_per_parent
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
```

The child cluster uses `n_per_parent` instead of `n_clusters` or `cluster_size`. This is how MCPower knows the classroom level is nested within the school level.

---

## Design Recommendations

MCPower assigns treatment at the individual level within clusters. Because the treatment contrast is estimated within each cluster, ICC and the allocation of observations across levels have minimal impact on fixed-effect power -- power depends primarily on total N.

> **Note:** This applies because MCPower generates treatment at the individual level within each cluster. In cluster-randomized trials where entire clusters receive the same treatment, the design effect would substantially reduce power.

The cluster structure matters for convergence and estimation stability:

- **Minimum:** 5 observations per classroom (enforced by MCPower; warning below 10).
- **Recommended:** 10+ top-level clusters and 2+ sub-groups per parent for stable random effect estimation.

### Convergence failure threshold

Nested models are the most challenging for convergence. Recommended thresholds:

| Design | Recommended threshold |
|--------|----------------------|
| 20+ schools, 3+ classrooms each | 0.10 - 0.20 |
| 10-20 schools | 0.20 - 0.30 |
| < 10 schools | 0.30 (and consider adding more schools) |

---

## Common Variations

### More sub-groups per parent

If each school has more classrooms:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("score ~ treatment + (1|school/classroom)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.15, n_clusters=10)
model.set_cluster("classroom", ICC=0.10, n_per_parent=5)  # 5 classrooms/school = 50 total
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.30)
model.find_power(sample_size=2500)  # 2500 / 50 = 50 per classroom
```

### Finding the required sample size

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("score ~ treatment + (1|school/classroom)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.15, n_clusters=15)
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.30)
model.find_sample_size(
    from_size=500,
    to_size=5000,
    by=500,
    target_test="treatment",
)
```

### Multiple fixed effects

Add covariates as needed:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("score ~ treatment + ses + motivation + (1|school/classroom)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.15, n_clusters=15)
model.set_cluster("classroom", ICC=0.10, n_per_parent=4)  # 60 classrooms
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, ses=0.2, motivation=0.3")
model.set_max_failed_simulations(0.30)
model.find_power(sample_size=3000)  # 3000 / 60 = 50 per classroom
```

---

## Next Steps

- **[Tutorial: Random Intercepts](mixed-intercepts.md)** -- The simpler single-level random intercept model
- **[Tutorial: Random Slopes](mixed-slopes.md)** -- When the treatment effect varies across clusters
- **[Mixed-Effects Models](../concepts/mixed-effects.md)** -- Overview of all mixed-effects features
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing standardized effect sizes
- **[API Reference](../api/index.md)** -- Full parameter documentation for `set_cluster()` and `find_power()`
