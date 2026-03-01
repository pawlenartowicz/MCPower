---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Correlated Predictors

**Goal:** Your predictors are correlated and you want to account for that in your power analysis.

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
- [Output Interpretation](#output-interpretation)
- [Using Matrix Format](#using-matrix-format)
- [Correlated vs. Uncorrelated: A Comparison](#correlated-vs-uncorrelated-a-comparison)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

A social science researcher is studying predictors of health outcomes. The model includes income, education, and social support -- variables that are correlated in the real world. Income and education are moderately correlated (r=0.50), and both are weakly correlated with social support.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model
model = MCPower("health = income + education + social_support")
model.set_simulations(400)

# 2. Set effect sizes
model.set_effects("income=0.30, education=0.25, social_support=0.40")

# 3. Set pairwise correlations between predictors
model.set_correlations(
    "(income, education)=0.50, "
    "(income, social_support)=0.20, "
    "(education, social_support)=0.30"
)

# 4. Find the required sample size
model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=40,
)
```

---

## Step-by-Step Walkthrough

### Lines 1-2: Model and effects

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("health = income + education + social_support")
model.set_simulations(400)
model.set_effects("income=0.30, education=0.25, social_support=0.40")
```

A three-predictor model. All variables are continuous by default:

- **income=0.30** -- a medium effect
- **education=0.25** -- a medium effect
- **social_support=0.40** -- a large effect

### Line 3: Set correlations (string format)

```{code-cell} ipython3
:tags: [remove-output]
model.set_correlations(
    "(income, education)=0.50, "
    "(income, social_support)=0.20, "
    "(education, social_support)=0.30"
)
```

This specifies three pairwise correlations:

| Pair | Correlation | Meaning |
|---|---|---|
| income, education | 0.50 | Moderately correlated (people with more education tend to earn more) |
| income, social_support | 0.20 | Weakly correlated |
| education, social_support | 0.30 | Weakly-to-moderately correlated |

The shorthand `(x, y)=r` is equivalent to the full form `corr(x, y)=r`. Any pair not specified defaults to 0 (independent).

MCPower uses Cholesky decomposition to generate data that matches this correlation structure. It also validates that the resulting correlation matrix is positive semi-definite -- contradictory correlations (e.g., A-B=0.9, A-C=0.9, B-C=-0.9) will raise an error.

### Line 4: Find sample sizes

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=40,
)
```

With `target_test="all"`, MCPower reports the minimum sample size for each individual predictor and the overall F-test.

---

## Output Interpretation

```
Sample Size Requirements:
Test                                     Required N
-----------------------------------------------------
overall                                  50
income                                   130
education                                180
social_support                           70
```

Key observations:

1. **education requires the largest N (180)** despite having a similar effect size to income (0.25 vs. 0.30). This is because education shares substantial variance with income (r=0.50), leaving less unique variance to detect its independent effect.

2. **social_support requires the smallest N (70)** because it has the largest effect size (0.40) and weaker correlations with the other predictors.

3. **The overall F-test requires only 50 participants** because it tests whether the model as a whole explains significant variance. This is almost always more powerful than individual coefficient tests.

4. **To power all individual tests at 80%, you need N=180** -- the maximum across all effects.

---

## Using Matrix Format

For models with many predictors, a numpy correlation matrix can be more convenient:

```{code-cell} ipython3
:tags: [remove-stderr]
import numpy as np
from mcpower import MCPower

model = MCPower("health = income + education + social_support")
model.set_simulations(400)
model.set_effects("income=0.30, education=0.25, social_support=0.40")

# Correlation matrix: rows/columns in formula order
corr_matrix = np.array([
    [1.00, 0.50, 0.20],   # income
    [0.50, 1.00, 0.30],   # education
    [0.20, 0.30, 1.00],   # social_support
])
model.set_correlations(corr_matrix)

model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=40,
)
```

The matrix must be:

- **Square** -- rows and columns match the number of non-factor predictors
- **Symmetric** -- `corr_matrix[i,j] == corr_matrix[j,i]`
- **Diagonal = 1.0** -- each variable correlates perfectly with itself
- **Positive semi-definite** -- the matrix represents a valid correlation structure
- **Ordered by formula** -- columns correspond to predictors in the order they appear in the formula (here: income, education, social_support)

---

## Correlated vs. Uncorrelated: A Comparison

To see the impact of correlations, compare the same model with and without them:

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# --- Without correlations (unrealistic but illustrative) ---
model_uncorr = MCPower("health = income + education + social_support")
model_uncorr.set_simulations(400)
model_uncorr.set_effects("income=0.30, education=0.25, social_support=0.40")
# No set_correlations call -- all correlations default to 0

model_uncorr.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=40,
)

# --- With correlations (realistic) ---
model_corr = MCPower("health = income + education + social_support")
model_corr.set_simulations(400)
model_corr.set_effects("income=0.30, education=0.25, social_support=0.40")
model_corr.set_correlations(
    "(income, education)=0.50, "
    "(income, social_support)=0.20, "
    "(education, social_support)=0.30"
)

model_corr.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=40,
)
```

**Typical results:**

| Test | Min N (uncorrelated) | Min N (correlated) | Increase |
|---|---|---|---|
| income | 100 | 130 | +30% |
| education | 140 | 180 | +29% |
| social_support | 60 | 70 | +17% |
| overall | 50 | 50 | 0% |

The more strongly two predictors are correlated, the more each one's required sample size increases. Income and education are the most correlated pair (r=0.50), and both see the largest increases.

**Key takeaway:** Ignoring predictor correlations overestimates power. You end up with an underpowered study.

---

## Common Variations

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Restore model for variations
model = MCPower("health = income + education + social_support")
model.set_simulations(400)
model.set_effects("income=0.30, education=0.25, social_support=0.40")
```

### Negative correlations

```python
model.set_correlations("(stress, coping)=-0.40")
```

Negative correlations are common in psychology (e.g., stress and coping strategies). They affect power the same way -- any non-zero correlation reduces power for the correlated predictors.

### Correlation with a binary variable

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("outcome = treatment + age + severity")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.50, age=0.20, severity=0.30")
model.set_correlations("(age, severity)=0.40, (treatment, age)=0.10")
```

Correlations between continuous and binary variables work the same way. Here, age and severity are moderately correlated, and treatment assignment has a slight correlation with age (perhaps older patients were more likely to accept the treatment).

### Combine correlations with scenario analysis

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Restore model for scenario analysis
model = MCPower("health = income + education + social_support")
model.set_simulations(400)
model.set_effects("income=0.30, education=0.25, social_support=0.40")
```

```{code-cell} ipython3
:tags: [remove-stderr]
model.set_correlations("(income, education)=0.50")

model.find_sample_size(
    target_test="income",
    from_size=50,
    to_size=400,
    by=40,
    scenarios=True,
)
```

Scenario analysis adds **correlation noise** on top of your specified correlations, simulating the possibility that your correlation estimates are slightly off. This provides a more robust sample size estimate.

### Correlations with uploaded data

If you have pilot data, you can let MCPower compute correlations directly from the data:

```python
import pandas as pd

data = pd.read_csv("pilot_data.csv")

model = MCPower("health = income + education + social_support")
model.upload_data(data[["income", "education", "social_support"]])
# In "strict" mode (the default), correlations are preserved via bootstrapping
# In "partial" mode, correlations are computed and can be overridden

model.set_effects("income=0.30, education=0.25, social_support=0.40")
model.find_sample_size(target_test="all", from_size=50, to_size=400, by=10)
```

See [Uploading Data](own-data.md) for details on correlation preservation modes.

### Check if correlations are causing problems

If power is unexpectedly low, check whether high correlations are the cause by running the same analysis without correlations and comparing the results. If the gap is large, consider whether you truly need both correlated predictors in the model, or whether one could be dropped.

---

## Next Steps

- **[Tutorial: Your First Power Analysis](first-analysis.md)** -- The basics of `find_power`
- **[Tutorial: Finding the Right Sample Size](sample-size.md)** -- Systematic sample size search
- **[Tutorial: Testing Interactions](interactions.md)** -- Interactions between correlated predictors
- **[Correlations](../concepts/correlations.md)** -- Full reference on correlation specification, matrix format, and limitations
- **[Uploading Data](own-data.md)** -- Using empirical data with preserved correlations
- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- Correlation noise and robustness testing
- **[API Reference](../api/index.md)** -- Full parameter documentation for `set_correlations`
