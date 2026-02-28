---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Using test_formula

**Goal:** Evaluate how model misspecification affects statistical power -- for example, what happens to your power when you omit a variable or ignore clustering.

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
- [How It Works](#how-it-works)
- [Use Case 1: Omitted Variable Bias](#use-case-1-omitted-variable-bias)
- [Use Case 2: Ignoring Clustering (LME to OLS)](#use-case-2-ignoring-clustering-lme-to-ols)
- [Use Case 3: Factor Variable Omission](#use-case-3-factor-variable-omission)
- [Interpreting Results](#interpreting-results)
- [Gotchas and Limitations](#gotchas-and-limitations)
- [See Also](#see-also)

---

## Full Working Example

You have a study with three predictors, but you suspect a reviewer will ask: "What if `x3` doesn't matter -- how does dropping it change your power?"

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# Define the full model (data generation)
model = MCPower("y = x1 + x2 + x3")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x3=0.3")

# Test with the reduced model (omit x3 from analysis)
model.find_power(100, test_formula="y = x1 + x2")
```

MCPower generates data from the full three-predictor model (where `x3` truly affects `y`), but fits the two-predictor model during analysis. This reveals the power impact of omitting a real predictor.

---

## How It Works

The `test_formula` parameter separates data generation from statistical testing:

1. **Data generation** always uses the formula you passed to `MCPower()`. All predictors contribute to the simulated outcome according to their effect sizes.
2. **Statistical testing** uses the `test_formula` instead. Only the variables in this formula are included in the fitted model.

This means the simulated data reflects reality (the full model), while the analysis reflects what a researcher would actually fit (the reduced model). The resulting power estimates tell you how well the reduced model detects effects when the full model is true.

The test formula must be a **subset** of the generation formula variables -- you cannot introduce new variables that were not part of data generation.

---

## Use Case 1: Omitted Variable Bias

The most common use case: you want to know how dropping a predictor affects power for the remaining effects.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2 + x3")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x3=0.3")

# Full model -- baseline power
model.find_power(100)

# Reduced model -- omit x3
model.find_power(100, test_formula="y = x1 + x2")
```

When `x3` is omitted but truly affects the outcome, its effect becomes part of the residual variance. This typically:

- Inflates the error term, reducing power for the remaining predictors
- May bias coefficient estimates if the omitted variable is correlated with included predictors

Compare the power output from both calls to quantify this impact.

---

## Use Case 2: Ignoring Clustering (LME to OLS)

You have clustered data (e.g., students nested in schools) but want to know what happens if you ignore the clustering and fit a simple OLS model instead.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# Full model: mixed-effects with clustering
model = MCPower("y ~ treatment + covariate + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5, covariate=0.3")
model.set_variable_type("treatment=binary")
model.set_max_failed_simulations(0.10)

# Correct analysis: mixed model
model.find_power(1000)

# Misspecified analysis: ignore clustering, fit OLS
model.find_power(1000, test_formula="y ~ treatment + covariate")
```

The data is generated with cluster-level variance (ICC = 0.2), but the test formula drops the random effect `(1|school)`. The OLS model treats all observations as independent, which:

- Underestimates standard errors (observations within a cluster are correlated)
- Can produce misleadingly high power -- the model is overconfident about detecting effects

This comparison helps justify including random effects in your analysis plan.

---

## Use Case 3: Factor Variable Omission

You want to see what happens when a categorical predictor is dropped from the model.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# Full model: continuous predictor + 3-level factor
model = MCPower("y = x1 + group")
model.set_simulations(400)
model.set_variable_type("group=(factor,3)")
model.set_effects("x1=0.5, group[2]=0.3, group[3]=0.4")

# Full model -- baseline
model.find_power(100)

# Reduced model -- drop the factor
model.find_power(100, test_formula="y = x1")
```

When `group` is omitted, the between-group variance it explains becomes part of the residual. The power for `x1` will change depending on whether `x1` and `group` are correlated.

---

## Interpreting Results

When comparing full-model and reduced-model power:

**Power decreases for remaining predictors:** This is the typical result when the omitted variable explains real outcome variance. The unexplained variance increases, making it harder to detect remaining effects.

**Power stays roughly the same:** The omitted variable contributes little to outcome variance, or it is uncorrelated with the remaining predictors. Omitting it has minimal impact.

**Power increases (rare but possible):** This can happen when the test formula has fewer parameters to estimate, freeing up degrees of freedom. It is more likely with small effect sizes for the omitted variable and no correlation with remaining predictors.

**Ignoring clustering inflates power:** When you drop random effects from a mixed model, the OLS analysis treats clustered observations as independent. Standard errors are underestimated, p-values are too small, and power appears higher than it truly is. This is a false gain -- the Type I error rate is inflated.

---

## Gotchas and Limitations

- **Test formula must be a subset.** Every variable in the test formula must also appear in the generation formula. You cannot add new variables that were not part of data generation.

- **Effect sizes apply to the generation formula.** The `set_effects()` call defines how data is generated. The test formula only controls which model is fitted during analysis -- it does not change the data.

- **Target tests must match the test formula.** If you use `target_test`, the requested effects must exist in the test formula, not just the generation formula. For example, if you omit `x3` from the test formula, you cannot request `target_test="x3"`.

- **Interactions and the test formula.** If the generation formula includes `x1:x2`, the test formula can omit the interaction. But if the test formula includes `x1:x2`, it should also include `x1` and `x2` as main effects (standard regression practice).

- **Mixed model convergence.** When the generation formula has random effects but the test formula does not, convergence is not an issue (OLS always converges). When both formulas have random effects, the test formula's random-effects structure determines convergence behavior.

---

## See Also

- [API: find_power()](../api/power-analysis.md) -- Full parameter reference including `test_formula`
- [API: find_sample_size()](../api/power-analysis.md) -- Sample size search with `test_formula`
- [Tutorial: Your First Power Analysis](first-analysis.md) -- Getting started with MCPower
- [Mixed-Effects Models](../concepts/mixed-effects.md) -- Clustered and longitudinal data
- [Concept: Effect Sizes](../concepts/effect-sizes.md) -- How to choose effect sizes
