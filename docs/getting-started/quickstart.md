---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Quick Start

This page walks you through your first Monte Carlo power analysis with MCPower, from defining a model to interpreting the results.

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Your First Power Analysis

Suppose you are planning a study to test whether a binary treatment improves satisfaction, controlling for motivation.

### Step 1: Define the Model

```{code-cell} ipython3
:tags: [remove-output]
from mcpower import MCPower

model = MCPower("satisfaction = treatment + motivation")
model.set_simulations(400)
```

MCPower accepts both `=` and `~` as separators. For full formula syntax details, see [Model Specification](../model-specification.md).

### Step 2: Set Effect Sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("treatment=0.5, motivation=0.3")
```

- **treatment=0.5** -- receiving the treatment increases satisfaction by 0.5 SD (medium-large effect)
- **motivation=0.3** -- each 1 SD increase in motivation increases satisfaction by 0.3 SD

For guidelines on choosing effect sizes, see [Effect Sizes](../concepts/effect-sizes.md).

### Step 3: Set Variable Types

```{code-cell} ipython3
:tags: [remove-output]
model.set_variable_type("treatment=binary")
```

This tells MCPower that `treatment` is a 0/1 variable. For all available types, see [Variable Types](../concepts/variable-types.md).

### Step 4: Check Power

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(sample_size=100, target_test="treatment")
```

The `target_test` parameter specifies which effect to test:

| Value | What It Tests |
|---|---|
| `"treatment"` | Power for the treatment coefficient |
| `"overall"` | F-test for the entire model |
| `"all"` | All individual coefficients + overall F-test |
| `"treatment, motivation"` | Multiple specific effects |

---

## Interpreting the Output

- **Power** -- percentage of simulations where the test was significant. 70.7 means the effect was detected in 70.7% of 1,600 simulated datasets.
- **Target** -- target power level (default: 80%).
- **Status** -- `✓` if power meets the target, `✗` if it falls short.

---

## Finding the Required Sample Size

Search for the smallest sample size that achieves your target power:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=200,
    by=20,
)
```

This evaluates power at each sample size from 50 to 200 and reports where target power is first reached.

Control the search grid:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=300,
    by=30,  # step size
)
```

Add `summary="long"` for a full table and power curve plot:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=200,
    by=20,
    summary="long",
)
```

---

## Using Variable Types

### Binary Variables

```{code-cell} ipython3
:tags: [remove-output]
model.set_variable_type("treatment=binary")          # 50/50 split
model.set_variable_type("treatment=(binary,0.3)")     # 30% receive treatment
```

### Factor Variables

Factor variables represent categorical predictors with 3+ levels. MCPower automatically creates dummy variables (level 1 is the reference).

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("outcome = condition + age")
model.set_simulations(400)
model.set_variable_type("condition=(factor,3)")
model.set_effects("condition[2]=0.4, condition[3]=0.6, age=0.2")
```

For custom group proportions:

```{code-cell} ipython3
:tags: [remove-output]
model.set_variable_type("condition=(factor,0.2,0.5,0.3)")  # 20%, 50%, 30%
```

For pairwise comparisons between factor levels, see [ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md). For all variable types, see [Variable Types](../concepts/variable-types.md).

---

## Scenario Analysis

Test how robust your power is under realistic conditions:

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Restore the satisfaction model for scenario analysis
model = MCPower("satisfaction = treatment + motivation")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3")
```

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=300,
    by=30,
    scenarios=True,
)
```

| Scenario | What It Simulates | When to Use |
|---|---|---|
| **Optimistic** | Your exact settings | Best-case planning |
| **Realistic** | Mild effect variations, small assumption violations | Recommended for most studies |
| **Doomer** | Larger variations, stronger violations | Conservative planning |

You can also add custom scenarios with `set_scenario_configs()` and run specific ones by name:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.set_scenario_configs({"extreme": {"heterogeneity": 0.6}})
model.find_power(sample_size=100, scenarios=["optimistic", "extreme"])
```

For details, see [Scenario Analysis](../concepts/scenario-analysis.md).

---

## Using Your Own Data

Upload pilot data so MCPower samples from empirical distributions:

```python
import pandas as pd

data = pd.read_csv("my_data.csv")

model = MCPower("mpg = hp + wt + cyl")
model.upload_data(data[["hp", "wt", "cyl"]])
model.set_effects("hp=0.5, wt=0.3, cyl[6]=0.2, cyl[8]=0.4")
model.find_power(sample_size=100)
```

MCPower auto-detects variable types based on unique values: 2 = binary, 3-6 = factor, 7+ = continuous. Override with `data_types` parameter. For details, see [Uploading Data](../tutorials/own-data.md).

---

## Programmatic Access to Results

Use `return_results=True` to get a Python dictionary:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
result = model.find_power(
    sample_size=100,
    target_test="all",
    return_results=True,
    print_results=False,
)

power_treatment = result["results"]["individual_powers"]["treatment"]
```

---

## Reproducibility

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.set_seed(42)
model.find_power(sample_size=100, target_test="treatment")
```

---

## Next Steps

- **[Model Specification](../model-specification.md)** -- Formula syntax, interactions
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Choosing and interpreting effect sizes
- **[Variable Types](../concepts/variable-types.md)** -- All distribution types
- **[Correlations](../concepts/correlations.md)** -- Setting predictor correlations
- **[ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md)** -- Pairwise comparisons
- **[Multiple Testing Corrections](../tutorials/multiple-testing.md)** -- Bonferroni, FDR, Holm
- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- Robustness testing
- **[Mixed-Effects Models](../concepts/mixed-effects.md)** -- Random intercepts, slopes, nested effects
- **[Uploading Data](../tutorials/own-data.md)** -- Using empirical data
- **[Performance & Backends](../concepts/performance.md)** -- C++ backend, parallelization
- **[API Reference](../api/index.md)** -- Full method documentation
