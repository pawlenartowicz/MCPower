---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Your First Power Analysis

**Goal:** You want to check if your planned sample size provides enough statistical power.

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
- [Adding Scenario Analysis](#adding-scenario-analysis)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

You are running a clinical trial to test whether a new drug improves patient recovery, while controlling for baseline health. You plan to recruit 100 participants and want to know if that is enough to detect the drug's effect.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model
model = MCPower("recovery = drug + baseline_health")
model.set_simulations(400)

# 2. Declare the treatment as a binary variable (drug vs. placebo)
model.set_variable_type("drug=binary")

# 3. Set expected effect sizes (standardized coefficients)
model.set_effects("drug=0.50, baseline_health=0.25")

# 4. Check power at N=100 for the drug effect
model.find_power(sample_size=100, target_test="drug")
```

---

## Step-by-Step Walkthrough

### Line 1: Import MCPower

```{code-cell} ipython3
:tags: [remove-output]
from mcpower import MCPower
```

`MCPower` is the only class you need. It handles model definition, simulation, and analysis.

### Line 2: Define the model

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("recovery = drug + baseline_health")
model.set_simulations(400)
```

This creates a regression model where `recovery` is the outcome and `drug` and `baseline_health` are predictors. The formula uses R-style syntax -- you can also write `recovery ~ drug + baseline_health`.

### Line 3: Set variable types

```{code-cell} ipython3
:tags: [remove-output]
model.set_variable_type("drug=binary")
```

By default, MCPower treats all predictors as continuous (standard normal). Since `drug` is a 0/1 treatment assignment, we declare it as binary. `baseline_health` remains continuous.

### Line 4: Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("drug=0.50, baseline_health=0.25")
```

Effect sizes are standardized regression coefficients (betas):

- **drug=0.50** -- receiving the drug increases recovery by 0.50 SD compared to placebo. This is a medium effect for a binary predictor.
- **baseline_health=0.25** -- each 1 SD increase in baseline health increases recovery by 0.25 SD. This is a medium effect for a continuous predictor.

### Line 5: Run the analysis

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=100, target_test="drug")
```

MCPower runs 1,600 Monte Carlo simulations (the default for OLS models). In each simulation it:

1. Generates a dataset of 100 participants
2. Fits the regression model
3. Tests whether `drug` is statistically significant at alpha = 0.05

The proportion of simulations where `drug` is significant is the estimated power.

The `target_test="drug"` parameter focuses the output on the drug coefficient. Without it, MCPower reports power for every predictor plus the overall F-test.

---

## Output Interpretation

```
Test                                     Power    Target   Status
-------------------------------------------------------------------
drug                                     70.7     80       ✗
```

| Column | Meaning |
|---|---|
| **Test** | The coefficient being tested (here, `drug`) |
| **Power** | Percentage of 1,600 simulations where the test was significant. 70.7% means 70.7% of simulated datasets detected the drug effect. |
| **Target** | The target power level (default: 80%) |
| **Status** | `✓` if power meets the target; `✗` if it falls short |

**Verdict:** With N=100 and a medium effect (0.50), you have 70.7% power -- below the 80% threshold. You may need a larger sample size.

---

## Adding Scenario Analysis

The basic analysis assumes perfect conditions. In reality, effects might be slightly different, distributions might not be exactly normal, and variance might not be constant. Use `scenarios=True` to test robustness:

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(sample_size=100, target_test="drug", scenarios=True)
```

| Scenario | What It Simulates |
|---|---|
| **Optimistic** | Your exact settings -- best case |
| **Realistic** | Mild effect variations, small assumption violations |
| **Doomer** | Larger variations, stronger violations -- worst case |

Here, none of the scenarios meet 80% power, and the **Doomer** scenario drops to 66.5%. If you want to be safe even under pessimistic conditions, consider increasing your sample size.

---

## Common Variations

### Test all effects at once

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=100, target_test="all")
```

This reports power for `drug`, `baseline_health`, and the overall F-test.

### Test multiple specific effects

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=100, target_test="drug, baseline_health")
```

### Test only the overall model

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=100, target_test="overall")
```

The overall F-test checks whether the model as a whole explains significant variance. It is almost always more powerful than individual coefficient tests.

### Get detailed output

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=100, target_test="drug", summary="long")
```

The `summary="long"` option prints additional detail about the model configuration and results.

### Get results as a Python dictionary

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
result = model.find_power(
    sample_size=100,
    target_test="drug",
    return_results=True,
    print_results=False,
)

power_drug = result["results"]["individual_powers"]["drug"]
print(f"Power for drug effect: {power_drug}%")
```

### Change the significance level

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.set_alpha(0.01)
model.find_power(sample_size=100, target_test="drug")
```

A stricter alpha (0.01 instead of 0.05) requires stronger evidence, so power will be lower.

### Set a reproducible seed

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.set_seed(42)
model.find_power(sample_size=100, target_test="drug")
```

Setting a seed ensures you get the same results each time you run the analysis.

---

## Next Steps

- **[Tutorial: Finding the Right Sample Size](sample-size.md)** -- Search for the minimum N that achieves your target power
- **[Tutorial: Testing Interactions](interactions.md)** -- Check if the effect of one variable depends on another
- **[Tutorial: Correlated Predictors](correlations.md)** -- Account for correlations between predictors
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing appropriate effect sizes
- **[Variable Types](../concepts/variable-types.md)** -- Binary, factor, and distribution options
- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- Understanding the three scenarios in depth
- **[API Reference](../api/index.md)** -- Full parameter documentation for `find_power` and all configuration methods
