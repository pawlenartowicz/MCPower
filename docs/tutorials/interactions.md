---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Testing Interactions

**Goal:** You want to test whether the effect of one variable depends on another.

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
- [Three-Way Interactions](#three-way-interactions)
- [Why Interactions Need Larger Samples](#why-interactions-need-larger-samples)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

A marketing team wants to know whether the effect of advertising spending on sales depends on the customer's age. Perhaps ads work better on younger audiences, or the relationship between spending and sales changes across age groups.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model with an interaction
model = MCPower("sales = advertising * age")
model.set_simulations(400)

# 2. Set effect sizes for main effects and the interaction
model.set_effects("advertising=0.40, age=0.25, advertising:age=0.15")

# 3. Check power for the interaction at N=200
model.find_power(sample_size=200, target_test="advertising:age")
```

---

## Step-by-Step Walkthrough

### Line 1: Define the model with `*`

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("sales = advertising * age")
model.set_simulations(400)
```

The `*` operator is shorthand for **main effects plus their interaction**. MCPower automatically expands this to:

```
sales = advertising + age + advertising:age
```

You do not need to write the expanded form. In fact, writing `MCPower("sales = advertising + age + advertising*age")` is redundant -- the `*` already includes the main effects.

| Formula Syntax | What It Creates |
|---|---|
| `advertising * age` | `advertising + age + advertising:age` |
| `advertising : age` | `advertising:age` only (no main effects) |

Use `*` when you want the full factorial model (main effects + interaction). Use `:` only when you have a specific reason to include the interaction without automatically adding main effects.

### Line 2: Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects("advertising=0.40, age=0.25, advertising:age=0.15")
```

You must set effect sizes for all three terms:

- **advertising=0.40** -- the main effect of advertising on sales (medium continuous effect)
- **age=0.25** -- the main effect of age on sales (medium continuous effect)
- **advertising:age=0.15** -- the interaction effect. This means: for each 1 SD increase in age, the effect of advertising on sales changes by 0.15 SD.

Interaction effects are typically smaller than main effects. A value of 0.15 is realistic for many social science applications.

### Line 3: Test the interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=200, target_test="advertising:age")
```

The `target_test="advertising:age"` parameter focuses the output on the interaction term. The colon notation (`:`) identifies interaction terms in both formulas and target tests.

---

## Output Interpretation

```
Test                                     Power    Target   Status
-------------------------------------------------------------------
advertising:age                          55.7     80       ✗
```

| Column | Meaning |
|---|---|
| **Test** | The interaction term being tested |
| **Power** | Only 55.7% of simulations detected the interaction |
| **Target** | 80% target power |
| **Status** | `✗` -- power falls short of the target |

**Verdict:** With N=200, you have only 55.7% power to detect the interaction. This is a common finding -- interactions are harder to detect than main effects. You need a larger sample.

To find the required sample size:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="advertising:age",
    from_size=100,
    to_size=500,
    by=45,
)
```

---

## Three-Way Interactions

You can test whether the interaction itself depends on a third variable. For example: does the advertising-by-age interaction differ between product categories?

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("sales = advertising * age * category")
model.set_simulations(400)
model.set_variable_type("category=binary")
model.set_effects(
    "advertising=0.40, age=0.25, category=0.30, "
    "advertising:age=0.15, advertising:category=0.10, age:category=0.10, "
    "advertising:age:category=0.10"
)
model.find_power(sample_size=400, target_test="advertising:age:category")
```

The `*` operator with three variables expands to all main effects, all two-way interactions, and the three-way interaction:

```
sales = advertising + age + category
      + advertising:age + advertising:category + age:category
      + advertising:age:category
```

You must set effect sizes for all seven terms. Three-way interactions require substantially larger samples than two-way interactions.

---

## Why Interactions Need Larger Samples

Interaction effects are inherently harder to detect for two reasons:

### 1. Interactions partition the sample

A main effect uses the full dataset. An interaction effect is essentially a "difference of differences" -- the signal is spread across combinations of the two predictors. With binary variables, each combination cell has roughly N/4 observations.

### 2. Interaction effects tend to be smaller

In most research fields, interaction effects are smaller than main effects. While a main effect of 0.40 is common, an interaction of 0.40 would be unusually large. Typical interaction effect sizes range from 0.10 to 0.20.

### Rule of thumb

To detect an interaction with the same power as a main effect of the same size, you typically need **4 times the sample size**. For a realistically smaller interaction effect, the multiplier is even higher.

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Compare: main effect power vs. interaction power at the same N

model = MCPower("y = x1 * x2")
model.set_simulations(400)
model.set_effects("x1=0.25, x2=0.25, x1:x2=0.25")

# At N=100: main effects will have good power, interaction will not
model.find_power(sample_size=100, target_test="all")
```

---

## Common Variations

### Binary-by-continuous interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("outcome = treatment * covariate")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.50, covariate=0.25, treatment:covariate=0.20")
model.find_power(sample_size=200, target_test="treatment:covariate")
```

Here, the interaction tests whether the treatment effect differs depending on the covariate level. For example, does a medication work better for patients with higher baseline severity?

### Binary-by-binary interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("y = drug * gender")
model.set_simulations(400)
model.set_variable_type("drug=binary, gender=binary")
model.set_effects("drug=0.50, gender=0.20, drug:gender=0.30")
model.find_power(sample_size=200, target_test="drug:gender")
```

### Factor-by-continuous interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("performance = condition * experience")
model.set_simulations(400)
model.set_variable_type("condition=(factor,3)")
model.set_effects(
    "condition[2]=0.40, condition[3]=0.60, experience=0.30, "
    "condition[2]:experience=0.15, condition[3]:experience=0.20"
)
model.find_power(sample_size=300, target_test="condition[2]:experience, condition[3]:experience")
```

When a factor is involved in an interaction, each dummy variable gets its own interaction term.

### Factor-by-factor interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("score = condition * group")
model.set_simulations(400)
model.set_variable_type("condition=(factor,3), group=binary")
model.set_effects(
    "condition[2]=0.40, condition[3]=0.60, group=0.30, "
    "condition[2]:group=0.15, condition[3]:group=0.25"
)
model.find_power(
    sample_size=300,
    target_test="condition[2]:group, condition[3]:group",
)
```

Each factor is expanded into dummies (reference level omitted), and the interaction produces the Cartesian product of the non-reference dummies. Here `condition` (3 levels) and `group` (binary) yield dummies `condition[2]`, `condition[3]`, and `group`, so the interaction terms are `condition[2]:group` and `condition[3]:group`. Use `target_test` with all interaction terms for a joint test.

### Test all effects together

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("sales = advertising * age")
model.set_simulations(400)
model.set_effects("advertising=0.40, age=0.25, advertising:age=0.15")

model.find_power(sample_size=300, target_test="all")
```

This shows power for `advertising`, `age`, `advertising:age`, and the overall F-test side by side. You will typically see that the main effects have high power while the interaction lags behind.

### Find the sample size for the interaction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("sales = advertising * age")
model.set_simulations(400)
model.set_effects("advertising=0.40, age=0.25, advertising:age=0.15")

model.find_sample_size(
    target_test="advertising:age",
    from_size=100,
    to_size=600,
    by=56,
    scenarios=True,
)
```

Use a wider search range than you would for main effects. Interaction tests commonly require N=300 or more.

---

## Next Steps

- **[Tutorial: Your First Power Analysis](first-analysis.md)** -- The basics of `find_power`
- **[Tutorial: Finding the Right Sample Size](sample-size.md)** -- Systematic sample size search
- **[Tutorial: Correlated Predictors](correlations.md)** -- Correlations between predictors further affect interaction power
- **[Model Specification](../model-specification.md)** -- Full formula syntax reference, including `*` and `:` operators
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing interaction effect sizes
- **[API Reference](../api/index.md)** -- Full parameter documentation for `find_power` and `find_sample_size`
