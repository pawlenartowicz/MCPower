---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Finding the Right Sample Size

**Goal:** You want to find the minimum sample size needed for adequate power.

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
- [Raising the Target to 90% Power](#raising-the-target-to-90-power)
- [Common Variations](#common-variations)
- [Next Steps](#next-steps)

---

## Full Working Example

You are designing an educational intervention study. A new teaching method is tested against the standard approach, while controlling for students' prior achievement. You need to determine the minimum number of students to recruit.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model
model = MCPower("test_score = teaching_method + prior_achievement")
model.set_simulations(400)

# 2. Set variable types
model.set_variable_type("teaching_method=binary")

# 3. Set expected effect sizes
model.set_effects("teaching_method=0.40, prior_achievement=0.25")

# 4. Search for the minimum sample size
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=300,
    by=28,
)
```

---

## Step-by-Step Walkthrough

### Lines 1-3: Model setup

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("test_score = teaching_method + prior_achievement")
model.set_simulations(400)
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.40, prior_achievement=0.25")
```

This is the same setup as a basic power analysis (see [Tutorial: Your First Power Analysis](first-analysis.md)). We define a model with one binary predictor (`teaching_method`) and one continuous covariate (`prior_achievement`), then set their expected effect sizes.

- **teaching_method=0.40** -- a small-to-medium effect for a binary predictor (between 0.20 small and 0.50 medium)
- **prior_achievement=0.25** -- a medium effect for a continuous predictor

### Line 4: Search for the minimum sample size

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=300,
    by=28,
)
```

Instead of checking power at a single sample size, `find_sample_size` evaluates power across a **search grid**:

| Parameter | Value | Meaning |
|---|---|---|
| `target_test` | `"teaching_method"` | Find the sample size needed to detect this specific effect |
| `from_size` | `50` | Start searching at N=50 |
| `to_size` | `300` | Stop searching at N=300 |
| `by` | `10` | Evaluate power every 10 participants: N=50, 60, 70, ..., 300 |

MCPower runs 1,600 simulations at each sample size in the grid and reports the smallest N where power first reaches the target (default: 80%).

### The search grid concept

The search grid is the set of sample sizes that MCPower evaluates:

```
N=50 → N=60 → N=70 → ... → N=200 (first ≥ 80%) → ... → N=300
```

MCPower evaluates every point in the grid. Smaller `by` values give a more precise answer but take longer. Larger `by` values run faster but might overshoot the true minimum.

**Practical advice:**

- Start with `by=10` or `by=25` for a rough estimate
- Narrow the range and reduce `by` for a precise answer
- The default grid (`from_size=30`, `to_size=200`, `by=5`) works well for many common designs

---

## Output Interpretation

```
Sample Size Requirements:
Test                                     Required N
-----------------------------------------------------
teaching_method                          200
```

| Column | Meaning |
|---|---|
| **Test** | The coefficient being tested |
| **Required N** | The smallest sample size in the grid where power meets the target |

**Verdict:** You need at least 200 students (100 per group) to detect the teaching method effect at alpha = 0.05.

If the search range is exhausted without reaching the target, increase `to_size` or reconsider your effect size assumptions.

---

## Adding Scenario Analysis

Use `scenarios=True` to see how the required sample size changes under realistic conditions:

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=300,
    by=28,
    scenarios=True,
)
```

| Scenario | Min N | Interpretation |
|---|---|---|
| **Optimistic** | 200 | If everything goes perfectly |
| **Realistic** | 200 | Under mild violations of assumptions |
| **Doomer** | 220 | Under strong violations of assumptions |

**Planning advice:** Use the **Realistic** estimate (N=200) for your primary justification. If you can afford N=220 (the Doomer estimate), your study is robust against substantial assumption violations.

---

## Raising the Target to 90% Power

The default target is 80%. To use 90% power instead:

```{code-cell} ipython3
:tags: [remove-stderr]
model.set_power(90)

model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=400,
    by=40,
)
```

Moving from 80% to 90% power typically requires 20-30% more participants.

---

## Common Variations

### Detailed output with power curve

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=300,
    by=28,
    summary="long",
)
```

The `summary="long"` option prints a full table showing power at every sample size in the grid, and displays a power curve plot (when running in an environment that supports plotting). This is useful for seeing how power grows across the range, rather than just the single threshold crossing.

### Test all effects

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=300,
    by=28,
)
```

This reports the minimum N for each predictor and the overall F-test. Different effects may require different sample sizes.

### Combine scenarios with corrections

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=400,
    by=40,
    scenarios=True,
    correction="bonferroni",
)
```

When testing multiple effects with a correction for multiple comparisons, the required sample size increases. Scenario analysis then shows how much additional buffer you need for assumption violations.

### Coarse-to-fine search

For efficient searching, start coarse and then zoom in:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Step 1: Rough scan
model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=500,
    by=50,
)
# Result: Min N ≈ 200

# Step 2: Precise scan around the rough estimate
model.find_sample_size(
    target_test="teaching_method",
    from_size=150,
    to_size=250,
    by=12,
)
# Result: Min N = 200
```

### Get results programmatically

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
result = model.find_sample_size(
    target_test="teaching_method",
    from_size=50,
    to_size=300,
    by=28,
    return_results=True,
    print_results=False,
)

min_n = result["results"]["first_achieved"]["teaching_method"]
print(f"Minimum sample size: {min_n}")
```

---

## Next Steps

- **[Tutorial: Your First Power Analysis](first-analysis.md)** -- Check power at a specific sample size
- **[Tutorial: Testing Interactions](interactions.md)** -- Interactions typically require larger samples
- **[Tutorial: Correlated Predictors](correlations.md)** -- Predictor correlations affect sample size requirements
- **[Effect Sizes](../concepts/effect-sizes.md)** -- Guidelines for choosing appropriate effect sizes
- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- Understanding the three scenarios in depth
- **[API Reference](../api/index.md)** -- Full parameter documentation for `find_sample_size`
