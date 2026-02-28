---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: ANOVA & Post-Hoc Pairwise Comparisons

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Goal

You want to compare specific groups (post-hoc pairwise comparisons) rather than just testing whether a factor has an overall effect.

---

## Full Working Example

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# ── Define a 3-group treatment study ──────────────────────────────
model = MCPower("pain_relief = treatment + baseline_pain")
model.set_simulations(400)
model.set_variable_type("treatment=(factor,3)")

# ── Name the factor levels (optional but recommended) ─────────────
model.set_factor_levels("treatment=placebo,low_dose,high_dose")

# ── Set effect sizes (vs reference = placebo) ─────────────────────
# placebo is the reference level (first listed)
# low_dose vs placebo: medium effect
# high_dose vs placebo: large effect
model.set_effects(
    "treatment[low_dose]=0.50, treatment[high_dose]=0.80, "
    "baseline_pain=0.25"
)

# ── Run all pairwise comparisons with Tukey correction ────────────
model.find_power(
    sample_size=150,
    target_test="all-posthoc",
    correction="tukey",
)
```

---

## Step-by-Step Walkthrough

### 1. Define the model with a factor variable

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("pain_relief = treatment + baseline_pain")
model.set_simulations(400)
model.set_variable_type("treatment=(factor,3)")
```

`treatment=(factor,3)` tells MCPower this is a categorical variable with 3 levels. MCPower automatically creates dummy variables using reference coding.

### 2. Name the factor levels (optional but recommended)

```{code-cell} ipython3
:tags: [remove-output]
model.set_factor_levels("treatment=placebo,low_dose,high_dose")
```

This assigns meaningful names to each level:

- `placebo` = reference level (first listed) -- no dummy variable
- `low_dose` = second level -- dummy `treatment[low_dose]`
- `high_dose` = third level -- dummy `treatment[high_dose]`

Without `set_factor_levels()`, dummies use integer indices instead: `treatment[2]` and `treatment[3]` (level 1 is the reference).

### 3. Set effect sizes

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects(
    "treatment[low_dose]=0.50, treatment[high_dose]=0.80, "
    "baseline_pain=0.25"
)
```

Each effect is the standardized difference **from the reference level** (placebo):

- `treatment[low_dose]=0.50` -- low dose improves pain relief by 0.50 SD over placebo
- `treatment[high_dose]=0.80` -- high dose improves by 0.80 SD over placebo
- The implicit contrast between low_dose and high_dose is `0.80 - 0.50 = 0.30` SD

Without named levels, you would write: `treatment[2]=0.50, treatment[3]=0.80`

### 4. Run post-hoc comparisons

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(
    sample_size=150,
    target_test="all-posthoc",
    correction="tukey",
)
```

- `target_test="all-posthoc"` generates all pairwise comparisons for every factor in the model.
- `correction="tukey"` applies Tukey HSD to control the family-wise error rate across all pairwise comparisons.

---

## Output Interpretation

```
================================================================================
MONTE CARLO POWER ANALYSIS RESULTS
================================================================================
Multiple comparison correction: tukey

Power Analysis Results (N=150):
Test                                     Power    Target   Status
-------------------------------------------------------------------
treatment[placebo] vs treatment[low_dose] 55.8     80       ✗
treatment[placebo] vs treatment[high_dose] 93.8     80       ✓
treatment[low_dose] vs treatment[high_dose] 19.4     80       ✗

With tukey correction:
Test                                     Power    Target   Status
-------------------------------------------------------------------
treatment[placebo] vs treatment[low_dose] 49.3     80       ✗
treatment[placebo] vs treatment[high_dose] 91.2     80       ✓
treatment[low_dose] vs treatment[high_dose] 14.8     80       ✗

Result: 1/3 tests achieved target power
```

- **treatment[placebo] vs treatment[low_dose]** -- placebo vs low_dose (effect = 0.50 SD). Moderate power.
- **treatment[placebo] vs treatment[high_dose]** -- placebo vs high_dose (effect = 0.80 SD). Excellent power.
- **treatment[low_dose] vs treatment[high_dose]** -- low_dose vs high_dose (implicit effect = 0.30 SD). Low power -- this is the hardest comparison to detect.
- The output shows two tables: uncorrected first, then Tukey-corrected. The Tukey correction reduces power slightly by adjusting for the number of pairwise comparisons using the Studentized Range distribution.

---

## Understanding the Dual Indexing

MCPower uses two different indexing systems that can be confusing at first:

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Setup for indexing examples
from mcpower import MCPower
model = MCPower("outcome = group + covariate")
model.set_simulations(400)
```

### Effect sizes: bracket notation with level names or integers starting at 2

When setting effects, you reference **dummy variables**. Without named levels:

```{code-cell} ipython3
:tags: [remove-output]
model.set_variable_type("group=(factor,3)")
model.set_effects("group[2]=0.4, group[3]=0.6")
# group[2] = second level vs reference
# group[3] = third level vs reference
```

With named levels:

```{code-cell} ipython3
:tags: [remove-output]
model.set_factor_levels("group=control,drug_a,drug_b")
model.set_effects("group[drug_a]=0.4, group[drug_b]=0.6")
```

### Post-hoc comparisons: 1-indexed levels (includes reference)

When specifying pairwise comparisons in `target_test`, levels are **1-indexed** (the reference level is included):

```python
target_test="group[1] vs group[2]"
# group[1] = reference level (level 1 / control)
# group[2] = second level (level 2 / drug_a)
# group[3] = third level (level 3 / drug_b)
```

### Summary table

For a 3-level factor `group=(factor,3)` with levels named `control, drug_a, drug_b`:

| Context | Level 1 (control) | Level 2 (drug_a) | Level 3 (drug_b) |
|---|---|---|---|
| **Effects** (integer) | reference (no effect) | `group[2]=0.4` | `group[3]=0.6` |
| **Effects** (named) | reference (no effect) | `group[drug_a]=0.4` | `group[drug_b]=0.6` |
| **Post-hoc** | `group[1]` | `group[2]` | `group[3]` |

---

## Common Variations

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Restore treatment model for variations
model = MCPower("pain_relief = treatment + baseline_pain")
model.set_simulations(400)
model.set_variable_type("treatment=(factor,3)")
model.set_factor_levels("treatment=placebo,low_dose,high_dose")
model.set_effects("treatment[low_dose]=0.50, treatment[high_dose]=0.80, baseline_pain=0.25")
```

### Specific pairwise comparisons

Instead of testing all pairs, request only the comparisons you care about:

```{code-cell} ipython3
:tags: [remove-stderr]
# Only compare the two active treatments
model.find_power(
    sample_size=150,
    target_test="treatment[low_dose] vs treatment[high_dose]",
    correction="tukey",
)
```

### Explicit post-hoc syntax

List specific comparisons separated by commas:

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(
    sample_size=150,
    target_test="treatment[placebo] vs treatment[low_dose], treatment[placebo] vs treatment[high_dose], treatment[low_dose] vs treatment[high_dose]",
    correction="tukey",
)
```

### Mixing standard tests with post-hoc comparisons

Combine overall F-test, individual coefficient tests, and post-hoc comparisons:

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(
    sample_size=150,
    target_test="overall, baseline_pain, treatment[placebo] vs treatment[high_dose]",
    correction="tukey",
)
```

When Tukey correction is active, non-contrast tests (`overall`, `baseline_pain`) show `"-"` in the corrected column because Tukey only applies to pairwise comparisons.

### Without named levels (integer indexing)

If you skip `set_factor_levels()`, effects use integer-indexed dummies:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("outcome = group + covariate")
model.set_simulations(400)
model.set_variable_type("group=(factor,3)")
model.set_effects("group[2]=0.4, group[3]=0.6, covariate=0.25")

model.find_power(
    sample_size=150,
    target_test="group[1] vs group[2], group[1] vs group[3], group[2] vs group[3]",
    correction="tukey",
)
```

### Finding required sample size

Search for the sample size needed to achieve 80% power on the hardest comparison:

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Restore treatment model for sample size search
model = MCPower("pain_relief = treatment + baseline_pain")
model.set_simulations(400)
model.set_variable_type("treatment=(factor,3)")
model.set_factor_levels("treatment=placebo,low_dose,high_dose")
model.set_effects("treatment[low_dose]=0.50, treatment[high_dose]=0.80, baseline_pain=0.25")
```

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_sample_size(
    target_test="treatment[low_dose] vs treatment[high_dose]",  # smallest effect (0.30 SD)
    from_size=100,
    to_size=600,
    by=56,
    correction="tukey",
)
```

### Using Holm or Bonferroni instead of Tukey

Standard corrections (Bonferroni, Holm, FDR) apply to all tests, not just post-hoc contrasts:

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_power(
    sample_size=150,
    target_test="overall, baseline_pain, treatment[placebo] vs treatment[low_dose], treatment[low_dose] vs treatment[high_dose]",
    correction="holm",
)
```

With Holm/Bonferroni/FDR, every test (including `overall` and `baseline_pain`) is part of the correction family.

### Four or more groups

The syntax scales to any number of levels:

```{code-cell} ipython3
:tags: [remove-stderr]
model = MCPower("score = condition + age")
model.set_simulations(400)
model.set_variable_type("condition=(factor,4)")
model.set_factor_levels("condition=placebo,low,medium,high")
model.set_effects(
    "condition[low]=0.20, condition[medium]=0.50, condition[high]=0.80, "
    "age=0.10"
)

# All 6 pairwise comparisons
model.find_power(
    sample_size=200,
    target_test="all-posthoc",
    correction="tukey",
)
```

---

## Next Steps

- **[Tutorial: Multiple Testing Corrections](multiple-testing.md)** -- controlling false positives across many tests
- **[Effect Sizes](../concepts/effect-sizes.md)** -- choosing appropriate effect sizes for factor levels
- **[Variable Types](../concepts/variable-types.md)** -- factor variable configuration details
- **[API Reference](../api/index.md)** -- full `find_power()` and `set_factor_levels()` parameter documentation
