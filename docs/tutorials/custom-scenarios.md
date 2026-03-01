---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Using Custom Scenarios

**Goal:** You want to customize the scenario analysis to match the specific assumption violations you expect in your field.

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
- [Adding Entirely New Scenarios](#adding-entirely-new-scenarios)
- [Custom Scenarios for Mixed Models](#custom-scenarios-for-mixed-models)
- [Common Patterns](#common-patterns)
- [Next Steps](#next-steps)

---

## Full Working Example

You are planning a study in educational psychology where heterogeneity in effect sizes is a major concern but distribution violations are unlikely (you have good measurement instruments). The default scenarios apply too much distribution noise and too little effect heterogeneity for your situation.

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

# 1. Define the model
model = MCPower("achievement = teaching_method + motivation + ses")
model.set_simulations(400)
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.40, motivation=0.30, ses=0.20")

# 2. Customize scenarios for your field
model.set_scenario_configs({
    "realistic": {
        "heterogeneity": 0.35,            # More effect heterogeneity
        "distribution_change_prob": 0.10,  # Less distribution noise
    },
    "doomer": {
        "heterogeneity": 0.60,            # Much more effect heterogeneity
        "distribution_change_prob": 0.20,  # Still mild distribution noise
        "correlation_noise_sd": 0.3,       # Moderate correlation instability
    },
})

# 3. Find required sample sizes under custom scenarios
model.find_sample_size(
    target_test="teaching_method",
    from_size=50, to_size=400, by=40,
    scenarios=True,
)
```

---

## Step-by-Step Walkthrough

### Lines 1-3: Model setup

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("achievement = teaching_method + motivation + ses")
model.set_simulations(400)
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.40, motivation=0.30, ses=0.20")
```

A standard three-predictor model. Nothing changes about model setup when using custom scenarios.

### Line 4: Customize scenario parameters

```{code-cell} ipython3
:tags: [remove-output]
model.set_scenario_configs({
    "realistic": {
        "heterogeneity": 0.35,
        "distribution_change_prob": 0.10,
    },
    "doomer": {
        "heterogeneity": 0.60,
        "distribution_change_prob": 0.20,
        "correlation_noise_sd": 0.3,
    },
})
```

This is the key step. `set_scenario_configs()` takes a dictionary mapping scenario names to configuration overrides. The merge rules are:

1. **You only specify what you want to change.** Any key you omit keeps its default value.
2. **Existing scenarios are updated.** Here, `"realistic"` and `"doomer"` already exist, so only the specified keys are overridden.
3. **The optimistic scenario is never modified.** It always uses your exact settings with zero perturbations.

For example, the realistic scenario above changes `heterogeneity` from 0.2 (default) to 0.35 and `distribution_change_prob` from 0.3 to 0.10, but keeps `heteroskedasticity` at 0.1, `correlation_noise_sd` at 0.2, and all other keys at their default values.

### Line 5: Run with scenarios

```{code-cell} ipython3
:tags: [remove-stderr]
model.find_sample_size(
    target_test="teaching_method",
    from_size=50, to_size=400, by=40,
    scenarios=True,
)
```

The `scenarios=True` flag runs the analysis under all configured scenarios (optimistic, realistic, doomer, plus any custom scenarios you've added). This works the same way with `find_power()`. You can also pass a list of scenario names to run only specific ones — see [Running Selected Scenarios](#running-selected-scenarios).

---

## Output Interpretation

```
Test                                     Optimistic   Realistic    Doomer
-------------------------------------------------------------------------------
teaching_method                          80           100          160
```

- **Optimistic (N=80):** Under ideal conditions, 80 participants suffice.
- **Realistic (N=100):** With your field-specific heterogeneity, you need 100 participants.
- **Doomer (N=160):** Under worst-case conditions, you need 160 participants.

**Recommendation:** Plan for N=100 (the realistic estimate). If the study is high-stakes, plan for N=160 to be safe under worst-case conditions.

---

## Adding Entirely New Scenarios

You are not limited to the three built-in scenarios. Add custom scenarios with any name:

```{code-cell} ipython3
:tags: [remove-input, remove-output]
# Setup model for new scenario examples
model = MCPower("outcome = treatment + covariate")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.40, covariate=0.25")
```

```{code-cell} ipython3
:tags: [remove-stderr]
model.set_scenario_configs({
    "field_specific": {
        "heterogeneity": 0.30,
        "heteroskedasticity": 0.05,
        "correlation_noise_sd": 0.15,
        "distribution_change_prob": 0.10,
    },
})

model.find_power(sample_size=150, target_test="treatment", scenarios=True)
# Runs: optimistic, realistic, doomer, AND field_specific
```

New scenarios inherit all keys from the optimistic baseline (all zeros), so you only specify the perturbations you want. The analysis now runs four scenarios instead of three.

### Multiple custom scenarios

```{code-cell} ipython3
:tags: [remove-stderr]
model.set_scenario_configs({
    "mild": {
        "heterogeneity": 0.10,
        "heteroskedasticity": 0.05,
    },
    "moderate": {
        "heterogeneity": 0.25,
        "heteroskedasticity": 0.15,
        "correlation_noise_sd": 0.20,
    },
    "severe": {
        "heterogeneity": 0.50,
        "heteroskedasticity": 0.30,
        "correlation_noise_sd": 0.40,
        "distribution_change_prob": 0.50,
    },
})

model.find_power(sample_size=200, scenarios=True)
# Runs: optimistic, realistic, doomer, mild, moderate, AND severe
```

Note that the built-in realistic and doomer scenarios still run alongside your custom ones.

### Running Selected Scenarios

You don't have to run all scenarios every time. Pass a list of scenario names (case-insensitive) to run only the ones you need:

```{code-cell} ipython3
:tags: [remove-stderr]
# Run only optimistic and your custom "severe" scenario
model.find_power(sample_size=200, scenarios=["optimistic", "severe"])
```

This saves computation when you only want to compare specific scenarios. Unknown names raise a `ValueError` listing all available scenarios.

---

## Custom Scenarios for Mixed Models

Mixed models have additional perturbation keys that control random effect and ICC violations. These keys are ignored for standard OLS models.

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("score ~ treatment + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.15, n_clusters=25)
model.set_effects("treatment=0.40")

model.set_scenario_configs({
    "realistic": {
        "heterogeneity": 0.25,
        "icc_noise_sd": 0.20,               # More ICC instability
        "random_effect_dist": "normal",      # Keep random effects normal
    },
    "doomer": {
        "heterogeneity": 0.45,
        "icc_noise_sd": 0.40,               # Heavy ICC instability
        "random_effect_dist": "heavy_tailed",
        "random_effect_df": 3,               # Very heavy tails
        "residual_change_prob": 0.6,
        "residual_df": 4,
    },
})
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(sample_size=750, scenarios=True)
```

### Mixed model configuration keys

| Key | What It Does | Default (Realistic) | Default (Doomer) |
|---|---|---|---|
| `icc_noise_sd` | Jitters the ICC per simulation (multiplicative noise on tau-squared) | 0.15 | 0.30 |
| `random_effect_dist` | Distribution of random effects: `"normal"`, `"heavy_tailed"`, or `"skewed"` | `"heavy_tailed"` | `"heavy_tailed"` |
| `random_effect_df` | Degrees of freedom when random effects are non-normal | 5 | 3 |

Lower `random_effect_df` means heavier tails (more extreme cluster deviations). These keys are silently ignored for models without cluster specifications.

---

## Common Patterns

### When effect sizes are well-established

If your field has strong meta-analytic estimates and the main concern is non-normality:

```{code-cell} ipython3
:tags: [remove-output]
model.set_scenario_configs({
    "realistic": {
        "heterogeneity": 0.05,              # Effects are well-known
        "residual_change_prob": 0.5,         # But distributions may be non-normal
        "residual_df": 8,
    },
    "doomer": {
        "heterogeneity": 0.10,
        "residual_change_prob": 0.8,
        "residual_df": 4,
    },
})
```

### When correlations are uncertain

If predictor correlations come from a small pilot study:

```{code-cell} ipython3
:tags: [remove-output]
model.set_scenario_configs({
    "realistic": {
        "correlation_noise_sd": 0.35,        # More noise than default (0.2)
    },
    "doomer": {
        "correlation_noise_sd": 0.50,        # Much more noise than default (0.4)
    },
})
```

### Disabling a specific perturbation

Set a key to its optimistic value (usually 0.0 or `"normal"`) to disable it:

```{code-cell} ipython3
:tags: [remove-output]
model.set_scenario_configs({
    "realistic": {
        "distribution_change_prob": 0.0,     # No distribution swaps
        "heteroskedasticity": 0.0,           # No heteroskedasticity
    },
})
```

### Combine custom scenarios with uploaded data

Custom scenarios work alongside `upload_data()`. The scenario perturbations (correlation noise, distribution swaps) are applied on top of the data-driven generation:

```python
import pandas as pd

data = pd.read_csv("pilot_data.csv")

model = MCPower("outcome = treatment + age + severity")
model.upload_data(data[["age", "severity"]])
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.50, age=0.20, severity=0.30")

model.set_scenario_configs({
    "realistic": {"heterogeneity": 0.30},
})

model.find_sample_size(
    target_test="treatment",
    from_size=50, to_size=300,
    scenarios=True,
)
```

---

## Configuration Reference

All available configuration keys with their defaults:

| Key | Type | Optimistic | Realistic | Doomer | Description |
|---|---|---|---|---|---|
| `heterogeneity` | float | 0.0 | 0.2 | 0.4 | SD of per-simulation effect size multiplier |
| `heteroskedasticity` | float | 0.0 | 0.15 | 0.35 | Correlation between predicted values and error SD |
| `correlation_noise_sd` | float | 0.0 | 0.15 | 0.30 | SD of noise added to predictor correlations |
| `distribution_change_prob` | float | 0.0 | 0.5 | 0.8 | Probability a variable's distribution is swapped |
| `new_distributions` | list | `["right_skewed", "left_skewed", "uniform"]` | same | same | Pool of replacement distributions |
| `residual_change_prob` | float | 0.0 | 0.5 | 0.8 | Probability residuals use non-normal distribution |
| `residual_df` | int | 10 | 8 | 5 | Degrees of freedom for non-normal residuals |
| `residual_dists` | list | `["heavy_tailed", "skewed"]` | same | same | Pool of non-normal residual distributions |
| `icc_noise_sd` | float | 0.0 | 0.15 | 0.30 | SD of multiplicative noise on ICC (mixed models only) |
| `random_effect_dist` | str | `"normal"` | `"heavy_tailed"` | `"heavy_tailed"` | Random effect distribution (mixed models only) |
| `random_effect_df` | int | 5 | 10 | 3 | DF for non-normal random effects (mixed models only) |

---

## Next Steps

- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- Conceptual guide to how scenarios work and when to use them
- **[Tutorial: Your First Power Analysis](first-analysis.md)** -- Basics of `find_power` with a simple scenario example
- **[Tutorial: Finding the Right Sample Size](sample-size.md)** -- Systematic sample size search
- **[Tutorial: Mixed Models - Random Intercepts](mixed-intercepts.md)** -- Cluster-based models where mixed model scenario keys apply
- **[API: set_scenario_configs](../api/scenarios.md)** -- Full parameter reference
- **[API Reference](../api/index.md)** -- All configuration methods
