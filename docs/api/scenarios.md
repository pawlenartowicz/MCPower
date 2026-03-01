---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Scenario Analysis

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

Methods for configuring scenario analysis and convergence failure tolerance.

---

(set-scenario-configs)=
## set_scenario_configs()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_scenario_configs
```

### Default Scenarios

Scenario analysis runs the power simulation under multiple assumption-violation profiles:

- **Optimistic** -- The user's exact settings with no perturbations (all values zero/normal).
- **Realistic** -- Moderate assumption violations.
- **Doomer** -- Severe assumption violations.

### Configuration Keys -- General (All Model Types)

| Key | Type | Default (Realistic) | Default (Doomer) | Description |
|---|---|---|---|---|
| `heterogeneity` | `float` | `0.2` | `0.4` | Effect size heterogeneity. SD of per-simulation effect multiplier. |
| `heteroskedasticity` | `float` | `0.15` | `0.35` | Correlation between predicted values and error SD. |
| `correlation_noise_sd` | `float` | `0.15` | `0.30` | SD of noise added to predictor correlations. |
| `distribution_change_prob` | `float` | `0.5` | `0.8` | Probability that a variable's distribution is swapped. |
| `new_distributions` | `list` | `["right_skewed", "left_skewed", "uniform"]` | same | Replacement distributions for swapped variables. |

### Configuration Keys -- Residual Perturbation (All Model Types)

| Key | Type | Default (Realistic) | Default (Doomer) | Description |
|---|---|---|---|---|
| `residual_dists` | `list[str]` | `["heavy_tailed", "skewed"]` | same | Pool of non-normal distributions. Per-simulation, one is randomly picked. |
| `residual_change_prob` | `float` | `0.5` | `0.8` | Probability that residuals use a non-normal distribution per simulation. |
| `residual_df` | `int` | `8` | `5` | Degrees of freedom for heavy-tailed (t) or skewed (chi-squared) residuals. |

### Configuration Keys -- Mixed Model (Cluster Models Only)

These keys are only consumed when cluster specifications are present. They are ignored for OLS analyses.

| Key | Type | Default (Realistic) | Default (Doomer) | Description |
|---|---|---|---|---|
| `icc_noise_sd` | `float` | `0.15` | `0.30` | SD of multiplicative noise on the ICC. |
| `random_effect_dist` | `str` | `"heavy_tailed"` | `"heavy_tailed"` | Distribution of random effects (`"normal"`, `"heavy_tailed"`, or `"skewed"`). |
| `random_effect_df` | `int` | `10` | `3` | Degrees of freedom when `random_effect_dist` is not `"normal"`. |

### Examples

**Override realistic scenario parameters:**

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("outcome = treatment + motivation")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3")

model.set_scenario_configs({
    "realistic": {
        "heterogeneity": 0.3,
        "heteroskedasticity": 0.15,
    },
})

model.find_sample_size(
    target_test="treatment",
    from_size=50, to_size=300, by=30,
    scenarios=True,
)
```

**Override mixed model parameters:**

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("y ~ treatment + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5")

model.set_scenario_configs({
    "realistic": {
        "icc_noise_sd": 0.10,
        "random_effect_dist": "normal",
    },
    "doomer": {
        "icc_noise_sd": 0.40,
        "random_effect_df": 2,
        "residual_df": 3,
    },
})

model.find_power(sample_size=1000, scenarios=True)
```

**Add a custom scenario** -- custom scenarios inherit all default (optimistic) values:

```python
model.set_scenario_configs({
    "extreme": {
        "heterogeneity": 0.6,
        "heteroskedasticity": 0.4,
        "correlation_noise_sd": 0.5,
        "distribution_change_prob": 0.8,
    },
})

model.find_power(sample_size=200, scenarios=True)
# Runs: optimistic, realistic, doomer, AND extreme
```

**Run only specific scenarios:**

```python
model.find_power(sample_size=200, scenarios=["optimistic", "extreme"])
# Runs only: optimistic and extreme (skips realistic and doomer)
```

Unknown names raise a `ValueError` listing the available scenarios.

### See Also

- [Tutorial: Custom Scenarios](../tutorials/custom-scenarios.md) -- Step-by-step guide with field-specific examples
- [Scenario Analysis](../concepts/scenario-analysis.md) -- Conceptual guide to scenario analysis

---

(set-max-failed-simulations)=
## set_max_failed_simulations()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_max_failed_simulations
```

### Common Thresholds

| Threshold | Meaning | Use Case |
|---|---|---|
| `0.03` | 3% (default) | Random intercept models with adequate sample sizes |
| `0.10` | 10% | Random intercept models with smaller samples or higher ICC |
| `0.20` | 20% | Random slope models |
| `0.30` | 30% | Complex nested models or random slopes with small clusters |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Random intercept model -- default threshold is usually fine
model = MCPower("satisfaction ~ treatment + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5")
model.find_power(sample_size=1000)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Random slopes -- relax the threshold
model = MCPower("y ~ x1 + (1 + x1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20,
                   random_slopes=["x1"], slope_variance=0.1)
model.set_effects("x1=0.5")
model.set_max_failed_simulations(0.20)
model.find_power(sample_size=1000)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Nested design -- more tolerance needed
model = MCPower("score ~ treatment + (1|school/classroom)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.15, n_clusters=10)
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
model.set_effects("treatment=0.5")
model.set_max_failed_simulations(0.30)
model.find_power(sample_size=1500)
```

### When Failures Are Expected

Convergence failures are more common when:

- **Small cluster sizes** -- Fewer than 10 observations per cluster increases failure rates.
- **High ICC** -- Strong within-cluster correlation (> 0.4) makes estimation harder.
- **Random slopes** -- More parameters to estimate per cluster.
- **Nested effects** -- Multiple variance components to estimate simultaneously.
- **Scenario analysis (doomer)** -- Perturbed distributions and ICC noise amplify convergence difficulty.

### Diagnosing High Failure Rates

If you encounter the error "Too many failed simulations", consider:

1. **Increasing the sample size** -- More observations per cluster improves convergence.
2. **Increasing `n_clusters`** -- More clusters are statistically more informative.
3. **Reducing ICC** -- If plausible for your research context.
4. **Relaxing the threshold** -- As a last resort, if you understand the trade-off.

For standard OLS (linear regression) models, convergence failures do not occur. This method has no practical effect on OLS analyses.

### See Also

- [Performance & Backends](../concepts/performance.md) -- Runtime considerations
- [Mixed-Effects Models](../concepts/mixed-effects.md) -- Complete guide to mixed-effects support
