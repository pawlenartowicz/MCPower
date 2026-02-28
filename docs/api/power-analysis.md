---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Power Analysis

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

The two main analysis methods: estimate power at a fixed sample size, or search for the minimum sample size that achieves a target power level.

---

(find-power)=
## find_power()

```{eval-rst}
.. automethod:: mcpower.MCPower.find_power
```

### Target Test Options

| Value | Meaning |
|---|---|
| `"all"` | Overall F-test plus all individual fixed effects (no post-hoc contrasts) |
| `"overall"` | Overall model F-test only |
| `"x1"` | A specific predictor by name |
| `"x1, x2"` | Comma-separated list of specific tests |
| `"group[1] vs group[2]"` | Post-hoc pairwise comparison between two factor levels |
| `"all-posthoc"` | All pairwise contrasts for every factor variable |
| `"-overall"` | Exclude a test (prefix with `-`); use with keywords like `"all, -overall"` |
| `"all, all-posthoc, -overall"` | Combine keywords and exclusions |

Duplicate tests in the resolved list raise a `ValueError`.

### Correction Options

| Value | Method |
|---|---|
| `None` | No correction |
| `"bonferroni"` | Bonferroni correction (conservative) |
| `"holm"` | Holm-Bonferroni step-down (less conservative than Bonferroni) |
| `"fdr"` or `"benjamini-hochberg"` | Benjamini-Hochberg false discovery rate |
| `"tukey"` | Tukey HSD (requires at least one post-hoc contrast in `target_test`) |

### Progress Callback

| Value | Behavior |
|---|---|
| `None` (default) + `print_results=True` | Automatic `PrintReporter` on stderr |
| `None` + `print_results=False` | Silent (no progress output) |
| `callable(current: int, total: int)` | Custom callback invoked periodically |
| `False` | Explicitly disable all progress reporting |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2 + x1:x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")

# Basic usage (prints results to stdout)
model.find_power(sample_size=100)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Programmatic access with correction
result = model.find_power(
    sample_size=200,
    target_test="x1, x2",
    correction="bonferroni",
    return_results=True,
    print_results=False,
)
```

### Test Formula Example

Generate data with a full model but test with a reduced model to evaluate model misspecification:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("y = x1 + x2 + x3")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x3=0.2")

# Full model (default)
model.find_power(100)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Reduced model (omit x3 from analysis)
result = model.find_power(100, test_formula="y = x1 + x2",
                          return_results=True, print_results=False)
# result contains power for x1 and x2 only
```

See [Tutorial: Using test_formula](../tutorials/test-formula.md) for more examples including mixed-model cross-testing.

### See Also

- {ref}`find_sample_size() <find-sample-size>` -- Find the minimum sample size for target power
- [Tutorial: Your First Power Analysis](../tutorials/first-analysis.md) -- Step-by-step walkthrough
- [Multiple Testing Corrections](../tutorials/multiple-testing.md) -- When and how to apply corrections
- [ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md) -- Post-hoc pairwise comparisons
- [Scenario Analysis](../concepts/scenario-analysis.md) -- Robustness testing under assumption violations

---

(find-sample-size)=
## find_sample_size()

```{eval-rst}
.. automethod:: mcpower.MCPower.find_sample_size
```

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = treatment + age")
model.set_simulations(400)
model.set_effects("treatment=0.4, age=0.2")
model.set_variable_type("treatment=binary")

# Search from 30 to 300 in steps of 30
model.find_sample_size(from_size=30, to_size=300, by=30)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Programmatic access
result = model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=500,
    by=50,
    return_results=True,
    print_results=False,
)
```

### Test Formula Example

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("y = x1 + x2 + x3")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x3=0.2")

# Full model (default)
model.find_sample_size(target_test="x1, x2, x3")
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Reduced model (omit x3 from analysis)
result = model.find_sample_size(target_test="x1, x2",
                                test_formula="y = x1 + x2",
                                return_results=True, print_results=False)
# result contains sample sizes for x1 and x2 only
```

See [Tutorial: Using test_formula](../tutorials/test-formula.md) for more examples including mixed-model cross-testing.

### Notes

- The target power level is set via `model.set_power()` (default: 80%).
- If no sample size in the range achieves target power, the results indicate this -- consider widening the range or increasing effect sizes.
- For mixed models, each sample size refers to the **total** number of observations across all clusters.
- Progress reporting spans all sample sizes multiplied by the number of simulations, so the total count is larger than a single `find_power()` call.

### See Also

- {ref}`find_power() <find-power>` -- Estimate power at a fixed sample size
- [Tutorial: Finding Sample Size](../tutorials/sample-size.md) -- Step-by-step sample size determination
- [Multiple Testing Corrections](../tutorials/multiple-testing.md) -- Correction methods explained
- [Scenario Analysis](../concepts/scenario-analysis.md) -- Robustness testing under assumption violations
