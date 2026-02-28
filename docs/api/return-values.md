---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Return Values

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

When `return_results=True` is passed to `find_power()` or `find_sample_size()`, the method returns a Python dictionary containing all analysis results and model metadata. By default, both methods print results to stdout and return `None`.

---

## Enabling Return Values

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")

# Print + return
result = model.find_power(sample_size=100, return_results=True)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Return only (no console output)
result = model.find_power(sample_size=100, return_results=True, print_results=False)
```

## Result Structure: find_power()

```python
result = model.find_power(
    sample_size=200,
    target_test="all",
    return_results=True,
    print_results=False,
)
```

### result["results"]

| Key | Type | Description |
|---|---|---|
| `"individual_powers"` | `dict` | Power values per test. Keys are effect names (e.g., `"x1"`, `"overall"`), values are floats (percentages). |
| `"individual_powers_corrected"` | `dict` | Present **only** when a `correction` is used. Same structure as `individual_powers` but with corrected values. |
| `"combined_probabilities"` | `dict` | Combined probabilities per test. |
| `"combined_probabilities_corrected"` | `dict` | Present **only** when a `correction` is used. |
| `"cumulative_probabilities"` | `dict` | Cumulative probabilities per test. |
| `"cumulative_probabilities_corrected"` | `dict` | Present **only** when a `correction` is used. |
| `"n_simulations_used"` | `int` | Number of simulations that completed successfully. |
| `"n_simulations_failed"` | `int` | Number of simulations that failed (convergence failures). **Only present for mixed-effects models.** |

```python
# Access individual power values
power_x1 = result["results"]["individual_powers"]["x1"]          # e.g. 85.2
power_overall = result["results"]["individual_powers"]["overall"]  # e.g. 96.1

# With correction
result = model.find_power(sample_size=200, correction="bonferroni",
                           return_results=True, print_results=False)
corrected_x1 = result["results"]["individual_powers_corrected"]["x1"]  # e.g. 71.3
```

### result["model"]

| Key | Type | Description |
|---|---|---|
| `"data_formula"` | `str` | The formula used for data generation (e.g., `"y = x1 + x2"`). |
| `"test_formula"` | `str` | The formula used for testing (empty string if same as data formula). |
| `"sample_size"` | `int` | The sample size tested. |
| `"alpha"` | `float` | The significance level (e.g., `0.05`). |
| `"target_power"` | `float` | The target power level (e.g., `80.0`). |
| `"n_simulations"` | `int` | The number of simulations run (e.g., `1600`). |
| `"model_type"` | `str` | `"linear"` or `"mixed"`. |
| `"target_tests"` | `list[str]` | List of tests that were requested. |
| `"correction"` | `str` or `None` | The correction method used (e.g., `"bonferroni"`). |
| `"parallel"` | `bool` or `str` | The parallelization mode used. |

```python
print(result["model"]["data_formula"])    # "y = x1 + x2"
print(result["model"]["sample_size"])     # 200
print(result["model"]["alpha"])           # 0.05
print(result["model"]["n_simulations"])   # 1600
```

## Result Structure: find_sample_size()

```python
result = model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=300,
    return_results=True,
    print_results=False,
)
```

### result["results"]

| Key | Type | Description |
|---|---|---|
| `"sample_sizes_tested"` | `list[int]` | List of sample sizes evaluated in the search grid. |
| `"powers_by_test"` | `dict` | Power values at each sample size. Keys are effect names, values are lists of floats (one per sample size). |
| `"powers_by_test_corrected"` | `dict` | Present **only** when a `correction` is used. Same structure as `powers_by_test` but with corrected values. |
| `"first_achieved"` | `dict` | First sample size where target power is reached. Keys are effect names, values are ints (-1 if target not reached). |
| `"first_achieved_corrected"` | `dict` | Present **only** when a `correction` is used. Same structure as `first_achieved`. |

```python
first_n = result["results"]["first_achieved"]["treatment"]
if first_n > 0:
    print(f"Required sample size: {first_n}")
else:
    print("Target power not reached in the search range.")
```

## Examples

### Basic Power Check

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("outcome = treatment + motivation + age")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3, age=0.2")

result = model.find_power(
    sample_size=200,
    target_test="all",
    return_results=True,
    print_results=False,
)

# Iterate over all power values
for test_name, power in result["results"]["individual_powers"].items():
    status = "ok" if power >= 80 else "underpowered"
    print(f"{test_name}: {power:.1f}% ({status})")
```

### With Correction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
result = model.find_power(
    sample_size=200,
    target_test="treatment, motivation, age",
    correction="holm",
    return_results=True,
    print_results=False,
)

uncorrected = result["results"]["individual_powers"]
corrected = result["results"]["individual_powers_corrected"]

for name in uncorrected:
    raw = uncorrected[name]
    adj = corrected.get(name, "-")
    print(f"{name}: raw={raw:.1f}%, corrected={adj}")
```

### Finding Sample Size

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
result = model.find_sample_size(
    target_test="treatment",
    from_size=50,
    to_size=500,
    by=50,
    return_results=True,
    print_results=False,
)

n = result["results"]["first_achieved"]["treatment"]
print(f"Required N = {n}")
print(f"Model: {result['model']['data_formula']}")
print(f"Simulations: {result['model']['n_simulations']}")
```

## Notes

- The `individual_powers` dictionary always includes an `"overall"` key for the omnibus F-test (when `target_test="all"`).
- When `correction="tukey"` is used, non-contrast tests appear in `individual_powers_corrected` with the value `float("nan")`. The console formatter displays this as `"-"`, but the returned dictionary contains `NaN`. Use `math.isnan()` to check for these values programmatically.
- The `result["model"]` section reflects the **actual** parameters used (after defaults and overrides), not just what the user set.

## See Also

- [Progress Reporting](progress.md) -- Monitoring long-running analyses
- [Multiple Testing Corrections](../tutorials/multiple-testing.md) -- Corrected power output
- [Correction Methods](corrections.md) -- Available correction methods
