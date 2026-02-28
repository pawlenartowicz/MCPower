---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Power Analysis with Your Own Data

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Goal

You have pilot data or a related dataset and want your power analysis to reflect real-world distributions instead of synthetic ones.

---

## Full Working Example

```python
import pandas as pd
from mcpower import MCPower

# ── Load your CSV ──────────────────────────────────────────────────
data = pd.read_csv("cars.csv")

# ── Define the model ───────────────────────────────────────────────
model = MCPower("mpg = hp + wt + cyl")

# ── Upload predictor columns ──────────────────────────────────────
model.upload_data(
    data[["hp", "wt", "cyl"]],
    preserve_correlation="strict",          # default — bootstrap whole rows
    preserve_factor_level_names=True,       # default — use original values as level names
)

# ── Set effect sizes (still required after upload) ────────────────
# cyl has values [4, 6, 8] → auto-detected as factor
# Reference level: 4 (first sorted value)
# Dummies: cyl[6] and cyl[8]
model.set_effects("hp=0.25, wt=0.40, cyl[6]=0.50, cyl[8]=0.80")

# ── Run the analysis ──────────────────────────────────────────────
model.find_power(sample_size=100, target_test="all")
```

---

## Step-by-Step Walkthrough

### 1. Load the data

```python
data = pd.read_csv("cars.csv")
```

Read your CSV into a pandas DataFrame. MCPower also accepts a plain dict of lists (see [Alternatives Without pandas](#alternatives-without-pandas) below).

### 2. Define the model

```python
model = MCPower("mpg = hp + wt + cyl")
```

Write the formula with the outcome on the left and predictors on the right, separated by `=` or `~`. Only **predictor columns** are uploaded; the outcome is always simulated.

### 3. Upload predictor data

```python
model.upload_data(
    data[["hp", "wt", "cyl"]],
    preserve_correlation="strict",
    preserve_factor_level_names=True,
)
```

Key points:

- **`upload_data()` returns `self`** -- it can be chained with other method calls.
- Pass only the predictor columns that appear in your formula.
- MCPower **auto-detects variable types** based on unique value counts:

| Unique Values | Detected Type |
|---|---|
| 1 | Dropped (constant) |
| 2 | Binary |
| 3--6 | Factor |
| 7+ | Continuous |

- **String columns** with 2--20 unique values are automatically detected as factors.

In this example, `hp` and `wt` have many unique values (continuous), while `cyl` has three unique values `[4, 6, 8]` (factor).

### 4. Set effect sizes

```python
model.set_effects("hp=0.25, wt=0.40, cyl[6]=0.50, cyl[8]=0.80")
```

Uploading data provides distributions, **not** effect sizes. You must still specify how strongly each predictor relates to the outcome.

With `preserve_factor_level_names=True` (the default), factor dummies use the **original data values** as level names:

- `cyl[6]` -- comparing cylinders=6 to the reference (cylinders=4)
- `cyl[8]` -- comparing cylinders=8 to the reference (cylinders=4)

The reference level is the first sorted unique value (here, 4).

### 5. Run the analysis

```python
model.find_power(sample_size=100, target_test="all")
```

`target_test="all"` reports power for every individual predictor plus the overall F-test.

---

## Output Interpretation

```
================================================================================
MONTE CARLO POWER ANALYSIS RESULTS
================================================================================

Power Analysis Results (N=100):
Test                                     Power    Target   Status
-------------------------------------------------------------------
overall                                  100.0    80       ✓
hp                                       24.7     80       ✗
wt                                       64.8     80       ✗
cyl[6]                                   32.8     80       ✗
cyl[8]                                   34.0     80       ✗

Result: 1/5 tests achieved target power
```

- **Power** -- percentage of 1,600 simulations where the test reached significance.
- **Target** -- your target power level (default: 80%).
- **Status** -- `✓` if power meets the target, `✗` if it falls short.
- In this example, `hp`, `wt`, `cyl[6]`, and `cyl[8]` all need a larger sample size to reach 80% power.

---

## Common Variations

### Alternatives Without pandas

Pass a dict of lists instead of a DataFrame:

```python
from mcpower import MCPower

data = {
    "hp": [110, 93, 175, 105, 245],
    "wt": [2.62, 2.32, 3.21, 3.15, 3.44],
    "cyl": [6, 4, 8, 6, 8],
}

model = MCPower("mpg = hp + wt + cyl")
model.upload_data(data)
model.set_effects("hp=0.25, wt=0.40, cyl[6]=0.50, cyl[8]=0.80")
model.find_power(sample_size=100)
```

### String Columns as Factors

String columns are automatically detected as factors when they have 2--20 unique values:

```python
data = pd.read_csv("cars_with_origin.csv")

model = MCPower("mpg = origin + hp")
model.upload_data(data[["origin", "hp"]])
# origin has values ["Europe", "Japan", "USA"] → factor
# Reference: "Europe" (first alphabetically)
# Dummies: origin[Japan], origin[USA]
model.set_effects("origin[Japan]=0.20, origin[USA]=0.50, hp=0.25")
model.find_power(sample_size=120)
```

### Correlation Preservation Modes

The `preserve_correlation` parameter controls how MCPower handles relationships between uploaded variables:

```python
# "strict" (default) — bootstrap whole rows, preserving exact relationships
model.upload_data(data, preserve_correlation="strict")

# "partial" — compute correlations from data, allow manual overrides
model.upload_data(data, preserve_correlation="partial")
model.set_correlations("corr(hp, wt)=0.6")  # override one pair

# "no" — ignore correlations from data entirely
model.upload_data(data, preserve_correlation="no")
model.set_correlations("corr(hp, wt)=0.3")  # set all manually
```

| Mode | Correlation Source | Best For |
|---|---|---|
| `"strict"` (default) | Bootstrapped rows | Most realistic simulation |
| `"partial"` | Data + manual overrides | Empirical baseline with adjustments |
| `"no"` | Manual only | Full manual control |

### Overriding Auto-Detection with `data_types`

If auto-detection classifies a variable incorrectly, override it:

```python
# "rating" has 5 unique values → auto-detected as factor
# Override to treat as continuous
model.upload_data(
    data[["group", "score", "rating"]],
    data_types={"rating": "continuous"},
)
```

You can also select the reference level for a factor:

```python
# Numeric reference level
model.upload_data(
    data[["hp", "wt", "cyl"]],
    data_types={"cyl": ("factor", 8)},  # cyl=8 becomes reference
)
# Dummies are now: cyl[4], cyl[6]

# String reference level
model.upload_data(
    data[["origin", "hp"]],
    data_types={"origin": ("factor", "USA")},  # USA becomes reference
)
# Dummies are now: origin[Europe], origin[Japan]
```

### Named Factor Levels Without Data

If you do not have data but want meaningful level names instead of integer indices, use `set_factor_levels()`:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("outcome = group + age")
model.set_simulations(400)
model.set_variable_type("group=(factor,3)")
model.set_factor_levels("group=placebo,low_dose,high_dose")

# Now effects use named levels:
model.set_effects("group[low_dose]=0.50, group[high_dose]=0.80, age=0.25")
model.find_power(sample_size=150)
```

The first listed level (`placebo`) becomes the reference. This is purely a labeling feature -- it does not change the statistical computation.

### Mixing Uploaded and Synthetic Variables

Variables in the formula that are **not** in the uploaded data are generated synthetically:

```python
model = MCPower("outcome = hp + wt + treatment")
model.upload_data(data[["hp", "wt"]])               # empirical distributions
model.set_variable_type("treatment=binary")           # synthetic variable
model.set_effects("hp=0.25, wt=0.40, treatment=0.50")
model.find_power(sample_size=100)
```

**Note:** In `"strict"` mode, cross-correlations between uploaded and non-uploaded variables are set to zero with a warning.

### Integer-Indexed Dummies

Set `preserve_factor_level_names=False` to use integer-indexed dummies instead of data values:

```python
model.upload_data(data[["hp", "wt", "cyl"]], preserve_factor_level_names=False)
# cyl dummies are now: cyl[2], cyl[3] instead of cyl[6], cyl[8]
model.set_effects("hp=0.25, wt=0.40, cyl[2]=0.50, cyl[3]=0.80")
```

The default (`True`) is recommended — it produces clearer, more readable output.

---

## Next Steps

- **[Tutorial: CSV Preparation](csv-preparation.md)** -- formatting your CSV file correctly
- **[Effect Sizes](../concepts/effect-sizes.md)** -- choosing appropriate effect sizes
- **[Variable Types](../concepts/variable-types.md)** -- all available variable types
- **[Correlations](../concepts/correlations.md)** -- setting predictor correlations
- **[API Reference](../api/index.md)** -- full `upload_data()` parameter documentation
