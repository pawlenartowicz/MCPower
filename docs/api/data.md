---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Data & Clustering

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

Methods for uploading empirical data, defining factor levels, and configuring mixed-effects clustering.

---

(upload-data)=
## upload_data()

```{eval-rst}
.. automethod:: mcpower.MCPower.upload_data
```

### Accepted Formats

| Format | How Names Are Determined |
|---|---|
| **pandas DataFrame** | Column names used automatically |
| **dict** `{str: array-like}` | Dictionary keys used as variable names |
| **numpy ndarray** (2D) | Use `columns` parameter, or auto-named `column_1`, `column_2`, ... |
| **numpy ndarray** (1D) | Single variable; use `columns` for its name |
| **list** (1D or 2D) | Same as numpy array |

### Auto-Detection Rules

Variable types are inferred from the number of unique values in each column:

| Unique Values | Detected Type | Notes |
|---|---|---|
| 1 | Dropped | Constant column, warned and removed |
| 2 | Binary | Mapped to 0/1 |
| 3--6 | Factor | Expanded into dummy variables |
| 7+ | Continuous | Normalized to mean=0, sd=1 |
| String column, 2--20 unique | Factor | String values become level names |
| String column, >20 unique | **Error** | Too many levels for a factor |

### Correlation Modes

**`"strict"` (default):** Bootstraps whole rows from the uploaded data to preserve the exact multivariate relationships. Uploaded variables bypass the normal Cholesky generation pipeline entirely.

**`"partial"`:** Computes a correlation matrix from the uploaded continuous variables and merges it with any user-specified correlations (via `set_correlations()`). User-specified correlations take precedence.

**`"no"`:** No correlation information is extracted. Binary and factor columns use detected proportions. Continuous columns use lookup-table-based generation.

### Type Overrides

The `data_types` parameter accepts a dictionary to override auto-detection:

```python
# Simple override
model.upload_data(df, data_types={"hp": "continuous", "cyl": "factor"})

# Override with reference level
model.upload_data(df, data_types={"cyl": ("factor", 8)})
# cyl has values [4, 6, 8] -> reference is 8, dummies: cyl[4], cyl[6]

# String reference level
model.upload_data(df, data_types={"origin": ("factor", "USA")})
# origin has values ["Europe", "Japan", "USA"] -> reference is "USA"
```

### Examples

```python
import pandas as pd
from mcpower import MCPower

df = pd.read_csv("study_data.csv")

model = MCPower("mpg = hp + wt + cyl")
model.upload_data(df[["hp", "wt", "cyl"]])
# Auto-detected: hp=continuous, wt=continuous, cyl=factor (3 unique values)
# cyl dummies use original values: cyl[6], cyl[8] (reference: cyl[4])

model.set_effects("hp=0.3, wt=0.4, cyl[6]=0.2, cyl[8]=0.5")
model.find_power(sample_size=100)
```

### Notes

- **Minimum data size:** The uploaded data must have at least 25 observations. A warning is issued for fewer than 30.
- **Large sample warning:** If the requested `sample_size` exceeds 3x the uploaded data count, a warning is printed about potential extrapolation.
- **Column matching:** Uploaded columns are matched to formula predictors by name. Columns not in the formula are ignored. Formula predictors not in the data use standard generation.
- **String columns:** Columns containing string values are auto-detected as factors (if 2--20 unique values). The first value in sorted order becomes the reference level by default.

### See Also

- [Tutorial: Using Your Own Data](../tutorials/own-data.md) -- Step-by-step walkthrough with real data
- {ref}`set_variable_type() <set-variable-type>` -- Manual type specification (alternative to auto-detection)
- {ref}`set_factor_levels() <set-factor-levels>` -- Define named levels without data
- {ref}`set_correlations() <set-correlations>` -- Manual correlation specification

---

(set-factor-levels)=
## set_factor_levels()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_factor_levels
```

### String Format

**Single factor** -- the first listed level is the reference (omitted from dummies):

```python
model.set_factor_levels("group=control,drug_a,drug_b")
# Creates dummies: group[drug_a], group[drug_b]
# Reference level: control
```

**Multiple factors** -- separate with `;`:

```python
model.set_factor_levels("group=control,drug_a,drug_b; dose=low,medium,high")
# group dummies: group[drug_a], group[drug_b]   (reference: control)
# dose dummies:  dose[medium], dose[high]        (reference: low)
```

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = group + age")
model.set_simulations(400)
model.set_variable_type("group=(factor,3)")
model.set_factor_levels("group=placebo,low_dose,high_dose")
model.set_effects("group[low_dose]=0.3, group[high_dose]=0.6, age=0.2")
model.find_power(sample_size=120)
```

### Notes

- The variable must already exist in the formula.
- The variable should be declared as a factor (via `set_variable_type()`) before or after calling `set_factor_levels()`. MCPower applies settings in the correct order regardless of call sequence.
- If you upload data with `upload_data()` and `preserve_factor_level_names=True` (the default), level names are extracted automatically. In that case, `set_factor_levels()` is typically unnecessary.
- Level names cannot contain commas, semicolons, or equals signs.

### See Also

- {ref}`set_variable_type() <set-variable-type>` -- Declare factor variables
- {ref}`set_effects() <set-effects>` -- Set effects using named level bracket notation
- {ref}`upload_data() <upload-data>` -- Automatic named levels from empirical data
- [ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md) -- Factor comparisons and post-hoc tests
- [Variable Types](../concepts/variable-types.md) -- Factor variable concepts

---

(set-cluster)=
## set_cluster()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_cluster
```

### Usage Patterns

**Random intercept only** -- the simplest mixed-effects structure:

```python
model = MCPower("y ~ treatment + (1|school)")
model.set_cluster("school", ICC=0.2, n_clusters=20)
```

Or specify cluster size instead of count:

```python
model.set_cluster("school", ICC=0.2, cluster_size=25)
```

**Random slopes** -- each cluster has its own intercept and slope:

```python
model = MCPower("y ~ x1 + (1 + x1|school)")
model.set_cluster("school", ICC=0.2, n_clusters=20,
                   random_slopes=["x1"],
                   slope_variance=0.1,
                   slope_intercept_corr=0.3)
```

**Nested random effects** -- call `set_cluster()` twice, parent first:

```python
model = MCPower("y ~ treatment + (1|school/classroom)")

# Parent level
model.set_cluster("school", ICC=0.15, n_clusters=10)

# Child level (3 classrooms per school = 30 total classrooms)
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
```

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5, motivation=0.3")
model.find_power(sample_size=600)  # 600 / 20 = 30 obs per cluster
```

### Notes

- **ICC range:** 0 (no clustering) or 0.1--0.9 for numerical stability. Values outside 0.1--0.9 (except 0) are rejected because extreme ICCs cause convergence issues in mixed models.
- **Minimum cluster size:** At least 5 observations per cluster (enforced). A warning is issued if cluster size falls below 10.
- **Design effect:** Clustering reduces effective sample size by `1 + (cluster_size - 1) * ICC`. Higher ICC or larger clusters require more total observations for the same power.
- **Convergence failures:** Complex cluster structures may cause some simulations to fail. Use `model.set_max_failed_simulations(0.10)` to allow up to 10% failures (default is 3%).

### See Also

- [Mixed-Effects Models](../concepts/mixed-effects.md) -- Full conceptual guide to clustering, ICC, and design effects
- [Tutorial: Mixed-Effects Models](../tutorials/mixed-intercepts.md) -- Step-by-step mixed model analysis
- [MCPower() Constructor](constructor.md) -- Random-effect formula syntax
- {ref}`find_power() <find-power>` -- Running the analysis after configuring clusters
