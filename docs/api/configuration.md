---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Configuration

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

Methods for configuring effect sizes, variable types, correlations, and simulation parameters.

---

(set-effects)=
## set_effects()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_effects
```

### String Format

Comma-separated `name=value` pairs:

```python
model.set_effects("x1=0.5, x2=0.3")
```

**Interaction effects** -- use `:` notation:

```python
model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
```

**Factor variables** -- assign different effects per level using bracket notation:

```python
# Integer-indexed levels (no uploaded data or named levels)
model.set_effects("group[2]=0.4, group[3]=0.6")

# Named levels (after set_factor_levels or upload_data)
model.set_effects("group[drug_a]=0.4, group[drug_b]=0.6")
```

### Updating Effects

After running an analysis, a new `set_effects()` updates (merges with) the previously applied effects:

```python
model.set_effects("x1=0.5, x2=0.3")
model.find_power(sample_size=100)
model.set_effects("x2=0.4")  # x1 remains 0.5, x2 is now 0.4
model.find_power(sample_size=100)  # uses x1=0.5, x2=0.4
```

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = treatment + motivation + treatment:motivation")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3, treatment:motivation=0.2")
model.find_power(sample_size=100)
```

### Notes

- Effect sizes are **standardized** -- they represent the change in outcome (in SDs) per 1 SD change in the predictor.
- For binary predictors, the effect size represents the difference between the two groups in standard deviation units (equivalent to Cohen's d).
- For factor variables, each dummy's effect size represents the difference between that level and the reference level.
- A common guideline: 0.2 = small, 0.5 = medium, 0.8 = large (Cohen's conventions).
- The method raises `TypeError` if the argument is not a string and `ValueError` if it is empty.

### See Also

- [Effect Sizes](../concepts/effect-sizes.md) -- Understanding and choosing effect sizes
- {ref}`set_factor_levels() <set-factor-levels>` -- Define named factor levels for bracket notation
- {ref}`set_variable_type() <set-variable-type>` -- Declare binary/factor variables before setting effects
- [Tutorial: Your First Power Analysis](../tutorials/first-analysis.md) -- Complete walkthrough including effect sizes

---

(set-variable-type)=
## set_variable_type()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_variable_type
```

### Supported Types

| Type String | Description | Generated Distribution |
|---|---|---|
| `normal` | Standard normal (default) | N(0, 1) |
| `binary` | Binary variable with 50/50 split | Bernoulli(0.5) |
| `(binary, p)` | Binary with custom proportion | Bernoulli(p), where 0 < p < 1 |
| `(factor, k)` | Factor with k levels, equal proportions | k-1 dummy variables |
| `(factor, p1, p2, ..., pk)` | Factor with custom level proportions | k-1 dummies; proportions are normalized to sum to 1 |
| `right_skewed` | Right-skewed (heavy right tail) | Chi-squared-like transform |
| `left_skewed` | Left-skewed (heavy left tail) | Mirrored right-skew |
| `high_kurtosis` | Heavy-tailed (leptokurtic) | t-distribution (df=3) |
| `uniform` | Uniform distribution | U(0, 1) transformed |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Basic type declarations
model = MCPower("y = treatment + condition + income")
model.set_simulations(400)
model.set_variable_type("treatment=binary, condition=(factor,3), income=right_skewed")
model.set_effects("treatment=0.5, condition[2]=0.3, condition[3]=0.4, income=0.2")
model.find_power(sample_size=150)
```

**Binary with custom proportion:**

```python
model.set_variable_type("treatment=(binary,0.3)")  # 30% in treatment group
```

**Factor with equal proportions:**

```python
model.set_variable_type("condition=(factor,3)")  # 3 levels, ~33% each
```

**Factor with custom proportions:**

```python
# 3 levels with proportions 20%, 50%, 30% -- must sum to 1.0
model.set_variable_type("group=(factor,0.2,0.5,0.3)")
```

**Updating types** -- calling again updates existing entries without clearing others:

```python
model.set_variable_type("x1=binary, x2=right_skewed")
model.set_variable_type("x2=normal")  # x1 remains binary, x2 is now normal
```

### Notes

- Factor variables create k-1 dummy variables (level 1 is the reference by default). After declaring a factor, use bracket notation in `set_effects()` to assign effects to each dummy.
- When `upload_data()` is used, variable types are auto-detected and typically do not need to be set manually. Use `set_variable_type()` to override auto-detection.
- Validation of types and proportions happens when `find_power()` or `find_sample_size()` is called.

### See Also

- [Variable Types](../concepts/variable-types.md) -- Full guide to variable types and distributions
- {ref}`set_effects() <set-effects>` -- Setting effect sizes for typed variables
- {ref}`set_factor_levels() <set-factor-levels>` -- Define named factor levels
- {ref}`upload_data() <upload-data>` -- Automatic type detection from empirical data

---

(set-correlations)=
## set_correlations()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_correlations
```

### Input Formats

**String format -- full syntax:**

```python
model.set_correlations("corr(x1, x2)=0.3, corr(x1, x3)=-0.2")
```

**String format -- shorthand** (the `corr()` wrapper is optional):

```python
model.set_correlations("(x1, x2)=0.3, (x1, x3)=-0.2")
```

**NumPy matrix** -- dimensions must match the number of non-factor predictors, in formula order:

```python
import numpy as np

# For a model with predictors x1, x2, x3 (all continuous)
model.set_correlations(np.array([
    [1.0, 0.3, -0.2],
    [0.3, 1.0,  0.1],
    [-0.2, 0.1, 1.0],
]))
```

### Correlation Values

- Valid range: -1 to 1 (exclusive of exact -1 and 1 for off-diagonal entries)
- Diagonal entries must be 1.0 (for matrix input)
- The matrix must be symmetric
- The matrix must be **positive semi-definite** (PSD) -- MCPower validates this and raises an error if the matrix is not PSD

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2 + x3")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x3=0.2")
model.set_correlations("(x1, x2)=0.4, (x2, x3)=0.2")
model.find_power(sample_size=100)
```

### Notes

- **Factor variables cannot be correlated.** Correlations are defined only between continuous and binary predictors.
- Unspecified pairs default to zero correlation (independence).
- When using `upload_data()` with `preserve_correlation="partial"`, correlations are computed from the data and merged with any user-specified values. With `preserve_correlation="strict"` (the default), the full row-bootstrap approach preserves the empirical correlation structure automatically.

### See Also

- [Correlations](../concepts/correlations.md) -- Full guide to predictor correlations
- {ref}`upload_data() <upload-data>` -- Automatic correlation preservation from empirical data
- [Tutorial: Your First Power Analysis](../tutorials/first-analysis.md) -- Adding correlations to a model

---

(set-alpha)=
## set_alpha()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_alpha
```

### Common Alpha Levels

| Alpha | Use Case |
|---|---|
| 0.05 | Standard threshold (default) |
| 0.01 | Stricter threshold, common in some fields |
| 0.005 | Proposed "redefine statistical significance" threshold |
| 0.10 | Exploratory research, pilot studies |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Use stricter significance threshold
model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")
model.set_alpha(0.01)
model.find_power(sample_size=100)
```

```{code-cell} ipython3
:tags: [remove-output]
# Chained
model = (
    MCPower("y = x1 + x2")
    .set_effects("x1=0.5, x2=0.3")
    .set_alpha(0.01)
)
```

---

(set-power)=
## set_power()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_power
```

### Common Power Targets

| Power | Use Case |
|---|---|
| 80% | Standard target (default). Accepted in most fields. |
| 90% | Higher confidence. Common for clinical trials and well-funded studies. |
| 95% | Very conservative. Requires substantially larger samples. |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Require 90% power instead of the default 80%
model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")
model.set_power(90)
model.find_sample_size(from_size=50, to_size=300, by=30)
```

---

(set-seed)=
## set_seed()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_seed
```

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")

# Reproducible results
model.set_seed(42)
model.find_power(sample_size=100)  # Always produces the same output
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Random seeding (different results each run)
model.set_seed(None)
model.find_power(sample_size=100)
```

### Notes

- The C++ backend uses `std::mt19937`, while Python uses `numpy.random`. The same seed produces **different** random sequences across backends, but statistical properties (power estimates) are comparable.
- The default seed is `2137`. Change it if you want a different reproducible sequence.

---

(set-simulations)=
## set_simulations()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_simulations
```

### Default Simulation Counts

| Model Type | Default Count |
|---|---|
| OLS (linear regression) | 1,600 |
| Mixed-effects models | 800 |

Mixed-effects models use fewer simulations by default because each simulation is more computationally expensive (LME fitting vs. OLS).

### Precision vs. Runtime

| Simulations | Approximate SE of Power Estimate | Use Case |
|---|---|---|
| 400 | ~2.5% | Quick exploration |
| 800 | ~1.8% | Mixed-model default |
| 1,600 | ~1.2% | OLS default; good for most analyses |
| 5,000 | ~0.7% | High-precision estimates |
| 10,000 | ~0.5% | Publication-quality precision |

The standard error of a power estimate at 80% power is approximately `sqrt(0.8 * 0.2 / n_sims)`.

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Set both OLS and mixed to the same count
model = MCPower("y = x1 + x2")
model.set_simulations(3200)
model.set_effects("x1=0.5, x2=0.3")
model.find_power(sample_size=100)
```

```{code-cell} ipython3
:tags: [remove-output]
# Set OLS and mixed independently
model = MCPower("satisfaction ~ treatment + (1|school)")
model.set_simulations(2000, model_type="linear")
model.set_simulations(1000, model_type="mixed")

# Method chaining
model = (
    MCPower("y = x1 + x2")
    .set_simulations(3200)
    .set_effects("x1=0.5, x2=0.3")
)
```

---

(set-parallel)=
## set_parallel()

```{eval-rst}
.. automethod:: mcpower.MCPower.set_parallel
```

### Parallelization Modes

| Value | Behavior |
|---|---|
| `True` | Parallel processing for all analyses (OLS and mixed). |
| `False` | Sequential processing only. |
| `"mixedmodels"` | Parallel only for mixed-model analyses; OLS stays sequential. **(Default)** |

### When to Use Each Mode

| Scenario | Recommended Mode | Reasoning |
|---|---|---|
| OLS with default 1,600 sims | `"mixedmodels"` (default) | C++ backend is fast enough; parallel overhead not worthwhile. |
| OLS with 5,000+ sims | `True` | High simulation count justifies parallelization overhead. |
| Mixed models | `"mixedmodels"` (default) | LME fitting is expensive; parallel processing helps substantially. |
| Debugging / profiling | `False` | Sequential execution is easier to reason about and profile. |
| Resource-constrained environment | `False` or low `n_cores` | Avoid saturating shared machines. |

### Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# Default: parallel for mixed models only
model = MCPower("y ~ treatment + (1|school)")
model.set_simulations(400)
model.set_cluster("school", ICC=0.2, n_clusters=20)
model.set_effects("treatment=0.5")
model.find_power(sample_size=1000)  # Runs in parallel automatically
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Force parallel for OLS (useful with very high simulation counts)
model = MCPower("y = x1 + x2 + x3 + x4 + x5")
model.set_effects("x1=0.5, x2=0.3, x3=0.2, x4=0.1, x5=0.4")
model.set_simulations(10000)
model.set_parallel(True, n_cores=4)
model.find_power(sample_size=200)
```

```{code-cell} ipython3
:tags: [remove-output]
# Disable parallelization entirely
model.set_parallel(False)
```

### Notes on n_cores

- When `n_cores` is `None`, MCPower uses half the available CPU cores (`cpu_count // 2`).
- Setting `n_cores=1` is equivalent to `enabled=False`.
- Using more cores than physically available provides no benefit and may hurt performance due to context-switching overhead.
- Requires `joblib` to be installed. If `joblib` is not available, MCPower falls back to sequential processing with a warning.

### See Also

- [Performance & Backends](../concepts/performance.md) -- Runtime considerations
- [Mixed-Effects Models](../concepts/mixed-effects.md) -- Complete guide to mixed-effects support
