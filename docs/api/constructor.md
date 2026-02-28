---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# MCPower() Constructor

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
```

```{eval-rst}
.. autoclass:: mcpower.MCPower
   :no-members:
```

---

## Formula Syntax

Both `=` and `~` are accepted as separators:

```python
MCPower("y = x1 + x2")   # equals sign
MCPower("y ~ x1 + x2")   # tilde (R-style)
```

### Formula Patterns

| Pattern | Meaning | Example |
|---|---|---|
| `y = x1 + x2` | Two main effects | Additive model with two predictors |
| `y = x1 * x2` | Main effects + interaction | Expands to `x1 + x2 + x1:x2` |
| `y = x1 + x2 + x1:x2` | Explicit interaction | Same as above, specified manually |
| `y ~ x + (1\|school)` | Random intercept | Mixed model with per-school intercepts |
| `y ~ x + (1+x\|school)` | Random intercept + slope | Mixed model with per-school intercepts and slopes for `x` |
| `y ~ x + (1\|school/classroom)` | Nested random effects | Classrooms nested within schools |

### Naming Rules

- Variable names can contain letters, digits, and underscores
- Must start with a letter or underscore (not a digit)
- Case-sensitive: `Treatment` and `treatment` are different variables
- Unicode letters are supported

## Examples

```{code-cell} ipython3
:tags: [remove-output]
from mcpower import MCPower

# Simple two-predictor model
model = MCPower("score = treatment + age")

# Full factorial with interaction
model = MCPower("y = x1 * x2")

# Mixed-effects model with random intercepts
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
```

## Defaults After Construction

| Attribute | Default Value |
|---|---|
| `seed` | `2137` |
| `alpha` | `0.05` |
| `power` | `80.0` (percent) |
| `n_simulations` | `1600` (OLS) |
| `n_simulations_mixed_model` | `800` |
| `parallel` | `"mixedmodels"` |
| `max_failed_simulations` | `0.03` (3%) |

## See Also

- [Model Specification](../model-specification.md) -- Full formula syntax reference
- [Tutorial: Your First Power Analysis](../tutorials/first-analysis.md) -- Step-by-step walkthrough
- [Quick Start](../getting-started/quickstart.md) -- Get running in 2 minutes
- [Mixed-Effects Models](../concepts/mixed-effects.md) -- Random-effect formula syntax
