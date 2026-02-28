---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Correction Methods

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

Multiple testing corrections are **not** configured via a `set_*` method. Instead, the `correction` parameter is passed directly to `find_power()` or `find_sample_size()` at analysis time.

```python
model.find_power(sample_size=200, correction="bonferroni")
model.find_sample_size(from_size=50, to_size=400, correction="holm")
```

This design allows you to compare the same model with different corrections without reconfiguring:

```python
model.find_power(sample_size=200, correction=None)          # No correction
model.find_power(sample_size=200, correction="bonferroni")   # Bonferroni
model.find_power(sample_size=200, correction="fdr")          # FDR
```

---

## Available Corrections

| Value | Aliases | Type | Description |
|---|---|---|---|
| `"bonferroni"` | -- | FWER | Divides alpha by the number of tests. Most conservative. |
| `"holm"` | -- | FWER | Step-down procedure. Uniformly more powerful than Bonferroni. |
| `"fdr"` | `"benjamini-hochberg"`, `"bh"` | FDR | Controls expected proportion of false discoveries. Least conservative. |
| `"tukey"` | -- | FWER | Tukey HSD for pairwise factor comparisons. Only applies to post-hoc contrasts. |
| `None` | -- | -- | No correction (default). |

### Correction Types

- **FWER (Family-Wise Error Rate)** -- Controls the probability of making **any** false positive across all tests.
- **FDR (False Discovery Rate)** -- Controls the expected **proportion** of false positives among significant results. Less conservative, allows more discoveries.

## How Corrections Interact with Tests

### The Correction Family

The overall F-test is **never** included in the correction family. Corrections apply only to individual t-tests (and post-hoc contrasts).

```python
model.find_power(
    sample_size=200,
    target_test="all",
    correction="bonferroni",
)
# Family = individual t-tests only (overall F-test excluded)
```

### Standard Corrections (Bonferroni, Holm, FDR)

All individual t-tests and post-hoc comparisons form a **single correction family**:

```python
model.find_power(
    sample_size=200,
    target_test="age, group[1] vs group[2], group[2] vs group[3]",
    correction="bonferroni",
)
# Family size = 3 (age + 2 post-hoc comparisons)
# Effective alpha per test = 0.05 / 3 = 0.0167
```

### Tukey Correction

Tukey HSD **only applies to post-hoc pairwise contrasts**. Non-contrast tests show `"-"` in the corrected power column:

```python
model.find_power(
    sample_size=200,
    target_test="age, group[1] vs group[2], group[2] vs group[3]",
    correction="tukey",
)
# age             -> corrected power = "-"
# group[1] vs group[2] -> Tukey-corrected
# group[2] vs group[3] -> Tukey-corrected
```

### Summary

| Correction | Applies To | Non-contrast Tests |
|---|---|---|
| Bonferroni / Holm / FDR | All t-tests + post-hoc contrasts together | Corrected |
| Tukey | Post-hoc contrasts only | Show `"-"` |
| None | -- | Raw p-values |

## Examples

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("outcome = treatment + motivation + age")
model.set_simulations(400)
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, motivation=0.3, age=0.2")

# Bonferroni correction
model.find_power(
    sample_size=200,
    target_test="treatment, motivation, age",
    correction="bonferroni",
)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Find sample size with Holm correction
model.find_sample_size(
    target_test="treatment, motivation, age",
    from_size=100, to_size=500, by=45,
    correction="holm",
)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Post-hoc with Tukey
model = MCPower("outcome = group + age")
model.set_simulations(400)
model.set_variable_type("group=(factor,3)")
model.set_effects("group[2]=0.4, group[3]=0.6, age=0.2")
model.find_power(
    sample_size=200,
    target_test="group[2] vs group[3], age",
    correction="tukey",
)
```

## Technical Details

MCPower precomputes correction-adjusted critical t-values **before** running simulations, so there is no runtime overhead from using corrections.

- **Bonferroni:** Critical value raised to `t(1 - alpha/(2m), df)` where `m` = number of t-based tests.
- **Holm (step-down):** An array of critical values, one per rank position.
- **FDR (step-up):** An array of critical values based on rank.
- **Tukey:** Uses the studentized range distribution for pairwise comparisons within each factor.

## References

- Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, *56*(293), 52--64.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, *6*(2), 65--70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, *57*(1), 289--300.
- Tukey, J. W. (1953). *The problem of multiple comparisons*. Unpublished manuscript, Princeton University.

## See Also

- [Multiple Testing Corrections](../tutorials/multiple-testing.md) -- Detailed guide with output examples
- [ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md) -- Post-hoc pairwise comparisons
- [Return Values](return-values.md) -- Accessing corrected power programmatically
