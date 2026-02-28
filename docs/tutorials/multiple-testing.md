---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Tutorial: Controlling for Multiple Testing

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Goal

You are testing multiple hypotheses and need to control the false positive rate so your power analysis reflects the correction you will use in your actual study.

---

## Full Working Example

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

# ── Medical study with 5 biomarkers ──────────────────────────────
# Some biomarkers have real effects, others are null (effect = 0)
model = MCPower("outcome = biomarker1 + biomarker2 + biomarker3 + biomarker4 + biomarker5")
model.set_simulations(400)

model.set_effects(
    "biomarker1=0.40, "    # large real effect
    "biomarker2=0.25, "    # medium real effect
    "biomarker3=0.00, "    # null — no true effect
    "biomarker4=0.10, "    # small real effect
    "biomarker5=0.00"      # null — no true effect
)

# ── Check power with Bonferroni correction ────────────────────────
model.find_power(
    sample_size=200,
    target_test="biomarker1, biomarker2, biomarker3, biomarker4, biomarker5",
    correction="bonferroni",
)
```

---

## Step-by-Step Walkthrough

### 1. Define the model

```{code-cell} ipython3
:tags: [remove-output]
model = MCPower("outcome = biomarker1 + biomarker2 + biomarker3 + biomarker4 + biomarker5")
model.set_simulations(400)
```

A study measuring five biomarkers as potential predictors of a health outcome.

### 2. Set effect sizes, including nulls

```{code-cell} ipython3
:tags: [remove-output]
model.set_effects(
    "biomarker1=0.40, "
    "biomarker2=0.25, "
    "biomarker3=0.00, "
    "biomarker4=0.10, "
    "biomarker5=0.00"
)
```

Setting an effect to `0.00` models a predictor that has **no real relationship** with the outcome. This is critical for understanding false positive rates -- with a correction, you expect these null effects to rarely reach significance.

### 3. Apply a correction

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(
    sample_size=200,
    target_test="biomarker1, biomarker2, biomarker3, biomarker4, biomarker5",
    correction="bonferroni",
)
```

The `correction` parameter adjusts significance thresholds to account for testing five effects simultaneously. Without correction, the chance of at least one false positive across 5 tests is about 23%.

---

## Output Interpretation

```
================================================================================
MONTE CARLO POWER ANALYSIS RESULTS
================================================================================
Multiple comparison correction: bonferroni

Power Analysis Results (N=200):
Test                                     Power    Target   Status
-------------------------------------------------------------------
biomarker1                               99.9     80       ✓
biomarker2                               92.9     80       ✓
biomarker3                               5.1      80       ✗
biomarker4                               29.1     80       ✗
biomarker5                               5.2      80       ✗

With bonferroni correction:
Test                                     Power    Target   Status
-------------------------------------------------------------------
biomarker1                               99.9     80       ✓
biomarker2                               81.2     80       ✓
biomarker3                               0.4      80       ✗
biomarker4                               13.4     80       ✗
biomarker5                               1.1      80       ✗

Result: 2/5 tests achieved target power
```

- The output now shows **two separate tables**: the first shows uncorrected power, the second shows power after Bonferroni adjustment. Corrected power is always lower than or equal to uncorrected power.
- **Status** -- `✓` if power meets the target, `✗` if it falls short. The final result is based on the **corrected** table.
- **biomarker3 and biomarker5** (null effects) -- the corrected false positive rates are 0.4% and 1.1%, well below the 5% threshold. The correction is working.
- **biomarker2** -- a medium effect needs a larger sample to survive Bonferroni correction.
- **biomarker4** -- a small effect has very low power after correction.

---

## Common Variations

### Correction methods compared

MCPower supports four correction methods (Bonferroni, Holm, FDR/Benjamini-Hochberg, and Tukey):

```python
# Bonferroni — most conservative FWER control
model.find_power(sample_size=200, target_test="all", correction="bonferroni")

# Holm — step-down FWER control (always >= Bonferroni power)
model.find_power(sample_size=200, target_test="all", correction="holm")

# FDR / Benjamini-Hochberg — controls false discovery rate (least conservative)
model.find_power(sample_size=200, target_test="all", correction="fdr")
# Aliases: "benjamini-hochberg" or "bh" also work

# Tukey HSD — for post-hoc pairwise factor comparisons only
model.find_power(
    sample_size=200,
    target_test="group[1] vs group[2], group[1] vs group[3], group[2] vs group[3]",
    correction="tukey",
)
```

### When to use which correction

| Situation | Recommended | Reasoning |
|---|---|---|
| Pre-registered confirmatory study | **Holm** or **Bonferroni** | Strict FWER control expected by reviewers |
| Few planned comparisons (2--3) | **Holm** | Controls FWER; more powerful than Bonferroni |
| Many comparisons (5+) in exploratory study | **FDR** | Less conservative; allows more discoveries |
| Strict error control required | **Bonferroni** | Simplest and most conservative |
| All pairwise comparisons within a factor | **Tukey** | Purpose-built for this case |
| Mixed regular + post-hoc tests | **Holm** | Corrects all tests together uniformly |

**General recommendation:** If unsure, use **Holm**. It controls the family-wise error rate and is always at least as powerful as Bonferroni.

### Focused testing strategy

Instead of testing all 5 biomarkers, focus on the ones you care most about. Fewer tests means less power loss from correction:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Only test the biomarkers you have specific hypotheses about
model.find_power(
    sample_size=200,
    target_test="biomarker1, biomarker2",
    correction="bonferroni",
)
# Family size = 2 instead of 5 → much less power loss
```

This is a legitimate strategy when you have **pre-registered hypotheses** for specific predictors. The other predictors are still in the model as covariates, but you only correct for the tests you report.

### Exploratory vs. confirmatory workflow

A common strategy is to run two analyses: an exploratory sweep with FDR to identify promising effects, then a confirmatory analysis with stricter correction:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# ── Step 1: Exploratory — which biomarkers are worth pursuing? ────
model.find_power(
    sample_size=200,
    target_test="biomarker1, biomarker2, biomarker3, biomarker4, biomarker5",
    correction="fdr",
)

# ── Step 2: Confirmatory — plan a study targeting the best candidates ──
model.find_sample_size(
    target_test="biomarker1, biomarker2",   # only the promising ones
    from_size=100,
    to_size=500,
    by=45,
    correction="holm",
)
```

The exploratory phase uses FDR (more lenient) to cast a wide net. The confirmatory phase uses Holm (strict FWER) on a focused set of pre-registered hypotheses.

### Finding required sample size with correction

Search for the sample size that achieves 80% **corrected** power:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="biomarker1, biomarker2, biomarker4",
    from_size=50,
    to_size=600,
    by=62,
    correction="holm",
)
```

The result accounts for the correction -- you need a larger sample than without correction.

### Combining corrections with scenario analysis

Test robustness under realistic assumptions:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_sample_size(
    target_test="biomarker1, biomarker2",
    from_size=100,
    to_size=500,
    by=45,
    correction="holm",
    scenarios=True,
)
```

The scenario analysis (Optimistic / Realistic / Doomer) is applied **on top of** the correction, giving you the most conservative planning estimate.

### Corrections with post-hoc comparisons

When combining standard tests and post-hoc comparisons, the correction method determines what gets corrected:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model = MCPower("outcome = treatment + covariate")
model.set_simulations(400)
model.set_variable_type("treatment=(factor,3)")
model.set_effects("treatment[2]=0.50, treatment[3]=0.80, covariate=0.25")

# Bonferroni: ALL tests form one correction family
model.find_power(
    sample_size=200,
    target_test="covariate, treatment[1] vs treatment[2], treatment[1] vs treatment[3]",
    correction="bonferroni",
)
# Family size = 3 → corrected threshold = 0.05/3

# Tukey: ONLY post-hoc contrasts are corrected
model.find_power(
    sample_size=200,
    target_test="covariate, treatment[1] vs treatment[2], treatment[1] vs treatment[3]",
    correction="tukey",
)
# covariate → corrected shows "-" (not applicable)
# post-hoc tests → Tukey-corrected
```

### How corrections affect the overall F-test

The overall F-test is never included in the correction family for Bonferroni/Holm/FDR corrections:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
model.find_power(
    sample_size=200,
    target_test="all",
    correction="bonferroni",
)
# "overall" F-test power is NOT reduced by the correction
# Only the individual t-tests are corrected
```

---

## Correction Methods: Technical Summary

| Method | Type | How It Works | Power |
|---|---|---|---|
| **Bonferroni** | FWER | Divides alpha by number of tests: alpha/m | Lowest |
| **Holm** | FWER | Step-down: tests ordered by p-value, thresholds get progressively less strict | >= Bonferroni |
| **FDR** | FDR | Step-up: controls the expected proportion of false discoveries | Highest |
| **Tukey** | FWER | Uses Studentized Range distribution for pairwise factor comparisons | Depends on factor levels |

- **FWER** (Family-Wise Error Rate) -- controls the probability of making **any** false positive. Use when even one false positive is unacceptable.
- **FDR** (False Discovery Rate) -- controls the **proportion** of false positives among significant results. Use when some false positives are tolerable.

MCPower precomputes correction-adjusted critical values **before** running simulations, so there is no runtime overhead from using a correction.

---

## Next Steps

- **[Tutorial: ANOVA & Post-Hoc Comparisons](anova-posthoc.md)** -- using Tukey correction with factor variables
- **[Scenario Analysis](../concepts/scenario-analysis.md)** -- combining corrections with robustness testing
- **[API Reference](../api/index.md)** -- full `find_power()` and `find_sample_size()` parameter documentation

---

## References

- Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, *56*(293), 52--64.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, *6*(2), 65--70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, *57*(1), 289--300.
- Tukey, J. W. (1953). *The problem of multiple comparisons*. Unpublished manuscript, Princeton University.
