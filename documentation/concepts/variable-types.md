# Variable types

Every predictor you set up has a **type** that tells MCPower how to draw its simulated values: **continuous** for numbers on a scale, **binary** for two 0/1 groups, or **factor** for a category with three or more levels. The type also decides how the predictor enters the model. A predictor you don't classify defaults to continuous standard normal.

## The three core types

- **Continuous** — numeric values on a scale. Default shape is standard normal, $N(0,1)$.
- **Binary** — two groups coded 0/1 (treatment/control, yes/no). Default split is 50/50; custom proportions are supported.
- **Factor** — a category with three or more levels, dummy-coded against a reference level. Each non-reference level becomes its own term with its own [[concepts/effect-sizes|effect size]].

A predictor left unspecified defaults to continuous standard normal.

## Continuous distribution shapes

Continuous predictors don't have to be normal. Choosing a realistic shape makes your power estimate robust to the messiness of real data.

| Shape | When to use |
|---|---|
| Normal | Default; symmetric, bell-shaped data. |
| Right-skewed | Income, reaction times, counts. |
| Left-skewed | Ceiling effects, negatively skewed scores. |
| High-kurtosis | Heavy-tailed data with outliers. |
| Uniform | Evenly spread. |

### Statistical properties

Every shape is standardized to mean 0 and variance 1, so swapping shapes changes *only* the shape — never the scale of your effect sizes. No synthetic shape ever emits a value beyond ±6 SD, so a single simulated point can't dominate the fit.

| Shape | Skewness | Excess kurtosis | Range (in SD) |
|---|---|---|---|
| Normal | 0 | 0 | ±5.4 |
| Right-skewed (exponential-based) | +1.9 | 4.9 | −1.0 to +6.0 |
| Left-skewed (mirror) | −1.9 | 4.9 | −6.0 to +1.0 |
| High-kurtosis (heavy-tailed t) | 0 | 6.4 | ±6.0 |
| Uniform | 0 | −1.2 | ±1.73 |

The skewed shapes sit at the upper edge of skewness observed in large surveys of real psychological and biomedical data, and high-kurtosis is the heaviest-tailed shape in the family.

## Factors and the reference level

A **factor** is a category with three or more levels, and one level is the **reference** that every other level is compared against — the first level you list, or the first sorted value in uploaded data. Each non-reference level becomes its own dummy variable with its own [[concepts/effect-sizes|effect size]]. When you simulate from [[concepts/upload-data|uploaded data]], dummies keep the original data values in their names — `cyl[6]`, `cyl[8]`, `origin[Japan]` — rather than integer indices, so the output reads the way your data does.

## How a type is determined

**Declared designs** (no uploaded data) involve no inference: every predictor is continuous standard normal unless you declare it binary, a factor, or a different continuous shape.

**Uploaded data** is auto-detected per column:

| Column content | Detected type |
|---|---|
| Any text values | Factor |
| Numeric, exactly 2 distinct values | Binary |
| Numeric, ≤ 7 distinct values with at least 15 rows per level | Factor |
| Numeric, otherwise | Continuous |

A factor — detected or declared — can have at most 20 levels; a column with more is rejected as a factor. You can override any detection with an explicit declaration. See [[concepts/upload-data|using empirical data]] for how detection and overrides work when simulating from a real dataset.

## Common patterns

| Scenario | Type | Notes |
|---|---|---|
| Treatment vs. control | binary | Default 50/50 split |
| Unbalanced treatment | binary, 30% | 30% treatment, 70% control |
| Three-group comparison | factor, 3 levels | Equal thirds |
| Weighted groups | factor with custom proportions | Proportions autosum to 1 |
| Income | right-skewed continuous | Positive skew |
| Rating scale | uniform | Evenly distributed |

> [!note] Rare levels need observations
> A level whose proportion times N falls below 5 observations cannot be
> estimated — MCPower excludes the factor at that N and flags it in the
> diagnostics; with the default exact allocation you are warned up front.
> See [[limitations#Sparse factor levels at small N]].

**Pin rule.** Declaring any explicit continuous distribution — including explicit `"normal"` — pins that predictor against scenario distribution swaps. An unpinned predictor (left at the default) may be swapped to a non-normal shape by the `distribution_change_prob` knob. If you want a predictor to stay normal regardless of scenario, declare it explicitly.

## Residual distribution

The **residual distribution** is the shape of the noise term — the unexplained scatter MCPower adds to each simulated outcome around its predicted value. The five choices are the same shapes as for predictors (`normal`, `right_skewed`, `left_skewed`, `high_kurtosis`, `uniform`), and the default is `normal`. Setting it with `set_residual_distribution(name)` pins the shape so scenario swaps leave it alone.

The five valid names:

| Name | Shape |
|---|---|
| `normal` | Standard normal (default if unset) |
| `right_skewed` | Right-skewed (exponential-based) |
| `left_skewed` | Left-skewed (mirror) |
| `high_kurtosis` | Heavy-tailed Student t; df comes from the active scenario's `residual_df` (optimistic 10 / realistic 8 / doomer 5) |
| `uniform` | Uniform |

There is no `df=` parameter on `set_residual_distribution`. The df for `high_kurtosis` is always scenario-supplied — set it via `set_scenario_configs({"your_scenario": {"residual_df": 5}})`. Calling `set_residual_distribution` with any name, including `"normal"`, **pins** the residual: scenario residual swaps leave it alone. The unpinned default — not calling `set_residual_distribution` at all — is the only state where swaps can fire.

The conceptual point is that **type, distribution shape, and residual shape are all part of your design assumptions** — perturbing the unpinned ones is exactly what [[concepts/scenario-analysis|scenario analysis]] does. For the exact syntax in each port, see the tutorials.

## Residuals and heteroskedasticity

**Heteroskedasticity** means the residual variance is not constant — the spread of the noise depends on a predictor instead of being the same everywhere. Linear models assume the opposite (constant variance), so this is an assumption violation worth stress-testing. The **driver** control names the continuous predictor whose value scales that variance; set it with `set_heteroskedasticity_driver(var)`, where `var` is a continuous predictor or the predicted value (the default).

The driver only says *which* predictor governs the spread. *How much* the variance changes along it is the ratio λ, which comes from the active scenario, not the model — so heteroskedasticity is exercised through scenario analysis. The driver is restricted to continuous predictors and applies to linear models only.

```python
model.set_heteroskedasticity_driver("x1")
```

```r
model$set_heteroskedasticity_driver("x1")
```

See the heteroskedasticity section of [[concepts/scenario-analysis|scenario analysis]] for what λ means, the preset values, and how to set a known ratio.
