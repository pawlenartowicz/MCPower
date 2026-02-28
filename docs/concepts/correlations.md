# Correlations

## What It Is

In real research, predictors are rarely independent. Income and education are correlated. Age and work experience move together. When predictors share variance, each one explains less *unique* variance in the outcome, which increases standard errors and reduces statistical power.

MCPower generates correlated predictor data using Cholesky decomposition, then transforms each variable to its target distribution while preserving the correlation structure. The effect of correlations on power depends on what you are testing:

- **Main effects:** Correlation between predictors *reduces* power (shared variance → less unique variance per predictor → larger standard errors).
- **Interaction effects:** Correlation between predictors *increases* power (correlated predictors produce a more variable interaction term → the interaction effect is easier to detect).

One important constraint: factor variables (categorical with 3+ levels) cannot be correlated through the standard correlation interface. If you need correlated categorical and continuous variables, upload empirical data with `preserve_correlation="strict"`, which bootstraps whole rows to preserve the exact relationships in your dataset.

---

## How It Works in MCPower

Set correlations with [`set_correlations()`](../api/configuration.md):

```python
model.set_correlations("corr(income, education)=0.5, corr(income, age)=0.3")
```

Or provide a full correlation matrix as a numpy array:

```python
model.set_correlations(np.array([[1.0, 0.4], [0.4, 1.0]]))
```

Unspecified pairs default to r = 0 (independent). The matrix must be positive semi-definite (PSD).

---

## Guidelines

### Correlation Magnitudes (Main Effects)

| Range | Label | Sample Size Impact | Action |
|---|---|---|---|
| 0.00--0.20 | Negligible | ~1.00x (no increase needed) | Safe to ignore |
| 0.20--0.40 | Small | 1.07--1.20x | Include if known |
| 0.40--0.60 | Moderate | 1.20--1.53x | Always include |
| 0.60--0.70 | Large | 1.53--1.87x | Include; consider dropping one predictor |
| 0.70+ | Very large | >1.87x; multicollinearity risk | Check if both predictors are needed |

### Positive Semi-Definite (PSD) Requirement

A correlation matrix is PSD when all the correlations are mutually consistent -- i.e., they could actually occur together in real data. MCPower checks this automatically and raises an error if not.

**Example of an invalid (non-PSD) combination:** If A and B are strongly correlated (r=0.9) and A and C are strongly correlated (r=0.9), then B and C must also be positively correlated -- they can't be negatively correlated (r=-0.9) because that contradicts the first two relationships.

If you get a PSD error, reduce the most extreme correlations or check that the signs are logically consistent.

### Constraints

- **Factor variables** cannot be correlated through `set_correlations()`.
- **Correlations are symmetric:** `corr(x1, x2)=0.3` and `corr(x2, x1)=0.3` are identical.
- **Matrix dimensions** must match the number of non-factor variables, in formula order.
- For correlated factors, use `upload_data()` with `preserve_correlation="strict"`.

---

## Common Patterns

### Correlation Preservation Modes with Uploaded Data

| Mode | Source | Best For |
|---|---|---|
| `"no"` | Manual only | Full manual control |
| `"partial"` | Computed from data + manual overrides | Empirical baseline with adjustments |
| `"strict"` (default) | Bootstrapped rows | Most realistic simulation from pilot data |

### Typical Correlations by Domain

| Domain | Predictor Pair | Typical r |
|---|---|---|
| Education | SES and test scores | 0.30--0.50 |
| Psychology | Anxiety and depression | 0.40--0.70 |
| Medicine | Age and blood pressure | 0.20--0.40 |
| Social science | Income and education | 0.40--0.60 |
| Marketing | Ad spend and brand awareness | 0.20--0.40 |

### Impact on Required Sample Size (Main Effects)

| Correlation (r) | Required N | Multiplier |
|---|---|---|
| 0.00 | 375 | 1.00x |
| 0.10 | 375 | 1.00x |
| 0.20 | 375 | 1.00x |
| 0.30 | 400 | 1.07x |
| 0.40 | 450 | 1.20x |
| 0.50 | 500 | 1.33x |
| 0.60 | 575 | 1.53x |
| 0.70 | 700 | 1.87x |

*Simulation: `y ~ x1 + x2`, both effects=0.15, 1600 simulations, seed=42.*

Correlations below 0.30 have negligible impact. Above 0.50, the sample size increase becomes substantial.

```python
# Try it yourself: correlation impact on required sample size
from mcpower import MCPower

for r in [0.0, 0.10, 0.30, 0.50, 0.70]:
    model = MCPower("y ~ x1 + x2")
    model.set_effects("x1=0.15, x2=0.15")
    if r > 0:
        model.set_correlations(f"corr(x1, x2)={r}")
    model.set_seed(42)
    model.set_simulations(1600)
    model.find_sample_size(
        from_size=50, to_size=2000, by=25,
        target_test="x1",
    )
```

### Impact on Interaction Power

For interactions, the effect is reversed -- correlation *helps*:

| Correlation (|r|) | Required N for interaction | Multiplier |
|---|---|---|
| 0.00 | 850 | 1.00x |
| 0.30 | 700 | 0.82x |
| 0.50 | 650 | 0.76x |

*Simulation: `y ~ x1 + x2 + x1:x2`, main effects=0.15, interaction=0.10, 1600 simulations, seed=42.*

```python
# Try it yourself: correlation impact on interaction power
from mcpower import MCPower

for r in [0.0, 0.30, 0.50]:
    model = MCPower("y ~ x1 + x2 + x1:x2")
    model.set_effects("x1=0.15, x2=0.15, x1:x2=0.10")
    if r > 0:
        model.set_correlations(f"corr(x1, x2)={r}")
    model.set_seed(42)
    model.set_simulations(1600)
    model.find_sample_size(
        from_size=50, to_size=3000, by=50,
        target_test="x1:x2",
    )
```

If your model has both main effects and interactions, correlation creates a trade-off: main effects need more observations while interactions need fewer. The sign of the correlation or the effects does not matter -- only |r|.

### Impact with Multiple Predictors

With three correlated predictors, the impact compounds:

| Pairwise Correlation | Required N | Multiplier |
|---|---|---|
| All r=0.00 | 350 | 1.00x |
| All r=0.30 | 400 | 1.14x |
| All r=0.50 | 550 | 1.57x |

*Simulation: `y ~ x1 + x2 + x3`, all effects=0.15, 1600 simulations, seed=42.*

---

## Learn More

- **[Uploading Data](../tutorials/own-data.md)** -- correlation preservation from empirical data
- **[API Reference: set_correlations](../api/configuration.md)** -- full parameter documentation
