# Variable Types

## What It Is

MCPower needs to know the statistical type of each predictor to generate realistic simulated data. The three main types are **continuous** (numeric values on a scale), **binary** (two groups, coded 0/1), and **factor** (categorical with three or more levels, dummy-coded).

Beyond these core types, continuous variables can follow different distributions: normal (default), right-skewed (e.g., income), left-skewed (e.g., ceiling effects), high-kurtosis (heavy tails with outliers), or uniform (evenly spread). These distribution shapes affect how robust your power estimates are to real-world data characteristics.

When you upload empirical data, MCPower auto-detects variable types based on the number of unique values. You can override any detection with explicit type declarations. String columns (e.g., "Europe", "Japan", "USA") are automatically recognized as factors.

---

## How It Works in MCPower

Set types manually with [`set_variable_type()`](../api/configuration.md):

```python
model.set_variable_type("treatment=binary, group=(factor,3), income=right_skewed")
```

Custom proportions are supported for binary and factor variables:

```python
model.set_variable_type("drug=(binary,0.3), dose=(factor,0.2,0.5,0.3)")
```

Name factor levels with [`set_factor_levels()`](../api/data.md):

```python
model.set_factor_levels("group=control,drug_a,drug_b")  # first = reference
```

---

## Guidelines

### Distribution Types

| Type | Syntax | When to Use |
|---|---|---|
| Normal | `var=normal` | Default. Symmetric bell-shaped data. |
| Binary | `var=binary` | Two groups (treatment/control, yes/no). |
| Binary (custom) | `var=(binary,0.3)` | Unequal split (30% in group 1). |
| Factor | `var=(factor,3)` | 3+ categorical levels, equal proportions. |
| Factor (custom) | `var=(factor,0.2,0.5,0.3)` | Custom proportions (must sum to 1). |
| Right skewed | `var=right_skewed` | Income, reaction times, counts. |
| Left skewed | `var=left_skewed` | Ceiling effects, negatively skewed scores. |
| High kurtosis | `var=high_kurtosis` | Heavy-tailed data with outliers. |
| Uniform | `var=uniform` | Evenly spread, no clustering around mean. |

### Auto-Detection from Uploaded Data

| Unique Values | Detected Type |
|---|---|
| 1 | Dropped (constant column) |
| 2 | Binary |
| 3--6 | Factor |
| 7+ | Continuous |
| String column, 2--20 unique | Factor |
| String column, >20 unique | Error (too many levels) |

Override auto-detection with the `data_types` parameter in `upload_data()`.

### Factor Variables

- Level 1 (or first sorted value from data) is the reference level.
- Each non-reference level becomes a dummy variable needing its own effect size.
- Use `data_types={"cyl": ("factor", 8)}` to pick a specific reference level.
- With uploaded data, dummies use original values: `cyl[6]`, `cyl[8]` (default `preserve_factor_level_names=True`).

---

## Common Patterns

| Scenario | Type String | Notes |
|---|---|---|
| Treatment vs. control | `treatment=binary` | Default 50/50 split |
| Unbalanced treatment | `treatment=(binary,0.3)` | 30% treatment, 70% control |
| 3-group comparison | `group=(factor,3)` | Equal thirds |
| Weighted groups | `group=(factor,0.2,0.5,0.3)` | Custom proportions |
| Income variable | `income=right_skewed` | Positive skew |
| Rating scale (1--7) | `rating=uniform` | Evenly distributed |
| Named conditions | `set_factor_levels("cond=placebo,low_dose,high_dose")` | Explicit level names |

### Variables left unset default to standard normal (continuous, mean=0, SD=1).

## Learn More

- **[Uploading Data](../tutorials/own-data.md)** -- auto-detection, correlation preservation, named levels
- **[ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md)** -- factor variables in ANOVA designs
- **[API Reference: set_variable_type](../api/configuration.md)** -- full parameter documentation
- **[API Reference: set_factor_levels](../api/data.md)** -- named levels without data
