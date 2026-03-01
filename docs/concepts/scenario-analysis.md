# Scenario Analysis

## What It Is

Traditional power analysis assumes ideal conditions: effects are exactly as specified, distributions are perfectly normal, and error variance is constant. Real studies never match these assumptions. Scenario analysis tests how robust your power calculations are when those assumptions break down.

MCPower provides three built-in scenarios. The **Optimistic** scenario uses your exact settings -- the best case. The **Realistic** scenario introduces moderate violations: effects vary slightly between participants, correlations fluctuate, and distributions deviate from normal. The **Doomer** scenario applies more severe violations to simulate worst-case conditions. Together, these give you a range of power estimates instead of a single optimistic number.

The practical recommendation is to plan around the **Realistic** estimate. If even the **Doomer** estimate meets your target power, your study design is highly robust. If the Optimistic estimate barely reaches 80%, you should increase your sample size regardless.

---

## How It Works in MCPower

Enable scenarios by adding `scenarios=True` to [`find_power()`](../api/power-analysis.md) or [`find_sample_size()`](../api/power-analysis.md):

```python
model.find_power(sample_size=200, target_test="treatment", scenarios=True)
```

Customize scenario parameters with [`set_scenario_configs()`](../api/scenarios.md):

```python
model.set_scenario_configs({
    "realistic": {"heterogeneity": 0.3, "heteroskedasticity": 0.15},
})
```

Scenarios multiply computation time (one full simulation run per scenario). With the three defaults, this triples computation time. You can also pass a list of scenario names to run only specific ones, e.g. `scenarios=["optimistic", "doomer"]`.

---

## Guidelines

### What Gets Varied

| Parameter | Optimistic | Realistic | Doomer |
|---|---|---|---|
| Effect size heterogeneity | None | Mild (0.20) | Moderate (0.40) |
| Heteroskedasticity | None | Moderate (0.15) | Substantial (0.35) |
| Correlation noise | None | Moderate (0.15) | Substantial (0.30) |
| Distribution perturbation | None | 50% chance | 80% chance |
| Residual non-normality | None | 50% chance | 80% chance |
| Residual df | — | 8 | 5 |

For **mixed models**, additional perturbations are applied:

| Mixed Model Parameter | Realistic | Doomer |
|---|---|---|
| ICC noise (SD) | Moderate (0.15) | Substantial (0.30) |
| Random effect distribution | Heavy-tailed (t, df=10) | Heavy-tailed (t, df=5) |

### When to Use Scenarios

| Use Scenarios | Skip Scenarios |
|---|---|
| Planning important or costly studies | Quick exploratory checks |
| Effect sizes are uncertain | Effect sizes are well-established |
| Working with messy real-world data | Tight computational budget |
| Reviewers expect robustness checks | Iterating on model specification |
| Grant applications | |

### Interpreting the Output

- **All three above target:** Strong, robust design.
- **Optimistic and Realistic above, Doomer below:** Adequate but not bulletproof. Consider a modest sample increase.
- **Only Optimistic above target:** Design is fragile. Increase sample size substantially.
- **None above target:** Fundamental redesign needed (larger effects, simpler model, or much larger N).

---

## Common Patterns

### Typical Power Drop Across Scenarios

| Effect Size | Optimistic | Realistic (typical drop) | Doomer (typical drop) |
|---|---|---|---|
| Large (0.40+) | 95% | -3 to -7 pp | -8 to -15 pp |
| Medium (0.25) | 80% | -5 to -10 pp | -12 to -20 pp |
| Small (0.10) | 60% | -5 to -12 pp | -15 to -25 pp |

*(pp = percentage points; actual drops depend on sample size and model complexity)*

### Custom Scenario Configurations

You can override individual parameters per scenario. Provided values merge with defaults:

```python
model.set_scenario_configs({
    "realistic": {"heterogeneity": 0.25},
    "doomer": {"heterogeneity": 0.50, "correlation_noise_sd": 0.3},
})
```

---

## Learn More

- **[Tutorial: Custom Scenarios](../tutorials/custom-scenarios.md)** -- customize scenario parameters for your field
- **[Tutorial: Your First Power Analysis](../tutorials/first-analysis.md)** -- scenario analysis in a complete workflow
- **[API: set_scenario_configs](../api/scenarios.md)** -- full parameter documentation
