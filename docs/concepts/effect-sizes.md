# Effect Sizes

## What It Is

In MCPower, effect sizes are **standardized regression coefficients** (betas). A beta of 0.30 means that a one standard deviation increase in the predictor is associated with a 0.30 standard deviation change in the outcome. This standardization makes effects comparable across variables measured on different scales.

For binary and factor predictors, the interpretation shifts slightly. A binary effect of 0.50 means that switching from the reference group (0) to the treatment group (1) produces a 0.50 SD change in the outcome. For factors, each dummy variable gets its own effect size representing the difference between that level and the reference level.

Effect sizes are the single most important input to a power analysis. Overestimating them leads to underpowered studies; underestimating them wastes resources. The best source is prior research or pilot data. When neither is available, Cohen's (1988) benchmarks provide a starting point.

---

## How It Works in MCPower

Set effect sizes with [`set_effects()`](../api/configuration.md):

```python
model.set_effects("treatment=0.50, age=0.25, group[2]=0.40, group[3]=0.60")
```

For factor variables, each non-reference level needs its own effect size using bracket notation (e.g., `group[2]=0.40`).

`set_effects()` must be called before running the analysis. Any predictors not explicitly assigned an effect size default to 0.0 (no effect).

---

## Guidelines

### Cohen's Benchmarks

| Predictor Type | Small | Medium | Large |
|---|---|---|---|
| Continuous | 0.10 | 0.25 | 0.40 |
| Binary / Factor | 0.20 | 0.50 | 0.80 |

### Why Different Benchmarks for Binary vs Continuous?

For **binary/factor** predictors, MCPower keeps binary variables as 0/1 (not standardized), so the beta coefficient equals Cohen's d directly. The benchmarks (0.20/0.50/0.80) are Cohen's (1988) standard conventions.

For **continuous** predictors, variables are standardized to N(0,1), so beta represents the change in Y per 1-SD change in X. These benchmarks are calibrated so that a continuous predictor at the "medium" threshold (beta = 0.25) produces approximately the same statistical power as a binary predictor at Cohen's medium effect (d = 0.50), holding sample size constant. This power-equivalence approach ensures the labels "small," "medium," and "large" carry consistent practical meaning across predictor types.

### Practical Interpretation

| Beta | What It Looks Like |
|---|---|
| 0.10 | Barely noticeable in raw data. Requires large samples to detect. |
| 0.20 | Small but real effect. Visible with careful measurement. |
| 0.30 | Moderate. A trained observer would notice the pattern. |
| 0.50 | Clearly visible. Obvious group differences in plots. |
| 0.80+ | Dramatic. Hard to miss even in small samples. |

### Interactions

Interaction effects (e.g., `x1:x2`) are typically smaller than main effects. Values of 0.10--0.20 are common. Plan for lower power when testing interactions.

### Factor Variables

Each dummy variable gets its own beta relative to the reference level. If two non-reference levels have different betas (e.g., 0.30 and 0.70), the implicit contrast between them is the difference (0.40). This matters for [post-hoc comparisons](../tutorials/anova-posthoc.md).

---

## Common Patterns

### Typical Effect Sizes by Field

| Field | Predictor | Typical Beta | Notes |
|---|---|---|---|
| Education | Teaching intervention | 0.20--0.40 | Medium effects common |
| Education | Socioeconomic status | 0.15--0.30 | Small-medium |
| Psychology | Therapy vs. control | 0.30--0.60 | Medium-large |
| Psychology | Personality trait | 0.10--0.25 | Small-medium |
| Medicine | Drug vs. placebo | 0.20--0.50 | Varies widely |
| Medicine | Lifestyle factor | 0.05--0.20 | Often small |
| Social science | Policy intervention | 0.10--0.30 | Small-medium |
| Marketing | Ad exposure | 0.05--0.15 | Typically small |

### Rules of Thumb

- When in doubt, use a **smaller** effect size. Overestimation is the more costly error.
- Use [Scenario Analysis](scenario-analysis.md) to test sensitivity to effect size uncertainty.
- If prior literature reports Cohen's *d*, it maps roughly to beta for a binary predictor.
- Main effects are almost always larger than interaction effects in the same model.

---

## Learn More

- **[Tutorial: Your First Power Analysis](../tutorials/first-analysis.md)** -- setting effects in a complete workflow
- **[ANOVA & Post-Hoc Tests](../tutorials/anova-posthoc.md)** -- factor effects and pairwise comparisons
- **[API Reference: set_effects](../api/configuration.md)** -- full parameter documentation

---

## References

- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.
