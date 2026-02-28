# Model Specification

This page covers MCPower's formula syntax for defining statistical models.

## Formula Syntax

MCPower accepts R-style formulas. Three equivalent forms:

```python
from mcpower import MCPower

model = MCPower("y = x1 + x2 + x1:x2")    # assignment style
model = MCPower("y ~ x1 + x2 + x1:x2")    # R-style formula
model = MCPower("x1 + x2 + x1:x2")         # predictors only (outcome auto-named)
```

The left side is the outcome variable; the right side lists predictors. The outcome name is optional -- if omitted, MCPower creates one automatically.

---

## Main Effects

List predictors separated by `+`:

```python
model = MCPower("satisfaction = treatment + motivation + age")
```

Each predictor becomes a term in the regression model. By default, all variables are treated as continuous (standard normal). Use `set_variable_type()` to change types. See [Variable Types](concepts/variable-types.md).

---

## Interactions

MCPower supports two interaction syntaxes:

### Star notation (`*`) -- main effects + interaction

```python
"x1*x2"         # Equivalent to: x1 + x2 + x1:x2
"x1*x2*x3"      # All main effects + all 2-way + 3-way interactions
```

### Colon notation (`:`) -- interaction only

```python
"x1:x2"          # Interaction term only (no main effects added)
```

### Examples

```python
# A/B test with interaction
model = MCPower("conversion = treatment + user_type + treatment:user_type")
model.set_variable_type("treatment=binary, user_type=binary")
model.set_effects("treatment=0.4, user_type=0.3, treatment:user_type=0.5")

# Equivalent using star notation
model = MCPower("conversion = treatment*user_type")

# Three-way interaction
model = MCPower("y = A*B*C")
# Expands to: A + B + C + A:B + A:C + B:C + A:B:C
```

Interaction effects must be set using the colon notation regardless of which formula syntax you used:

```python
model.set_effects("treatment:user_type=0.2")
```

---

## Mixed-Effects Formulas

MCPower supports R-style random effect specifications for clustered data:

| Syntax | Structure |
|---|---|
| `(1\|school)` | Random intercept per school |
| `(1 + x\|school)` | Random intercept and slope for x per school |
| `(1\|school/classroom)` | Nested random intercepts (classroom within school) |

```python
# Random intercept
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")

# Random slopes
model = MCPower("y ~ x1 + (1 + x1|school)")

# Nested effects
model = MCPower("y ~ treatment + (1|school/classroom)")
```

Random effects require additional configuration via `set_cluster()`. See [Mixed-Effects Models](concepts/mixed-effects.md) for full documentation.

---

## Common Patterns

| Study Design | Formula |
|---|---|
| Simple regression | `"y = x1 + x2"` |
| Binary treatment + covariate | `"outcome = treatment + baseline"` |
| Interaction | `"y = treatment*covariate"` |
| Multi-group (factor) | `"wellbeing = group + age"` |
| Two factors + interaction | `"y = A*B + covariate"` |
| Mixed model (random intercept) | `"y ~ x + (1\|school)"` |
| Mixed model (random slopes) | `"y ~ x + (1 + x\|school)"` |
| Mixed model (nested) | `"y ~ x + (1\|school/classroom)"` |
