---
title: "Model formula syntax for power analysis"
description: "Formula syntax for power analysis models: main effects, interactions, mixed-effects random terms, and test-formula misspecification."
---
# Model specification

A **formula** is your statistical model written as text — an outcome, a `~` or `=`, and the predictors that explain it, as in `y ~ x1 + x2`. Join terms with `+`; use `*` to add two predictors *and* their interaction (`x1*x2` means `x1 + x2 + x1:x2`), or `:` for the interaction term alone. The same formula string works in every port, so you learn the syntax once and reuse it everywhere.

## Three equivalent forms

```
y = x1 + x2 + x1:x2     # assignment style
y ~ x1 + x2 + x1:x2     # R-style formula
x1 + x2 + x1:x2         # predictors only (outcome auto-named)
```

The left side is the outcome; the right side lists predictors. The outcome name is optional — omit it and MCPower names one for you.

## Main effects

List predictors separated by `+`:

```
satisfaction = treatment + motivation + age
```

Each predictor becomes a term in the model. By default every variable is continuous standard normal; change that with [[concepts/variable-types|variable types]].

## Interactions

Two notations, with an important distinction:

- **Star `*` — main effects *and* interaction.** `x1*x2` expands to `x1 + x2 + x1:x2`. `x1*x2*x3` expands to all three main effects, all three two-way interactions, and the three-way term. Don't also write the expanded terms yourself — `*` already includes them.
- **Colon `:` — interaction only.** `x1:x2` adds the product term *without* the main effects.

```
conversion = treatment*user_type
# same as: treatment + user_type + treatment:user_type

y = A*B*C
# expands to: A + B + C + A:B + A:C + B:C + A:B:C
```

> [!note] Naming an interaction's effect
> However you wrote the formula, an interaction's [[concepts/effect-sizes|effect size]] is always referred to with colon notation — e.g. the effect of `treatment:user_type`.

## Mixed-effects formulas

For clustered data, MCPower accepts an R-style random-intercept term —
`(1|school)` gives each school its own baseline:

```
satisfaction ~ treatment + motivation + (1|school)
```

The syntax also extends to random slopes `(1 + x|school)` and nested groupings `(1|school/classroom)`. Random effects need a little extra configuration; see [[concepts/mixed-effects|mixed-effects models]].

## Test-formula misspecification

A test formula lets you **fit a different model than the one that generated the data** — the data come from your full model, but power is measured on a smaller analysis model you name separately. Use it to study the power cost of misspecifying your analysis: dropping a covariate, ignoring an interaction, or otherwise testing a leaner model than the truth.

Pass it as `test_formula` on `find_power` / `find_sample_size`. Every term in the test formula must already exist in the model formula (give an omitted-but-real predictor an effect of `0` so it shapes the data but you can still drop it from the fit).

```python
# data generated from y = treatment + covariate + treatment:covariate,
# but power measured for a model that omits the interaction
model.find_power(sample_size=200, test_formula="y = treatment + covariate")
```

```r
# same study in R
model$find_power(sample_size = 200, test_formula = "y = treatment + covariate")
```

For the *why* — how testing the wrong model manufactures spurious effects or drains real power — see [[concepts/model-misspecification|model misspecification]].

## Common patterns

| Study design | Formula |
|---|---|
| Simple regression | `y = x1 + x2` |
| Binary treatment + covariate | `outcome = treatment + baseline` |
| Interaction | `y = treatment*covariate` |
| Multi-group (factor) | `wellbeing = group + age` |
| Two factors + interaction | `y = A*B + covariate` |
| Mixed (random intercept) | `y ~ x + (1\|school)` |
