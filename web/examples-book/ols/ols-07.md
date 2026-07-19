---
title: "Power for a two-group t-test (OLS)"
description: "Power & sample-size analysis by Monte Carlo simulation for a two-group t-test as OLS regression (pain_score ~ treatment). Free, Python & R."
---
# Two-group comparison (independent-samples t-test)

You have one continuous outcome and two groups — control vs treatment — and you
want to know whether their mean pain scores differ. This is the independent-samples
t-test, written here as a regression on a single binary predictor:
`pain_score ~ treatment`.

## Variations

- **Smaller or larger gap.** The effect is on the binary benchmark scale —
  swap `treatment=0.50` (medium) for `treatment=0.20` (small) or `treatment=0.80`
  (large) to see how the expected separation moves power.
- **Unbalanced groups.** If you expect a lopsided split rather than 50/50,
  set the treatment proportion when you declare the variable type (e.g. a 30/70
  allocation) — unbalanced cells cost power for the same total N.
- **Solve for N instead.** Replace `find_power(sample_size=120, …)` with
  `find_sample_size(target_test="treatment", from_size=30, to_size=300, by=10)` to
  get the minimum N that reaches 80% power.
- **Same design, other fields:**
  - Ecology: `abundance ~ habitat` — does species abundance differ between two habitat types (disturbed vs undisturbed)?
  - Social science: `wage ~ gender` — does wage differ between two gender groups?

## Not this setup?

- [[ols/ols-08|Two groups with a baseline covariate (ANCOVA)]]
- [[anova/anova-01|One-way ANOVA omnibus (3+ groups)]]
- [[glm/glm-02|Two-group comparison on a binary outcome]]
- [[ols/ols-01|Single continuous predictor]]

## If you'd rather have…

- [[ols/ols-08|Two-group comparison with a baseline covariate]] — add a baseline
  covariate to the comparison (ANCOVA) for a more precise group effect.
- [[ols/ols-12|A 3+-level categorical predictor as regression]] — more than two
  groups: a single categorical predictor with 3+ levels.
- [[anova/anova-01|One-way ANOVA omnibus test]] — the same group comparison
  framed as a one-way ANOVA omnibus test (3+ groups).
- [[ols/ols-10|Binary group plus a continuous predictor]] — keep the binary
  group predictor but add a continuous predictor (parallel slopes).
- [[glm/glm-02|Two-group comparison on a binary outcome]] — the same two-group
  comparison on a binary outcome (logistic / chi-square recast).

## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:ols-07 -->
```python
from mcpower import MCPower

# Two-group mean comparison: does pain_score differ between the two treatment groups?
model = MCPower("pain_score = treatment")

# treatment is a binary two-level predictor (0 = control, 1 = treatment).
model.set_variable_type("treatment=binary")

# Expected effect on the binary benchmark scale: 0.50 = a medium group gap.
model.set_effects("treatment=0.50")

model.find_power(sample_size=120, target_test="treatment")
```
<!-- /chunk:py:ols-07 -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:ols-07 -->
```r
suppressMessages(library(mcpower))

# Two-group mean comparison: does pain_score differ between the two treatment groups?
model <- MCPower$new("pain_score ~ treatment")

# treatment is a binary two-level predictor (0 = control, 1 = treatment).
model$set_variable_type("treatment=binary")

# Expected effect on the binary benchmark scale: 0.50 = a medium group gap.
model$set_effects("treatment=0.50")

invisible(model$find_power(sample_size = 120, target_test = "treatment"))
```
<!-- /chunk:r:ols-07 -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/ols-07-setup.png|600|theme-light]]
![[assets/ols-07-setup-dark.png|600|theme-dark]]

</details>
