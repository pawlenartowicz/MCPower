# Validation

How do you know a simulation-based power tool is *correct*? You can't check it against
a textbook formula — the whole point of MCPower is to handle designs that have no
clean closed form. So MCPower proves correctness a different way: by splitting the
pipeline in two and verifying each half against a trusted reference.

## What "right" means

A simulation power tool is correct if **both** halves of its pipeline are correct:

1. **The data generator is honest.** The synthetic datasets it builds really embody
   the design you requested — the right effect sizes, distributions, factor
   proportions, and clustering.
2. **The solver is faithful.** Given a fixed dataset, it computes the same
   coefficients, standard errors, test statistics, and thresholds that a trusted
   reference — standard **R** — computes on that exact same dataset.

If both hold, the full *generate → fit → count* pipeline is correct **by
composition**. There is no need for an end-to-end "true power" oracle, because each
link in the chain has been checked against something we already trust.

## How each half is checked

- **Data generation.** We generate data from a formula with **known true
  coefficients**, then analyse it in R and confirm that the recovered moments, effect
  sizes, and intraclass correlations match what was requested. Here the **specification
  is the oracle** — the data must reproduce the design that defined it.
- **Solving.** We fit the **exact same dataset** twice: once with standard R (`lm`,
  `glm`, and `lme4::lmer` with REML) and once with MCPower's engine. The coefficients,
  standard errors, test statistics, and thresholds must agree closely. Because both
  solvers see *identical* data, there is no sampling noise to hide behind — any
  disagreement is a real difference in the solver, not luck of the draw.

## Coverage

The checks span the families MCPower supports — single and multiple continuous
predictors, interactions, factors, logistic regression (GLM), and mixed / multilevel
models (LME) — at both small and larger effect sizes, so the evidence covers the range
of designs you're likely to analyse.

## The reports

Each report below is a detailed, formula-by-formula evidence document. Open the one
that matches what you want to confirm:

| Report | What it shows |
|---|---|
| [Data generation](validation_data_generation.md) | the data generator — generated data matches the requested design |
| [OLS solving](validation_OLS_solving.md) | the OLS solver against R's `lm` on identical data |
| [GLM solving](validation_GLM_solving.md) | the logistic / GLM solver against R's `glm` |
| [MLE solving](validation_MLE_solving.md) | the mixed-effects (LME) solver against `lme4::lmer` (REML) |
| [Effect recovery](validation_get_effects.md) | end-to-end round-trip: specify a design → simulate it → recover the effects |
| [Required N](validation_crossing.md) | the model-based required-N estimate — the default search grid against a dense 100,000-simulation ground truth |
| [Scenario perturbations](validation_scenarios.md) | heterogeneity, heteroskedasticity, correlation noise, and distribution swaps — each knob validated in isolation and in combination |
| [Uploaded data](validation_upload.md) | the upload path — data simulated from a user-provided frame reproduces its moments and correlations across draws |

## How to read them

These are evidence documents, not dashboards. Each opens with a plain "what this
shows" section, then walks through the designs one formula at a time. Skim the
at-a-glance results if you just want reassurance, or drill into a specific design if
you want to see the numbers behind it.

---

See also [[internals/index|what's inside MCPower]] and
[[concepts/limitations|where MCPower's results need caution]].
