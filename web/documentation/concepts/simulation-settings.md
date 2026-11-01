---
title: "Simulation settings for power analysis"
description: "Simulation settings for power analysis: simulation count, alpha, random seed, and convergence tolerance for mixed-effects models."
---
# Simulation settings

Power in MCPower is estimated by **simulation** — the engine generates many synthetic datasets from your design, analyses each one, and reports how often the effect is detected. Four expert settings govern that run: the **simulation count** (accuracy of the estimate), **alpha** (how strict each significance test is), the **seed** (reproducibility), and the **failed-simulation tolerance** (how many non-converging fits to allow). The defaults suit almost every study; reach for these only when you need more precision, a different alpha, an exactly reproducible run, or more headroom for hard-to-fit models.

## Simulation count

The number of simulated datasets sets the **precision** of the estimate: more simulations narrow the confidence interval around the power number but take longer. The defaults balance the two — **1,600** simulations for OLS and other single-fit designs, **800** for mixed-effects models with a continuous outcome and **400** for mixed-effects models with a binary or count outcome (each fit costs more). Raising the count helps when two designs report nearly identical power and you need to tell them apart.

## Alpha

**Alpha (α)** is the significance threshold — the false-positive rate you accept on each test. The default is **α = 0.05**. Lowering it makes significance harder to reach and so lowers power at a fixed sample size; if you plan to correct for multiple tests, set the correction rather than shrinking alpha by hand. See [[concepts/multiple-testing|multiple testing]].

## Seed

The **seed** fixes the random-number stream so a run is exactly **reproducible** — same seed and inputs, same numbers. The default seed is **2137**. Change it only to confirm a result is stable across different random draws.

## Failed-simulation tolerance

Some fits don't converge — most often mixed-effects models on small or highly clustered samples. MCPower tolerates a small fraction of failed fits, reports the rate, and errors only if the fraction exceeds the tolerance you allow. OLS designs essentially never fail, so the tolerance has no effect there. See [[concepts/mixed-effects|mixed-effects models]].

## Estimation mode

For a **clustered binary or count model** (a logit, probit, or Poisson GLMM — see [[concepts/supported-families|supported families]]), two more expert settings on `find_power`/`find_sample_size` govern how the fit is done: `wald_se` (which standard-error method) and `agq` (how many quadrature nodes approximate the random effect). Both are ignored for OLS, linear mixed models, and unclustered GLMs, which already have exact standard errors.

**`wald_se`** picks between two ways to get the fixed effects' Wald standard errors:

- `"rx"` (the default) — a Schur-complement shortcut, matching lme4's `vcov(use.hessian = FALSE)`. Faster, and the right choice for most runs, including large grids.
- `"hessian"` — inverts the full per-fit finite-difference Hessian, matching lme4's `vcov(use.hessian = TRUE)`. Slightly slower and slightly more conservative (it doesn't assume the fixed effects and the random-effect variance are orthogonal, an assumption `"rx"` makes and a GLMM's IRLS weight coupling can break). Use it when you want the more conservative SE, or when validating against an `lme4` fit that used `use.hessian = TRUE`.

**`agq`** sets the number of adaptive Gauss–Hermite quadrature nodes used to integrate out the random effect, in place of the default Laplace approximation:

- `1` (the default) — Laplace approximation (`nAGQ = 1` in `glmer` terms). Works for any clustered design.
- an odd integer `> 1`, up to `25` — adaptive Gauss–Hermite quadrature, more accurate than Laplace at small cluster sizes but more expensive. Only allowed for a binary/count GLMM with a **single grouping factor** and **at most 3 random effects per group**. A design outside that envelope (crossed/nested groupings, too many random effects, an even node count, or a value above 25) warns and silently falls back to `agq=1` — the run still completes, on Laplace.

The desktop and web app surface both controls as one three-way switch: **Fast** (`wald_se="rx"`), **Accurate** (`wald_se="hessian"`), and **AGQ** (`agq=k`, k configurable), next to the model's other advanced settings.

## Denominator degrees of freedom

For any **mixed-effects model** (LME or clustered GLMM), `df_method` on
`find_power`/`find_sample_size` selects the test for each fixed effect. Each
value other than `"auto"` names one test:

- `"auto"` (the default): picks the test from the design (table below).
- `"between_within"`: tests each fixed effect against a *t* distribution
  with between-within denominator degrees of freedom: a predictor declared
  cluster-level (see [[concepts/mixed-effects|mixed-effects models]]) gets
  the between-cluster df, others get the within-cluster df. The joint Wald
  test uses `k·F(k, df)` with the smaller df across the target set. It needs
  a single grouping factor: with a crossed or nested grouping the
  between-within split is not defined, and the run stops with an error.
- `"satterthwaite"`: tests each fixed effect against a *t* at each simulated
  fit's Satterthwaite degrees of freedom, as lmerTest computes them. Holm and
  Benjamini–Hochberg corrections are applied to the per-target p-values, as
  `p.adjust` would; the joint Wald test uses `k·F(k, df)` with the smallest
  df among the targets. A binary or count mixed model has no Satterthwaite
  df, so for a GLMM the run stops with an error.
- `"asymptotic"`: tests each fixed effect against a Wald z.

What `"auto"` picks:

| Model | Grouping | Random slope on the first grouping | Test |
|---|---|---|---|
| LME | single | no | between-within df |
| LME | single | yes | Satterthwaite df |
| LME | crossed or nested | either | Satterthwaite df |
| GLMM | single | either | between-within df |
| GLMM | crossed or nested | either | Wald z |

A predictor with a random slope varies within clusters, but its effect varies
across them too, so its test rests on the number of clusters. Between-within
df give it the within-cluster df however few clusters there are, and the test
then behaves almost like the Wald z; Satterthwaite df follow the clusters.
Without a random slope the two agree closely on a single grouping, and
between-within df cost nothing per fit. A slope counts whatever its variance.
There is no overall (omnibus) F for mixed models under any setting.

When a simulated fit collapses (every random-effect variance estimated at
zero, lme4's "singular fit"), the between-within split no longer describes
the fitted model, so under between-within df that fit is tested against the
collapsed model's own reference instead: Student *t* on the residual df
N − p for an LME, Wald z for a GLMM. Satterthwaite df reach the residual df
there on their own. A fit whose Satterthwaite df cannot be computed is tested
against the Wald z.

Under between-within df, when the design leaves fewer than one between-cluster
df (`g − p_between < 1`: at least as many cluster-level columns, intercept
included, as clusters) or fewer than one within-cluster df, the run stops with
an error naming the counts; pick `df_method="asymptotic"` or add clusters.
`df_method` has no effect on OLS or unclustered GLM, which already use their
exact reference distributions. Its calibration against lmerTest Satterthwaite
is on the [[validation/validation_df_calibration|denominator-df calibration
page]]. The desktop and web app surface it as the **Denominator df** setting
on the run panel, alongside the estimation-mode switch above, with a
per-predictor cluster-level toggle; the setting offers only the tests the
design supports.
