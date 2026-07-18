---
title: "Simulation settings for power analysis"
description: "Simulation settings for power analysis: simulation count, alpha, random seed, and convergence tolerance for mixed-effects models."
---
# Simulation settings

Power in MCPower is estimated by **simulation** — the engine generates many synthetic datasets from your design, analyses each one, and reports how often the effect is detected. Four expert settings govern that run: the **simulation count** (accuracy of the estimate), **alpha** (how strict each significance test is), the **seed** (reproducibility), and the **failed-simulation tolerance** (how many non-converging fits to allow). The defaults suit almost every study; reach for these only when you need more precision, a different alpha, an exactly reproducible run, or more headroom for hard-to-fit models.

## Simulation count

The number of simulated datasets sets the **precision** of the estimate: more simulations narrow the confidence interval around the power number but take longer. The defaults balance the two — **1,600** simulations for OLS and other single-fit designs, **800** for mixed-effects models (each fit costs more). Raising the count helps when two designs report nearly identical power and you need to tell them apart.

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
