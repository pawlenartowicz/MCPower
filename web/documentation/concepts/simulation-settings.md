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
