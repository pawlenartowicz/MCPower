---
title: "Supported outcome families"
description: "The four outcome families MCPower supports — linear, logit, probit, and Poisson — their link functions, baseline setters, and estimators."
---
# Supported outcome families

MCPower fits four outcome families, chosen with the `family=` argument to the constructor (the app exposes the same choice as an **Outcome type** dropdown: Linear, Logit, Probit, Poisson). Each family has its own link function and its own way of setting the reference-level baseline; everything else — the formula, effect sizes, `find_power`/`find_sample_size` — works the same way across all four.

| Outcome type | `family=` | Link | Baseline setter | Estimator | Typical use |
|---|---|---|---|---|---|
| Linear | `"ols"` (default) | identity | none — effects are direct shifts on the outcome's SD scale | OLS (clustered: linear mixed model, `family="lme"`) | continuous scores, indices, reaction times |
| Logit | `"logit"` | logit (log-odds) | [[#baseline-probability-logit-and-probit\|`set_baseline_probability(p)`]] | GLM (clustered: GLMM) | yes/no outcomes where effects read naturally as odds ratios |
| Probit | `"probit"` | probit (`Φ⁻¹`) | [[#baseline-probability-logit-and-probit\|`set_baseline_probability(p)`]] | GLM (clustered: GLMM) | yes/no outcomes where a normal-latent-variable model is preferred |
| Poisson | `"poisson"` | log | [[#baseline-rate-poisson\|`set_baseline_rate(λ)`]] | GLM (clustered: GLMM) | counts and event rates |

`"lme"` is not a fifth row here — it is the clustered form of the continuous (linear) family, selected the same way clustered logit/probit/Poisson are: add a `(1|group)` term to the formula and call `set_cluster`. See [[concepts/mixed-effects|mixed-effects models]] for the clustered path in general.

## Baseline probability (logit and probit)

Both binary families need one extra number the linear family doesn't: the outcome probability when every predictor sits at its reference level. Set it with `set_baseline_probability(p)`, `p` strictly between 0 and 1 — required for `family="logit"` and `family="probit"`, since without it the engine has no reference point to generate realistic binary outcomes from.

The two families turn that same `p` into different intercepts: logit maps it through the log-odds transform (`log(p / (1 - p))`), probit through the normal inverse CDF (`Φ⁻¹(p)`). Effect sizes are standardised on each family's own link scale, so a given standardised effect is not numerically identical between logit and probit — the two links have different scale factors — but the two families answer the same question (does this predictor move a binary outcome) and typically agree closely on power for the same design. Both start from the same small/medium/large 0.20 / 0.50 / 0.80 Cohen benchmarks (see [[concepts/effect-sizes#cohens-benchmarks|Cohen's benchmarks]]), but only logit additionally offers the odds-ratio preset set (beta = log(OR)) — a probit beta is already Cohen's *d* on the latent scale, not a log-odds, so `exp(beta)` there is not an odds ratio. See [[concepts/effect-sizes#Logistic regression|logistic effect sizes]] for the odds-ratio benchmarks.

## Baseline rate (Poisson)

A count outcome needs its own reference point: the expected count when every predictor is at its reference level. Set it with `set_baseline_rate(lambda)`, `lambda > 0` — required for `family="poisson"` for the same reason `set_baseline_probability` is required for the binary families: without a baseline the engine cannot generate realistic counts. Internally `lambda` becomes the log-link intercept, `ln(lambda)`.

Poisson effects are shifts on the **log-rate** scale — a predictor's coefficient is a log-rate-ratio, and `exp(beta)` is the multiplicative change in the expected count per unit (or per level, for binary/factor predictors). MCPower reuses the logit family's odds-ratio preset numbers (small 0.41 / medium 0.92 / large 1.39, Chen et al. 2010) as rate-ratio anchors for Poisson — same beta values, read as `RR = exp(beta)` instead of an odds ratio. See [[concepts/effect-sizes#Logistic regression|logistic effect sizes]] for where those numbers come from.

## Clustered counts and probits

Clustering works for all four families, but the random-intercept is sized differently depending on the link:

- **Logit / probit (clustered):** `set_cluster(grouping, ICC=..., n_clusters=...)` — the ICC is read on the latent scale, using each link's own latent-residual variance (logit: `π²/3`; probit: `1`). You still just pass an ICC; the conversion is internal.
- **Poisson (clustered):** Poisson has no natural latent-scale ICC, so the random intercept is sized directly by its **raw variance**: `set_cluster(grouping, tau_squared=..., n_clusters=...)`. Passing `ICC=` for `family="poisson"` is an error, and `tau_squared=` is only accepted for Poisson.

See [[concepts/mixed-effects#clustered-logistic-glmm|Clustered logistic (GLMM)]] for the worked binary case; the Poisson case follows the same pattern with `tau_squared` in place of `ICC`.

## Learn more

- [[concepts/effect-sizes|Effect sizes]] — what a standardised effect means per family.
- [[concepts/mixed-effects|Mixed-effects models]] — clustering, ICC, and GLMM convergence.
- [[concepts/simulation-settings#estimation-mode|Estimation mode]] — the Fast/Accurate/AGQ controls for clustered binary and count models.
