---
title: "Effect sizes for power analysis"
description: "Standardized effect sizes (betas) for power analysis: continuous, binary, and logistic regression benchmarks, Cohen conventions, and odds-ratio anchors for logit models."
---
# Effect sizes

An **effect size** in MCPower is a **standardized regression coefficient** (a beta): how strongly one predictor moves the outcome, on a scale that is comparable across predictors. You set one per predictor — the effect you expect it to have. It is the single most important input to a power analysis, since everything else (sample size, correlations, scenarios) is calibrated against it.

Overestimate your effects and you run an underpowered study; underestimate them and you waste resources on a sample larger than you need. The sections below explain how a beta is interpreted, what benchmarks to reach for, and how the interpretation shifts for binary outcomes.

## Standardized betas

In MCPower an effect size is a **standardized regression coefficient** (a beta). A beta of 0.30 means a one–standard-deviation increase in the predictor is associated with a 0.30 SD change in the outcome. Standardizing this way makes effects comparable across predictors measured on different scales.

For binary and factor predictors the interpretation shifts slightly:

- A **binary** effect of 0.50 is the change in the outcome (in SD) when switching from the reference group (0) to the treatment group (1).
- A **factor** predictor contributes one effect per non-reference level — each is the difference between that level and the reference. See [[concepts/variable-types|variable types]].

Predictors you don't assign an effect to are treated as having no effect (beta = 0).

## Continuous vs binary outcomes

What an effect size means depends on your outcome. A **continuous** outcome (a measured quantity — a score, a price, a reaction time) is fit with linear regression, and the effect is a standardized beta: the SD change in the outcome per 1-SD change in the predictor, as described in [Standardized betas](#standardized-betas) above. A **binary** outcome (yes/no, success/failure) is fit with logistic regression, where the effect is a beta on the **log-odds** scale and you also set a **baseline probability**.

These two cases use different effect-size benchmarks — see [Cohen's benchmarks](#cohens-benchmarks). For how logit betas relate to odds ratios, see [Logistic regression](#logistic-regression); for what the baseline anchors, see [Baseline probability](#baseline-probability).

## Cohen's benchmarks

With no pilot data or prior literature to go on, Cohen's (1988) conventions are a starting point:

| Predictor type | Small | Medium | Large |
|---|---|---|---|
| Continuous | 0.10 | 0.25 | 0.40 |
| Binary / factor | 0.20 | 0.50 | 0.80 |

> [!note] Why two different scales?
> For **binary/factor** predictors MCPower keeps the variable as 0/1, so the beta equals Cohen's *d* directly — the 0.20/0.50/0.80 benchmarks are Cohen's standard conventions. For **continuous** predictors the variable is standardized to $N(0,1)$, so beta is the change in $Y$ per 1-SD change in $X$. The continuous benchmarks are calibrated so that a "medium" continuous effect ($\beta = 0.25$) yields about the same power as a "medium" binary effect ($d = 0.50$) at the same sample size. The labels *small / medium / large* therefore carry consistent practical meaning across predictor types.

### What a beta looks like in the data

| Beta | What it looks like |
|---|---|
| 0.10 | Barely noticeable in raw data; needs large samples to detect. |
| 0.20 | Small but real; visible with careful measurement. |
| 0.30 | Moderate; a trained observer would notice the pattern. |
| 0.50 | Clearly visible; obvious group differences in plots. |
| 0.80+ | Dramatic; hard to miss even in small samples. |

## Interactions

Interaction effects are typically **smaller** than main effects — values of 0.10–0.20 are common. Plan for lower power when an interaction is your target, and see [[concepts/correlations|correlations]] for why correlated predictors actually make interactions *easier* to detect.

## Factor contrasts

Each factor level carries its own beta relative to the reference. If two non-reference levels have betas of 0.30 and 0.70, the implicit contrast between them is the difference (0.40) — which is exactly what a post-hoc pairwise comparison tests. See [[concepts/multiple-testing|multiple testing]] for how those contrasts are corrected.

## Logistic regression

The **odds** of an event are the chance it happens divided by the chance it doesn't (a 30% chance is odds of $0.3 / 0.7 \approx 0.43$); **log-odds** are the natural log of that. For a logistic (logit) model, betas are **standardized log-odds** coefficients — not raw odds ratios. A continuous predictor at $\beta = 0.5$ multiplies the odds of the outcome by $e^{0.5} \approx 1.65$ per 1-SD increase; a binary predictor at $\beta = 0.7$ multiplies them by $e^{0.7} \approx 2.0$ when switching from 0 to 1. The same log-odds beta interpretation applies when the binary outcome is **clustered** — a [[concepts/mixed-effects#Clustered logistic (GLMM)|clustered logistic GLMM]] — not just plain logistic regression.

### Odds-ratio benchmarks (beta)

When it is easier to think in **odds ratios** than in log-odds betas, MCPower offers an odds-scale benchmark set, following Chen, Cohen & Chen (2010):

| Level | Odds ratio | Beta = log(OR) |
|---|---|---|
| Small | 1.5 | 0.41 |
| Medium | 2.5 | 0.92 |
| Large | 4.0 | 1.39 |

This is a **beta** feature. The values are stored as the log-odds beta MCPower already works in, so nothing about data generation changes — only the benchmark you reach for. In the app, the small/medium/large preset buttons switch to this odds set for every predictor whenever the outcome is **logit**, and each effect input shows its live odds ratio ($\mathrm{OR} = e^{\beta}$) beside the value. In Python and R the printed summary echoes the OR beside each beta you set and adds an **OR** column to the per-test power table.

Read the odds ratio on the **same scale as the beta**: **per 1 SD** for a continuous predictor, **per category** (reference → level) for a binary or factor predictor. The number $e^{\beta}$ is the same either way; only the wording of the unit differs.

> [!note] Probit and Poisson read this table differently
> **Probit** does not use this odds-scale set at all: a probit beta's latent variable has variance exactly 1, so the beta already *is* Cohen's *d* — the app keeps the plain continuous/binary Cohen presets for probit and shows no OR readout ($e^{\beta}$ there is not an odds ratio). **Poisson** reuses the same three beta values (0.41 / 0.92 / 1.39) as **rate-ratio** anchors, since its beta is a log-rate-ratio: $\mathrm{RR} = e^{\beta}$ replaces the OR readout, small/medium/large corresponding to RR 1.5 / 2.5 / 4.0. The anchors are the same numbers Chen et al. (2010) derived for odds ratios, borrowed here as a multiplicative scale for rates — the citation doesn't carry over, only the magnitudes do.

To convert an odds ratio reported per **raw unit** of $x$ in the literature:

$$
\beta = \log(\mathrm{OR}) \times \mathrm{sd}(x_{\text{raw}})
$$

Working in probabilities makes the size of a beta concrete: starting from a 30% baseline, a binary predictor at $\beta = 0.5$ raises the event probability to roughly 41%.

> [!warning] Implausibly large logit effects
> A standardized log-odds beta above $|\beta| > 3$ (OR ≈ 20) is rare in practice, and above $|\beta| > 5$ (OR ≈ 150) almost always signals a misread odds ratio or a unit error. MCPower flags these.

## Baseline probability

For a logistic model you also set the **baseline probability** — the expected probability of the outcome when every predictor sits at its reference level (continuous predictors at their mean, binary and factor predictors at 0). This single number fixes the model's intercept, anchoring where on the S-shaped logistic curve your data sits. It matters for power: a baseline near 0.5 leaves the most room for predictors to move the outcome, while a baseline close to 0 or 1 compresses that room — rare or near-certain outcomes need larger samples to reach the same power.

## Typical effect sizes by field

| Field | Predictor | Typical beta | Notes |
|---|---|---|---|
| Education | Teaching intervention | 0.20–0.40 | Medium effects common |
| Education | Socioeconomic status | 0.15–0.30 | Small–medium |
| Psychology | Therapy vs. control | 0.30–0.60 | Medium–large |
| Psychology | Personality trait | 0.10–0.25 | Small–medium |
| Medicine | Drug vs. placebo | 0.20–0.50 | Varies widely |
| Medicine | Lifestyle factor | 0.05–0.20 | Often small |
| Social science | Policy intervention | 0.10–0.30 | Small–medium |
| Marketing | Ad exposure | 0.05–0.15 | Typically small |

> [!tip] Rules of thumb
> - When in doubt, use a **smaller** effect size — overestimation is the more costly error.
> - Use [[concepts/scenario-analysis|scenario analysis]] to test how sensitive your power is to effect-size uncertainty.
> - If prior work reports Cohen's *d*, it maps roughly onto the beta of a binary predictor.
> - Main effects are almost always larger than interaction effects in the same model.

## References

- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Chen, H., Cohen, P., & Chen, S. (2010). How big is a big odds ratio? Interpreting the magnitudes of odds ratios in epidemiological studies. *Communications in Statistics — Simulation and Computation*, 39(4), 860–864.
