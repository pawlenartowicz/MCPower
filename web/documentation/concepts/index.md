---
title: "Statistical power analysis concepts"
description: "Introduction to statistical power analysis by simulation - how power, sample size, effect size, alpha, and model choices interact."
---
# Concepts: a power-analysis walkthrough

**Statistical power** is the probability that your study detects an effect that is really there. You pick a target — conventionally **80%**, the MCPower default — and MCPower finds the sample size that reaches it, or the power a given sample size buys. At 80% power a real effect is found four times in five; the missing fifth is the false-negative risk.

Formally, power is the chance of rejecting the null hypothesis when it is genuinely false, so the missing 20% at 80% power is the false-negative risk you accept in exchange for a manageable sample. 80% is convention, not law — chasing 90% or 95% shrinks that risk but demands a larger N, while a lower target accepts more missed effects for a smaller study. Set the target to whatever your field expects and your resources allow.

## What power depends on

Power is never a single number — it is the product of every modeling choice you make. Each driver below has its own page; together they trace the path from a research idea to a defensible sample size.

- **Outcome family** — linear, logit, probit, or Poisson; each has its own link function and baseline setter. See [[concepts/supported-families|supported families]].
- **Effect size** — how large the effect you expect is; the single most important input. See [[concepts/effect-sizes|effect sizes]].
- **Sample size (N)** — more observations mean more power; this is the lever a power analysis usually solves for.
- **Significance level (α)** — the false-positive rate you allow, **0.05** by default; a stricter α lowers power. See [[concepts/simulation-settings|simulation settings]].
- **Correlations** — how predictors move together, which can raise or lower power. See [[concepts/correlations|correlations]].
- **Variable types** — whether each predictor is continuous, binary, or a factor. See [[concepts/variable-types|variable types]].
- **Multiple testing** — how many coefficients you test, and how you correct for it. See [[concepts/multiple-testing|multiple testing]].
- **Clustering** — grouped or repeated-measures data and its intraclass correlation. See [[concepts/mixed-effects|mixed-effects models]].
- **Robustness** — how sensitive your power is to all of the above. See [[concepts/scenario-analysis|scenario analysis]].
- **Model specification** — the model you *test*, not just the data you generate, decides what you can detect. See [[concepts/model-misspecification|model misspecification]].
- **Required N** — how the sample-size answer itself is estimated, and what the curve and CI mean. See [[concepts/required-sample-size|how required N is estimated]].
