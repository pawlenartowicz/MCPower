---
title: "Power Analysis with Correlated Predictors in R"
description: "Statistical power analysis with correlated predictors in R - string and matrix forms, and why correlation erodes power."
---
# Correlations

So far the model has assumed that `stress`, `exercise`, and `sleep` are drawn
independently. Real predictors rarely are — and ignoring their correlations
means your power estimate could be off. This rung shows how to tell MCPower
about predictor correlations and why it matters.

## Why correlation erodes power

When two predictors are correlated they carry overlapping information about the
outcome. The regression must disentangle those overlapping signals, and that
disentanglement costs statistical precision. The more correlated the predictors,
the harder each one is to detect — see [[concepts/correlations|why correlation
matters]] for the derivation.

The effect is not always symmetric: a predictor with a modest correlation to one
strong predictor may lose only a little power, while one caught between two
strongly correlated companions loses substantially more.

## The study

We model self-reported wellbeing as a function of three lifestyle predictors:
`stress` (negative), `exercise`, and `sleep`. All three have plausible medium
effects. Because they are lifestyle variables they are correlated in real
populations — stressed people sleep less, and they also exercise less.

## Two ways to describe the same structure

MCPower accepts predictor correlations in two equivalent forms. Pick whichever
fits your workflow.

### 1. String form — good for a sparse set of pairs

Name each pair explicitly with `corr(a, b)=value`:

<!-- example:03-correlations-string -->
```r
library(mcpower)

model <- MCPower$new("wellbeing ~ stress + exercise + sleep")
model$set_effects("stress=-0.3, exercise=0.25, sleep=0.3")
model$set_correlations("corr(stress, exercise)=-0.3, corr(stress, sleep)=0.4, corr(exercise, sleep)=0.2")

result <- model$find_power(sample_size = 150, target_test = "all", verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: wellbeing ~ stress + exercise + sleep
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: stress=-0.30, exercise=0.25, sleep=0.30

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
Overall F            99.9%      80%
stress               85.8%      80%
exercise             74.8%      80%
sleep                87.1%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            99.9%   [99.6%,  100%]
stress               85.8%   [84.0%, 87.4%]
exercise             74.8%   [72.6%, 76.8%]
sleep                87.1%   [85.3%, 88.6%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.4%       100%
1        6.8%      99.6%
2       37.8%      92.9%
3       55.1%      55.1%
────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```
<!-- /example -->

`stress` reaches **85.8%** power, `sleep` **87.1%**, but `exercise` falls just
short at **74.8%** — exercise is the most "squeezed" predictor, sandwiched
between two variables it correlates with in opposite directions.

### 2. Matrix form — good for a complete correlation structure

Pass a base `matrix(...)` whose rows and columns follow the predictor order in
the formula (`stress`, `exercise`, `sleep`):

<!-- example:03-correlations-matrix -->
```r
library(mcpower)

model <- MCPower$new("wellbeing ~ stress + exercise + sleep")
model$set_effects("stress=-0.3, exercise=0.25, sleep=0.3")

# rows/cols follow the predictors in formula order: stress, exercise, sleep
correlation_matrix <- matrix(c(
   1.0, -0.3, 0.4,
  -0.3,  1.0, 0.2,
   0.4,  0.2, 1.0
), nrow = 3, byrow = TRUE)
model$set_correlations(correlation_matrix)

result <- model$find_power(sample_size = 150, target_test = "all", verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: wellbeing ~ stress + exercise + sleep
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: stress=-0.30, exercise=0.25, sleep=0.30

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
Overall F            99.9%      80%
stress               85.8%      80%
exercise             74.8%      80%
sleep                87.1%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            99.9%   [99.6%,  100%]
stress               85.8%   [84.0%, 87.4%]
exercise             74.8%   [72.6%, 76.8%]
sleep                87.1%   [85.3%, 88.6%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.4%       100%
1        6.8%      99.6%
2       37.8%      92.9%
3       55.1%      55.1%
────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```
<!-- /example -->

The per-test powers are identical to the string form: **85.8% / 74.8% / 87.1%**.
The two inputs describe exactly the same correlation structure, so you can use
whichever is more convenient — a sparse string for a handful of pairs, a full
matrix when you have a complete correlation table from a pilot study.

> [!note]
> Whichever form you use, MCPower validates it before running: each correlation
> must be in $[-1, 1]$, a full matrix must be square and symmetric with an all-1s
> diagonal, and the structure as a whole must be positive semi-definite.

## Reading the output

The joint significance distribution shows that all three effects are
simultaneously detected in **55.1%** of simulations at N = 150. That is the
number to budget for when you want to claim that *every* predictor is
significant — not just any one of them.

> [!tip]
> If a single underpowered predictor is dragging down the joint probability, try
> `find_sample_size` to find the n at which it clears 80%. The other predictors
> will already be above target by then, so the joint bar rises with the weakest
> test.

next → [[04_logistic-regression|Logistic regression]]
