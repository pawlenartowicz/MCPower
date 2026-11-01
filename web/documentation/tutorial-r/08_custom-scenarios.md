---
title: "Custom Scenarios for Sensitivity Analysis in R"
description: "Robustness power analysis in R: built-in scenario sweeps and custom scenario configs to stress-test assumptions with MCPower."
---
# Custom scenarios

Every power number rests on a set of assumptions — specific effect sizes, a
well-behaved residual structure, predictors drawn from the distributions you
specified. [[concepts/sensitivity-analysis|Sensitivity analysis]] asks a harder
question: *if those assumptions are optimistic, how much do we lose?*

MCPower answers it by re-running your analysis under alternative configurations
called **scenarios**, adding a column per scenario to the power table so you can
read robustness at a glance.

## The built-in sweep

Pass `scenarios = TRUE` to activate the built-in three-scenario sweep —
**optimistic**, **realistic**, and **doomer**. The built-in profiles are
calibrated to represent a plausible range of real-world conditions without
requiring you to guess at parameter values:

> [!note]
> Sensitivity analysis works for [[04_logistic-regression|logistic models]] too —
> the same call with `family = "logit"`. See
> [[concepts/sensitivity-analysis|Sensitivity analysis]] for which knobs apply per
> model family. Mixed models do not support scenarios yet.

<!-- example:08-builtin -->
```r
library(mcpower)

model <- MCPower$new("satisfaction ~ treatment + age")
model$set_variable_type("treatment=binary")
model$set_effects("treatment=0.5, age=0.3")

result <- model$find_power(sample_size = 150, target_test = "all", scenarios = TRUE, verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: satisfaction ~ treatment + age
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: treatment=0.50, age=0.30

Per-test power
─────────────────────────────────────────────────────────────
Test                 optimistic   realistic   doomer   Target
─────────────────────────────────────────────────────────────
Overall F                 98.9%       97.9%    94.9%      80%
treatment                 86.4%       82.9%    76.2%      80%
age                       95.2%       90.6%    82.2%      80%
─────────────────────────────────────────────────────────────

Power & 95% CI — optimistic
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            98.9%   [98.2%, 99.3%]
treatment            86.4%   [84.6%, 88.0%]
age                  95.2%   [94.1%, 96.2%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Power & 95% CI — realistic
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            97.9%   [97.0%, 98.5%]
treatment            82.9%   [81.0%, 84.7%]
age                  90.6%   [89.0%, 91.9%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Power & 95% CI — doomer
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            94.9%   [93.8%, 95.9%]
treatment            76.2%   [74.1%, 78.3%]
age                  82.2%   [80.2%, 84.0%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.9%       100%
1       16.6%      99.1%
2       82.5%      82.5%
────────────────────────

Robustness  (Δ power vs baseline: optimistic)
─────────────────────────────────────────
Test                 realistic     doomer
─────────────────────────────────────────
treatment              -3.4 pp   -10.1 pp
age                    -4.7 pp   -13.1 pp
─────────────────────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```
<!-- /example -->

The power table gains three columns, one per scenario. Power erodes as
conditions worsen, and the two coefficients erode differently. `age` starts at
95.2% under optimistic conditions, eases to 90.2% under realistic, and falls to
81.4% under doomer; `treatment` slides from 86.4% to 82.2% to 76.4% — dipping
below the 80% target by the doomer scenario. The **Robustness** section at the
bottom makes the deltas explicit: by the doomer scenario `age` has lost 13.9
percentage points and `treatment` 9.9. `age` is the more sensitive to departures
from the idealised setup, but `treatment` — starting lower — is the one that
slips under target first.

> [!tip]
> The built-in sweep is a fast sanity check. If you need to represent
> domain-specific assumptions — e.g. a known degree of measurement noise or a
> near-certain distributional shift — use custom scenarios instead.

## Custom scenario configs

`model$set_scenario_configs` accepts a named list of named lists. Each name is a
scenario name; each inner list is a set of **knobs** to override. The merge has
two branches: overriding a built-in name (`"optimistic"`, `"realistic"`,
`"doomer"`) *updates that preset* — knobs you don't set keep their preset
values — while a brand-new name inherits the optimistic values (except
`truth_start`, which starts off) and applies only the overrides you give.

The available knobs are:

| Knob | What it perturbs |
|---|---|
| `heterogeneity` | The true effect varies from study to study — on average it's the effect you set, but any given simulation might draw 80% or 120% of it (SD = knob × effect). At large values this puts a hard cap on achievable power, no matter the sample size — see [[concepts/limitations|Limitations]] |
| `heteroskedasticity_ratio` | Residual-variance ratio λ between high and low predicted values (1.0 = homoskedastic) |
| `correlation_noise_sd` | Jitter added to the predictor correlation structure |
| `distribution_change_prob` | Probability that a normal predictor is redrawn from a non-normal distribution |
| `sampled_factor_proportions` | Factor group sizes: fixed at the requested proportions (within one or two observations, the same in every simulation) when `FALSE` (the default), random per-observation assignment when `TRUE` |

Knob names are validated up front: an unknown or misspelled key raises an
error listing the valid keys. See
[[concepts/sensitivity-analysis|Sensitivity analysis]] for every knob's full
semantics.

`sampled_factor_proportions` applies to every factor at once. To fix one
factor's group sizes whatever the scenario says, use `set_factor_sampling`. It
works for declared factors and for factors from uploaded data:

```r
model$set_factor_sampling("group", "exact")    # exact group sizes in every scenario
model$set_factor_sampling("group", "sampled")  # random group sizes in every scenario
model$set_factor_sampling("group", NULL)       # follow the scenario again
```

After registering configs, pass a character vector of names to `scenarios` to
choose which ones to run. Names that appear in the vector but have no registered
config are treated as aliases for the built-in profiles — so `"optimistic"` here
uses the built-in optimistic baseline:

<!-- example:08-custom -->
```r
library(mcpower)

model <- MCPower$new("satisfaction ~ treatment + age")
model$set_variable_type("treatment=binary")
model$set_effects("treatment=0.5, age=0.3")

model$set_scenario_configs(list(
  realistic   = list(heteroskedasticity_ratio = 3.0, correlation_noise_sd = 0.20),
  stress_test = list(heterogeneity = 0.5, heteroskedasticity_ratio = 5.0,
                     distribution_change_prob = 0.9)
))

result <- model$find_power(sample_size = 150, target_test = "all",
                           scenarios = c("optimistic", "realistic", "stress_test"), verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: satisfaction ~ treatment + age
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: treatment=0.50, age=0.30

Per-test power
──────────────────────────────────────────────────────────────────
Test                 optimistic   realistic   stress_test   Target
──────────────────────────────────────────────────────────────────
Overall F                 98.9%       97.8%         93.0%      80%
treatment                 86.4%       82.9%         73.5%      80%
age                       95.2%       90.3%         78.2%      80%
──────────────────────────────────────────────────────────────────

Power & 95% CI — optimistic
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            98.9%   [98.2%, 99.3%]
treatment            86.4%   [84.6%, 88.0%]
age                  95.2%   [94.1%, 96.2%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Power & 95% CI — realistic
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            97.8%   [96.9%, 98.4%]
treatment            82.9%   [81.0%, 84.6%]
age                  90.3%   [88.8%, 91.7%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Power & 95% CI — stress_test
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            93.0%   [91.6%, 94.1%]
treatment            73.5%   [71.3%, 75.6%]
age                  78.2%   [76.2%, 80.2%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.9%       100%
1       16.6%      99.1%
2       82.5%      82.5%
────────────────────────

Robustness  (Δ power vs baseline: optimistic)
────────────────────────────────────────────
Test                 realistic   stress_test
────────────────────────────────────────────
treatment              -3.5 pp      -12.9 pp
age                    -4.9 pp      -17.0 pp
────────────────────────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```

![[examples/08-scenarios-r.png|600]]
<!-- /example -->

The chart below shows how power evolves across the three scenarios for each
test. Both coefficients slope downward as the knobs bite: `age` falls from
95.2% to 78.2% and `treatment` from 86.4% to 73.5% by the `stress_test` (high
heterogeneity, heavy heteroskedasticity, near-certain distributional shift),
which pushes both below the 80% target.

> [!note]
> A brand-new scenario name (like `stress_test`) inherits the optimistic
> baseline and applies only the keys you provide — omitted knobs stay at their
> optimistic defaults, except `truth_start`, which starts off. Overriding a
> built-in name (like `realistic` above) instead updates that preset: omitted
> knobs keep their *preset* values.

## Reading the output

Compare the `08-builtin` and `08-custom` results side by side. The `realistic`
column in both is nearly identical (~82% / ~90% for treatment/age) because
overriding a built-in name *updates* the preset: this custom `realistic` keeps
the realistic values for every knob it doesn't set and only raises
`heteroskedasticity_ratio` (2.0 → 3.0) and `correlation_noise_sd` (0.15 → 0.20).
The `stress_test` column replaces `doomer` and pushes harder on `heterogeneity`
and `distribution_change_prob`, landing near the doomer numbers for `age` (~80%
in both) while being explicitly designed around what you know about your domain.
That is the point of custom scenarios: you control what "pessimistic" means,
rather than accepting a one-size-fits-all profile.

next → [[09_upload-data|Upload data]]
