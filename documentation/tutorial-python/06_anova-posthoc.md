# ANOVA & post-hoc

Factorial designs test whether group means differ and, when they do, which
specific pairs drive the difference. MCPower handles both questions in one call.

## The two questions

The **omnibus F test** answers: "do the group means differ at all?". It is a
single test — the power is the probability of detecting any non-zero treatment
effect.

**Pairwise post-hoc contrasts** answer: "which specific pairs differ?". Each
pair is a separate test with its own power, and the set of all `C(k,2)` pairs
for a `k`-level factor is a **family** to which [[concepts/multiple-testing|multiple testing]]
correction applies.

## Setting up a factorial design

Define the formula and mark the grouping variable as a factor with `set_variable_type("dose_group=(factor,0.34,0.33,0.33)")`. The `(factor, p1, p2, …)` notation sets group proportions that must sum to 1. Without uploaded data, levels are integer-labelled starting at 1 (level 1 = reference). With uploaded data the labels come from the column's distinct values.

Set effects as standardised mean differences vs the reference group using bracket notation: `set_effects("dose_group[2]=0.5, dose_group[3]=0.8")`. Effect-size benchmarks for factor predictors: small 0.20, medium 0.50, large 0.80.

> [!note] Sparse levels
> If any group's proportion times your sample size gives fewer than 5
> observations, MCPower warns before simulating and excludes that factor at
> that N (its effects report power 0). Details:
> [[limitations#Sparse factor levels at small N]].

## Omnibus F + all pairwise contrasts

Pass `target_test="overall, all-contrasts"` to see the omnibus F alongside
every pairwise comparison. The Overall F row (98.8%) is the omnibus test — the
probability of detecting that the group means differ at all:

<!-- example:06-contrasts -->
```python
from mcpower import MCPower

model = MCPower("pain_reduction = dose_group")
model.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")
model.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

result = model.find_power(sample_size=200, target_test="overall, all-contrasts", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: pain_reduction = dose_group
estimator: OLS  N=200  sims=1600  α=0.05  target=80%
effects: dose_group[2]=0.50, dose_group[3]=0.80

Per-test power
───────────────────────────────────────
Test                     Power   Target
───────────────────────────────────────
Overall F                98.8%      80%
dose_group  (pairwise)
  2 vs 1                 82.9%      80%
  3 vs 1                 99.5%      80%
  3 vs 2                 40.6%      80%
───────────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            98.8%   [98.1%, 99.2%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.4%       100%
1        5.1%      99.6%
2       65.7%      94.6%
3       28.9%      28.9%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

The pairwise contrasts nest under a `dose_group  (pairwise)` header — one row
per pair in canonical order. With integer labels `2 vs 1` means level 2 vs the
reference (level 1). With uploaded data the labels are the data values
(e.g. `low_dose vs placebo`).

The three contrasts tell different stories: `3 vs 1` (large effect, 0.80
standardised) is powered at 99.5%, `2 vs 1` (medium effect, 0.50) at 82.9%,
and `3 vs 2` (difference of two active doses, 0.30) at only 40.6% — well
below the 80% target. The omnibus F at 98.8% oversells the picture: you are
almost certain to detect *something*, but not the 3 vs 2 contrast.

> [!note]
> `"all-contrasts"` does not include marginal coefficient tests. Use
> `"all, all-contrasts"` to add those, or `"overall, all-contrasts"` for
> omnibus + pairwise without individual β rows.

## Family-wise correction with Tukey HSD

Without correction each pair is tested at α = 0.05, inflating the family-wise
error rate. Adding `correction="tukey"` applies Tukey HSD across the pairwise
family — the standard choice for balanced factorial designs:

<!-- example:06-tukey -->
```python
from mcpower import MCPower

model = MCPower("pain_reduction = dose_group")
model.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")
model.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

result = model.find_power(sample_size=200, target_test="overall, all-contrasts",
                          correction="tukey", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: pain_reduction = dose_group
estimator: OLS  N=200  sims=1600  α=0.05  target=80%
effects: dose_group[2]=0.50, dose_group[3]=0.80
correction: tukey

Per-test power
─────────────────────────────────────────────────────────
Test                     uncorrected   corrected   Target
─────────────────────────────────────────────────────────
Overall F                      98.8%      (same)      80%
dose_group  (pairwise)
  2 vs 1                       82.9%       71.2%      80%
  3 vs 1                       99.5%       98.7%      80%
  3 vs 2                       40.6%       26.4%      80%
─────────────────────────────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            98.8%   [98.1%, 99.2%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.4%       100%
1        5.1%      99.6%
2       65.7%      94.6%
3       28.9%      28.9%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

The omnibus F shows `(same)` in the corrected column — it is a single test and
is never part of the correction family. All three contrast powers drop: `2 vs 1`
falls from 82.9% to 71.2%, `3 vs 1` from 99.5% to 98.7%, and `3 vs 2` from
40.6% to 26.4%. Corrected values are always ≤ uncorrected.

Other correction methods (`bonferroni`, `holm`, `bh`) also apply per-family — pass e.g. `correction="bonferroni"` to `find_power`.

## Multi-factor designs

With more than one factor, `"all-contrasts"` generates a separate pairwise
family for **each** factor. The table renders one `<name>  (pairwise)` group
per factor, and correction is applied independently within each family:

<!-- example:06-multifactor -->
```python
from mcpower import MCPower

model = MCPower("score = treatment + region")
model.set_variable_type("treatment=(factor,0.5,0.5), region=(factor,0.34,0.33,0.33)")
model.set_effects("treatment[2]=0.4, region[2]=0.2, region[3]=0.5")

result = model.find_power(sample_size=150, target_test="overall, all-contrasts",
                          correction="tukey", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: score = treatment + region
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: treatment[2]=0.40, region[2]=0.20, region[3]=0.50
correction: tukey

Per-test power
────────────────────────────────────────────────────────
Test                    uncorrected   corrected   Target
────────────────────────────────────────────────────────
Overall F                     82.8%      (same)      80%
treatment  (pairwise)
  2 vs 1                      66.9%       66.9%      80%
region  (pairwise)
  2 vs 1                      18.3%        9.7%      80%
  3 vs 1                      69.6%       55.2%      80%
  3 vs 2                      31.2%       18.4%      80%
────────────────────────────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            82.8%   [80.8%, 84.5%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        8.9%       100%
1       26.8%      91.1%
2       34.6%      64.4%
3       28.9%      29.8%
4        0.9%       0.9%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

The output shows two `(pairwise)` groups — one for `treatment` (one pair,
since it is a two-level factor) and one for `region` (three pairs). Each group
has its own Tukey family, so the Tukey correction for `treatment` is applied
only over the single `treatment` pair, and the Tukey correction for `region`
only over the three `region` pairs.

next → [[07_multiple-testing|Multiple testing]]
