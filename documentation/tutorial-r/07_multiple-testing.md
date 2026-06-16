# Multiple testing

So far every rung has targeted one or two tests in isolation. Real studies often
report several coefficients — a panel of biomarkers, a battery of outcomes, a set
of independent predictors — and each additional test is another chance to claim a
false positive. Testing *m* hypotheses at α = 0.05 without any adjustment inflates
the family-wise error rate well above 5%. A correction controls it, but the price
is lower power on each individual test. See [[concepts/multiple-testing|multiple
testing]] for the underlying logic.

This rung uses a model with three continuous biomarkers (`biomarker1`, `biomarker2`,
`biomarker3`) predicted to have small-to-medium effects (0.30, 0.25, 0.20). We first
look at the uncorrected picture, then add `correction = "bonferroni"` to `find_power`
and read the difference.

> [!note]
> `correction =` is a `find_power` argument, not a setter. You can call the same
> model with and without a correction without rebuilding anything.

## Without correction

<!-- example:07-uncorrected -->
```r
library(mcpower)

model <- MCPower$new("response ~ biomarker1 + biomarker2 + biomarker3")
model$set_effects("biomarker1=0.3, biomarker2=0.25, biomarker3=0.2")

result <- model$find_power(sample_size = 200, target_test = "all", verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: response ~ biomarker1 + biomarker2 + biomarker3
estimator: OLS  N=200  sims=1600  α=0.05  target=80%
effects: biomarker1=0.30, biomarker2=0.25, biomarker3=0.20

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
Overall F            99.9%      80%
biomarker1           98.8%      80%
biomarker2           92.8%      80%
biomarker3           80.2%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            99.9%   [99.5%,  100%]
biomarker1           98.8%   [98.1%, 99.2%]
biomarker2           92.8%   [91.4%, 94.0%]
biomarker3           80.2%   [78.2%, 82.1%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.1%       100%
1        2.4%      99.9%
2       23.1%      97.5%
3       74.4%      74.4%
────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```
<!-- /example -->

All three biomarkers clear 80% power at N = 200. `biomarker3`, the weakest effect
(0.20), lands at **80.2%** — just above the target. The overall F test is nearly
certain at 99.9%, and the joint significance distribution shows all three detected
together 74.4% of the time.

## With Bonferroni correction

Bonferroni divides α by the number of tests (here, three). It is the most
conservative of the available options and the simplest to explain to reviewers:

<!-- example:07-corrected -->
```r
library(mcpower)

model <- MCPower$new("response ~ biomarker1 + biomarker2 + biomarker3")
model$set_effects("biomarker1=0.3, biomarker2=0.25, biomarker3=0.2")

result <- model$find_power(sample_size = 200, target_test = "all",
                           correction = "bonferroni", verbose = FALSE)
print(summary(result))
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: response ~ biomarker1 + biomarker2 + biomarker3
estimator: OLS  N=200  sims=1600  α=0.05  target=80%
effects: biomarker1=0.30, biomarker2=0.25, biomarker3=0.20
correction: bonferroni

Per-test power
─────────────────────────────────────────────────────
Test                 uncorrected   corrected   Target
─────────────────────────────────────────────────────
Overall F                  99.9%      (same)      80%
biomarker1                 98.8%       96.2%      80%
biomarker2                 92.8%       85.4%      80%
biomarker3                 80.2%       67.1%      80%
─────────────────────────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F            99.9%   [99.5%,  100%]
biomarker1           96.2%   [95.2%, 97.1%]
biomarker2           85.4%   [83.6%, 87.0%]
biomarker3           67.1%   [64.8%, 69.4%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.1%       100%
1        2.4%      99.9%
2       23.1%      97.5%
3       74.4%      74.4%
────────────────────────

Plots: plot(result) to view, plot(result, 'chart.png') to save.
```
<!-- /example -->

The table now shows both an **uncorrected** and a **corrected** column. The overall
F test is unaffected — it is a single omnibus test, not part of the corrected family.
For the coefficient tests the drop is real:

| Test        | Uncorrected | Bonferroni |
|-------------|-------------|------------|
| biomarker1  | 98.8%       | 96.2%      |
| biomarker2  | 92.8%       | 85.4%      |
| biomarker3  | 80.2%       | 67.1%      |

`biomarker3` is hit hardest because it sits closest to the 80% threshold; the
correction pushes it to **67.1%**, well below target. If you need to maintain 80%
power for all three biomarkers under Bonferroni, you must increase N.

> [!tip]
> Two alternatives soften the penalty. **Holm** (`correction = "holm"`) is uniformly
> more powerful than Bonferroni while still controlling the family-wise error rate —
> it should be your default whenever Bonferroni is on the table.
> **Benjamini–Hochberg** (`correction = "bh"`) controls the false-discovery rate
> rather than the family-wise error rate, which is appropriate when you are
> comfortable allowing a small proportion of false positives among significant
> findings.

## Distinguishing rung 06 from this rung

Rung 06 applied Tukey's HSD to the **pairwise contrasts of a single factor** —
a single categorical variable with multiple levels, where the family is the set of
level comparisons. Here the family is **several separate coefficients in the same
regression**, each an independent substantive question. The correction mechanism
is the same idea; the family is defined differently.

next → [[08_custom-scenarios|Custom scenarios]]
