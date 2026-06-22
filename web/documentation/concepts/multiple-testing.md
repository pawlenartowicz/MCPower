---
title: "Multiple testing corrections and power"
description: "How multiple testing corrections (Bonferroni, Holm, FDR, Tukey) reduce power and how to size studies for corrected tests."
---
# Multiple testing

When you test several coefficients in one model, the chance of at least one false positive grows with the number of tests. A **correction** controls that inflated error rate — but it does so by raising the bar for significance, which *lowers power*. Multiple-testing corrections are therefore a power-analysis concern, not an afterthought: the correction you plan to use belongs in the analysis that sizes your study.

## Choosing which tests to target

Targeting picks **which coefficients your study must detect** — the effects your hypothesis is really about. MCPower lets you target every coefficient, only the first predictor, or a custom subset, and reports power for exactly the tests you select. Narrowing the targets often buys power: a smaller family of tests means a gentler [[concepts/multiple-testing#available-corrections|correction]].

## Two things you might control

- **FWER (family-wise error rate)** — the probability of making **any** false positive across all tests. Strict.
- **FDR (false discovery rate)** — the expected **proportion** of false positives among the results you call significant. Less strict, allows more discoveries.

## Available corrections

A correction adjusts the significance bar so that running many tests doesn't inflate your false-positive rate. The practical choice: **Bonferroni** (simplest, most conservative), **Holm** (always at least as powerful as Bonferroni — a safe default), **FDR / Benjamini–Hochberg** (most permissive, controls the false-discovery proportion), or **None** (raw p-values). **Tukey HSD** is a special case for pairwise factor comparisons only.

| Correction | Type | Behaviour |
|---|---|---|
| None | — | Raw p-values (default). |
| Bonferroni | FWER | Divides alpha by the number of tests. Most conservative. |
| Holm | FWER | Step-down procedure; uniformly more powerful than Bonferroni. |
| FDR (Benjamini–Hochberg) | FDR | Controls the expected proportion of false discoveries. Least conservative. |
| Tukey HSD | FWER | Pairwise factor comparisons only — post-hoc contrasts. |

## What forms the correction family

> [!note] The omnibus F-test is never corrected
> The overall F-test stands outside the correction family. Corrections apply only to the individual coefficient t-tests and post-hoc contrasts. The omnibus is reported for OLS and logistic (GLM) models only; [[concepts/mixed-effects|mixed-effects models]] have no overall test.

For Bonferroni, Holm, and FDR, all the individual t-tests and post-hoc comparisons you request form **one** family. If you test three things, the effective alpha per test under Bonferroni is $0.05 / 3 = 0.0167$ — a noticeably higher bar that your sample size has to clear.

**Tukey** is different: it applies **only** to post-hoc pairwise contrasts (the comparisons between [[concepts/effect-sizes|factor levels]]). A non-contrast test — a continuous covariate, say — has no Tukey-corrected power and is reported as not applicable.

| Correction | Applies to | Non-contrast tests |
|---|---|---|
| Bonferroni / Holm / FDR | all t-tests + post-hoc contrasts together | corrected |
| Tukey | post-hoc contrasts only | not applicable |
| None | — | raw p-values |

## Contrasts and post-hoc

A **pairwise contrast** tests whether two specific factor levels differ — group A versus group B — rather than asking the blanket "does this factor matter at all?" question a single coefficient answers. Add one when your hypothesis is about a *particular* comparison (treatment B vs the control level, say), and MCPower reports power for that exact difference alongside your other targets.

You can request contrasts two ways. Name a specific pair — `("treatment[B]", "treatment[A]")` — and MCPower powers just that comparison. Or request **post-hoc** comparisons for a factor, which power *all* pairs of its levels at once (the C(k,2) comparisons among k levels); pair these with the **Tukey HSD** [[concepts/multiple-testing#available-corrections|correction]], which is built for exactly this all-pairs case. Each contrast joins the correction family, so every comparison you add raises the bar on the rest.

## Power, not just p-values

Because a correction shrinks the alpha each test sees, the **corrected power** is what you should size a multi-coefficient study against — not the uncorrected number. MCPower computes both, so you can see exactly how much power a correction costs. The correction is also effectively free at runtime: the adjusted critical values are worked out once, before the simulations run, so comparing corrections doesn't slow anything down.

Related: [[concepts/effect-sizes|factor contrasts]] (where post-hoc comparisons come from) and [[concepts/scenario-analysis|scenario analysis]] (robustness of the corrected estimate). For the exact call and worked output, see the multiple-testing tutorial for your port.

## References

- Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52–64.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Tukey, J. W. (1953). *The problem of multiple comparisons*. Unpublished manuscript, Princeton University.
