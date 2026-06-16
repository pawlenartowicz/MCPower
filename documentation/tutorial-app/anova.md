# ANOVA power (app)

ANOVA builds its model from structured **factor** and **covariate** cards, not a typed formula. The panel has two sections — **ANOVA factors** and **Covariates** — each with its own Add button. Top to bottom:

## 1. Add factors

Click **Add factor** to create a factor card; factors are auto-named F1, F2, …. Each card's header contains:

- An **editable name** input — rename the factor here.
- A static **factor** kind badge (primary ANOVA factors are always factors).
- A **levels stepper** — set the number of levels (minimum 2). Each non-reference level becomes one comparison, so a factor with k levels contributes k − 1 effect rows.
- An **Advanced** (⚙) button — opens the Advanced dialog for that factor (see below).
- A **remove** (trash) button.

Inside the card, effect rows list each comparison in order: the reference level appears first as disabled text, then one labelled effect row per non-reference level (`[2]`, `[3]`, …). Set the standardised effect size on each row — **0.20 / 0.50 / 0.80** (small / medium / large); prefer a value from prior evidence. See [[concepts/effect-sizes|effect sizes]] and [[concepts/variable-types|variable types]].

### Factor Advanced dialog

Click ⚙ on a factor card to open its Advanced dialog. Here you can:

- **Set level labels** — rename the levels from their default index labels.
- **Set shares** — the relative size of each level. Shares are weights: they don't need to sum to 100%, the app rescales them automatically. The optional **Rescale shares to sum to 100%** button just normalises the displayed numbers.
- **Choose the reference level** — the level all others are compared against.
- **Sampled-shares toggle** — simulate group proportions as random draws rather than fixed values.

> [!note] Sparse levels
> If any group's proportion times your sample size gives fewer than 5
> observations, MCPower warns before simulating and excludes that factor at
> that N (its effects report power 0). The Diagnostics panel names the factor
> and the minimum N needed. Details:
> [[concepts/limitations#Sparse factor levels at small N]].

## 2. Add covariates (optional)

Click **Add covariate** to create a covariate card; covariates are auto-named cov1, cov2, …. Unlike factor cards, the **kind** badge is switchable — choose **continuous**, **binary**, or **factor**. Set the standardised effect size in the card (continuous benchmarks: **0.10 / 0.25 / 0.40**; binary or factor: **0.20 / 0.50 / 0.80**). This is an ANCOVA design: the covariate is included as a predictor to adjust for. See [[concepts/variable-types|variable types]].

## 3. Robustness scenarios

The **Robustness scenarios** toggle in the status bar repeats every run under three perturbation sets — **Optimistic** (your exact settings, no perturbations), **Realistic** (moderate assumption violations), and **Doomer** (severe violations, a worst case) — so you get a *range* of power instead of one optimistic number. If even Doomer clears your target, the design is robust; if only Optimistic reaches it, increase the sample size. Each set's knobs are editable under **Settings → Scenarios**. See [[concepts/scenario-analysis|scenario analysis]].

## 4. Optional settings

- **Tests & post-hoc corrections** — pick the omnibus test and any pairwise comparisons between factor levels. **Tukey** is the standard all-pairwise post-hoc (Bonferroni, Holm also available); correcting controls false positives at the cost of power. [[concepts/multiple-testing|multiple testing]]
- **Advanced** — number of simulations (ANOVA defaults to 1,000), α (0.05), seed (2137), and the failed-simulation tolerance.
