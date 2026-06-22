---
title: "Upload pilot data - MCPower app"
description: "Drive MCPower simulations from pilot data - upload a CSV to set predictor distributions, choose resampling mode, and borrow effect sizes from real data."
---
# Upload data (app)

When you have **pilot data** — a small study, a prior experiment, a public dataset — you can hand it to the app instead of describing each predictor from scratch, and let the real distributions and dependence structure drive the simulation. As everywhere in MCPower, the outcome is still simulated from the effect sizes you set: uploading shapes the **predictors**, not the outcome. See [[concepts/upload-data|using empirical data]] for the statistics. The **Upload data** section appears for the **Regression** and **Mixed** panels (not ANOVA). Top to bottom:

## Load a file

Expand the collapsible **Upload data** section above Model and pick a `.csv` file. The app parses it in place — nothing leaves your machine. A type-summary panel then lists every column with its detected **Type** and **Levels**: a factor shows its count and labels (`cyl` → `3 (4, 6, 8)`, `origin` → `3 (Europe, Japan, USA)`), a binary column shows `2`, a continuous column shows `—`, with the row and column totals beneath. Use the **+** next to any column to add it to your formula (it greys out once the column is already there). See [[concepts/variable-types|variable types]].

## Matched predictors lock to the data

Any predictor whose name matches an uploaded column gets a **from data** badge, and its type, levels, and share controls lock — the data fixes the variable's class, so the simulation can't model it as something the column isn't. Only the **effect size** stays editable: you still choose what to detect, the data only supplies the distribution.

## Choose a mode

Three buttons set **how faithfully** the synthetic predictors follow the real ones — **Partial** is the default:

| Mode | Reproduces | Draws |
|------|------------|-------|
| **None** | each predictor's **marginal** distribution | fresh synthetic rows, predictors independent |
| **Partial** *(default)* | marginals **plus** the measured correlations among continuous predictors | fresh synthetic rows |
| **Strict** | the **full empirical joint** — every value combination as observed | resamples whole rows (bootstrap) |

**Strict** draws rows with replacement, so some pilot rows repeat within a simulated dataset; it is the only mode that preserves nonlinear dependence and the joint structure of categorical predictors. When the sample size grows past roughly twice the uploaded row count, the Summary pane warns that bootstrap reuse is high — at that point prefer Partial or None, which generate fresh rows at any size. See [[concepts/upload-data|the three modes]].

## Correlations follow the data

In **Partial** mode the Correlations triangle previews the pairwise correlations measured from your data for each continuous pair, as editable starting points — leave a cell and the data drives it, or type a value to override that pair. Correlations are continuous-only; binary and factor predictors are drawn from their marginals. In **Strict** mode the uploaded predictors drop out of the triangle entirely (whole-row resampling already preserves their dependence). See [[concepts/correlations|predictor correlations]].

## Borrow starting effects

With a file loaded, a **Fit from data** button appears in the Predictors section (continuous, binary, or mixed outcomes — not ANOVA). Clicking it fits your model to the data and shows a **preview**: each recovered effect listed as `name = value`, plus a note naming the estimator used (standardized OLS, logistic log-odds, or mixed-model fixed effects). For a **mixed** model the preview also reports the **estimated ICC** recovered from the random-intercept fit, and for a **binary** outcome the **estimated baseline probability** from the fitted intercept. Nothing changes until you click **Apply**, which writes the effects onto the predictor rows, the ICC onto the cluster card, and the baseline onto the baseline-probability input. These are an **approximation, not a target** — they carry the pilot's sampling error, so treat them as a first guess to vary, and edit before you trust them. See [[concepts/effect-sizes|effect sizes]].

## Run, and clear

Run **Find power** or **Find sample size** as usual — the uploaded data is included automatically. After a Strict run, the Summary pane reports the bootstrap reuse fraction (and the high-reuse warning when it applies). The **Clear** link next to the filename removes the data and restores the file picker.
