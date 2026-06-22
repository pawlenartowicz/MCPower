---
title: "Using pilot data in power analysis"
description: "Using pilot data in power analysis: distribution mapping vs. bootstrap modes, column typing, and borrowing starting effect sizes."
---
# Using empirical data

Most of this guide assumes you describe your design from scratch: you pick each
predictor's distribution, set correlations, and choose effect sizes. If you
already have **pilot data** — a small study, a previous experiment, a public
dataset — you can hand it to MCPower instead and let the real distributions and
dependence structure drive the simulation.

This page explains *what* uploading data does to the analysis. For the code, see
[[tutorial-python/09_upload-data|the Python walkthrough]] (or the
[[tutorial-r/09_upload-data|R]] / [[tutorial-app/upload-data|app]] equivalents).

## What uploaded data changes — and what it doesn't

Uploading data shapes the **predictors only** — the `X` side of the model. The
outcome is *always* simulated from the effect sizes you set, in every mode. Your
pilot's outcome column is never copied into the simulation; power is still the
question "if the true effects are *these*, how often would a study of size *N*
detect them?", and you remain in control of "these".

The one exception is [[concepts/upload-data#borrowing-starting-effects|`get_effects_from_data`]],
which reads an outcome column purely to *suggest* starting effect sizes — it
does not feed that column into the power simulation.

So uploading answers a narrower question than it might first appear: *given my
predictors look and co-vary like this real sample, how much power do my assumed
effects buy me?*

## Preparing your data

MCPower expects a **tidy** table:

- **one row per observation**, one column per variable;
- a **header row** naming each column — use the same names in your formula;
- **factors as their labels** (`"Japan"`, `"USA"`, `"Europe"`), not pre-encoded
  dummy columns;
- **no index/ID column** — drop row numbers and identifiers, or they will be
  read as a predictor.

**CSV is the canonical form**, but the *same shape* applies to an in-memory
frame — the tutorials show both paths (a file path or a dataframe you have
already loaded). Files are read locally by the package itself, never uploaded
anywhere — the native engine only ever sees validated, typed columns.

### How many rows

- **Minimum 20 rows** — a **hard limit**. The package rejects smaller uploads,
  because with fewer rows the empirical quantile-matching that drives the
  synthetic draws becomes unreliable.
- **Maximum 1,000,000 rows** for the Python and R packages and the desktop app — a
  **hard limit**: larger uploads are rejected. The simulation gets slow for very little
  statistical gain well before that, so in practice you rarely want anywhere near the cap.
- **The browser (web) app caps lower — 10,000 rows — and enforces it.** It runs
  in a single browser worker with a smaller memory budget, so it rejects files
  above that limit. If you need to drive a larger dataset, use the desktop app
  instead.

> [!note] Upload row caps at a glance
> The desktop app, Python, and R **hard-cap at 1,000,000 rows** — larger files are rejected
> (and the simulation is slow long before that). The browser app caps at **10,000 rows** and
> *enforces* it too: files above that limit are rejected. To drive a larger dataset in the
> browser, switch to the desktop app.

### How columns are typed

On upload, each column is classified and a short summary is printed so you can
confirm the engine read your data the way you intended:

- **continuous** — a numeric column with many distinct values;
- **binary** — exactly two distinct values;
- **factor** — a text column, or a numeric column with only a few distinct
  levels (a rows-per-level guard keeps genuinely continuous columns from being
  mistaken for factors).

Factor **level names are preserved**, so a `cyl` column with values 4/6/8 becomes
the coefficients `cyl[6]` and `cyl[8]` (relative to the reference level), and an
`origin` column becomes `origin[Japan]`, `origin[USA]`. You refer to those exact
names when you set effects or read results.

## Three modes

When you `upload_data(mode=...)`, that `mode` string sets how faithfully your real
predictors are reproduced — choose one of three. `none` keeps only each
predictor's shape (its marginal) and draws fresh synthetic rows; `partial` (the
default) does the same but also keeps the correlations measured among your
continuous predictors; `strict` resamples whole real rows, preserving the full
joint distribution.

Underneath, those three modes are **two** mechanisms. `none` and `partial` both
**map the distribution** of your data and draw fresh synthetic rows from it;
`strict` instead **bootstraps whole rows** of the real data. Everything else
about uploading follows from which mechanism a mode uses.

| `mode=` string | Mechanism | What happens |
|---|---|---|
| `none` | **Distribution mapping** | Estimate each predictor's marginal; draw **fresh synthetic rows**. Predictors independent unless you set correlations yourself. |
| `partial` *(default)* | **Distribution mapping** | Same, **plus** fit the correlation matrix among continuous predictors (NORTA) and install it as an overridable default. |
| `strict` | **Strict bootstrap** | Resample **whole real rows** with replacement. Preserves the full empirical joint. |

### Distribution mapping — `none` and `partial`

Distribution mapping estimates the **marginal** of each predictor and draws
**fresh synthetic rows** from it, so it can produce a sample of any size, not
just copies of what you uploaded.

- **`none`** matches the *shape* of each predictor one at a time (its empirical
  marginal) but treats predictors as independent unless you set correlations
  yourself.
- **`partial`** (the default) additionally measures the **correlation matrix**
  among **continuous** predictors and installs it as a starting point. An
  explicit `set_correlations(...)` always overrides the measured value,
  regardless of the order you call things in — measured correlations are a
  *default*, your values are a *decision*.

  Because each continuous predictor is redrawn through its own empirical
  distribution (a rank-preserving transform), the quantity `partial` reproduces
  faithfully is the **rank (Spearman) correlation** of your data, not the Pearson
  correlation. MCPower measures the Spearman correlation and reproduces it
  exactly in the generated data; the realized Pearson correlation is then
  *approximate* — it is whatever your predictors' marginal shapes imply, and
  matches the uploaded Pearson only when the dependence is close to Gaussian. A
  value you set yourself with `set_correlations(...)` is treated as an ordinary
  (latent) correlation parameter, not converted.

The measured correlations in `partial` cover **continuous predictors only**.
Binary and factor predictors are reproduced from their **marginals** — their
proportions are faithful — but on this synthetic-draw path they are drawn
**independently** of the other predictors, even if they were associated in your
data. This is deliberate: forcing a target correlation *through* a binary or
categorical variable produces unnatural, hard-to-interpret joint behaviour, so
MCPower does not attempt it. If the **joint** dependence between a categorical
predictor and the rest of your design genuinely matters for your power question,
that is exactly what the strict bootstrap preserves.

### Strict bootstrap — `strict`

`strict` stops modelling the distribution and instead **resamples whole rows**
from your data, with replacement. It is the only mechanism that preserves
nonlinear dependence, higher moments, and exact correlations among **all**
predictors — including the categorical joint structure that distribution mapping
approximates — because it never builds a correlation matrix at all. The flip
side: it can only ever produce predictor combinations that appear in the data.

Because `strict` draws rows **with replacement**, some rows from your pilot will
appear more than once in any given simulated dataset. MCPower prints a reuse
diagnostic whenever you run in strict mode:

```
Strict mode: estimated reuse ~26% (N=100, U=100 uploaded rows)
```

The expected fraction of uploaded rows that appear **more than once** (i.e. are
reused) within a single simulated dataset of N rows drawn from U uploaded rows
is given by:

```
g = 100 · [1 − (1 − 1/U)^N − (N/U)(1 − 1/U)^(N-1)]
```

Two golden benchmarks: when **N = U** (simulated sample equals the upload size)
about **26%** of uploaded rows are reused; when **N = 2U** about **59%** are
reused. The reuse itself is not a problem — bootstrap resampling is well-founded
— but it becomes a concern when N greatly exceeds U, because then many synthetic
rows are identical copies of a few real observations.

> [!warning] When N > 2·U
> MCPower prints a warning when the simulated sample size is more than twice the
> number of uploaded rows. At that ratio, `strict` introduces heavy repetition
> that can make your synthetic predictor space look narrower than the real one.
> Consider switching to distribution mapping (`partial` or `none`), which
> generates fresh synthetic rows at any sample size.

One [[concepts/scenario-analysis|scenario]] knob interacts with the choice of
mechanism: `sampled_factor_proportions`. Under distribution mapping, factors from your
data obey it as usual — their empirical proportions are the target, hit exactly
(`false`) or with multinomial jitter (`true`). Under `strict` the knob has no
effect on uploaded predictors: group sizes follow whatever rows the bootstrap
draws, because planning exact cell counts would break apart the very joint
structure strict preserves. Only a factor you declare on top of the upload —
one that is not a column of your data — is still simulated and still obeys the
knob.

### Which do I pick?

**Distribution mapping** scales to any N and is smooth — reach for it when you
want realistic predictor variation at a sample size larger than your pilot, and
the continuous correlation structure is the dependence that matters. **Strict
bootstrap** is the faithful-joint choice — reach for it when categorical
dependence or some complex joint structure matters and your target N is not far
above U. Keeping both on one page is deliberate: the side-by-side is the whole
decision.

## Borrowing starting effects

`get_effects_from_data(y)` fits your specified model to the uploaded data and
returns a ready-to-paste `set_effects(...)` string, with `y` naming the outcome
column. It is a convenience for turning a pilot into a starting point — the
values come back already on MCPower's standardized scale.

Two things to keep in mind:

- The numbers are an **approximation, not a target**: they carry the pilot's
  sampling error and the usual standardization assumptions. Treat them as a
  sensible first guess to then vary, not as ground truth. The values are
  **never auto-applied** — you read them, decide, and set them yourself.
- It works for **every outcome family** — continuous (OLS), binary (logistic),
  and mixed models. For a clustered model, include the grouping column in the
  upload: the fixed effects come back in the string, and the note also reports
  the **estimated ICC** recovered from the data (latent log-odds scale for
  logistic) with a ready-to-paste `set_cluster(...)` snippet — so you need not
  guess the ICC either. For a binary outcome it also reports the **estimated
  baseline probability** with a `set_baseline_probability(p)` snippet. Like the
  effects, the ICC and baseline are approximations and are never auto-applied.

## When to reach for it

Uploading shines when you have a representative sample and want the predictors to
look real — skewed covariates, lopsided group sizes, correlated measurements.
For a continuous predictor, recall the usual standardized-effect benchmarks
(small 0.10, medium 0.25, large 0.40; binary or factor contrasts 0.20 / 0.50 /
0.80) when you decide what effects to test on top of the borrowed distributions.

If you do not have data yet, you lose nothing: describe the design directly with
[[variable-types|variable types]] and [[correlations|correlations]] and MCPower
simulates from your assumptions.
