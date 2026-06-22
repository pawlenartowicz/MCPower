---
title: "Upload Pilot Data for Power Analysis - Python"
description: "Use a pilot dataset or pandas DataFrame to shape predictor distributions in Python MCPower so your Monte Carlo power analysis reflects the actual population."
---
# Upload data

Every rung so far generated predictor values from scratch — a normal distribution here,
a binary split there. That works well when you're designing a study from theory. But
if you already have a pilot dataset or an existing dataset that resembles your target
population, you can hand it directly to MCPower and let the simulation draw predictors
that match it. This rung shows how.

## What uploading does — and doesn't do

Uploading shapes the **predictor side** of the simulation. The engine reads each
column named in your formula, learns its marginal distribution and the correlation
structure across all matched columns, and draws from that joint distribution for each
simulated dataset. See [[concepts/upload-data|upload data]] for the technical detail.

What uploading does *not* change: the outcome is still generated from your formula
and your effect sizes. `set_effects(...)` is still required, still means the same
thing, and still controls what you are powered to detect. The data gives you
realistic predictor variation; the rest is yours to specify.

## How to upload: a file or in-memory data

You can hand MCPower either a path to a file or data you already have in memory:

- **A CSV file path** — `model.upload_data("pilot.csv")`. The file is read
  locally by the package; nothing is uploaded anywhere.
- **A pandas DataFrame** — `model.upload_data(df)`. Use this when the data is
  already in memory after earlier preprocessing steps.

A plain dict of columns is accepted too, as is a NumPy array if you have numpy
installed (numpy and pandas are optional — `pip install mcpower[optional]`).

## The three modes (quick preview)

The `mode=` argument controls how faithfully the synthetic predictors follow your
real ones:

- **`mode="none"`** — match each predictor's marginal distribution only.
- **`mode="partial"`** (the default) — marginals **plus** the measured
  correlations among matched predictors.
- **`mode="strict"`** — bootstrap whole rows; the faithful joint.

See [[concepts/upload-data]] for the two mechanisms behind these (distribution
mapping vs. strict bootstrap). The example below uses the default.

## Uploaded columns set their own type

When a column in your formula matches an uploaded column, its **type** (continuous, binary,
or factor) is determined by what the native engine detects in the data — you cannot override
it to a different class. For example, if `am` is detected as `binary`, calling
`set_variable_type("am", "continuous")` raises a `ValueError`:

```
Column 'am' was detected as binary from your uploaded data; it can't be modeled as
continuous. Uploaded columns take their type from the data.
```

A matched **continuous** column may still have its distribution overridden (e.g.
`right_skewed`); only the class (continuous / binary / factor) is locked. Factor levels are
always taken from the data.

## Only formula columns are used

The engine reads only the columns that appear in your formula. Every other column in
the file is reported as `(extra)` and ignored — you can upload your full dataset
without trimming it first.

## The cars example

We model fuel efficiency (`mpg`) as a function of horsepower (`hp`), weight (`wt`),
and transmission type (`am`), using the classic 32-car `mtcars` dataset — bundled
with MCPower, so `from mcpower import mtcars` makes it available — as a predictor
template. Effects are set to medium-range standardised values: `hp` and `am` at ±0.3,
`wt` at −0.4, reflecting the expectation that heavier cars and higher horsepower
reduce efficiency while a manual transmission improves it.

<!-- example:09-upload -->
```python
from mcpower import MCPower, mtcars

model = MCPower("mpg = hp + wt + am")
model.upload_data(mtcars)
model.set_effects("hp=-0.3, wt=-0.4, am=0.3")

result = model.find_power(sample_size=150, target_test="all", verbose=False)
print(result.summary())
```

```
Uploaded 32 rows, 11 columns.
  mpg: continuous (extra)
  cyl: continuous (extra)
  disp: continuous (extra)
  hp: continuous (matched)
  drat: continuous (extra)
  wt: continuous (matched)
  qsec: continuous (extra)
  vs: binary (extra)
  am: binary (matched)
  gear: continuous (extra)
  carb: continuous (extra)
==================================================
  MCPower · Power Analysis
==================================================
formula: mpg = hp + wt + am
estimator: OLS  N=150  sims=1600  α=0.05  target=80%
effects: hp=-0.30, wt=-0.40, am=0.30

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
Overall F             100%      80%
hp                   59.0%      80%
wt                   85.0%      80%
am                   43.4%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
Overall F             100%   [99.8%,  100%]
hp                   59.0%   [56.6%, 61.4%]
wt                   85.0%   [83.2%, 86.7%]
am                   43.4%   [41.0%, 45.9%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0        0.2%       100%
1       31.5%      99.8%
2       48.8%      68.2%
3       19.4%      19.4%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

## Reading the output

The upload confirmation lists every column in your data and labels each one
`(matched)` or `(extra)`. Here `hp`, `wt`, and `am` are matched; the remaining eight
columns are ignored.

At N = 150 the three predictors tell different stories:

- **`wt`** reaches **85.0%** — comfortably powered. Weight is the strongest and
  most consistently measured predictor in this dataset.
- **`hp`** lands at **59.0%** — well below the 80% target. The marginal distribution
  of horsepower in the 32-car sample is right-skewed, which reduces effective power
  compared to a symmetric predictor of the same standardised effect.
- **`am`** comes in at **43.4%** — substantially underpowered. The binary
  transmission variable has a very uneven split in the cars data (roughly 40/60
  manual/automatic), which makes it harder to detect than a balanced binary
  predictor would be.

> [!note]
> The split matters for binary predictors: a 40/60 split is less efficient than a
> 50/50 split at the same sample size and effect size. When your pilot dataset has an
> uneven binary, the uploaded simulation captures that penalty automatically — which
> is exactly why uploading is valuable.

The joint significance distribution shows that detecting all three effects in the
same study at N = 150 happens only 19.4% of the time. If detecting `am` matters for
your study, you would need to either increase N substantially or revisit whether a
medium effect for `am` is realistic given the imbalanced split.

## Borrowing a starting point

The example above typed the effect sizes by hand. If the outcome is in your
upload too, you don't have to guess: `get_effects_from_data("mpg")` fits your
model to the data and returns a ready-to-paste `set_effects(...)` string, the
argument naming the outcome column. The values come back on MCPower's
standardised scale, and it works for every family — OLS, logistic, and mixed.
For a clustered model (include the grouping column in the upload) the printed
note also reports the **estimated ICC** with a `set_cluster(...)` snippet, so you
need not guess that either. For a **binary (logistic)** outcome it additionally
reports the **estimated baseline probability** with a `set_baseline_probability(p)`
snippet, recovered from the fitted intercept.

Treat them as a **first guess, not a target**: they carry the pilot's sampling
error, and they are never applied automatically — you read the string, decide,
and call `set_effects` yourself. See
[[concepts/upload-data#borrowing-starting-effects|borrowing starting effects]].

next → [[10_model-misspecification|Model misspecification]]
