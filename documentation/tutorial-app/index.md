# The MCPower app

The app is MCPower without code: the same native engine as the
[[tutorial-python/index|Python]] and [[tutorial-r/index|R]] packages, behind a
point-and-click interface. It ships in two forms with an identical interface —
the **web app** runs in your browser with nothing to install, and the
**desktop app** is a download ([[tutorial-app/install-desktop|install guide]]).

The window splits in two: **configuration on the left, results on the right**
(a narrow window collapses them into one pane with a toggle). A tour, top to
bottom:

## Pick a family

The ribbon in the header selects the analysis family — **ANOVA**,
**Regression** (continuous or binary outcome), or **Mixed effects**. Each
family keeps its own configuration, so switching back and forth loses nothing.
Next to the ribbon sit **Settings** (appearance, run configuration, and the
scenario definitions), **History** (a searchable list of past runs — click one
to replay it), and **Tutorial**, which opens these pages.

## Describe the model

The left panel is the model description, as collapsible cards in workflow
order:

- **Upload data** (Regression and Mixed effects only) — hand the simulation a
  pilot CSV instead of describing every predictor from scratch.
- **Model** — the formula box, then one card per predictor: its type
  (continuous, binary, or factor) and the standardised effect size to detect.
  The panel adapts to the family: Regression adds a Continuous/Binary outcome
  toggle (Binary adds a baseline-probability field), Mixed effects adds the
  cluster card (cluster name, ICC, number of clusters), and ANOVA swaps the
  formula for structured factor and covariate rows.
- **Correlations** (not ANOVA) — optional: pairwise correlations among
  continuous predictors.
- **Run** — target power, α, which coefficients to test and the
  multiple-testing correction, plus the inputs for the two run modes: **Find
  power** takes a fixed sample size `n`; **Find sample** simulates a from/to
  grid and reports the `n` where the fitted power curve reaches the target
  ([[concepts/required-sample-size|how required N is estimated]]).

## Run it

The action bar above the results pane summarises the current target, α, and
sample-size range, and holds the two run buttons — **Find power** and **Find
sample** — alongside the **Scenarios** toggle (re-run the same design under
the optimistic / realistic / doomer assumption sets; see
[[concepts/scenario-analysis|scenario analysis]]) and a status badge (*Ready
to run*, *Running…*, *Last run done*). While a run is in flight the buttons
are replaced by **Cancel**.

## Read the results

Before the first run, the results pane shows a get-started checklist and a
short guide to the active family. Each run then lands in its own **tab** with
four views:

- **Summary** — the power table and plot. With Scenarios on, one chip per
  scenario plus an **⧉ Overlay** chip shows all scenarios side by side on a
  3-column grid. For a **Find sample** run the table's Required N is the
  model-based estimate read off the fitted power curve, with a 95% CI column
  in single-scenario runs — see
  [[concepts/required-sample-size|how required N is estimated]] for the
  `≤` / `≥` / `appr.` markers and the non-monotone warning.
- **Joint dist** — how many of your tested effects reach significance
  *together* in the same simulated study. When ≥ 2 effects are tested, two
  charts appear: **At least k** (P(≥ k significant) vs N) and **Exactly k**
  (P(exactly k significant) vs N, including k = 0).
- **Script** — the equivalent script for the analysis you just configured,
  ready to copy when you outgrow the GUI. A **Python | R** toggle next to the
  Copy button switches the output language; the choice is remembered globally
  and persists across sessions.
- **Export** — pick a plot block from the selector and save it as a PNG or SVG
  file. Saved files use the print style (white background, black axes,
  colourblind-safe palette) — the same as `save_plot()` in Python and R. The
  on-screen charts keep the app's live colour theme.

## Export

The **Export** tab saves a run's charts as image files. Pick which chart from the **Chart** selector — the power-by-effect plot, the power curve, any per-scenario panel, or the overlay — choose **PNG** or **SVG**, and click **Save**; PNG adds a **Scale** field (1–4×) for higher-resolution output. Saved files use the print theme (white background, colour-blind-safe palette), independent of the app's on-screen colours, matching `save_plot()` in Python and R.

## When something goes wrong

The app never fails silently:

- **A run that fails** shows an **error card** in the results area with the engine's
  message and a **Copy details** button — so you can read (and copy) exactly why it
  stopped. Dismiss it to return to your previous results, which are kept. (This replaces
  the old, dead-end "see console" status badge.)
- **A formula or model problem** is flagged **inline, under the formula box**, the moment
  it can't be turned into a valid model — fix it there and the run buttons re-enable.
- **A background hiccup** (for example, settings or history that couldn't be saved) raises
  a brief **toast** in the corner with a Details option, rather than losing your change
  without a word.

## The panel guides

One page per panel, in reading order:

1. [[tutorial-app/upload-data|Upload data]] — driving the simulation from a CSV.
2. [[tutorial-app/regression|Regression]] — OLS and logistic power, panel by panel.
3. [[tutorial-app/mixed-models|Mixed models]] — clustered designs in the GUI.
4. [[tutorial-app/anova|ANOVA]] — factorial designs and post-hoc power.
