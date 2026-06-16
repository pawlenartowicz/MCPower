# Debug mode (R)

When a power result surprises you, debug mode lets you open up **one scenario** and
inspect the engine's intermediate state — the generated data, the model it chose, the
full sampling distribution, and the significance threshold — instead of seeing only
the final power number. It answers questions like *"is this the dataset I think I
asked for?"* and *"where does that power number actually come from?"*.

> [!note]
> Debug mode is **R-only**. The Python package has no equivalent — if you need to
> introspect a scenario, do it from R.

## Setting it up

`MCPowerDebug` is a drop-in subclass of `MCPower`: you build it with the same formula
and the same `set_*` chain you already know from the [[tutorial-r/index|R tutorial]].
The only extra arguments fix the size of the inspection.

```r
library(mcpower)

model <- MCPowerDebug$new(
  "satisfaction ~ treatment + age",
  family = "ols",       # "ols" (default), "logit", or "lme"
  debug_n = 50,         # sample size used for every debug call
  debug_n_sims = 200    # number of simulations used for every debug call
)
model$set_variable_type("treatment=binary")
model$set_effects("treatment=0.5, age=0.3")
```

`debug_n` and `debug_n_sims` pin the sample size and simulation count for the
inspection (smaller than a real run, so it's quick to read). The seed is inherited
from `MCPower` — the default **2137**. Because all four stages below share that fixed
seed, sample size, and simulation count, they describe **one consistent run**.

## The stages

Each method below exposes one step of the pipeline, in the order the engine runs them.

### Data — `create_data()`

```r
d <- model$create_data()
```

Returns the actual generated **design matrix**, **outcome**, and **cluster IDs** for
one simulation. *Is this the dataset I think I asked for?* — check the columns, the
distributions, and (for multilevel designs) the cluster structure against your intent.

### Dispatch — `dispatch()`

```r
model$dispatch()
```

Returns the engine's **routing decision**: the kind of outcome (continuous or binary)
and the estimator it picked (`ols`, `glm`, or `mle`). *Did the engine choose the model
I expected?*

### Statistics and power — `raw_statistics()`

```r
s <- model$raw_statistics()
s$power   # the same power the regular analysis reports for this n / n_sims
```

Returns the full **sampling distribution of the test statistic** across all
simulations, the **convergence** flags, and the engine's **reported power** — the very
number a normal `find_power` call would give for this `debug_n` / `debug_n_sims`.
*Where does the power number actually come from?*

### Critical value — `critical_value()`

```r
model$critical_value()
```

Returns the **significance threshold for each tested coefficient**, together with α
and the degrees of freedom. *What bar did each test have to clear?*

## Bring your own data — `load_data()`

`load_data()` is the inverse of `create_data()`: hand it a dataset and see the
estimates the engine produces from it. Pass the list `create_data()` returned, or your
own pilot data with the same `$design` / `$outcome` shape.

```r
fit <- model$load_data(d)        # d from create_data(), or your own pilot data
fit$targets                      # per-coefficient beta, se, statistic, critical value
```

You get back MCPower's own coefficients, standard errors, statistics, and thresholds
for that exact data — handy for **calibrating effect sizes from pilot data**. See
[[concepts/upload-data|uploading data]] for working from empirical datasets.

## Good to know

- **Always available.** Debug mode needs no special build or flag — it's part of the
  installed package.
- **One coherent run.** Everything is fixed by `debug_n`, `debug_n_sims`, and the
  seed, so the four stages all describe the *same* simulated run. You can take the
  statistics and the threshold and **hand-recompute the power yourself** — you'll get
  the same number the engine reports.
- **Synthesized column labels.** The predictor columns carry positional names
  (`col_1`, `col_2`, `factor_dummy_k`), not your formula's variable names — the engine
  works below the level where human labels exist.
- **The statistic has no sign.** It's reported as a **non-negative magnitude**, so you
  compare it directly against the threshold — including for two-sided tests, where the
  threshold already accounts for both tails.

---

See also the [[tutorial-r/index|R tutorial]], [[concepts/simulation-settings|simulation
settings]], [[concepts/upload-data|uploading data]], and
[[internals/index|what's inside MCPower]].
