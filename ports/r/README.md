# MCPower

```
███╗   ███╗  ██████╗ ██████╗ 
████╗ ████║ ██╔════╝ ██╔══██╗ ██████╗ ██╗    ██╗███████╗██████╗ 
██╔████╔██║ ██║      ██║  ██║██╔═══██╗██║    ██║██╔════╝██╔══██╗
██║╚██╔╝██║ ██║      ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝
██║ ╚═╝ ██║ ██║      ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
██║     ██║ ╚██████╗ ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═════╝ ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
```

**Power analysis by simulation — any design from t-test to mixed models, in your browser, on your desktop, or in Python and R.**

[![R-universe](https://pawlenartowicz.r-universe.dev/badges/mcpower)](https://pawlenartowicz.r-universe.dev/mcpower)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16502734-blue)](https://doi.org/10.5281/zenodo.16502734)

## Why MCPower?

- **MCPower covers anything from ANOVA to generalized linear to mixed
  models.** Analytical power formulas exist for a few textbook designs and are
  correct only when all their assumptions are met (they aren't). Monte Carlo
  is the ground truth they approximate.
- **Fast enough to mean it.** A purpose-built engine, 100–1000× faster
  than a hand-written R/Python simulation loop — even the most complex power
  analysis runs in seconds, not hours or even days for mixed models. Speed
  stops being the reason to avoid simulation.
- **Robustness built in.** Stress-tests your design against the messy,
  non-ideal data that formulas assume away, so you catch under-powering before
  you collect.
- **Easy, and everywhere.** A few-line API across four bindings — Python, R,
  desktop app, browser. Free and open source.

## Install

```r
# From R-universe (binaries):
install.packages("mcpower", repos = "https://pawlenartowicz.r-universe.dev")

# Or from source:
# install.packages("remotes")
remotes::install_github("pawlenartowicz/mcpower", subdir = "mcpower/ports/r")
```

MCPower is also being submitted to CRAN; the R-universe build is the primary install path.

## Quickstart

```r
library(mcpower)

# 1. Define the model with an R formula.
model <- MCPower$new("patient_outcome ~ treatment + baseline_score")

# 2. Expected effect sizes (standardised).
#    treatment=0.5      -> therapy shifts outcomes by 0.5 SD (a medium effect).
#    baseline_score=0.3 -> baseline moderately predicts the outcome.
model$set_effects("treatment=0.5, baseline_score=0.3")

# 3. Variable types — treatment is binary (0=control, 1=therapy).
model$set_variable_type("treatment=binary")

# 4. Short form (printed automatically). One row per effect with Power,
#    95% CI, and a check/cross marker against the target power.
invisible(model$find_power(sample_size = 120, target_test = "treatment"))

# 5. Long form via summary(). verbose=FALSE suppresses the auto short form;
#    target_test="all" adds the omnibus "Overall" row and the joint-
#    significance distribution.
result <- model$find_power(sample_size = 120, target_test = "all", verbose = FALSE)
print(summary(result))

# 6. Robustness — run every built-in scenario at once.
invisible(model$find_power(sample_size = 120, target_test = "all", scenarios = TRUE))

# 7. Programmatic access — as_tibble() gives a tidy (test x scenario) frame;
#    plot() renders the power-at-N chart as a Vega-Lite widget.
if (requireNamespace("tibble", quietly = TRUE)) {
  print(tibble::as_tibble(result))
}
```

## More examples

**Sample size search:**

```r
library(mcpower)

# 1. Define the model.
model <- MCPower$new("test_score ~ intervention + prior_knowledge + motivation")

# 2. Expected effects from literature / pilot data.
model$set_effects("intervention=0.4, prior_knowledge=0.35, motivation=0.3")

# 3. Variable types — intervention is binary (0=control, 1=intervention).
model$set_variable_type("intervention=binary")

# 4. Short form (printed automatically). find_sample_size() sweeps a grid and
#    reports the smallest N that reaches the target power (Required N column).
invisible(model$find_sample_size(
  target_test = "intervention", from_size = 30, to_size = 300, by = 10
))
```

**Mixed-effects / LME (clustered data):**

```r
library(mcpower)

# 1. Declare a mixed model. The (1|classroom) term adds a random intercept per
#    classroom; family = "lme" fits it by maximum likelihood (MLE estimator).
model <- MCPower$new("score ~ teaching_method + prior_gpa + (1|classroom)",
                     family = "lme")
model$set_variable_type("teaching_method=binary")
model$set_effects("teaching_method=0.4, prior_gpa=0.18")

# 2. Describe the clustering: ICC=0.15 (15% of variance is between-classroom)
#    across 30 classrooms. At N=300 that is 10 students per classroom.
model$set_cluster("classroom", ICC = 0.15, n_clusters = 30)

# 3. Short form — target the fixed effects directly.
invisible(model$find_power(sample_size = 300, target_test = "teaching_method, prior_gpa"))

# 4. Sample-size search for the clustered design.
invisible(model$find_sample_size(
  target_test = "teaching_method, prior_gpa", from_size = 120, to_size = 420, by = 30
))
```

See [examples/](examples/) (01–11) and the [R tutorial](https://docs.mcpower.app/tutorial-r/index) for interactions, correlations, factors/ANOVA, logistic regression, your-own-data upload, custom scenarios, and plotting.

## API at a glance

`set_*` methods chain; add `scenarios = TRUE` to either `find_*` for optimistic/realistic/doomer robustness; results support `summary()`, `as_tibble()`, and `plot()` (Vega-Lite).

| Call | What it does |
|---|---|
| `MCPower$new("y ~ x1 + x2", family = "ols")` | Define the model (R formula; `family` ∈ `"ols"`/`"logit"`/`"lme"`) |
| `$set_effects("x1=0.5, x2=0.3")` | Standardised effect sizes |
| `$set_variable_type("x1=binary, g=(factor,3)")` | Predictor distributions |
| `$set_correlations("corr(x1, x2)=0.3")` | Correlations between predictors |
| `$set_cluster("group", ICC = 0.2, n_clusters = 20)` | Random-effects structure (`family = "lme"`) |
| `$set_baseline_probability(0.3)` | Event rate at reference (`family = "logit"`) |
| `$upload_data(df)` | Use your own data frame instead of synthetic |
| `$get_effects_from_data("y")` | Borrow starting effect sizes from uploaded data (approximate) |
| `$set_seed(2137)` · `$set_alpha(0.05)` · `$set_power(80)` · `$set_simulations(n)` | Tuning knobs |
| `$find_power(sample_size = 200, target_test = "all")` | Power at a fixed N |
| `$find_sample_size(target_test = "x1", from_size = 50, to_size = 400)` | Smallest N for target power |

## Docs

Full documentation: [https://docs.mcpower.app](https://docs.mcpower.app).

## Citation & License

GPL v3. If you use MCPower in research, please cite:

Lenartowicz, P. (2025). MCPower: Monte Carlo Power Analysis for Complex Statistical Models [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.16502734

```bibtex
@software{mcpower2025,
  author    = {Lenartowicz, Pawe{\l}},
  title     = {{MCPower}: Monte Carlo Power Analysis for Complex Statistical Models},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16502734},
  url       = {https://doi.org/10.5281/zenodo.16502734}
}
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
