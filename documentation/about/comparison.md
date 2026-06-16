# MCPower vs other tools

Most power calculators are either formula-based (fast but limited to the designs
their formulas cover) or simulation-based for one narrow family. MCPower is
simulation-based across **OLS, logistic regression, mixed-effects models, and
ANOVA** in one tool — so you do not have to switch calculators when your design
has a binary outcome, clustered observations, or an interaction.

## Feature comparison

| Feature | MCPower | G\*Power | pwr | superpower | simr | WebPower |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Simulation-based (not formula-based) | ✓ | — | — | ✓ | ✓ | — |
| OLS / multiple regression | ✓ | partial | ✓ | — | — | partial |
| Logistic regression (GLM) | ✓ | — | — | — | ✓ | partial |
| Mixed-effects / multilevel models | ✓ | — | — | — | ✓ | — |
| Factorial ANOVA + post-hoc | ✓ | ✓ | partial | ✓ | — | partial |
| Correlated predictors | ✓ | — | — | — | — | — |
| Non-normal / skewed predictors | ✓ | — | — | — | — | — |
| Pilot data upload (CSV / dataframe) | ✓ | — | — | — | — | — |
| Robustness test (scenarios) | ✓ | — | — | — | — | — |
| Multiple-testing correction (FWER/FDR) | ✓ | partial | — | ✓ | — | — |
| No-code GUI (browser + desktop) | ✓ | ✓ | — | — | — | ✓ |
| Python package | ✓ | — | — | — | — | — |
| R package | ✓ | — | ✓ | ✓ | ✓ | ✓ |

"partial" means the tool covers a restricted version of the feature (e.g. a
single-family formula, Bonferroni only, or a web form with limited model types).

### Tool-by-tool notes

**G\*Power** is the long-standing reference for formula-based power. It covers
dozens of statistical tests very quickly, and its GUI is widely taught. It does
not do simulation, so it cannot handle correlated predictors, non-standard
distributions, or custom interaction structures — and mixed-effects designs are
out of scope entirely.

**pwr** (R package) offers the same closed-form approach in code: fast, exact for
the supported tests (t-tests, chi-squared, one-way ANOVA, correlation, linear
models), and zero configuration. The right choice for those tests. It stops at
simple designs.

**superpower** (R package) is simulation-based for factorial ANOVA designs and
handles multiple-comparison corrections well within that family. It does not cover
regression or mixed-effects models.

**simr** (R package) is the closest relative for mixed-effects power: genuinely
simulation-based, driven by a fitted `lmer` / `glmer` model object. Its strength
is that the simulation parameters come straight from a pilot fit; its limit is
that you need a pilot model to start from, and the interface is R-only with no
GUI. MCPower covers overlapping ground while also supporting pilot-data upload,
GUI access, and the other model families.

**WebPower** (R package + web app) is formula-based for a wide range of tests and
exposes them through a web form. It covers mixed models for some standard
structures. It does not do simulation.

## Speed

MCPower's native engine runs simulations in a compiled, multi-core Rust kernel,
so it is typically faster than pure-R or pure-Python simulation tools for the
same number of simulations.

> **[needs measurement]** Quantitative benchmarks — wall-clock time per 1,000
> simulations for a common OLS, logistic, and LME design — have not yet been
> measured against each alternative. Any future figure must state the machine
> (CPU, core count), the simulation count, and the exact designs timed alongside
> the number. Self-reported figures in tool documentation are not compared here.

For the browser app, the same engine runs in a worker pool inside the browser
tab — no server round-trips, no installation required. Simulation count versus
precision is the only knob you control; see
[[internals/optimizations|why it's fast]] for how that trade-off works.
