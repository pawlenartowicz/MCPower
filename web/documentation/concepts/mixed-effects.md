---
title: "Power analysis for mixed-effects models"
description: "Power analysis for mixed-effects models: ICC, cluster sizing, random slopes, repeated measures, and clustered logistic (GLMM)."
---
# Mixed-effects models

Mixed-effects models (also called multilevel or hierarchical models) handle data where observations are grouped into clusters — students in schools, patients in hospitals, repeated measures within participants. Whenever observations within a group are more alike than observations across groups, you have clustered data, and ignoring that structure **inflates Type I error** and produces misleadingly precise estimates.

## Fixed and random effects

"Mixed" refers to two kinds of effect:

- **Fixed effects** are the predictors you care about (treatment, age, motivation) — the relationships you want to test.
- **Random effects** capture shared variation within clusters. A **random intercept** gives each cluster its own baseline; a **random slope** lets a predictor's effect vary across clusters.

Power in MCPower is always computed for **fixed effects** — whether a beta differs from zero — never for the random effects themselves.

## The supported structure

MCPower models clustering with a **random intercept** for a single grouping
factor — name the cluster in the formula as `(1|school)`:

$$
y_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta} + b_j + e_{ij}, \quad b_j \sim \mathcal{N}(0, \tau^2), \quad e_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

See [[concepts/model-specification|model specification]] for the formula syntax.

> [!note] Multiple groupings and random slopes
> The full supported structure includes **crossed** groupings (e.g. subjects × stimuli
> in a psycholinguistics design), **nested** groupings (e.g. classrooms within schools),
> and **random slopes** `(1 + x|group)`. Declare the structure in the formula and
> call `set_cluster` once per grouping factor; add slopes with `random_slopes`. See
> [[tutorial-python/05_mixed-models|the mixed-models tutorial]] for worked examples.

> [!note] No overall test for mixed models
> The omnibus / overall test — the OLS F-test or GLM likelihood-ratio test that
> asks "does the model explain anything at all" — is reported only for plain OLS
> and logistic (GLM) models. Mixed-effects fits (LME and clustered-logistic GLMM)
> report power for each **fixed effect** individually but have no overall test:
> the default reports every fixed effect without it, and an explicit
> `target_test="overall"` raises. Request specific effects instead, e.g.
> `target_test="x1"`. (Fitting clustered data with plain OLS —
> `estimator="ols"` — does keep the F-test.)

## The ICC

The **intraclass correlation (ICC)** is the share of an outcome's variance that lives *between* clusters rather than within them — the higher it is, the more alike two observations from the same cluster are. Set `0` for no clustering, or a value in `0.05–0.95`.

A clustered design is configured by three things: the cluster itself (named by the formula's `(1|name)` term), the number of clusters (or, equivalently, the observations per cluster), and the ICC. Values strictly between 0 and 0.05 are rejected for numerical stability.

> [!note] Conditional, not observed
> The ICC you set is the **conditional** (residual) ICC — the between-cluster share of the variance that remains *after the predictors are accounted for*. If you instead compute the raw correlation of the generated outcome, it will be **lower**, because the fixed effects explain part of the total variance (the stronger the effects, the larger the gap). This is the standard conditional-vs-marginal distinction: the value you set is recovered conditionally, as intended.

| Domain | Grouping | Typical ICC |
|---|---|---|
| Education | Schools | 0.10–0.25 |
| Healthcare | Hospitals / clinics | 0.05–0.20 |
| Psychology | Therapists / sites | 0.10–0.30 |
| Organizational | Companies / teams | 0.10–0.25 |

## Cluster sizing regimes

A clustered design is pinned by total N plus either the **number of clusters** or the **observations per cluster** — you fix one and the other follows from N. Pass `n_clusters` to `set_cluster` and the cluster size grows as N increases; pass `cluster_size` and the number of clusters grows instead.

`n_clusters` (Regime A, "fixed clusters"): the cluster count is constant across any sample size you test; each cluster gets `N / n_clusters` observations, so clusters grow as N grows. `cluster_size` (Regime B, "fixed size"): every cluster has exactly `cluster_size` observations; the number of clusters is `N / cluster_size`, which grows as N grows. The two params are mutually exclusive — pass exactly one.

> [!important] N is the number of measurements, not the number of participants
> In a repeated-measures design the cluster is the **participant** and total N counts *measurements*: `N = participants × trials-per-participant`. With 60 participants on 100 trials each, set `n=6000` (not `n=60`). See [Repeated measures](#repeated-measures) for the full walk-through.

```python
# Regime A — 30 schools, cluster size scales with N
model.set_cluster("school", ICC=0.20, n_clusters=30)

# Regime B — 20 observations per school, cluster count scales with N
model.set_cluster("school", ICC=0.20, cluster_size=20)
```

```r
# Regime A — 30 schools, cluster size scales with N
model$set_cluster("school", ICC = 0.20, n_clusters = 30)

# Regime B — 20 observations per school, cluster count scales with N
model$set_cluster("school", ICC = 0.20, cluster_size = 20)
```

The choice matters most for cluster-level predictors (see [Cluster-level predictors](#cluster-level-predictors) below): their precision is governed by the number of distinct clusters, so fixing `n_clusters` makes that precision constant while fixing `cluster_size` lets it grow with N.

## Design recommendations

| Recommendation | Rationale |
|---|---|
| 10–20 clusters minimum | Below 10, random-effect estimation is unstable. |
| ≥ 5 observations per cluster | Enforced; warning below 10. |

> [!note] What actually drives power
> When treatment is assigned at the individual level within clusters, ICC and cluster allocation have only a minor effect on fixed-effect power — power depends mostly on total $n$.

### Constraints

- ICC must be 0 or in 0.05–0.95.
- With crossed or nested extra groupings, total N must be an exact multiple of the **atom** — the product of primary cluster count and each extra-grouping's level count. `set_cluster` reports the atom size when it does not divide evenly.
- Random slopes (`random_slopes`) can be placed on any grouping factor — the primary or any crossed/nested extra grouping.
- `cluster_level_vars` names predictors that are constant within each cluster — typically a cluster-assigned treatment. Their standard error scales with `1/sqrt(n_clusters)`, not `1/sqrt(N)`; the design-effect difference is large when clusters are few.

## Cluster-level predictors

Some predictors are assigned at the cluster level — a treatment given to an entire school, a policy applied to a whole site. These variables carry one distinct value per cluster, so their information content is `n_clusters`, not total N. Forgetting to declare them overstates power: the standard SE calculation uses N rather than the smaller effective sample.

Mark them with `cluster_level_vars` in `set_cluster`:

```python
model.set_cluster("site", ICC=0.20, n_clusters=30,
                  cluster_level_vars=["treatment"])
```

```r
model$set_cluster("site", ICC = 0.20, n_clusters = 30,
                  cluster_level_vars = c("treatment"))
```

**D3/D4 design-effect distinction.** A within-cluster predictor (measured on each individual) has its SE reduced by within-cluster replication; a cluster-level predictor has no such benefit — only the number of distinct clusters matters. At fixed total N, halving the number of clusters roughly doubles the SE of the cluster-level predictor while barely changing the SE of the within-cluster predictor. Run the same model at your planned `n_clusters` and at half that count to see how sensitive your power estimate is to the number of clusters (not just their size).

> [!warning] Cluster-level predictors and ICC
> If your primary predictor of interest is cluster-level, the ICC has almost no
> effect on its power — the cluster-level predictor's SE is already governed by
> `n_clusters`. The ICC matters for within-cluster predictors, where higher ICC
> reduces the effective N. Plan your number of clusters first; plan cluster size
> second.

## Repeated measures

A **repeated-measures** (within-subjects) design — each participant measured many times across trials, time points, or conditions — is an ordinary clustered design with the **participant as the cluster**. Name it in the formula as `(1|participant)`; the random intercept gives each person a stable baseline, so the fixed-effect test respects the fact that one person's measurements are not independent.

The sample size is counted in **measurements, not participants**: `N = participants × trials-per-participant`. Sixty participants on 100 trials each is `n=6000` — the participant count is `n_clusters` and the trials per participant is `cluster_size`, the two [sizing regimes](#cluster-sizing-regimes). Setting `n=60` instead models one measurement per person and throws away the repeated-measures advantage. Here the ICC reads as the **correlation between two measurements from the same participant**: high ICC means the repeated trials are partly redundant, low ICC means each trial adds nearly independent information (reaction-time and questionnaire designs commonly sit around 0.2–0.5).

Whether extra trials buy power depends on **where the predictor varies**:

| Predictor kind | Varies… | Effective sample size | More power from… |
|---|---|---|---|
| **Within-participant** (condition, time, stimulus) | across trials *inside* each person | total N | more trials *or* more participants |
| **Between-participant** (group, age, sex) | only *across* people | `n_clusters` (participant count) | more **participants only** |

A within-participant manipulation is estimated against the within-person residual, so each added trial sharpens it — which is why within-subjects experiments need few people. A between-participant factor carries one value per person, so adding trials does nothing for it; declare such predictors with `cluster_level_vars` so the engine uses the smaller effective sample (see [Cluster-level predictors](#cluster-level-predictors)). Worked examples: [[tutorial-python/05_mixed-models#repeated-measures|Python]] · [[tutorial-r/05_mixed-models#repeated-measures|R]].

## Multiple groupings (crossed and nested)

Real multi-level designs often have more than one random source of variation. A psycholinguistics study has by-subject and by-item variability (crossed); a school study has by-classroom variability nested within by-school variability (nested). MCPower supports both: name every grouping factor in the formula, then call `set_cluster` once per factor.

```python
# Crossed: subjects × items — formula "rt ~ frequency + (1|subject) + (1|item)"
model.set_cluster("subject", ICC=0.20, n_clusters=24)
model.set_cluster("item", ICC=0.15, n_clusters=12)

# Nested: classrooms within schools — formula "score ~ ... + (1|school/classroom)"
# The nested child uses n_per_parent (child clusters per parent cluster).
model.set_cluster("school", ICC=0.15, n_clusters=10)
model.set_cluster("school:classroom", ICC=0.10, n_per_parent=4)
```

```r
# Crossed: subjects × items — formula "rt ~ frequency + (1|subject) + (1|item)"
model$set_cluster("subject", ICC = 0.20, n_clusters = 24)
model$set_cluster("item", ICC = 0.15, n_clusters = 12)

# Nested: classrooms within schools — formula "score ~ ... + (1|school/classroom)"
model$set_cluster("school", ICC = 0.15, n_clusters = 10)
model$set_cluster("school:classroom", ICC = 0.10, n_per_parent = 4)
```

Total N must be an exact multiple of the atom (primary clusters × extra-grouping levels for crossed, or primary clusters × nested-children for nested). Each grouping's ICC sizes its own between-level variance component, independently of the others.

## Random slopes

A random slope `(1 + x|group)` lets the effect of predictor `x` vary across groups. Add the slope term to the formula and name the predictor in `random_slopes`:

```python
# Formula: "y ~ x + (1 + x|school)"
model.set_cluster("school", ICC=0.20, n_clusters=30,
    random_slopes=["x"], slope_variance=0.10, slope_intercept_corr=0.30)
```

```r
# Formula: "y ~ x + (1 + x|school)"
model$set_cluster("school", ICC = 0.20, n_clusters = 30,
    random_slopes = list(list(predictor = "x", variance = 0.10,
                              corr_with_intercept = 0.30)))
```

`random_slopes` names the predictor(s) whose effect varies across groups — each must appear in the formula's random-slope term, e.g. `(1 + x|school)`. `slope_variance` is the between-group variance of the slope; `slope_intercept_corr` is the correlation between each group's random intercept and its random slope.

> [!note] Slopes on extra groupings
> Random slopes compose freely across groupings — each crossed or nested extra
> grouping can carry its own random slopes, just like the primary.

## Convergence failures

Unlike OLS, a mixed-model fit can fail to converge on a given simulated dataset. MCPower tolerates a small fraction of failed fits and reports the rate; you can raise the tolerance for harder designs — for random-intercept models a tolerance of **3–10%** is reasonable.

Failures get more common with **small clusters** (< 10 obs) and **high ICC** (> 0.4). If a run reports too many failures, the fixes in order of preference are: increase the sample size, add clusters, lower the ICC if plausible, and only as a last resort relax the tolerance. OLS designs never hit this — the tolerance has no effect there.

## Clustered logistic (GLMM)

When the outcome is binary and observations are clustered, MCPower fits a **generalised linear mixed model (GLMM)** — a logistic regression with a cluster-level random intercept. Specify it the same way as any other mixed model: a formula with a `(1|group)` term, `family="logit"`, and `set_cluster`.

```python
model = MCPower("y ~ treatment + (1|site)", family="logit")
model.set_baseline_probability(0.20)
model.set_cluster("site", ICC=0.15, n_clusters=30)
result = model.find_power(sample_size=600, target_test="treatment")
```

```r
model <- MCPower$new("y ~ treatment + (1|site)", family = "logit")
model$set_baseline_probability(0.20)
model$set_cluster("site", ICC = 0.15, n_clusters = 30)
result <- model$find_power(sample_size = 600, target_test = "treatment")
```

The engine fits this via a **Laplace approximation (nAGQ=1)** — the cluster random effects are integrated out at their conditional modes — the same approximation `glmer` uses at its default settings.

### Latent-scale ICC convention

On the binary path the ICC you specify maps to a random-intercept variance on the **latent (log-odds) scale**: `τ² = ICC / (1 − ICC) · π²/3` (the Snijders & Bosker latent-scale formula for the logistic link, where `π²/3 ≈ 3.29` is the logistic variance). You still specify ICC directly — literature ICC estimates for binary outcomes (commonly reported on the latent scale by logistic multilevel models) plug in without conversion; the `π²/3` scaling is internal.

### Laplace-bias warning

At small cluster sizes the Laplace approximation can be optimistic. MCPower warns when both hold: (1) the estimated between-cluster variance `τ̂²` (averaged across converged fits) exceeds the configured threshold (default 1.0), and (2) the minimum cluster size is below the recommended floor (default 10 observations per cluster). Cluster sizes below 5 are rejected outright, so the warning targets the "allowed but risky" 5–9-observations-per-cluster band. When it fires, interpret GLMM power with caution and consider larger clusters. With ≥10 observations per cluster the approximation is generally reliable at the ICCs common in social and health research (0.05–0.30).

### SE method and quadrature

A clustered GLMM (binary or count) has two more expert controls: `wald_se` (which standard-error method) and `agq` (Laplace vs adaptive Gauss–Hermite quadrature). The default `wald_se="rx"` is the fast Schur-complement shortcut; `wald_se="hessian"` is the slower, slightly more conservative per-fit Hessian inversion. `agq=1` (default) is the Laplace approximation used above; an odd `agq` up to 25 switches to adaptive quadrature for a single-grouping-factor design. See [[concepts/simulation-settings#estimation-mode|Estimation mode]] for the full explanation, including the lme4 correspondence and AGQ's eligibility rules.

These controls only affect the **clustered binary/count GLMM**. OLS, Gaussian mixed models, and unclustered GLMs already have exact standard errors, so both settings are ignored there.

## Clustered count (Poisson GLMM) and clustered probit

The same clustered-GLMM machinery covers a clustered **count** outcome (`family="poisson"`) and a clustered **probit** outcome (`family="probit"`) — swap the family and, for Poisson, size the random intercept by raw variance instead of ICC: `set_cluster(grouping, tau_squared=..., n_clusters=...)`. See [[concepts/supported-families|supported families]] for the baseline setters and the ICC-vs-`tau_squared` distinction across all four families.

## Learn more

- [[validation/index|Validation]] — how MCPower's mixed-model power is checked against `lme4`.
- [[internals/optimizations|What makes it fast]] — why the native engine matters most for mixed models.
