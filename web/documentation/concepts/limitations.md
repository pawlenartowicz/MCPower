---
title: "MCPower limitations and edge cases"
description: "Known limitations of MCPower: simulation noise, large-sample tests, mixed-model depth limits, and heterogeneity power ceilings."
---
# Limitations

MCPower's numbers are only as good as the assumptions and the method behind
them. This page collects the places where MCPower is the wrong tool for the
question, or where a result deserves a second look before it goes into a grant.
None of these are hidden failure modes — each is a documented property of how
the tool works, listed here so you can plan around it.

## Power estimates carry simulation noise

Every power number is a **Monte Carlo estimate**, not an exact value. An
estimate from $n_\text{sims}$ simulations carries a standard error of about
$\sqrt{p(1-p)/n_\text{sims}}$ — at the default 1,600 simulations and 50% power,
roughly ±1.25 percentage points. So don't over-read small differences: 78.9%
vs 80.2% from separate runs is noise, not a finding. When two designs land
close together and the distinction matters, raise the simulation count — see
[[concepts/simulation-settings|simulation settings]].

## Results can differ from G*Power — by design

MCPower simulates the **predictors anew in every dataset** (a *random-X*,
unconditional design), the way data actually arrives in observational and
survey research. Analytic calculators such as G*Power instead treat the design
matrix as **fixed** and known in advance (*fixed-X*), which is natural for
fully controlled experiments. The two answer slightly different questions, so
their power numbers can legitimately differ — most visibly at small samples.
That gap is a framing difference, not an error in either tool.

## The tests are large-sample tests (logistic and mixed)

OLS power uses the exact Student-*t* test, at any sample size. Logistic and
mixed-model power use **Wald z tests** — the standard large-sample
approximation — not likelihood-ratio tests or small-sample degree-of-freedom
corrections (such as Satterthwaite). At reasonable sample sizes the difference
is negligible; with very small samples, or a mixed design with only a handful
of clusters, the z approximation runs slightly optimistic. Treat power
estimates at those edges with extra caution, and prefer more clusters over
more rows per cluster when you can.

## Mixed-effects structure has depth limits

The mixed-model structure is bounded in two ways. They cover the designs people
actually run, but stop short of full `lme4` generality:

- **One level of nesting.** A single nested grouping such as `(1|school/class)`
  is supported; a deeper chain like `(1|school/class/student)` is not.
- **Crossed factors need a fixed number of clusters.** When you add a crossed
  grouping — `(1|subject) + (1|item)` — the primary grouping must be sized *by
  number of clusters*, not *by cluster size*, because a crossed factor is crossed
  against a fixed count of primary clusters. The app disables the "by cluster
  size" toggle in that case.

(There are also generous ceilings — at most seven extra grouping factors, and
eight random-effect terms on the primary cluster — that realistic designs do not
reach.)

## Outcome families are limited

The engine fits three model families: **continuous outcomes** (OLS), **binary
outcomes** (logistic regression), and **clustered continuous outcomes**
(mixed models). Counts, ordinal scales, survival times, and multinomial
outcomes are out of scope — power for those needs a different tool.

## Some tests are only available for some models

Two test types are tied to particular model families:

- **Post-hoc pairwise comparisons are OLS-only.** Tukey-style all-pairs
  comparisons between a factor's levels are produced for continuous-outcome
  (OLS) models. They are not offered for logistic or mixed models, whose
  pairwise corrections behave differently. *This could be extended to those
  families on request.*
- **The overall (omnibus) test covers OLS and unclustered logistic only.** The
  single "is the model as a whole significant?" test — the *F*-test for OLS, the
  likelihood-ratio test for plain logistic regression — is not produced for
  mixed-effects or clustered-logistic models, where a well-behaved omnibus needs
  a different construction. Test individual terms, or a joint test of a chosen
  set of terms, instead. *An omnibus for the mixed families could be added on
  request.*

## Correlations are between continuous and binary predictors only

The predictor correlation structure applies to **continuous and binary**
predictors. Multi-level categorical factors cannot be entered into the
correlation matrix — their dependence with other predictors is not something you
set directly. Specify the correlations among the continuous and binary
predictors you need linked, and leave factors out of the correlation
specification.

## Heterogeneity imposes a power ceiling

With heterogeneity turned on — the realistic and doomer scenarios, or any
custom `heterogeneity > 0` — each simulated study draws its own true effect. At
large values some of those studies draw essentially no effect (or the wrong
sign), and those just can't be detected at any sample size. That puts a **hard
ceiling on the maximum power you can reach**, and it also makes the last stretch
*up* to that ceiling much harder. More data won't help — the ceiling is
structural, not estimation noise, so keep it in mind.

| heterogeneity | rough power ceiling | notes |
|---|---|---|
| 0.0 | none | default optimistic |
| 0.2 | ~99.99% | default realistic |
| 0.4 | ~99% | default doomer |
| 0.5 | ~98% | custom |
| 1.0 | ~84% | custom (extreme) |

This bites most on stringent designs. If you need very high power — say ≥ 99% —
at a strict alpha (for example α = 0.01 with a Bonferroni correction across ten
tests), a non-zero heterogeneity scenario can put your target *above* the
ceiling, and a required-sample-size search will report it as unreachable at any
N. When that happens, ask whether per-study heterogeneity is the right
assumption: for a tightly controlled, single-population, single-protocol study
where the effect is plausibly homogeneous, set `heterogeneity = 0` in that
scenario (see [[concepts/scenario-analysis|scenario analysis]]) rather than
chasing a sample size that cannot move the ceiling.

## Same seed, different parallelism — slightly different numbers

Reproducibility in MCPower is a **per-machine, per-seed** guarantee: same
machine, same seed, same inputs, same numbers, every time. It is *not* a
promise that every product walks the same random path — a run split across a
different number of workers (the browser app, most visibly) draws different
random numbers and lands on a slightly different estimate. The two results are
statistically equivalent, within the Monte Carlo noise above, but not
byte-identical. See
[[internals/engine-architecture|why two runs aren't byte-identical]].

## App, Python, and R agree — to the last decimal that matters

The same analysis run on the desktop app, in Python, and in R gives the same
answer. Within any one of them, a seeded run reproduces *exactly* — same seed,
same inputs, same numbers, every time. Across them, results are identical for
every practical purpose: the only difference is floating-point rounding in the
last bits of each number, far below anything you would report.

The lone observable consequence would be a single significance call landing
*exactly* on the α boundary — a *p*-value tied to α to the last bit — and
flipping by one decision between two faces. With continuous estimates that
coincidence is vanishingly rare: on the order of once in a billion years of
running. It is written down here for honesty, not because it is a concern you
need to plan around — treat App, Python, and R as interchangeable.

## Sparse factor levels at small N

A factor level needs observations to be estimable. If any level of a factor
would receive fewer than **5 observations** at a given sample size, MCPower
excludes the whole factor from the model in that run: its effects report
power 0, the other predictors are still analysed, and the result carries a
diagnostic naming the factor and how often it was excluded.

With the default exact group allocation this is deterministic — you are told
**before the simulation starts** which factor is affected and, in a sample-size
search, the smallest N in the searched range that clears the minimum. (With
sampled allocation or uploaded data the counts vary per run, so there is no
up-front warning — the post-run diagnostics still report any exclusion.) As a
rule of thumb a level with proportion *p* needs roughly `N ≥ 5 / p`: a 5%
level needs about 100 observations just to be estimable, well before it has
any power.

If you see this warning: increase N, raise the sparse level's proportion, or
merge rare levels. See [[variable-types]] for setting factor proportions.

## Upload size depends on where MCPower runs

Pilot-data uploads are capped by platform: up to **1,000,000 rows** in the
desktop app and the Python and R packages, but **10,000 rows** in the browser
app, whose memory budget as a browser tab is tighter. A dataset larger than
10,000 rows has to use the desktop app or a package. The same engine, models,
and numbers run everywhere; only the upload ceiling differs — see
[[about/app-vs-python-vs-R|which MCPower to use]].

## Uploaded data is a description, not a guarantee

Uploading pilot data shapes the data-generating process — the simulated
predictors inherit your sample's distributions and dependence. They also
inherit its flaws: a tiny, biased, or unrepresentative pilot produces a
faithful simulation of an unrepresentative world, and the power number carries
that bias forward. Uploading makes assumptions *concrete*; it cannot make them
*correct*. See [[concepts/upload-data|using empirical data]].

---

For how the parts that *are* in scope get verified, see
[[validation/index|Validation]].
