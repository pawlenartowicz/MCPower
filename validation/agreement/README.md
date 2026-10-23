# MCPower power agreement

Does MCPower's power match independent references? Each case is run through
MCPower and through a reference (a hand-written simulation loop, and where one
covers the design, a dedicated tool: simr, Superpower, simglm). The two power
estimates are compared point by point. The check is rerunnable and is not a
pass/fail gate.

## The cases — `cases.json`

`cases.json` is the single source of cases, read by `harness.R`. Sixteen designs
across four families (`ols`, `logit`, `lme`, `glmm`). `lme_ex2` is a random
intercept and slope design with a cluster-level binary predictor and 60 clusters.
Each case names its `targets` and the dedicated tool
that covers it (`tool`), or `null` when none does. The `defaults.agreement` block
holds the per-grid-point simulation counts.

## Same inference in every tier

Every reference applies MCPower's default decision rule, so all tiers estimate
the same quantity:

- OLS: *t* at the residual df *n − (P + 1)*.
- Logit GLM: Wald *z*.
- LMM and GLMM: the Wald statistic against a *t* at the between-within df.
  A target that varies within clusters: *n − (P + 1) − (g − p_between)*, with
  *g* the number of clusters and *p_between* the cluster-level fixed effects,
  intercept included. A target constant within clusters: *g −* (number of
  cluster-level fixed effects, intercept included). LMM fits use REML; GLMM fits
  use Laplace with RX-based standard errors (`wald_se = "rx"`).
- Mixed fits with every variance component at zero use the single-level
  reference: *t* at *n − (P + 1)* for LMM, Wald *z* for GLMM.
- A failed fit counts as not significant, and power is divided by all
  simulations. A singular fit is not a failure.
- Superpower is scored on the pairwise contrast equal to MCPower's dummy
  coefficient, not its omnibus *F*. simr stores *z* p-values, so its summary is
  taken at the alpha that makes *p < alpha* the same event as *|z| >* the *t*
  critical value.

What still differs by design: simr draws the predictors once and reuses them in
every simulation, and Superpower allocates exact cell counts. MCPower and the
loops sample predictors and factor levels anew each simulation (`harness.R` sets
sampled factor proportions).

## Running

From this directory, with the `mcpower` R port installed plus `lme4`,
`lmerTest`, `simr`, `Superpower`, `simglm`:

```bash
Rscript harness.R --case all --budget agreement --methods mcpower_find_power,loop_best,tool --out results/agreement_r.json
../../../.venv/bin/python power_agreement.py --in results/agreement_r.json
Rscript harness.R --case ols_multi --budget agreement --out results/ols_multi.json   # single case
```

The snapshot takes `results/*_r.json` and their `<stem>_points.csv` /
`<stem>_summary.csv`, never smoke outputs.

`--budget agreement` uses the per-grid-point counts from `cases.json`
(`defaults.agreement`): MCPower 10,000; loop 2,000; simglm and Superpower 500;
simr 200. Seed is `2137 + n`. `--scale 0.1` runs a fraction of the simulations
for a quick preview. Every results JSON is `{meta, records}`, with the mcpower
version in `meta.packages.mcpower`.

Three example scripts check the documentation examples against independent
loops. Each writes a `{meta, records}` JSON to `results/`, which
`power_agreement.py --in` reads like the harness output. `SMOKE=1` runs 2% of the
simulations to check the calls, not the numbers.

```bash
Rscript example_upload.R    # uploaded pilot (pilot.csv), strict mode, resampling loop
Rscript example_misspec.R   # misspecified analysis models, seven cases
Rscript example_knobs.R     # scenario knobs, seven cases
SMOKE=1 Rscript example_upload.R
```

## The agreement statistic

Per (case, target, n, comparator), with *x* = round(power × n_sims):

    z = (x_mc/n_mc − x_ref/n_ref) / sqrt(p̃_mc(1 − p̃_mc)/n_mc + p̃_ref(1 − p̃_ref)/n_ref)

where *p̃* = (*x* + 1) / (*n* + 2) in the variances only; the numerator uses the
raw powers. MCPower and its comparator are paired by target name, never by index.
A point is flagged when |*z*| > qnorm(0.95) (per-test 0.1, no multiple-testing
correction). Flags are leads to investigate, not a verdict, and the threshold is
not tuned after seeing results.

`power_agreement.py --in <file>.json` writes, beside the input (`<stem>` is the
file name without extension):

- `<stem>_points.csv` — one row per point: case, family, target, n, comparator,
  both powers and sim counts, `z`, `flagged`, mcpower version.
- `<stem>_summary.csv` — one row per (case, comparator): points, flags, expected
  flags (0.1 × points), mean *z*, SE of the mean (1/sqrt(points)), largest
  |power difference|.
- `<stem>_scatter.png`, `<stem>_curves_<k>.png` — scatter and per-case curves.

Near power 0 or 1 the counts are discrete and *z* shrinks toward 0, so 0.1 ×
points is an upper bound on the expected flags. A case's mean *z* over 10–16
points detects a systematic offset of about 1 point against the loop; the tool
tier detects only gross errors (about 10 points).

## Snapshot

`snapshot.sh <name>` copies the results into `../snapshots/<name>/` once. It
refuses if that directory exists, if the mcpower versions in the JSON files
differ, or if any JSON has `meta.n_sims_scale` other than 1.

## Files

| file | role |
|------|------|
| `cases.json` | the 16 cases + per-family defaults (single source) |
| `harness.R` | R harness: runs MCPower, the loops and the tools; CLI |
| `loops_r.R` | R reference loops (naive + best) |
| `tools_r.R` | dedicated-tool adapters: simr, Superpower, simglm |
| `smoke_tools.R` | on-request smoke for the tool adapters |
| `power_agreement.py` | the agreement statistic, CSVs and plots |
| `test_power_agreement.py` | unit tests for `power_agreement.py` |
| `example_upload.R`, `example_misspec.R`, `example_knobs.R` | example checks; `pilot.csv` is the upload pilot |
| `snapshot.sh` | writes a frozen snapshot |
| `results/` | rolling outputs (git-ignored) |
