# MCPower cross-port speed benchmark

A single, JSON-driven benchmark that times **MCPower vs dedicated power tools
vs DIY simulation loops**, in Python and R, across fifteen power-analysis
designs. It answers two questions:

1. **MCPower vs the alternatives, per language** — how much faster is
   MCPower's native engine than the dedicated R tools (simr, Superpower,
   simglm) and than hand-written simulation loops, naive and optimized.
2. **MCPower-Python vs MCPower-R** — both ports wrap the *same* native engine,
   so this is both a speed check (do they cost the same?) and a correctness
   cross-check (same seed + single thread ⇒ identical power numbers).

It also carries a **Julia** loop tier (`loop_naive` + `loop_best`, all four
families incl. GLMM) as a third, competitor-only language. Julia removes the
interpreted-language confound: a JIT-compiled tight loop runs at near-native
speed, so "Rust engine vs Julia loop" is a language-neutral test of whether
MCPower's `find_sample_size` grid-search advantage (one call, draws shared
across the grid) survives against an opponent whose per-sim loop is as fast as
ours. Julia has no MCPower engine port — so no `mcpower:jl`, no
`find_sample_size:jl`, and no Julia `tool` tier.

It is timing-oriented. Power values are reported only as a sanity signal — for
statistical validation see `mcpower/validation/`.

## The cases — `benchmark_cases.json`

`benchmark_cases.json` is the single source of cases, read by both harnesses
(`cases.py` / `load_cases` in `harness.R`). Fifteen designs across four
families:

| family | cases | shapes covered |
|--------|-------|----------------|
| `ols` (6)    | `ols_simple` … `anova_oneway4` | 1–5 predictors, correlated predictors, large n, 2×2 / one-way-4 ANOVA |
| `logit` (3)  | `glm_simple` … `glm_rare` | the same shapes on the logit scale (baseline P(y=1)=0.3), plus rare events (0.05) |
| `lme` (3)    | `lme_simple` … `lme_factor_inter` | random intercept, ICC=0.2, 20–100 clusters |
| `glmm` (3)   | `glmm_simple` … `glmm_multislope` | random intercept + random slopes on the logit scale, ICC=0.2, 20–30 clusters |

Each family carries its own default `n` grid and per-tier simulation counts
(`defaults` block): OLS sweeps `n = 20..200` step 20, logit `50..500` step 50,
LME / GLMM `100..1000` step 100 — several cases override the grid. MCPower runs
10,000 sims per point (1,000 for GLMM); the loop and tool tiers run fewer
(30–500, per the `n_sims` block) since they are far slower per sim. Each case
names the dedicated tool that covers it (`tool`), or `null` when no tool covers
the design (the cliff band).

## Methods measured

Five recorded methods per case:

- **`mcpower_find_power`** — the native engine, multi-core (plus a 1-thread
  variant via `--threads 1`), in Python and R.
- **`mcpower_find_sample_size`** — one call evaluating the full n grid from
  shared draws. A different unit from the per-n rows (one row per case);
  combine.py reports it in its own grid-vs-grid table.
- **`tool`** (R only; the Python harness prints a skip) — the dedicated tool
  named by the case, used as its docs show: `simr` (LME + GLMM), `Superpower`
  (ANOVA), `simglm` (OLS/GLM). Adapters in `tools_r.R`.
- **`loop_best`** — hand-rolled DIY loop with precomputed critical values and
  a manual Wald decision, matching MCPower's decision rule (OLS *t*; GLM/LME/
  GLMM Wald *z* — not Satterthwaite), parallelized across cores. Python: numpy /
  IRLS / statsmodels `MixedLM`. R: manual `lm`-style fit / `glm.fit` /
  `lme4::lmer` / `lme4::glmer`. Julia: hand-rolled OLS/IRLS / `MixedModels.jl`
  (LMM + Laplace GLMM), `Base.Threads`-parallel, BLAS pinned to 1 thread.
- **`loop_naive`** — off-the-shelf API the way someone writes it first,
  serial. Python: `statsmodels` `OLS`/`GLM`/`MixedLM`. R: `lm` /
  `glm(binomial)` / `lmerTest::lmer` (Satterthwaite p-values). Julia:
  `GLM.jl` / `MixedModels.jl`.

Julia records **only** the two loop tiers (no mcpower/tool/find_sample_size
rows) and covers **all four families including GLMM** — the family the Python
harness skips. R already supplies a `glmer` GLMM loop, so Julia GLMM is a second
comparator, not a gap-filler.

**Timing hygiene:** one discarded warm-up call per (method, case, n), then the
median of 3 reps for the fast tiers (`mcpower_find_power`,
`mcpower_find_sample_size`, `loop_best`); a single rep for `loop_naive`; a
micro warm-up (nsim 4, 10 for Superpower) + a single timed rep for the tools.

**Metric:** per-sim time is the headline (so the different sim-count tiers are
comparable); raw wall-clock is recorded too.

## Running

From the `mcpower/benchmarks/` directory. `./run_all.sh` runs the full
pipeline — both harnesses, a single-thread mcpower pass each, then combine +
plot, then the power-agreement cross-check (`power_agreement.py`, printed right
after the timing tables) — skipping any harness stage whose results file already
exists (delete `results/*.json` to re-run).

Python (needs the workspace `.venv` active):

```bash
source ../../.venv/bin/activate
python harness.py --case all --out results/py.json          # all 15 cases
python harness.py --case ols_multi                          # one case
python harness.py --case ols_multi --threads 1              # 1-thread variant
python harness.py --case glm_multi --methods mcpower_find_power,loop_best
```

R (uses the installed `mcpower` package + `lme4`/`lmerTest`/`simr`/
`Superpower`/`simglm`):

```bash
Rscript harness.R --case all --out results/r.json
Rscript harness.R --case ols_multi --threads 1
```

Julia (needs the `benchmarks/` project instantiated once —
`julia --project=. -e 'using Pkg; Pkg.instantiate()'` — and `GLM`,
`MixedModels`, `StatsModels`, `Distributions`, `JSON3` from `Project.toml`):

```bash
julia --project=. -t auto harness.jl --case all --out results/jl.json   # all cases, all 4 families
julia --project=. -t auto harness.jl --case glmm_simple --methods loop_best
julia --project=. runtests.jl                                            # case-load + design-parse + loop smoke
```

Julia records only the loop tiers (no engine), so there is no `--threads 1`
twin and no `jl_1t.json`.

Combine all languages' results (optionally with the 1-thread runs and a
chart):

```bash
python combine.py results/py.json results/r.json
python combine.py results/py.json results/r.json \
  --py-1t results/py_1t.json --r-1t results/r_1t.json \
  --plot results/summary_fp.png --plot-fss results/summary_fss.png
```

`--threads 1` re-execs the harness under `RAYON_NUM_THREADS=1` (both
languages); the default runs multi-core. Each invocation is one thread mode.

The power-agreement cross-check (run automatically as the last pipeline step,
or standalone) prints the per-tool `|MCPower − tool|` agreement and the largest
disagreements, and writes the scatter / per-case curve plots:

```bash
python power_agreement.py        # reads results/r.json
```

## The combined output

`combine.py` prints, per `(case, n)`, the per-sim times and ratios:

- `mc-py`, `mc-r` — MCPower `find_power` per-sim seconds, each language.
- `py/r` — MCPower-R ÷ MCPower-Python (≈1.0 expected: same engine).
- `tool`, `tool×r` — the covering tool and its per-sim time ÷ MCPower-R.
- `naive×py` / `naive×r` / `naive×jl`, `best×py` / `best×r` / `best×jl` — the
  loop tiers' per-sim time ÷ MCPower. The `:jl` columns are normalized to the
  **py** engine (Julia has no engine; py/r per-sim ≈ equal, same engine).

The loop tiers also surface as `loop_best:jl` / `loop_naive:jl` series in the
per-family aggregates and grid-search projection. Caveat: the per-family
aggregate and the two plots restrict to each family's **tool-covered** cases
(GLMM is simr-covered like LME); a family with *no* tool-covered cases falls
back to all its measured cases, labelled "no tool coverage" under the family
group instead of the usual coverage count.

Below the table: per-family geometric-mean aggregates normalized to
`mcpower:py = 1` (each tool averaged over its own covered cases), a per-family
summary projecting the full power curve at 10,000 sims/point, the
`find_sample_size`-vs-grid table (the one call evaluating the full grid vs the
summed per-point `find_power` calls), and the **find-sample-size grid-search**
table — the real task users run: locate the n that hits target power by
searching the grid. MCPower answers the whole grid in one `find_sample_size`
call (budget S sims **total**, draws shared across the grid); every alternative
must rerun a full power simulation at each of the G grid points (S per point,
S×G total), so it is projected per-sim to the same precision and summed over the
grid. Each tier's seconds are reported with the total sims it spends, so the
"fewer total sims by design" advantage is explicit, not hidden. The `--threads
1` runs appear as the `mcpower-1t` series in the aggregates and the summary
plot.

The two charts are written as **separate files**: `--plot` writes the
`find_power` per-sim comparison (`summary_fp.png`) and `--plot-fss` writes the
`find_sample_size` grid-search comparison (`summary_fss.png`) — the latter pits
`find_sample_size` against the dedicated tools and DIY loops only (the
within-MCPower `find_power`-grid win lives in the `find_sample_size`-vs-grid
table, on a different scale). Each is a per-family log-y bar chart.

## Notes

- **Correctness cross-check (MCPower-py vs MCPower-R).** At the same seed and a
  single worker the two ports return *identical* power — they wrap the same
  native engine. (The benchmark uses `seed = 2137 + n`, computed identically in
  both harnesses.)
- **Loop tiers cross-language are timing-only.** Python (PCG64) and R
  (Mersenne-Twister) draw different values at the same seed, so the loop power
  numbers differ by Monte-Carlo noise across languages — only the wall-clock
  is comparable.
- **Factor allocation is sampled in every tier.** The loops and tools draw
  factor levels randomly per sim, so the harnesses flip mcpower's optimistic
  baseline from exact-count to sampled allocation
  (`set_scenario_configs`) — all tiers answer the same random-allocation
  question.
- **Port overhead is negligible.** A bulk `find_power` call (one call, many
  sims) costs essentially the same in both ports: per-sim times sit within a
  small constant factor (`py/r` ≈ 1; the spread is thread-scheduling and
  measurement noise at microsecond scale), not the large gap that real FFI
  marshalling would produce. At a single thread with the same seed the two
  ports return *identical* power. Per-call marshalling would only matter for
  many-small-call patterns such as a `find_sample_size` grid.
- **Results are scratch.** `results/*.json` is git-ignored; the harnesses,
  `loops_*` / `tools_r.R`, `combine.py`, and `benchmark_cases.json` are
  committed so the speed claims are reproducible.

## Files

| file | role |
|------|------|
| `benchmark_cases.json` | the 15 cases + per-family defaults (single source) |
| `cases.py`             | loads the cases, merges family defaults (Python) |
| `harness.py`           | Python harness: timing runner, CLI |
| `loops_py.py`          | Python DIY-loop baselines (naive + best), keyed by family |
| `harness.R`            | R harness (mirrors `harness.py`) |
| `loops_r.R`            | R DIY-loop baselines (mirrors `loops_py.py`) |
| `tools_r.R`            | dedicated-tool adapters: simr, Superpower, simglm |
| `smoke_tools.R`        | on-request smoke for the tool adapters |
| `cases.jl`             | loads the cases, merges family defaults (Julia; mirrors `cases.py`) |
| `loops_jl.jl`          | Julia DIY-loop baselines, all 4 families incl. GLMM (mirrors `loops_py.py` + `loops_r.R` glmm) |
| `harness.jl`           | Julia harness: loop-tier timing runner, CLI (mirrors `harness.py`, loops only) |
| `runtests.jl`          | Julia unit tests (case-load, design-parse, loop smoke, harness) |
| `Project.toml` / `Manifest.toml` | pinned Julia dependency environment for the `.jl` tier |
| `combine.py`           | merges results into the tables, aggregates, summary + grid-search plots |
| `power_agreement.py`   | power cross-check vs tools + DIY loops (printed after the tables); scatter / curve plots |
| `run_all.sh`           | full pipeline with results-file caching |
| `test_benchmark.py`    | unit tests (case loading, build, loops, combine) |
