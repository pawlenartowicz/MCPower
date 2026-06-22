// The bench's scaling case: drive the real worker-pool wrapper (findPower)
// at forced 1/2/4/8/12/16 workers in a real browser — Web Worker postMessage is a
// browser phenomenon Node would mismeasure. No core pinning (Chrome is
// many-threaded); ratios of the SAME spec absorb the absolute noise.
import { findPower } from '../../src/index';
import type { AppSpec } from '../../src/types';

// AppSpec approximations of two native grid rows. The per-case rows are
// single-sourced from --dump-cases; these hand copies are sanctioned because
// only the worker-count ratios (1→2→4→8→12→16, matching WORKER_COUNTS) of
// each spec are ever reported.
//
// ols_multi: 5 standard-normal predictors, x1=x2=0.25, n=200, 10k sims —
// the FASTEST grid case (~0.4 s single-core), so its ratios measure pool
// fixed costs, not compute scaling. glmm_intercept: the heavy arm (binary +
// random intercept, ~490 fits/s single-core wasm) at the app's mixed default
// 800 sims — compute-dominated, so its ratios measure real scaling.
const OLS_SPEC = {
  family: 'linear',
  parsed_formula: { outcome: 'y', predictors: ['x1', 'x2', 'x3', 'x4', 'x5'], interaction_terms: [] },
  var_types: [
    { kind: 'numeric', name: 'x1' },
    { kind: 'numeric', name: 'x2' },
    { kind: 'numeric', name: 'x3' },
    { kind: 'numeric', name: 'x4' },
    { kind: 'numeric', name: 'x5' },
  ],
  effects: [
    { name: 'x1', value: 0.25 },
    { name: 'x2', value: 0.25 },
    { name: 'x3', value: 0 },
    { name: 'x4', value: 0 },
    { name: 'x5', value: 0 },
  ],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 10000,
  seed: 2137,
  tests: { kind: 'all' },
  correction: 'none',
  scenarios: [],
  csv: null,
  report_overall: false,
  contrasts: [],
} as unknown as AppSpec;

// Mirrors throughput.rs glmm_intercept: glm_spec(1, 0.3) + FixedClusters{8},
// τ²=0.822 latent ⇔ icc 0.2 (assembler scales ×π²/3), x1=0.5, n=480; n_sims
// is the app's mixed default 800 (native row uses 1000 — ratios don't care).
const GLMM_SPEC = {
  family: 'mixed',
  parsed_formula: { outcome: 'y', predictors: ['x1'], interaction_terms: [] },
  var_types: [{ kind: 'numeric', name: 'x1' }],
  effects: [{ name: 'x1', value: 0.5 }],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 800,
  seed: 2137,
  tests: { kind: 'all' },
  correction: 'none',
  scenarios: [],
  csv: null,
  report_overall: false,
  contrasts: [],
  cluster_name: 'g',
  icc: 0.2,
  cluster_dim: { kind: 'n_clusters', value: 8 },
  cluster_level_vars: [],
  extra_groupings: [],
  slopes: [],
  outcome: { kind: 'binary', baseline_probability: 0.3 },
} as unknown as AppSpec;

const CASES: Record<string, { spec: AppSpec; sampleSize: number }> = {
  ols_multi: { spec: OLS_SPEC, sampleSize: 200 },
  glmm_intercept: { spec: GLMM_SPEC, sampleSize: 480 },
};

// Default-route latency arm: a default-sized OLS (1,600 sims) routed with NO
// `workers` override falls to the size-based heuristic's 1-worker anchor and
// borrows the persistent warm worker (index.ts). Compared against forced:1
// (spawn-per-run, terminated after — cold) to expose the per-run spawn + wasm
// instantiation fixed cost the warm worker eliminates on short interactive runs.
// Gate: warm end-to-end fixed cost ≤ 10 ms over single-core compute (the compute
// number comes from the throughput bench, not this page).
const DEFAULT_OLS_SPEC = { ...(OLS_SPEC as Record<string, unknown>), n_sims: 1600 } as unknown as AppSpec;

async function timeDefault(forced?: number): Promise<number> {
  const t0 = performance.now();
  await findPower(DEFAULT_OLS_SPEC, 200, forced === undefined ? undefined : { workers: forced });
  return (performance.now() - t0) / 1000;
}

const WORKER_COUNTS = [1, 2, 4, 8, 12, 16];
const REPS = 3;

async function timeRun(spec: AppSpec, sampleSize: number, workers: number): Promise<number> {
  const t0 = performance.now();
  await findPower(spec, sampleSize, { workers });
  return (performance.now() - t0) / 1000;
}

document.getElementById('run')!.addEventListener('click', async () => {
  const out = document.getElementById('out')!;
  const caseId = (document.getElementById('case') as HTMLSelectElement).value;
  const { spec, sampleSize } = CASES[caseId]!;
  out.textContent = `case: ${caseId}\nrunning…\n`;
  const mins = new Map<number, number>();
  try {
    for (const w of WORKER_COUNTS) {
      const warm = await timeRun(spec, sampleSize, w); // discarded warm-up: worker spawn + per-worker wasm init + JIT tiering
      out.textContent += `${w} worker(s): warm-up ${warm.toFixed(3)} s\n`;
      let best = Infinity;
      for (let i = 0; i < REPS; i++) best = Math.min(best, await timeRun(spec, sampleSize, w));
      mins.set(w, best);
      out.textContent += `${w} worker(s): min ${best.toFixed(3)} s over ${REPS} reps\n`;
    }
  } catch (err) {
    out.textContent += `\nERROR: ${String(err)}\n`;
    throw err;
  }
  const t1 = mins.get(1)!;
  out.textContent += '\nscaling ratios (ideal = worker count):\n';
  for (const w of WORKER_COUNTS) out.textContent += `  1→${w}: ${(t1 / mins.get(w)!).toFixed(2)}x\n`;
  out.textContent += '\npool+merge overhead share at w (w·t_w − t_1)/t_1:\n';
  for (const w of WORKER_COUNTS.slice(1)) {
    out.textContent += `  w=${w}: ${((((w * mins.get(w)!) - t1) / t1) * 100).toFixed(1)}%\n`;
  }
});

document.getElementById('run-default')!.addEventListener('click', async () => {
  const out = document.getElementById('out')!;
  out.textContent = 'default-route latency — 1,600-sim OLS (warm vs cold)\nrunning…\n';
  const reps = 3;
  try {
    await timeDefault();                     // discarded warm-up: spawns + inits the warm worker
    let warm = Infinity;
    for (let i = 0; i < reps; i++) warm = Math.min(warm, await timeDefault());
    await timeDefault(1);                     // discarded warm-up
    let cold = Infinity;
    for (let i = 0; i < reps; i++) cold = Math.min(cold, await timeDefault(1));
    out.textContent += `warm (default route, reused worker): min ${(warm * 1000).toFixed(1)} ms over ${reps}\n`;
    out.textContent += `cold (forced:1, spawn-per-run):      min ${(cold * 1000).toFixed(1)} ms over ${reps}\n`;
    out.textContent += `spawn + wasm-init recovered by warm: ${((cold - warm) * 1000).toFixed(1)} ms\n`;
    out.textContent += 'gate: warm − single-core compute ≤ 10 ms (compute from the throughput bench)\n';
  } catch (err) {
    out.textContent += `\nERROR: ${String(err)}\n`;
    throw err;
  }
});
