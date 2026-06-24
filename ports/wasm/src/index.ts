// @mcpower/engine-wasm public client — same surface as ports/app's api/engine.ts.
// Spawns a worker per n_sims share, merges parts, generates plots, aggregates
// progress. Cancellation = worker.terminate().
//
// INVARIANTS that must not break:
//  - warmWorker is never multi-borrowed: warmBusy guards all borrow sites;
//    concurrent runs that would need it always spawn fresh workers instead.
//  - every run settles its fanOut promise on every exit path — part-complete
//    (resolve), worker {kind:'error'} message, transport-level worker error
//    (onerror/onmessageerror), and cancelRun (reject) — and deletes its
//    activeRuns entry, so a run never hangs and the entry never leaks.
//  - a warm worker hit by an error or a cancel is torn down and respawned
//    (teardownWorkers), never returned to the pool; only a cleanly-completed
//    borrow is freed (release) and kept warm.
//  - merge_power_results / merge_sample_size_results must run on the main
//    thread (wasm-bindgen single-instance; workers each hold their own init).
import init, {
  merge_power_results,
  merge_sample_size_results,
  power_plot_specs_json,
  sample_size_plot_specs_json,
  parse_formula as wasm_parse_formula,
  get_effects_from_data as wasm_get_effects_from_data,
  effect_skeleton as wasm_effect_skeleton,
} from '../vendor/engine-wasm/engine_wasm.js';
import { classifyCase, poolSize, splitSims, workerSeeds } from './seeds';
import { ProgressAggregator } from './progress';
import { validateUploadRows } from './upload';
import { reuseFraction, strictReuseWarning } from './reuse';
import type {
  AppSpec, CsvData, EffectDescriptor, EffectsFromData, FindPowerResponse, FindSampleSizeResponse, PlotSpecs, ProgressEvent, SampleSizeMethod,
} from './types';
import htmlTemplate from '$configs/plot-html-template.html?raw';
import plotThemes from '$configs/plot-themes.json';

/** Deep-merge `overlay` into `base` (mutates base). Mirrors Python's `_deep_merge`. */
function deepMerge(base: Record<string, unknown>, overlay: Record<string, unknown>): Record<string, unknown> {
  for (const [key, value] of Object.entries(overlay)) {
    const existing = base[key];
    if (existing !== null && typeof existing === 'object' && !Array.isArray(existing) &&
        value !== null && typeof value === 'object' && !Array.isArray(value)) {
      deepMerge(existing as Record<string, unknown>, value as Record<string, unknown>);
    } else {
      base[key] = value;
    }
  }
  return base;
}

/** Derive `corrected` from an AppSpec JSON object. Mirrors `AppSpec::is_corrected()`. */
function isCorrected(spec: AppSpec): boolean {
  const correction = (spec as Record<string, unknown>).correction;
  return typeof correction === 'string' && correction !== 'none';
}

let mainReady: Promise<unknown> | null = null;
// Exactly one persistent worker, spawned at first API touch (page start in the
// app) so its spawn + wasm init + JIT tiering precede the user's first analysis.
// Runs routed to a single worker (§routing in seeds.ts) borrow it instead of
// spawning, killing the per-run spawn cost on short interactive runs. Heavier
// runs still spawn per-run fan-out (spawn cost is noise there, and not keeping
// W workers idle bounds memory).
let warmWorker: Worker | null = null;
let warmBusy = false;

/** Spawn the warm worker if it is absent. Browser-only: Node/test envs have no
 *  `Worker` global, so this no-ops there (a test that stubs `globalThis.Worker`
 *  gets the stub spawned, as intended). */
function ensureWarm(): void {
  if (warmWorker === null && typeof Worker !== 'undefined') {
    warmWorker = newWorker();
    warmBusy = false;
  }
}

const ensureMain = () => {
  ensureWarm();
  return (mainReady ??= init());
};

const listeners = new Set<(e: ProgressEvent) => void>();
const emit = (e: ProgressEvent) => listeners.forEach((l) => l(e));

/**
 * Subscribe to pool-wide progress events. Multiple listeners are allowed; each
 * receives every ProgressEvent in order. Returns an unsubscribe function —
 * call it to stop receiving events (idempotent).
 */
export async function onProgress(listener: (e: ProgressEvent) => void): Promise<() => void> {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const activeRuns = new Map<string, { workers: Worker[]; reject: (reason: Error) => void }>();

/**
 * Cancel an active run. `runId` is the UUID from the `run_started` event
 * (`run_id` field). Returns `false` if no run with that id is active — not
 * an error; the run may have already finished.
 */
export async function cancelRun(runId: string): Promise<boolean> {
  const run = activeRuns.get(runId);
  if (!run) return false;
  // Cancellation stays terminate(): tear down all workers (recycling the warm
  // worker if it was borrowed) then reject the in-flight fanOut so the awaiting
  // caller settles. Without the reject the awaited run never returns, so the
  // caller's cleanup (e.g. the app store's `finally` clearing currentRunId)
  // never runs. The app treats a reject while runState is 'cancelled' as a clean
  // cancel, mirroring the Tauri path; the 'cancelled' event is emitted first so
  // that state is set before the rejection microtask lands.
  teardownWorkers(run.workers);
  activeRuns.delete(runId);
  emit({ kind: 'cancelled' });
  run.reject(new Error('run cancelled'));
  return true;
}

/**
 * Parse a formula string into the engine's parsed-formula tree. Returns the
 * parsed object. Throws (from the wasm layer) on syntax errors — callers
 * should catch and surface to the user.
 */
export async function parseFormula(formula: string): Promise<unknown> {
  await ensureMain();
  return JSON.parse(wasm_parse_formula(formula));
}

/**
 * The engine's index-only effect skeleton (β-column aligned) for `spec`. Render
 * result names from this + your own factor-label store: a run's `target_indices`
 * index into the returned array directly (index 0 = intercept). Mirrors the
 * `skeleton` the Python/R bridges return and the Tauri `effect_skeleton` command.
 */
export async function effectSkeleton(spec: AppSpec): Promise<EffectDescriptor[]> {
  await ensureMain();
  return JSON.parse(wasm_effect_skeleton(JSON.stringify(spec))) as EffectDescriptor[];
}

/**
 * Recover standardized effect sizes from uploaded data by fitting the spec's
 * estimator (OLS / GLM / MLE) to the columns in `spec.csv`, plus the estimated
 * cluster ICC (mixed) and binary-outcome baseline probability. Mirrors Python's
 * `get_effects_from_data` and the Tauri engine.ts export of the same name.
 * The wasm shim returns the `EffectsFromData` object — not a bare array.
 */
export async function getEffectsFromData(spec: AppSpec): Promise<EffectsFromData> {
  await ensureMain();
  return JSON.parse(wasm_get_effects_from_data(JSON.stringify(spec))) as EffectsFromData;
}

function newWorker(): Worker {
  // Vite resolves this URL form to a bundled worker chunk.
  return new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
}

/** Terminate all of a run's workers; if the warm worker is among them, respawn a
 *  replacement in the background so the next routed-single run still finds a warm
 *  worker. Used by cancel and by fanOut's error paths so a warm worker that
 *  errored or was cancelled is never reused — unlike `release`, which keeps the
 *  warm worker alive after a clean completion. */
function teardownWorkers(workers: Worker[]): void {
  let killedWarm = false;
  workers.forEach((w) => {
    if (w === warmWorker) killedWarm = true;
    w.terminate();
  });
  if (killedWarm) {
    warmWorker = null;
    warmBusy = false;
    ensureWarm();
  }
}

/** Route a run to a worker count via the size-based heuristic (seeds.ts). `n`
 *  is the run's sample size (a sample-size grid passes its largest N). */
function routedWorkers(spec: AppSpec, n: number, nSims: number, forced?: number): number {
  const scenarios = (spec as unknown as { scenarios?: unknown[] }).scenarios;
  return poolSize({
    caseClass: classifyCase(spec),
    hasScenarios: Array.isArray(scenarios) && scenarios.length > 0,
    n,
    nSims,
    hardwareConcurrency: navigator.hardwareConcurrency,
    forced,
  });
}

/** Run one share per worker; resolve when all parts are in (or reject on cancel/error).
 *  `warm` (only ever set for a single-worker run) borrows the persistent warm
 *  worker as worker 0: it is freed (busy flag cleared) instead of terminated, so
 *  it stays warm for the next routed-single run — an identity merge over the one
 *  part, not a special path. */
function fanOut(
  runId: string,
  shares: number[],
  seeds: bigint[],
  makeMsg: (share: number, seed: bigint) => unknown,
  agg: ProgressAggregator,
  warm: boolean,
): Promise<string[]> {
  const workers = shares.map((_, i) => (warm && i === 0 ? warmWorker! : newWorker()));
  if (warm) warmBusy = true;
  const parts: string[] = new Array(shares.length);

  const release = (w: Worker) => {
    if (warm && w === warmWorker) warmBusy = false;
    else w.terminate();
  };

  let rejectRun!: (reason: Error) => void;
  const promise = new Promise<string[]>((resolve, reject) => {
    rejectRun = reject;
    let remaining = shares.length;
    // Any worker failure tears the whole run down (recycling a borrowed warm
    // worker so it isn't reused) and rejects, so the awaiting caller never hangs.
    const fail = (err: Error) => {
      teardownWorkers(workers);
      activeRuns.delete(runId);
      reject(err);
    };
    workers.forEach((w, i) => {
      w.onmessage = (e: MessageEvent) => {
        const m = e.data;
        if (m.kind === 'progress') {
          const ev = m.event as ProgressEvent;
          agg.workerEvent(i, ev);
          if (ev.kind === 'sims_completed') agg.workerProgress(i, ev.completed);
        } else if (m.kind === 'part') {
          parts[i] = m.part;
          agg.workerDone(i);
          release(w);
          if (--remaining === 0) {
            activeRuns.delete(runId);
            resolve(parts);
          }
        } else if (m.kind === 'error') {
          fail(new Error(m.message));
        }
      };
      // A worker that fails to load, throws during wasm init before its own
      // onmessage try/catch runs, or can't (de)serialize a message never posts a
      // 'part'/'error' — without these the run's promise would never settle.
      w.onerror = (e: ErrorEvent) => fail(new Error(`worker error: ${e.message || 'failed to load or initialize'}`));
      w.onmessageerror = () => fail(new Error('worker message deserialization failed (structured clone)'));
      w.postMessage(makeMsg(shares[i]!, seeds[i]!));
    });
  });
  // Registered after the executor runs (synchronously, so rejectRun is set);
  // workers reply asynchronously, so no handler can fire before this.
  activeRuns.set(runId, { workers, reject: rejectRun });
  return promise;
}

/** Whether a routed run should borrow the warm worker: routed (not `forced`) to
 *  a single worker, and the warm worker is spawned and idle. Otherwise spawn
 *  fresh — including the busy fallback when a concurrent run holds it. */
function borrowWarm(nWorkers: number, forced: number | undefined): boolean {
  return forced === undefined && nWorkers === 1 && warmWorker !== null && !warmBusy;
}

export async function findPower(
  spec: AppSpec,
  sampleSize: number,
  opts?: { uploadRows?: number; workers?: number },
): Promise<FindPowerResponse> {
  await ensureMain();
  if (opts?.uploadRows !== undefined) validateUploadRows(opts.uploadRows); // reject oversized uploads before touching the engine

  const runId = crypto.randomUUID();
  emit({ kind: 'run_started', run_id: runId });

  const nSims = (spec as unknown as { n_sims: number }).n_sims;
  const nWorkers = routedWorkers(spec, sampleSize, nSims, opts?.workers);
  const shares = splitSims(nSims, nWorkers);
  const seeds = workerSeeds(BigInt((spec as unknown as { seed: number | string }).seed), nWorkers);

  // The aggregator's `started` is driven by the workers' real `started` event
  // (carries the true total_scenarios); no eager fake-`started` here.
  const agg = new ProgressAggregator(nSims, shares, sampleSize, emit);

  const specJson = JSON.stringify(spec);
  const parts = await fanOut(
    runId, shares, seeds,
    (share, seed) => ({ kind: 'power', spec: specJson, sampleSize, nSims: share, seed: seed.toString() }),
    agg, borrowWarm(nWorkers, opts?.workers),
  );

  const merged = merge_power_results(`[${parts.join(',')}]`);
  const plots = JSON.parse(power_plot_specs_json(merged, spec.target_power, isCorrected(spec))) as PlotSpecs;
  agg.finish();

  const warnings: string[] = [];
  const csv = (spec as unknown as { csv: CsvData | null }).csv;
  if (csv?.mode === 'strict' && csv.n_rows > 0) {
    const U = csv.n_rows;
    const frac = reuseFraction(U, sampleSize);
    const diagLine = `[strict bootstrap] N=${sampleSize}, uploaded rows U=${U}: ~${Math.round(frac)}% of rows reused per simulated dataset.`;
    warnings.push(diagLine);
    console.warn(diagLine);
    const w = strictReuseWarning(U, sampleSize);
    if (w !== null) {
      warnings.push(w);
      console.warn(w);
    }
  }

  return { run_id: runId, result: JSON.parse(merged), plots, warnings };
}

export async function findSampleSize(
  spec: AppSpec,
  bounds: [number, number],
  method: SampleSizeMethod,
  opts?: { uploadRows?: number; workers?: number },
): Promise<FindSampleSizeResponse> {
  await ensureMain();
  if (opts?.uploadRows !== undefined) validateUploadRows(opts.uploadRows);

  const runId = crypto.randomUUID();
  emit({ kind: 'run_started', run_id: runId });

  const nSims = (spec as unknown as { n_sims: number }).n_sims;
  const nWorkers = routedWorkers(spec, bounds[1], nSims, opts?.workers);
  const shares = splitSims(nSims, nWorkers);
  const seeds = workerSeeds(BigInt((spec as unknown as { seed: number | string }).seed), nWorkers);

  // `started` is driven by the workers' real envelope (true total_grid_points).
  const agg = new ProgressAggregator(nSims, shares, bounds[0], emit);

  const specJson = JSON.stringify(spec);
  const boundsJson = JSON.stringify(bounds);
  const methodJson = JSON.stringify(method);
  const parts = await fanOut(
    runId, shares, seeds,
    (share, seed) => ({
      kind: 'sample_size', spec: specJson, bounds: boundsJson, method: methodJson,
      nSims: share, seed: seed.toString(),
    }),
    agg, borrowWarm(nWorkers, opts?.workers),
  );

  const merged = merge_sample_size_results(`[${parts.join(',')}]`);
  const plots = JSON.parse(sample_size_plot_specs_json(merged, spec.target_power, isCorrected(spec))) as PlotSpecs;
  agg.finish();

  const warnings: string[] = [];
  const csvSS = (spec as unknown as { csv: CsvData | null }).csv;
  if (csvSS?.mode === 'strict' && csvSS.n_rows > 0) {
    const U = csvSS.n_rows;
    // Per-target per-scenario diagnostics, mirroring Python's find_sample_size path.
    const parsedResult = JSON.parse(merged) as { scenarios: [string, { first_achieved: (number | null)[] }][] };
    for (const [, inner] of parsedResult.scenarios) {
      const fa = inner.first_achieved ?? [];
      fa.forEach((achievedN, pos) => {
        if (achievedN === null) return;
        const frac = reuseFraction(U, achievedN);
        const diagLine = `[strict bootstrap] target ${pos}: first N=${achievedN}, uploaded rows U=${U}: ~${Math.round(frac)}% of rows reused per simulated dataset.`;
        warnings.push(diagLine);
        console.warn(diagLine);
        const w = strictReuseWarning(U, achievedN);
        if (w !== null) {
          warnings.push(w);
          console.warn(w);
        }
      });
    }
  }

  return { run_id: runId, result: JSON.parse(merged), plots, warnings };
}

/**
 * Wrap a Vega-Lite spec JSON string in a self-contained HTML document using
 * the shared CDN template (`configs/plot-html-template.html`).  The `light-print`
 * theme is deep-merged into `spec.config` before embedding — this is a saved
 * artifact, so it gets the light-print theme.  The returned string can be written to
 * a `.html` file or opened in a browser tab via a Blob URL.
 *
 * Mirrors Python's `_write_stacked_html` (theme=`"light-print"`) in
 * `ports/py/mcpower/output/plotting.py`.
 * Mirror checklist: if the `<\/` escape strategy or the `"light-print"` theme-key
 * changes in `_write_stacked_html`, update this function to match.
 */
export function plotHtml(spec: string): string {
  const specObj = JSON.parse(spec) as Record<string, unknown>;
  const printTheme = (plotThemes as Record<string, Record<string, unknown>>)['light-print']!;
  const config = (specObj['config'] !== null && typeof specObj['config'] === 'object' && !Array.isArray(specObj['config']))
    ? specObj['config'] as Record<string, unknown>
    : {};
  specObj['config'] = deepMerge(config, printTheme);
  // Escape "</" → "<\/" in the substituted payload so a spec value containing a
  // literal "</script>" cannot break out of the inline <script> block.
  // The backslash is inert inside a JS string; the HTML parser no longer sees a
  // closing tag. Mirrors Python's `.replace("</", "<\\/")` in _write_stacked_html.
  const specsJson = JSON.stringify([specObj]).replace(/<\//g, '<\\/');
  return htmlTemplate.replace('{{SPECS}}', specsJson);
}

/**
 * Trigger a browser file-download of a Vega-Lite spec JSON string.
 * Creates a temporary anchor element, clicks it, then cleans up.
 * Browser-only — in Node/vitest environments mock `document`, `Blob`, and `URL`.
 */
export function downloadPlot(spec: string, filename: string): void {
  const blob = new Blob([spec], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
