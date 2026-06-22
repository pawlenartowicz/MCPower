// Progress aggregator for the WASM worker pool.
//
// BARRIER INVARIANTS:
//  - workerEvent must be called before workerProgress for the same worker tick
//    (index.ts routes sims_completed via workerProgress only after workerEvent).
//  - finish() must be called exactly once, after all workers have completed
//    their shares (fanOut resolves first, then the caller calls agg.finish()).
//  - fitsPerSim defaults to 1 until the first `started` event arrives — safe
//    because `started` always precedes any `sims_completed` from the same worker.
import type { ProgressEvent } from './types';

/**
 * Reconstructs ONE native-semantics progress stream from N per-worker streams.
 *
 * Each worker runs `single_core_find_*` over the *full* scenario slice (workers
 * split `n_sims`, not scenarios), so every worker emits a complete, identically
 * shaped envelope: `started` → per scenario (`scenario_started`, grid
 * `n_point_completed`*, `scenario_completed`) → `completed`. Naive forwarding
 * would therefore emit each envelope event N times. This aggregator collapses
 * them into the single stream the Tauri shell delivers, so the shared run store
 * sees identical event semantics on both shells:
 *
 *  - `started` once, with the *true* total_sims (pool total, not one worker's
 *    share) and the worker's real total_scenarios / total_grid_points (never 1);
 *  - `scenario_started` once per scenario idx (first worker to reach it);
 *  - `n_point_completed` once per (scenario idx, n) — only after ALL workers
 *    have reported that N-point (barrier across the pool);
 *  - `scenario_completed` once per idx — only after ALL workers finish it;
 *  - `sims_completed` monotone, summed across workers' cumulative fit counts.
 *  - `completed` once, after every worker's share is in (driven by `finish`).
 *
 * Unit accounting: the progress unit is one model fit, and a worker's total is
 * `share × M` where `M = n_scenarios × grid_points`. The grid is built engine-
 * side (atom snapping, bounds rounding) so M is NOT derivable client-side;
 * instead it is recovered exactly from any worker's `started` event as
 * `started.total_sims / share` — an exact integer division because the engine
 * computed that product from the same share.
 */
export class ProgressAggregator {
  private readonly perWorker: number[];
  private readonly nWorkers: number;
  private startedEmitted = false;
  /** Fits per base sim (`n_scenarios × grid_points`); derived from the first
   *  `started`. 1 until known — safe because each worker's `started` precedes
   *  all of its progress signals. */
  private fitsPerSim = 1;
  /** scenario idx → workers that have emitted scenario_started for it. */
  private readonly scenarioStarted = new Set<number>();
  /** scenario idx → count of workers that have emitted scenario_completed. */
  private readonly scenarioCompletedCount = new Map<number, number>();
  private readonly scenarioCompletedEmitted = new Set<number>();
  /** "idx:n" → count of workers that have reported that N-point. */
  private readonly nPointCount = new Map<string, number>();
  private readonly nPointEmitted = new Set<string>();
  /** Per-worker open scenario idx, so n_point_completed keys by the right idx
   *  even if workers interleave (each worker advances its own scenario in order). */
  private readonly workerScenario: number[];

  constructor(
    /** Pool-wide base sim count (Σ shares) — multiplied by `fitsPerSim` once known. */
    private readonly baseSims: number,
    private readonly shares: number[],
    private readonly n: number,
    private readonly sink: (e: ProgressEvent) => void,
  ) {
    this.perWorker = shares.map(() => 0);
    this.workerScenario = shares.map(() => 0);
    this.nWorkers = shares.length;
  }

  /** A full per-worker engine event (forwarded verbatim by the worker). */
  workerEvent(worker: number, ev: ProgressEvent): void {
    switch (ev.kind) {
      case 'started': {
        // Recover fits-per-sim from this worker's share-scoped total (exact:
        // the engine computed total_sims = share × n_scenarios × grid_points).
        const share = this.shares[worker]!;
        if (share > 0) this.fitsPerSim = ev.total_sims / share;
        if (!this.startedEmitted) {
          this.startedEmitted = true;
          // Override the share-scoped total_sims with the pool total; keep the
          // worker's real scenario/grid counts (each worker sees the full slice).
          this.sink({
            kind: 'started',
            total_sims: this.baseSims * this.fitsPerSim,
            total_scenarios: ev.total_scenarios,
            total_grid_points: ev.total_grid_points,
          });
        }
        break;
      }
      case 'scenario_started':
        this.workerScenario[worker] = ev.idx;
        if (!this.scenarioStarted.has(ev.idx)) {
          this.scenarioStarted.add(ev.idx);
          this.sink(ev);
        }
        break;
      case 'n_point_completed': {
        const idx = this.workerScenario[worker]!;
        const key = `${idx}:${ev.n}`;
        const seen = (this.nPointCount.get(key) ?? 0) + 1;
        this.nPointCount.set(key, seen);
        // Barrier: emit once every worker has crossed this N-point.
        // n_point_completed carries per-scenario power arrays = the in-progress
        // worker's partial values, ignored by the UI (only the post-merge
        // ScenarioResult is rendered).
        if (seen === this.nWorkers && !this.nPointEmitted.has(key)) {
          this.nPointEmitted.add(key);
          this.sink(ev);
        }
        break;
      }
      case 'scenario_completed': {
        const seen = (this.scenarioCompletedCount.get(ev.idx) ?? 0) + 1;
        this.scenarioCompletedCount.set(ev.idx, seen);
        if (seen === this.nWorkers && !this.scenarioCompletedEmitted.has(ev.idx)) {
          this.scenarioCompletedEmitted.add(ev.idx);
          this.sink(ev);
        }
        break;
      }
      // `completed` is driven by `finish` (after merge); per-worker `completed`
      // and `cancelled` are absorbed here.
      default:
        break;
    }
  }

  /**
   * A worker reported `completed` fits so far (cumulative across its share).
   * The value is clamped to share × fitsPerSim — the engine can emit a final
   * tick that slightly exceeds the computed ceiling due to rounding in the
   * share-to-total scaling; the clamp keeps the pool sum monotone and bounded.
   */
  workerProgress(worker: number, completed: number): void {
    this.perWorker[worker] = Math.min(completed, this.shares[worker]! * this.fitsPerSim);
    this.emitSum();
  }

  /** A worker finished its whole share. */
  workerDone(worker: number): void {
    this.perWorker[worker] = this.shares[worker]! * this.fitsPerSim;
    this.emitSum();
  }

  finish(): void {
    this.sink({ kind: 'completed' });
  }

  private emitSum(): void {
    const completed = this.perWorker.reduce((a, b) => a + b, 0);
    this.sink({
      kind: 'sims_completed',
      n: this.n,
      completed,
      total: this.baseSims * this.fitsPerSim,
    });
  }
}
