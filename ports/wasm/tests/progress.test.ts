import { describe, it, expect } from 'vitest';
import { ProgressAggregator } from '../src/progress';
import type { ProgressEvent } from '../src/types';

function collect() {
  const events: ProgressEvent[] = [];
  return { events, sink: (e: ProgressEvent) => events.push(e) };
}

/** One worker's full single-core envelope for a find_power run. `single_core_find_power`
 *  emits no n_point (single N, reported via the merged result), only the scenario envelope. */
function powerEnvelope(share: number, totalScenarios: number): ProgressEvent[] {
  const evs: ProgressEvent[] = [
    { kind: 'started', total_sims: share * totalScenarios, total_scenarios: totalScenarios, total_grid_points: 0 },
  ];
  for (let idx = 0; idx < totalScenarios; idx++) {
    evs.push({ kind: 'scenario_started', label: `s${idx}`, idx, total: totalScenarios });
    evs.push({ kind: 'scenario_completed', label: `s${idx}`, idx });
  }
  evs.push({ kind: 'completed' });
  return evs;
}

/** One worker's full single-core envelope for a find_sample_size run: each
 *  scenario emits an n_point per grid point. Fit accounting: the engine's
 *  share-scoped total is share × scenarios × grid points. */
function sampleSizeEnvelope(share: number, totalScenarios: number, grid: number[]): ProgressEvent[] {
  const evs: ProgressEvent[] = [
    { kind: 'started', total_sims: share * totalScenarios * grid.length, total_scenarios: totalScenarios, total_grid_points: grid.length },
  ];
  for (let idx = 0; idx < totalScenarios; idx++) {
    evs.push({ kind: 'scenario_started', label: `s${idx}`, idx, total: totalScenarios });
    for (const n of grid) {
      evs.push({ kind: 'n_point_completed', n, power_uncorrected: [0.5], power_corrected: [0.5] });
    }
    evs.push({ kind: 'scenario_completed', label: `s${idx}`, idx });
  }
  evs.push({ kind: 'completed' });
  return evs;
}

describe('ProgressAggregator — native-semantics reconstruction', () => {
  it('emits started once with the pool total_sims and the real total_scenarios', () => {
    const { events, sink } = collect();
    const agg = new ProgressAggregator(/*totalSims*/ 1600, /*shares*/ [800, 800], 80, sink);
    const env = (s: number) => powerEnvelope(s, 2);
    // Worker 0 then worker 1 deliver their envelopes.
    env(800).forEach((e) => agg.workerEvent(0, e));
    env(800).forEach((e) => agg.workerEvent(1, e));
    agg.workerDone(0);
    agg.workerDone(1);
    agg.finish();

    const started = events.filter((e) => e.kind === 'started');
    expect(started).toHaveLength(1);
    // Fit accounting: 1600 base sims × 2 scenarios = 3200 fits.
    expect(started[0]).toMatchObject({ kind: 'started', total_sims: 3200, total_scenarios: 2 });
  });

  it('emits exactly one scenario_started/scenario_completed per scenario, in order', () => {
    const { events, sink } = collect();
    const agg = new ProgressAggregator(1600, [800, 800], 80, sink);
    // Interleave the two workers to prove dedup is worker-agnostic.
    const e0 = powerEnvelope(800, 2);
    const e1 = powerEnvelope(800, 2);
    const max = Math.max(e0.length, e1.length);
    for (let i = 0; i < max; i++) {
      if (e0[i]) agg.workerEvent(0, e0[i]!);
      if (e1[i]) agg.workerEvent(1, e1[i]!);
    }
    agg.finish();

    const starts = events.filter((e) => e.kind === 'scenario_started') as Extract<ProgressEvent, { kind: 'scenario_started' }>[];
    const dones = events.filter((e) => e.kind === 'scenario_completed') as Extract<ProgressEvent, { kind: 'scenario_completed' }>[];
    expect(starts.map((e) => e.idx)).toEqual([0, 1]);
    expect(dones.map((e) => e.idx)).toEqual([0, 1]);
  });

  it('emits n_point_completed only after ALL workers cross that N (barrier)', () => {
    const { events, sink } = collect();
    const agg = new ProgressAggregator(400, [200, 200], 30, sink);
    const grid = [30, 40, 50];
    const e0 = sampleSizeEnvelope(200, 1, grid);
    // Worker 0 streams its whole envelope first — no n_point should fire yet
    // (the barrier needs both workers).
    e0.forEach((e) => agg.workerEvent(0, e));
    expect(events.filter((e) => e.kind === 'n_point_completed')).toHaveLength(0);

    const e1 = sampleSizeEnvelope(200, 1, grid);
    e1.forEach((e) => agg.workerEvent(1, e));
    const nps = events.filter((e) => e.kind === 'n_point_completed') as Extract<ProgressEvent, { kind: 'n_point_completed' }>[];
    expect(nps.map((e) => e.n)).toEqual(grid); // one per grid point, in order, once
  });

  it('aggregates sims_completed monotonically and ends with a single completed', () => {
    const { events, sink } = collect();
    const agg = new ProgressAggregator(1600, [800, 800], 80, sink);
    powerEnvelope(800, 1).forEach((e) => agg.workerEvent(0, e));
    powerEnvelope(800, 1).forEach((e) => agg.workerEvent(1, e));
    agg.workerDone(0);
    agg.workerDone(1);
    agg.finish();

    const sims = events.filter((e) => e.kind === 'sims_completed') as Extract<ProgressEvent, { kind: 'sims_completed' }>[];
    const counts = sims.map((e) => e.completed);
    // Monotone non-decreasing, reaching the full total.
    for (let i = 1; i < counts.length; i++) expect(counts[i]!).toBeGreaterThanOrEqual(counts[i - 1]!);
    expect(counts.at(-1)).toBe(1600);

    expect(events.filter((e) => e.kind === 'completed')).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ kind: 'completed' });
  });

  it('derives the fit multiplier from started and scales totals, ticks, and clamps', () => {
    const { events, sink } = collect();
    // 400 base sims split 200/200; 1 scenario × 3 grid points → M = 3.
    const agg = new ProgressAggregator(400, [200, 200], 30, sink);
    agg.workerEvent(0, { kind: 'started', total_sims: 600, total_scenarios: 1, total_grid_points: 3 });
    expect(events.at(-1)).toMatchObject({ kind: 'started', total_sims: 1200 });

    // Cumulative mid-run tick from worker 0: 300 of its 600 fits.
    agg.workerProgress(0, 300);
    expect(events.at(-1)).toMatchObject({ kind: 'sims_completed', completed: 300, total: 1200 });

    // A tick beyond the worker's scaled share clamps to share × M.
    agg.workerProgress(0, 9999);
    expect(events.at(-1)).toMatchObject({ kind: 'sims_completed', completed: 600, total: 1200 });

    // Worker 1 finishes its whole share → pool total reached.
    agg.workerDone(1);
    expect(events.at(-1)).toMatchObject({ kind: 'sims_completed', completed: 1200, total: 1200 });
  });

  it('per-worker completed events are absorbed (no duplicate terminal)', () => {
    const { events, sink } = collect();
    const agg = new ProgressAggregator(800, [400, 400], 80, sink);
    powerEnvelope(400, 1).forEach((e) => agg.workerEvent(0, e)); // includes a per-worker `completed`
    powerEnvelope(400, 1).forEach((e) => agg.workerEvent(1, e));
    agg.finish();
    expect(events.filter((e) => e.kind === 'completed')).toHaveLength(1);
  });
});
