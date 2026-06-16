import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('$lib/api/engine', () => ({
  findPower: vi.fn(),
  findSampleSize: vi.fn(),
  cancelRun: vi.fn(),
  onProgress: vi.fn(async () => () => {}),
  effectSkeleton: vi.fn(async () => []),
}));

import { findPower, findSampleSize } from '$lib/api/engine';
import { runStore } from './run.svelte';
import type { AppSpec } from '$lib/domain/app-spec';

const sampleSpec: AppSpec = {
  family: 'linear',
  parsed_formula: { outcome: 'y', predictors: ['x1'], interaction_terms: [] },
  var_types: [{ kind: 'numeric', name: 'x1' }],
  effects: [{ name: 'x1', value: 0.3 }],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 100,
  seed: 1,
  tests: { kind: 'all' },
  correction: 'none',
  scenarios: [],
  csv: null,
  report_overall: false,
  contrasts: [],
};

const sampleResult = {
  scenarios: [['s1', {
    n: 80, n_sims: 100, target_indices: [0],
    power_uncorrected: [0.5], power_corrected: [0.5],
    ci_uncorrected: [{ lo: 0.4, hi: 0.6 }],
    ci_corrected: [{ lo: 0.4, hi: 0.6 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
  }]],
};

describe('runStore.startFindPower', () => {
  beforeEach(() => {
    runStore.clearTabs();
    runStore.runState = 'idle';
    runStore.lastError = null;
    vi.clearAllMocks();
  });

  it('transitions idle → running → done and appends a tab', async () => {
    vi.mocked(findPower).mockResolvedValueOnce({ run_id: 'r1', result: sampleResult as any, plots: { blocks: [] } });
    expect(runStore.runState).toBe('idle');
    const p = runStore.startFindPower(sampleSpec, 80);
    expect(runStore.runState).toBe('running');
    await p;
    expect(runStore.runState).toBe('done');
    expect(runStore.runTabs.length).toBe(1);
    const tab = runStore.runTabs[0];
    expect(tab?.kind).toBe('find-power');
    expect(tab?.id).toBe('r1');
  });

  it('clears the previous run\'s progress synchronously at start (no stale bar)', async () => {
    // Simulate a finished prior run left at 100%.
    runStore.progress.completed = 4800;
    runStore.progress.total = 4800;
    vi.mocked(findPower).mockResolvedValueOnce({ run_id: 'r1', result: sampleResult as any, plots: { blocks: [] } });

    const p = runStore.startFindPower(sampleSpec, 80);
    // Reset must be synchronous — before the strip renders or any await resolves —
    // otherwise the bar inherits the prior 100% and animates down from it.
    expect(runStore.progress.completed).toBe(0);
    expect(runStore.progress.total).toBe(0);
    await p;
  });
});

describe('runStore run failure → lastError', () => {
  beforeEach(() => {
    runStore.clearTabs();
    runStore.runState = 'idle';
    runStore.lastError = null;
    vi.clearAllMocks();
  });

  it("a failed run sets runState=error and a 'run' lastError carrying the engine message", async () => {
    vi.mocked(findPower).mockRejectedValueOnce(new Error("factor 'group' has only one observed level"));
    await runStore.startFindPower(sampleSpec, 80);
    expect(runStore.runState).toBe('error');
    expect(runStore.lastError?.severity).toBe('run');
    expect(runStore.lastError?.message).toContain('only one observed level');
    // the full text is preserved for "Copy details" (lost to console.error before)
    expect(runStore.lastError?.detail).toBeTruthy();
  });

  it('clears a previous lastError when the next run starts', async () => {
    vi.mocked(findPower).mockRejectedValueOnce(new Error('boom'));
    await runStore.startFindPower(sampleSpec, 80);
    expect(runStore.lastError).not.toBeNull();

    vi.mocked(findPower).mockResolvedValueOnce({ run_id: 'r2', result: sampleResult as any, plots: { blocks: [] } });
    await runStore.startFindPower(sampleSpec, 80);
    expect(runStore.runState).toBe('done');
    expect(runStore.lastError).toBeNull();
  });
});

const sampleSizeResult = {
  scenarios: [['s1', {
    grid_or_trace: [{ n: 80, n_sims: 100, target_indices: [0], power_uncorrected: [0.82], power_corrected: [0.82], ci_uncorrected: [{ lo: 0.7, hi: 0.9 }], ci_corrected: [{ lo: 0.7, hi: 0.9 }], convergence_rate: 1.0, boundary_hit: [], estimator_extras: { estimator: 'ols' }, success_count_histogram_uncorrected: [0, 100], success_count_histogram_corrected: [0, 100] }],
    first_achieved: [80],
    first_joint_achieved: [80],
    target_power: 0.8,
    method: { Grid: { by: { Fixed: 10 }, mode: 'Linear' } },
  }]],
};

describe('runStore.startFindSampleSize', () => {
  beforeEach(() => {
    runStore.clearTabs();
    runStore.runState = 'idle';
    runStore.lastError = null;
    vi.clearAllMocks();
  });

  it('transitions idle → running → done and appends a find-sample-size tab', async () => {
    vi.mocked(findSampleSize).mockResolvedValueOnce({
      run_id: 'r2',
      result: sampleSizeResult as any,
      plots: { blocks: [] },
    });
    const p = runStore.startFindSampleSize(sampleSpec, [30, 200], { Grid: { by: { Fixed: 10 }, mode: 'Linear' } });
    expect(runStore.runState).toBe('running');
    await p;
    expect(runStore.runState).toBe('done');
    expect(runStore.runTabs.length).toBe(1);
    expect(runStore.runTabs[0]?.kind).toBe('find-sample-size');
  });

  it('runs find-sample-size in cluster-size mode (the engine grids N by the cluster size)', async () => {
    vi.mocked(findSampleSize).mockResolvedValueOnce({
      run_id: 'r3',
      result: sampleSizeResult as any,
      plots: { blocks: [] },
    });
    const mixedClusterSizeSpec: AppSpec = {
      ...sampleSpec,
      family: 'mixed',
      cluster_name: 'school',
      cluster_dim: { kind: 'cluster_size', value: 20 },
      icc: 0.1,
    } as unknown as AppSpec;
    await runStore.startFindSampleSize(mixedClusterSizeSpec, [30, 200], { Grid: { by: { Fixed: 10 }, mode: 'Linear' } });
    expect(runStore.runState).toBe('done');
    expect(vi.mocked(findSampleSize)).toHaveBeenCalledWith(mixedClusterSizeSpec, [30, 200], { Grid: { by: { Fixed: 10 }, mode: 'Linear' } });
  });
});
