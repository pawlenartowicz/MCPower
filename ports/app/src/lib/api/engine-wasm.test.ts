// Gate-wiring tests for the browser engine seam: spec.csv.n_rows must be
// forwarded as opts.uploadRows so the wasm pool's validateUploadRows gate
// (ports/wasm/src/upload.ts) fires before any worker spawns. The Tauri seam
// (engine.ts) must NOT gain the gate — native ports never enforce max_rows
// host-side by design. The error-surfacing half of the chain (engine rejection
// → runStore.lastError → RunErrorCard) is covered in run.svelte.test.ts.
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@mcpower/engine-wasm', () => ({
  findPower: vi.fn(),
  findSampleSize: vi.fn(),
  cancelRun: vi.fn(),
  onProgress: vi.fn(),
  parseFormula: vi.fn(),
  getEffectsFromData: vi.fn(),
  effectSkeleton: vi.fn(),
}));
vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));
vi.mock('@tauri-apps/api/event', () => ({ listen: vi.fn() }));

import { findPower as poolFindPower, findSampleSize as poolFindSampleSize } from '@mcpower/engine-wasm';
import { invoke } from '@tauri-apps/api/core';
import { findPower as wasmFindPower, findSampleSize as wasmFindSampleSize } from './engine-wasm';
import { findPower as tauriFindPower } from './engine';
import type { AppSpec } from '$lib/domain/app-spec';
import type { SampleSizeMethod } from '$lib/domain/result';

const baseSpec: AppSpec = {
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

const csvSpec: AppSpec = {
  ...baseSpec,
  csv: { mode: 'partial', n_rows: 20000, columns: [] },
};

const method: SampleSizeMethod = { Grid: { by: { Fixed: 10 }, mode: 'Linear' } };

const okResponse = { run_id: 'r1', result: { scenarios: [] }, plots: { blocks: [] }, warnings: [] };

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(poolFindPower).mockResolvedValue(okResponse as never);
  vi.mocked(poolFindSampleSize).mockResolvedValue(okResponse as never);
  vi.mocked(invoke).mockResolvedValue(okResponse as never);
});

describe('engine-wasm upload row gate wiring', () => {
  it('findPower forwards spec.csv.n_rows as opts.uploadRows', async () => {
    await wasmFindPower(csvSpec, 80);
    expect(poolFindPower).toHaveBeenCalledWith(csvSpec, 80, { uploadRows: 20000 });
  });

  it('findPower without uploaded data passes no opts (gate skipped)', async () => {
    await wasmFindPower(baseSpec, 80);
    expect(poolFindPower).toHaveBeenCalledWith(baseSpec, 80, undefined);
  });

  it('findSampleSize forwards spec.csv.n_rows as opts.uploadRows', async () => {
    await wasmFindSampleSize(csvSpec, [30, 200], method);
    expect(poolFindSampleSize).toHaveBeenCalledWith(csvSpec, [30, 200], method, { uploadRows: 20000 });
  });

  it('findSampleSize without uploaded data passes no opts (gate skipped)', async () => {
    await wasmFindSampleSize(baseSpec, [30, 200], method);
    expect(poolFindSampleSize).toHaveBeenCalledWith(baseSpec, [30, 200], method, undefined);
  });

  it("propagates the pool's UploadRejected so runStore surfaces it as lastError", async () => {
    vi.mocked(poolFindPower).mockRejectedValueOnce(
      new Error('Upload has 20000 rows; the browser limit is 10000.'),
    );
    await expect(wasmFindPower(csvSpec, 80)).rejects.toThrow('browser limit');
  });
});

describe('Tauri seam stays ungated (native ports never enforce max_rows host-side)', () => {
  it('findPower passes an oversized upload straight to invoke without a row gate', async () => {
    await expect(tauriFindPower(csvSpec, 80)).resolves.toBeDefined();
    expect(invoke).toHaveBeenCalledWith('find_power_cmd', { spec: csvSpec, sampleSize: 80 });
  });
});
