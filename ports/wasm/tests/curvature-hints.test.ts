// Shared curvature hints: with several workers the main thread computes the
// hints once (at the master seed) and posts the same opaque string to every
// worker; with one worker it computes none and the worker runs its own pilot.
// Mocks the wasm module (no wasm build) + stubs Worker to capture posted messages.
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../vendor/engine-wasm/engine_wasm.js', () => ({
  default: vi.fn().mockResolvedValue(undefined), // init()
  merge_power_results: vi.fn().mockReturnValue('{"scenarios":[["s",{}]]}'),
  power_plot_specs_json: vi.fn().mockReturnValue('{"blocks":[]}'),
  merge_sample_size_results: vi.fn().mockReturnValue('{"scenarios":[["s",{}]]}'),
  sample_size_plot_specs_json: vi.fn().mockReturnValue('{"blocks":[]}'),
  curvature_hints_power: vi.fn().mockReturnValue('[[]]'),
  curvature_hints_sample_size: vi.fn().mockReturnValue('[[]]'),
  parse_formula: vi.fn(),
  get_effects_from_data: vi.fn(),
  effect_skeleton: vi.fn(),
}));

import type { AppSpec, SampleSizeMethod } from '../src/types';

const SPEC = {
  family: 'mixed', target_power: 0.8, n_sims: 6, seed: 2137, correction: 'none',
} as unknown as AppSpec;
const METHOD = { Grid: { by: { Fixed: 10 }, mode: 'Linear' } } as SampleSizeMethod;

let posted: unknown[] = [];

class FakeWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  constructor() {}
  postMessage(msg: unknown) {
    posted.push(msg);
    Promise.resolve().then(() => this.onmessage?.({ data: { kind: 'part', part: '{}' } } as MessageEvent));
  }
  terminate() {}
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks(); // the mocked module is shared across tests: reset its call counts
  posted = [];
  (globalThis as Record<string, unknown>).Worker = FakeWorker;
  Object.defineProperty(globalThis, 'navigator', { value: { hardwareConcurrency: 4 }, configurable: true, writable: true });
  Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: () => 'run-1' }, configurable: true, writable: true });
});

const hintsOf = (m: unknown) => (m as { hints?: string }).hints;

describe('shared curvature hints', () => {
  it('findPower with workers: 2 sends the same hints string to both workers', async () => {
    const mod = await import('../src/index');
    const wasm = await import('../vendor/engine-wasm/engine_wasm.js');
    await mod.findPower(SPEC, 100, { workers: 2 });

    expect(posted.length).toBe(2);
    for (const m of posted) expect(hintsOf(m)).toBe('[[]]');
    expect(wasm.curvature_hints_power).toHaveBeenCalledTimes(1);
    expect(wasm.curvature_hints_power).toHaveBeenCalledWith(JSON.stringify(SPEC), 100, 2137n);
  });

  it('findPower with workers: 1 sends no hints', async () => {
    const mod = await import('../src/index');
    const wasm = await import('../vendor/engine-wasm/engine_wasm.js');
    await mod.findPower(SPEC, 100, { workers: 1 });

    expect(posted.length).toBe(1);
    expect(hintsOf(posted[0])).toBeUndefined();
    expect(wasm.curvature_hints_power).not.toHaveBeenCalled();
  });

  it('findSampleSize with workers: 2 sends the same hints string to both workers', async () => {
    const mod = await import('../src/index');
    const wasm = await import('../vendor/engine-wasm/engine_wasm.js');
    await mod.findSampleSize(SPEC, [50, 100], METHOD, { workers: 2 });

    expect(posted.length).toBe(2);
    for (const m of posted) expect(hintsOf(m)).toBe('[[]]');
    expect(wasm.curvature_hints_sample_size).toHaveBeenCalledTimes(1);
    expect(wasm.curvature_hints_sample_size).toHaveBeenCalledWith(
      JSON.stringify(SPEC), JSON.stringify([50, 100]), JSON.stringify(METHOD), 2137n,
    );
  });

  it('findSampleSize with workers: 1 sends no hints', async () => {
    const mod = await import('../src/index');
    const wasm = await import('../vendor/engine-wasm/engine_wasm.js');
    await mod.findSampleSize(SPEC, [50, 100], METHOD, { workers: 1 });

    expect(posted.length).toBe(1);
    expect(hintsOf(posted[0])).toBeUndefined();
    expect(wasm.curvature_hints_sample_size).not.toHaveBeenCalled();
  });
});
