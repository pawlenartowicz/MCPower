// wald_se pass-through: findPower forwards the spec's wald_se value to every
// worker verbatim and never injects a wald_se_denom (the asymp calibrate-once
// machinery was removed; the engine computes the per-fit Hessian SE inline).
// Mocks the wasm module (no wasm build) + stubs Worker to capture posted specs.
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../vendor/engine-wasm/engine_wasm.js', () => ({
  default: vi.fn().mockResolvedValue(undefined), // init()
  merge_power_results: vi.fn().mockReturnValue('{"scenarios":[["s",{}]]}'),
  power_plot_specs_json: vi.fn().mockReturnValue('{"blocks":[]}'),
  merge_sample_size_results: vi.fn(),
  sample_size_plot_specs_json: vi.fn(),
  parse_formula: vi.fn(),
  get_effects_from_data: vi.fn(),
  effect_skeleton: vi.fn(),
}));

import type { AppSpec } from '../src/types';

function makeSpec(waldSe: 'hessian' | 'rx'): AppSpec {
  return {
    family: 'mixed', target_power: 0.8, n_sims: 6, seed: 2137,
    correction: 'none', wald_se: waldSe,
  } as unknown as AppSpec;
}

let posted: unknown[] = [];

class FakeWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  terminated = false;
  constructor() {}
  postMessage(msg: unknown) {
    posted.push(msg);
    Promise.resolve().then(() => this.onmessage?.({ data: { kind: 'part', part: '{}' } } as MessageEvent));
  }
  terminate() { this.terminated = true; }
}

beforeEach(() => {
  vi.resetModules();
  posted = [];
  (globalThis as Record<string, unknown>).Worker = FakeWorker;
  Object.defineProperty(globalThis, 'navigator', { value: { hardwareConcurrency: 4 }, configurable: true, writable: true });
  Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: () => 'run-1' }, configurable: true, writable: true });
});

describe('wald_se pass-through (no denom injection)', () => {
  for (const mode of ['hessian', 'rx'] as const) {
    it(`forwards wald_se='${mode}' to every worker and injects no wald_se_denom`, async () => {
      const mod = await import('../src/index');
      await mod.findPower(makeSpec(mode), 100, { workers: 3 });

      expect(posted.length).toBe(3);
      for (const m of posted) {
        const spec = JSON.parse((m as { spec: string }).spec) as {
          wald_se?: unknown;
          wald_se_denom?: unknown;
        };
        expect(spec.wald_se).toBe(mode);
        expect(spec.wald_se_denom).toBeUndefined();
      }
    });
  }
});
