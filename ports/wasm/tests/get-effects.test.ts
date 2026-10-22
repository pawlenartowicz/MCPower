// Tests for getEffectsFromData (G1) and findPower's `corrected` plot flag.
// The wasm module is not compiled in the test environment, so we mock the
// vendor module and assert on the TS wrapper behaviour only.
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the wasm vendor module before any imports that pull it in.
vi.mock('../vendor/engine-wasm/engine_wasm.js', () => ({
  default: vi.fn().mockResolvedValue(undefined), // init()
  get_effects_from_data: vi.fn(),
  merge_power_results: vi.fn(),
  power_plot_specs_json: vi.fn(),
  merge_sample_size_results: vi.fn(),
  sample_size_plot_specs_json: vi.fn(),
  parse_formula: vi.fn(),
}));

import * as wasmMod from '../vendor/engine-wasm/engine_wasm.js';
import { getEffectsFromData, findPower } from '../src/index';
import type { EffectsFromData, CsvData, AppSpec } from '../src/types';

// Minimal Linear spec (no csv) used as a base for both suites.
const BASE_SPEC: AppSpec = {
  family: 'linear',
  parsed_formula: { outcome: 'y', predictors: ['x'], interaction_terms: [] },
  var_types: [{ kind: 'numeric', name: 'x' }],
  effects: [{ name: 'x', value: 0.0 }],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 4,
  seed: 2137,
  tests: { kind: 'all' },
  correction: 'none',
  active_scenario: null,
  csv: null,
  report_overall: false,
  contrasts: [],
};

// ── G1: getEffectsFromData ────────────────────────────────────────────────────

describe('getEffectsFromData', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(wasmMod.default).mockResolvedValue({} as wasmMod.InitOutput);
  });

  it('calls wasm get_effects_from_data with the JSON-serialized spec and returns the parsed EffectsFromData object', async () => {
    // The shim returns the EffectsFromData object — effects plus the ICC and
    // baseline scalars — NOT a bare array. Here a mixed-binary-style preview:
    // an ICC number and a baseline number both survive the JSON round-trip.
    const mockPreview: EffectsFromData = {
      effects: [{ name: 'x', value: 0.42 }],
      cluster_icc: 0.27,
      baseline_probability: 0.31,
    };
    vi.mocked(wasmMod.get_effects_from_data).mockReturnValue(JSON.stringify(mockPreview));

    const specWithCsv: AppSpec = {
      ...BASE_SPEC,
      csv: {
        mode: 'strict',
        n_rows: 3,
        columns: [
          { name: 'x', col_type: 'continuous', values: [1.0, 2.0, 3.0], labels: [] },
          { name: 'y', col_type: 'continuous', values: [2.1, 4.0, 5.9], labels: [] },
        ],
      } as CsvData,
    };

    const result = await getEffectsFromData(specWithCsv);

    // Must have called wasm with the exact JSON we passed in.
    expect(wasmMod.get_effects_from_data).toHaveBeenCalledOnce();
    const calledArg = vi.mocked(wasmMod.get_effects_from_data).mock.calls[0]![0] as string;
    expect(JSON.parse(calledArg)).toMatchObject({ family: 'linear', csv: { mode: 'strict', n_rows: 3 } });

    // The whole object is parsed back; ICC and baseline survive as numbers.
    expect(result).toEqual(mockPreview);
    expect(result.effects).toEqual([{ name: 'x', value: 0.42 }]);
    expect(result.cluster_icc).toBe(0.27);
    expect(result.baseline_probability).toBe(0.31);
  });

  it('propagates wasm errors as JS errors', async () => {
    vi.mocked(wasmMod.get_effects_from_data).mockImplementation(() => {
      throw new Error('get_effects_from_data: no csv data attached to this spec');
    });

    await expect(getEffectsFromData(BASE_SPEC)).rejects.toThrow('no csv data');
  });
});

// ── findPower: the `corrected` flag reaching the plot emitter ────────────────
// Verbatim spec forwarding to the workers is pinned in wald_se.test.ts; the
// TS-side logic here is deriving `corrected` from spec.correction.

describe('findPower corrected flag', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(wasmMod.default).mockResolvedValue({} as wasmMod.InitOutput);
  });

  it("passes corrected = (correction !== 'none') to power_plot_specs_json", async () => {
    // Stub Worker globally so fanOut's new Worker() works in Node.
    (globalThis as Record<string, unknown>).Worker = class {
      onmessage: ((e: MessageEvent) => void) | null = null;
      postMessage() {
        Promise.resolve().then(() => {
          if (this.onmessage) {
            this.onmessage({ data: { kind: 'part', part: '{}' } } as MessageEvent);
          }
        });
      }
      terminate = vi.fn();
    };

    // Stub navigator for poolSize (read-only in Node — use defineProperty).
    Object.defineProperty(globalThis, 'navigator', {
      value: { hardwareConcurrency: 1 },
      configurable: true,
      writable: true,
    });
    // Stub crypto.randomUUID.
    Object.defineProperty(globalThis, 'crypto', {
      value: { randomUUID: () => 'test-uuid' },
      configurable: true,
      writable: true,
    });

    // merge + plot stubs.
    vi.mocked(wasmMod.merge_power_results).mockReturnValue(
      JSON.stringify({ scenarios: [['s1', {}]] }),
    );
    vi.mocked(wasmMod.power_plot_specs_json).mockReturnValue(JSON.stringify({ blocks: [] }));

    await findPower({ ...BASE_SPEC, correction: 'none' }, 10);
    await findPower({ ...BASE_SPEC, correction: 'holm' }, 10);

    const corrected = vi.mocked(wasmMod.power_plot_specs_json).mock.calls.map((c) => c[2]);
    expect(corrected).toEqual([false, true]);

    // Cleanup globals.
    delete (globalThis as Record<string, unknown>).Worker;
    Object.defineProperty(globalThis, 'navigator', { value: undefined, configurable: true, writable: true });
    Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true, writable: true });
  });
});
