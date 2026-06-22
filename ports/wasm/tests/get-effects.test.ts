// Tests for getEffectsFromData (G1) and csv-driven findPower (G2).
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

// Minimal Linear spec (no csv) used as a base for both G1 and G2 tests.
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
    vi.mocked(wasmMod.default).mockResolvedValue(undefined);
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

  it('preserves null cluster_icc / baseline_probability through the JSON round-trip', async () => {
    // The regression/Gaussian preview: both scalars null. They must come back
    // as null (not undefined / dropped) so the host's Apply flow skips them.
    const mockPreview: EffectsFromData = {
      effects: [{ name: 'x', value: 0.9 }],
      cluster_icc: null,
      baseline_probability: null,
    };
    vi.mocked(wasmMod.get_effects_from_data).mockReturnValue(JSON.stringify(mockPreview));

    const result = await getEffectsFromData(BASE_SPEC);
    expect(result.cluster_icc).toBeNull();
    expect(result.baseline_probability).toBeNull();
  });

  it('propagates wasm errors as JS errors', async () => {
    vi.mocked(wasmMod.get_effects_from_data).mockImplementation(() => {
      throw new Error('get_effects_from_data: no csv data attached to this spec');
    });

    await expect(getEffectsFromData(BASE_SPEC)).rejects.toThrow('no csv data');
  });
});

// ── G2: csv passes through findPower ─────────────────────────────────────────

describe('findPower csv passthrough (G2)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(wasmMod.default).mockResolvedValue(undefined);
  });

  function makeWorkerMock(partJson: string) {
    // Minimal Worker stub: immediately posts a part message on postMessage.
    const listeners: Record<string, ((e: MessageEvent) => void)> = {};
    return {
      onmessage: null as ((e: MessageEvent) => void) | null,
      set onmessage(fn: ((e: MessageEvent) => void) | null) {
        if (fn) listeners['message'] = fn;
      },
      postMessage(msg: unknown) {
        // Yield so onmessage is set first, then reply synchronously.
        Promise.resolve().then(() => {
          if (listeners['message']) {
            listeners['message']({ data: { kind: 'part', part: partJson } } as MessageEvent);
          }
        });
      },
      terminate: vi.fn(),
    };
  }

  it('serializes csv field into the spec JSON sent to workers', async () => {
    // Capture the spec JSON passed to the worker.
    const capturedSpecs: string[] = [];

    // Stub Worker globally so fanOut's new Worker() works in Node.
    (globalThis as Record<string, unknown>).Worker = class {
      onmessage: ((e: MessageEvent) => void) | null = null;
      postMessage(msg: Record<string, unknown>) {
        capturedSpecs.push(msg['spec'] as string);
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
    const partResult = JSON.stringify({ scenarios: [['s1', {}]] });
    vi.mocked(wasmMod.merge_power_results).mockReturnValue(partResult);
    vi.mocked(wasmMod.power_plot_specs_json).mockReturnValue(
      JSON.stringify({ blocks: [] }),
    );

    const csvData: CsvData = {
      mode: 'partial',
      n_rows: 2,
      columns: [
        { name: 'x', col_type: 'continuous', values: [0.5, -0.5], labels: [] },
        { name: 'y', col_type: 'continuous', values: [1.0, -1.0], labels: [] },
      ],
    };

    const specWithCsv: AppSpec = { ...BASE_SPEC, csv: csvData };
    const specNoCsv: AppSpec = { ...BASE_SPEC, csv: null };

    // Fire both calls; we only need to see the serialized spec.
    await findPower(specWithCsv, 10);
    await findPower(specNoCsv, 10);

    expect(capturedSpecs).toHaveLength(2);

    const parsedWith = JSON.parse(capturedSpecs[0]!) as { csv: unknown };
    const parsedWithout = JSON.parse(capturedSpecs[1]!) as { csv: unknown };

    // csv-present spec carries the csv block.
    expect(parsedWith.csv).not.toBeNull();
    expect((parsedWith.csv as CsvData).mode).toBe('partial');

    // csv-absent spec does not.
    expect(parsedWithout.csv).toBeNull();

    // correction: 'none' in BASE_SPEC → corrected=false reaches the plot emitter (D6).
    expect(wasmMod.power_plot_specs_json).toHaveBeenCalledWith(
      expect.anything(), expect.anything(), false);

    // Cleanup globals.
    delete (globalThis as Record<string, unknown>).Worker;
    Object.defineProperty(globalThis, 'navigator', { value: undefined, configurable: true, writable: true });
    Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true, writable: true });
  });
});
