// Warm-1 worker lifecycle: a persistent worker spawned at first API touch,
// borrowed by routed-single runs (no spawn/terminate per run), respawned after a
// cancel kills it. We count Worker constructions through a stub.
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

// A routed-single spec (simple case, small N + n_sims, no scenarios → 1 worker).
const SPEC = { family: 'linear', target_power: 0.8, n_sims: 4, seed: 2137, correction: 'none' } as unknown as AppSpec;

let constructed = 0;
let autoReply = true;
let instances: FakeWorker[] = [];

class FakeWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  terminated = false;
  constructor() { constructed++; instances.push(this); }
  postMessage() {
    if (!autoReply) return; // leave the run in-flight (cancel test)
    Promise.resolve().then(() => this.onmessage?.({ data: { kind: 'part', part: '{}' } } as MessageEvent));
  }
  terminate() { this.terminated = true; }
}

beforeEach(() => {
  vi.resetModules(); // fresh index.ts → warmWorker resets to null
  constructed = 0;
  autoReply = true;
  instances = [];
  (globalThis as Record<string, unknown>).Worker = FakeWorker;
  Object.defineProperty(globalThis, 'navigator', { value: { hardwareConcurrency: 1 }, configurable: true, writable: true });
  Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: () => 'run-1' }, configurable: true, writable: true });
});

describe('warm-1 worker', () => {
  it('borrows the warm worker across two sequential routed-single runs (one construction, never terminated)', async () => {
    const mod = await import('../src/index');
    await mod.findPower(SPEC, 10);
    await mod.findPower(SPEC, 10);
    expect(constructed).toBe(1);          // spawned once at first ensureMain, reused
    expect(instances[0]!.terminated).toBe(false); // freed, not terminated, between runs
  });

  it('respawns the warm worker after a cancel terminates it', async () => {
    const mod = await import('../src/index');
    const seen: string[] = [];
    await mod.onProgress((e) => { if (e.kind === 'run_started') seen.push(e.run_id); });

    autoReply = false; // run stays in-flight so we can cancel it mid-flight
    void mod.findPower(SPEC, 10).catch(() => {}); // never settles after terminate
    for (let i = 0; i < 10 && seen.length === 0; i++) await new Promise((r) => setTimeout(r, 0));

    expect(constructed).toBe(1);
    expect(await mod.cancelRun(seen[0]!)).toBe(true);
    expect(instances[0]!.terminated).toBe(true); // warm worker killed by cancel
    expect(constructed).toBe(2);                 // replacement respawned in the background
  });
});
