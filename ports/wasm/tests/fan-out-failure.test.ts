// fanOut failure paths: a worker {kind:'error'} message, a transport-level
// onerror, and onmessageerror each reject the run with the right error, terminate every
// worker of the run, drop the run from the cancel registry, and recycle a borrowed warm
// worker instead of returning it to the pool.
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../vendor/engine-wasm/engine_wasm.js', () => ({
  default: vi.fn().mockResolvedValue(undefined),
  merge_power_results: vi.fn(),
  power_plot_specs_json: vi.fn(),
  merge_sample_size_results: vi.fn(),
  sample_size_plot_specs_json: vi.fn(),
  curvature_hints_power: vi.fn().mockReturnValue('[[]]'),
  curvature_hints_sample_size: vi.fn().mockReturnValue('[[]]'),
  parse_formula: vi.fn(),
  get_effects_from_data: vi.fn(),
  effect_skeleton: vi.fn(),
}));

import type { AppSpec } from '../src/types';

const SPEC = { family: 'linear', target_power: 0.8, n_sims: 4, seed: 2137, correction: 'none' } as unknown as AppSpec;

class FakeWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  onmessageerror: ((e: MessageEvent) => void) | null = null;
  terminated = false;
  posted = 0;
  constructor() { instances.push(this); }
  postMessage() { this.posted++; }
  terminate() { this.terminated = true; }
}

let instances: FakeWorker[] = [];

beforeEach(() => {
  vi.resetModules();
  instances = [];
  (globalThis as Record<string, unknown>).Worker = FakeWorker;
  Object.defineProperty(globalThis, 'navigator', { value: { hardwareConcurrency: 4 }, configurable: true, writable: true });
  Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: () => 'run-1' }, configurable: true, writable: true });
});

/** Start a run and wait until every run worker has been sent its share. */
async function startRun(workers: number | undefined) {
  const mod = await import('../src/index');
  const run = mod.findPower(SPEC, 10, workers === undefined ? undefined : { workers });
  const settled = run.then(() => 'resolved', (e: Error) => e);
  for (let i = 0; i < 20 && !instances.some((w) => w.posted > 0); i++) await new Promise((r) => setTimeout(r, 0));
  return { mod, settled };
}

const FAILURES = [
  {
    name: "{kind:'error'} message",
    trigger: (w: FakeWorker) => w.onmessage!({ data: { kind: 'error', message: 'boom in worker' } } as MessageEvent),
    message: 'boom in worker',
  },
  {
    name: 'onerror',
    trigger: (w: FakeWorker) => w.onerror!({ message: 'script load failed' } as ErrorEvent),
    message: 'worker error: script load failed',
  },
  {
    name: 'onerror without a message',
    trigger: (w: FakeWorker) => w.onerror!({ message: '' } as ErrorEvent),
    message: 'worker error: failed to load or initialize',
  },
  {
    name: 'onmessageerror',
    trigger: (w: FakeWorker) => w.onmessageerror!({} as MessageEvent),
    message: 'worker message deserialization failed (structured clone)',
  },
];

describe.each(FAILURES)('fanOut — $name', ({ trigger, message }) => {
  it('rejects, terminates every run worker, leaves the idle warm worker alone, and clears the run', async () => {
    const { mod, settled } = await startRun(2);
    // instances[0] is the warm worker (not borrowed by a forced multi-worker run).
    const [warm, w0, w1] = instances;
    expect(instances).toHaveLength(3);

    trigger(w1!);

    const err = (await settled) as Error;
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toBe(message);
    expect(w0!.terminated).toBe(true);
    expect(w1!.terminated).toBe(true);
    expect(warm!.terminated).toBe(false);
    expect(await mod.cancelRun('run-1')).toBe(false); // activeRuns entry deleted
  });

  it('recycles a borrowed warm worker: terminated, and a replacement is spawned', async () => {
    const { mod, settled } = await startRun(undefined); // routed single → borrows warm
    expect(instances).toHaveLength(1);
    const warm = instances[0]!;

    trigger(warm);

    expect(((await settled) as Error).message).toBe(message);
    expect(warm.terminated).toBe(true);
    expect(instances).toHaveLength(2); // respawned warm worker
    expect(instances[1]!.terminated).toBe(false);
    expect(await mod.cancelRun('run-1')).toBe(false);
  });
});
