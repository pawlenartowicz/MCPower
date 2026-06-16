// Module worker: init the wasm module once, run one worker's share, post
// progress events and the final part. The main thread (index.ts) owns merge,
// plots, and cancellation (worker.terminate()).
import init, { find_power, find_sample_size } from '../vendor/engine-wasm/engine_wasm.js';

type RunMsg =
  | { kind: 'power'; spec: string; sampleSize: number; nSims: number; seed: string }
  | { kind: 'sample_size'; spec: string; bounds: string; method: string; nSims: number; seed: string };

let ready: Promise<unknown> | null = null;
function ensureInit() {
  // wasm-pack --target web: init() fetches+instantiates the .wasm once per worker.
  return (ready ??= init());
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;

const onProgress = (json: string) => ctx.postMessage({ kind: 'progress', event: JSON.parse(json) });

ctx.onmessage = async (e: MessageEvent<RunMsg>) => {
  const msg = e.data;
  try {
    await ensureInit();
    let part: string;
    if (msg.kind === 'power') {
      part = find_power(msg.spec, msg.sampleSize, msg.nSims, BigInt(msg.seed), onProgress);
    } else {
      part = find_sample_size(msg.spec, msg.bounds, msg.method, msg.nSims, BigInt(msg.seed), onProgress);
    }
    ctx.postMessage({ kind: 'part', part });
  } catch (err) {
    ctx.postMessage({ kind: 'error', message: String(err) });
  }
};
