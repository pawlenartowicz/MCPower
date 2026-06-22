// RunAdapter interface + startRun entry point for the DEMO run path (StatusBar's
// demo button) — always the fake adapter. Real analysis runs go through
// `runStore` → `$lib/api/engine` (Tauri) / `engine-wasm` (browser), not here.
import type { DemoPowerResult } from '$lib/domain/result';
import { demoStore } from '$lib/stores/demo.svelte';

export type RunKind = 'find-power' | 'find-sample-size';

export interface ProgressEvent {
  completed: number;
  total: number;
}

export interface RunCallbacks {
  onProgress: (p: ProgressEvent) => void;
  onDone: (result: DemoPowerResult) => void;
  onError?: (err: Error) => void;
}

export interface RunHandle {
  cancel: () => void;
}

export interface RunAdapter {
  start(kind: RunKind, cb: RunCallbacks): RunHandle;
}

let adapterPromise: Promise<RunAdapter> | null = null;

function loadAdapter(): Promise<RunAdapter> {
  if (adapterPromise) return adapterPromise;
  adapterPromise = import('./run-fake').then((m) => m.fakeAdapter);
  return adapterPromise;
}

export function startRun(kind: RunKind): RunHandle {
  demoStore.runState = 'running';
  demoStore.progress.completed = 0;
  demoStore.progress.total = 0;

  let cancel: () => void = () => {};
  let canceled = false;

  loadAdapter().then((adapter) => {
    if (canceled) return;
    const handle = adapter.start(kind, {
      onProgress: (p) => {
        demoStore.progress.completed = p.completed;
        demoStore.progress.total = p.total;
      },
      onDone: (result) => {
        demoStore.runState = 'done';
        demoStore.pushTab({
          label:
            kind === 'find-power'
              ? `Run ${demoStore.runTabs.length + 1}`
              : `n-search ${demoStore.runTabs.length + 1}`,
          kind,
          subView: 'bars',
          result,
        });
      },
      onError: (err) => {
        console.error('run failed', err);
        demoStore.runState = 'idle';
      },
    });
    cancel = handle.cancel;
  });

  return {
    cancel: () => {
      canceled = true;
      cancel();
      if (demoStore.runState === 'running') demoStore.runState = 'idle';
    },
  };
}
