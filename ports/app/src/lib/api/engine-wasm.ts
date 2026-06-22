// Browser engine seam — drop-in for $lib/api/engine under VITE_TARGET=wasm.
// Delegates to the @mcpower/engine-wasm worker pool (resolved via a Vite alias
// to ports/wasm/src/index.ts). Exports the same surface as engine.ts so the
// alias swap is transparent to every caller.
import type { AppSpec, EffectDescriptor, EffectSize, EffectsFromData } from '$lib/domain/app-spec';
export type { EffectDescriptor, EffectSize, EffectsFromData };
import type { SampleSizeMethod, ProgressEvent, FormulaParse } from '$lib/domain/result';
import {
  findPower as poolFindPower,
  findSampleSize as poolFindSampleSize,
  cancelRun as poolCancel,
  onProgress as poolOnProgress,
  parseFormula as poolParseFormula,
  getEffectsFromData as poolGetEffectsFromData,
  effectSkeleton as poolEffectSkeleton,
} from '@mcpower/engine-wasm';
import type { FindPowerResponse, FindSampleSizeResponse } from './engine';

export async function findPower(spec: AppSpec, sampleSize: number): Promise<FindPowerResponse> {
  // Cast is safe: AnovaSpec has csv: null by type; the gate only runs when uploadRows is defined.
  const uploadRows = (spec as { csv?: { n_rows: number } | null }).csv?.n_rows;
  return poolFindPower(spec as never, sampleSize, uploadRows !== undefined ? { uploadRows } : undefined) as Promise<FindPowerResponse>;
}

export async function findSampleSize(
  spec: AppSpec,
  bounds: [number, number],
  method: SampleSizeMethod,
): Promise<FindSampleSizeResponse> {
  const uploadRows = (spec as { csv?: { n_rows: number } | null }).csv?.n_rows;
  return poolFindSampleSize(spec as never, bounds, method as never, uploadRows !== undefined ? { uploadRows } : undefined) as Promise<FindSampleSizeResponse>;
}

export async function cancelRun(runId: string): Promise<boolean> {
  return poolCancel(runId);
}

export async function onProgress(listener: (e: ProgressEvent) => void): Promise<() => void> {
  return poolOnProgress(listener as never);
}

export async function parseFormula(formula: string): Promise<FormulaParse> {
  return poolParseFormula(formula) as Promise<FormulaParse>;
}

export async function getEffectsFromData(spec: AppSpec): Promise<EffectsFromData> {
  return poolGetEffectsFromData(spec as never) as Promise<EffectsFromData>;
}

// Mirrors engine.ts effectSkeleton — see its JSDoc for the return contract.
export async function effectSkeleton(spec: AppSpec): Promise<EffectDescriptor[]> {
  return poolEffectSkeleton(spec as never) as Promise<EffectDescriptor[]>;
}

// WASM workers are single-core by design — thread-count control is n/a.
// Stub matches the Tauri engine.ts surface so shared code can import setNThreads
// without a VITE_TARGET branch; callers still guard with VITE_TARGET before calling.
export async function setNThreads(_n: number): Promise<void> {}
