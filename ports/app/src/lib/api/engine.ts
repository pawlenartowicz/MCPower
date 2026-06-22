// Tauri engine API bridge: routes find_power/find_sample_size/cancel/progress calls to the Tauri backend, or to the E2E mock when VITE_E2E=true.
import type { AppSpec, EffectDescriptor, EffectSize, EffectsFromData } from '$lib/domain/app-spec';
export type { EffectDescriptor, EffectSize, EffectsFromData };
import type {
  PowerResult,
  SampleSizeResult,
  SampleSizeMethod,
  ScenarioResult,
  ProgressEvent,
  PlotSpecs,
  FormulaParse,
} from '$lib/domain/result';

import * as mock from './engine-e2e-mock';

// Vite statically replaces this, so the E2E branch compiles out of the real build.
const E2E = import.meta.env.VITE_E2E === 'true';

let _invoke: typeof import('@tauri-apps/api/core').invoke | null = null;
let _listen: typeof import('@tauri-apps/api/event').listen | null = null;

async function api() {
  if (!_invoke) ({ invoke: _invoke } = await import('@tauri-apps/api/core'));
  if (!_listen) ({ listen: _listen } = await import('@tauri-apps/api/event'));
  return { invoke: _invoke!, listen: _listen! };
}

export interface FindPowerResponse {
  run_id: string;
  result: ScenarioResult<PowerResult>;
  plots: PlotSpecs;
}
export interface FindSampleSizeResponse {
  run_id: string;
  result: ScenarioResult<SampleSizeResult>;
  plots: PlotSpecs;
}

export async function findPower(spec: AppSpec, sampleSize: number): Promise<FindPowerResponse> {
  if (E2E) return mock.findPower(spec, sampleSize);
  const { invoke } = await api();
  return invoke<FindPowerResponse>('find_power_cmd', { spec, sampleSize });
}

export async function findSampleSize(
  spec: AppSpec,
  bounds: [number, number],
  method: SampleSizeMethod,
): Promise<FindSampleSizeResponse> {
  if (E2E) return mock.findSampleSize(spec, bounds, method);
  const { invoke } = await api();
  return invoke<FindSampleSizeResponse>('find_sample_size_cmd', { spec, bounds, method });
}

/**
 * Cancel an active run by its run_id (the opaque string returned by find_power / find_sample_size).
 * Returns `true` if the cancellation was issued; `false` if there is no active run with that id (not an error).
 */
export async function cancelRun(runId: string): Promise<boolean> {
  if (E2E) return mock.cancelRun(runId);
  const { invoke } = await api();
  return invoke<boolean>('cancel_run_cmd', { runId });
}

export async function onProgress(listener: (e: ProgressEvent) => void): Promise<() => void> {
  if (E2E) return mock.onProgress(listener);
  const { listen } = await api();
  const unlisten = await listen<ProgressEvent>('progress', ({ payload }) => listener(payload));
  return unlisten;
}

/**
 * Parse `formula` and return the structured `FormulaParse` tree (outcome, predictors,
 * interaction_terms). Throws on a syntactically invalid formula.
 */
export async function parseFormula(formula: string): Promise<FormulaParse> {
  if (E2E) return mock.parseFormula(formula);
  const { invoke } = await api();
  return invoke<FormulaParse>('parse_formula_cmd', { formula });
}

/**
 * Recover standardized effect sizes from the uploaded data in `spec.csv`, plus
 * the estimated cluster ICC (mixed, converged fit) and binary-outcome baseline
 * probability for the Apply flow. The fitted estimator follows the spec family
 * (OLS/GLM/MLE).
 */
export async function getEffectsFromData(spec: AppSpec): Promise<EffectsFromData> {
  const { invoke } = await api();
  return invoke<EffectsFromData>('get_effects_from_data_cmd', { spec });
}

/**
 * The engine's index-only effect skeleton (β-column aligned) for `spec`. Render
 * result names from this + the factor-label store: a run's `target_indices`
 * index into the returned array directly (index 0 = intercept). Mirrors the
 * `effectSkeleton` export of the WASM port and `effect_skeleton_json` in the Rust crate.
 */
export async function effectSkeleton(spec: AppSpec): Promise<EffectDescriptor[]> {
  if (E2E) return mock.effectSkeleton(spec);
  const { invoke } = await api();
  return invoke<EffectDescriptor[]>('effect_skeleton_cmd', { spec });
}

/**
 * Configure the rayon thread pool. Must be called before the first find_power /
 * find_sample_size invocation; a second call (pool already initialised) returns
 * an error. n = null is a no-op (callers guard before invoking).
 * Tauri-only — the WASM shell uses single-core workers by design.
 */
export async function setNThreads(n: number): Promise<void> {
  const { invoke } = await api();
  return invoke<void>('set_n_threads_cmd', { n });
}
