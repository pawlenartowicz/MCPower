// Wire types mirrored from ports/app/src/lib/domain/result.ts. Same shapes the
// Tauri path produces, so the worker-pool client is a drop-in for engine.ts.
export type ProgressEvent =
  | { kind: 'run_started'; run_id: string }
  | { kind: 'started'; total_sims: number; total_scenarios: number; total_grid_points: number }
  | { kind: 'scenario_started'; label: string; idx: number; total: number }
  | { kind: 'sims_completed'; n: number; completed: number; total: number }
  | { kind: 'n_point_completed'; n: number; power_uncorrected: number[]; power_corrected: number[] }
  | { kind: 'scenario_completed'; label: string; idx: number }
  | { kind: 'cancelled' }
  | { kind: 'completed' };

// ---- upload types (serde-identical to Rust; mirror ports/app/src/lib/domain/app-spec.ts) ----

/** Mirrors `engine_spec_builder::input::UploadMode`. */
export type UploadMode = 'none' | 'partial' | 'strict';

/** Mirrors `engine_spec_builder::input::UploadColumnType`. */
export type UploadColumnType = 'continuous' | 'binary' | 'factor';

/** Mirrors `engine_spec_builder::input::UploadColumn`. */
export interface UploadColumn {
  name: string;
  col_type: UploadColumnType;
  values: number[];
  /** Level labels for factor columns (code i → labels[i]). Empty for continuous/binary. */
  labels: string[];
}

/** Mirrors `engine_app_spec::CsvData`. */
export interface CsvData {
  mode: UploadMode;
  n_rows: number;
  columns: UploadColumn[];
}

/** Mirrors `engine_app_spec::EffectSize`. */
export interface EffectSize {
  name: string;
  value: number;
}

/** Mirrors `engine_app_spec::EffectsFromData` — the effect-recovery preview:
 *  fitted effects plus the estimated cluster ICC (mixed, converged fit only)
 *  and binary-outcome baseline probability. Both scalars are `null` when not
 *  applicable. snake_case keys match the Rust serde shape. */
export interface EffectsFromData {
  effects: EffectSize[];
  cluster_icc: number | null;
  baseline_probability: number | null;
}

/**
 * One element of the engine's index-only `EffectSkeleton`
 * (`engine_spec_builder::EffectDescriptor`), β-column aligned. A run's
 * `target_indices` index into the skeleton array directly (index 0 = intercept).
 * `level` is the 0-based index into the factor's full label list (reference
 * included); the consumer renders `factor[labels[level]]` from its own store.
 */
export type EffectDescriptor =
  | { kind: 'intercept' }
  | { kind: 'continuous'; predictor: string }
  | { kind: 'factor_level'; factor: string; level: number }
  | { kind: 'interaction'; components: EffectDescriptor[] };

// ---- spec / result types ----

// Opaque to the pool — passed through from the app, handed to the wasm shell as JSON.
export type AppSpec = Record<string, unknown> & { target_power: number };

/** A single named Vega-Lite plot block. `spec` is a theme-naked JSON string. */
export interface PlotBlock { key: string; spec: string }
/** Ordered plot blocks for a find_power or find_sample_size result. */
export type PlotSpecs = { blocks: PlotBlock[] };
export type ScenarioResult<T> = { scenarios: [string, T][] };

export interface FindPowerResponse {
  run_id: string;
  result: ScenarioResult<unknown>;
  plots: PlotSpecs;
  /** Strict-bootstrap diagnostics (strict mode only). Empty array otherwise. */
  warnings: string[];
}
export interface FindSampleSizeResponse {
  run_id: string;
  result: ScenarioResult<unknown>;
  plots: PlotSpecs;
  /** Strict-bootstrap diagnostics (strict mode only). Empty array otherwise. */
  warnings: string[];
}
export type SampleSizeMethod = { Grid: { by: unknown; mode: 'Linear' | 'Log' } };
