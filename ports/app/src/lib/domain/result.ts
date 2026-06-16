// Wire result types mirroring engine-orchestrator serde shapes; demo aliases kept separate so canonical names belong to the wire format.
// INVARIANT: optional/absent fields must stay optional (older engine payloads); removing a `?` is a silent break.
import type { Entrypoint } from './family';

// =========================================================================
// Demo result types — kept under `Demo*` names so the canonical
// `PowerResult` / `SampleSizeResult` names belong to the wire format below.
// Demo consumers (demo.svelte.ts, run-fake.ts, etc.) use these aliases; not used in live runs.
// =========================================================================

export interface DemoEffectPower {
  name: string;
  power: number;
  ci: [number, number];
}

export interface DemoPowerResult {
  family: Entrypoint;
  kind: 'find-power' | 'find-sample-size';
  n: number;
  effects: DemoEffectPower[];
  ranAt: string;
  simulations: number;
}

// =========================================================================
// Wire result types — mirror engine-orchestrator's serde shape byte-for-byte.
// =========================================================================

export interface Ci { lo: number; hi: number; }

/** Mirrors `engine_orchestrator::EstimatorExtras` (serde tag "estimator", snake_case). */
export type EstimatorExtras =
    | { estimator: 'ols' }
    | { estimator: 'glm'; baseline_prob_realized: number;
        singular_fit_rate?: number;
        /** Mean τ̂² across converged fits — GLMM (Glm + cluster) numerics, zero for plain GLM. */
        tau_squared_hat_mean?: number;
        baseline_prob_sum?: number; baseline_prob_n?: number }
    | { estimator: 'mle'; tau_estimate: number; boundary_hits: number;
        joint_uncorrected_rate: number; joint_corrected_rate: number;
        singular_fit_rate?: number;
        /** Per-diagonal-component pin rate (fraction of converged fits that
         *  pinned component k), ordered [intercept, slope_0, …]. Empty for
         *  non-cluster or intercept-only Mle. Additive (absent in older payloads). */
        boundary_rate_per_component?: number[];
        tau_sum?: number; tau_n?: number;
        joint_uncorrected_count?: number; joint_corrected_count?: number };

export interface PowerResult {
  n: number;
  n_sims: number;
  target_indices: number[];
  /** Pairwise contrast identities `[positive, negative]` as β-column indices in
   *  the same 1-based space as `target_indices`. The power/ci arrays hold the
   *  marginals first, then one entry per contrast (length =
   *  `target_indices.length + contrast_pairs.length`). Absent in older payloads. */
  contrast_pairs?: [number, number][];
  power_uncorrected: number[];
  power_corrected: number[];
  ci_uncorrected: Ci[];
  ci_corrected: Ci[];
  convergence_rate: number;
  boundary_hit: number[];
  estimator_extras: EstimatorExtras;
  overall_significant_rate?: number | null;
  overall_significant_ci?: Ci | null;
  success_count_histogram_uncorrected?: number[];
  success_count_histogram_corrected?: number[];
  /** Per-factor sparse-exclusion counts (count / n_sims = exclusion rate). Parallel to spec factor_names order. */
  factor_exclusion_counts?: number[];
  /** Per-factor GLM separation-fallback drop counts. Parallel to spec factor_names order. */
  factor_separation_counts?: number[];
  /** Preflight and grid warnings (e.g. cluster-snap, sparse-level notices). */
  grid_warnings?: string[];
}

/** Mirrors `engine_orchestrator::ByValue` (externally-tagged serde). */
export type ByValue = { Fixed: number } | { Auto: { count: number } };

/** Mirrors `engine_orchestrator::SampleSizeMethod` (externally-tagged serde). */
export type SampleSizeMethod =
  | { Grid: { by: ByValue; mode: 'Linear' | 'Log' } };

/**
 * Mirrors `engine_orchestrator::CrossingFit` (serde status-tagged union).
 * Present only in payloads from engine versions that emit model-based crossing fits.
 */
export type CrossingFit =
  | { status: 'fitted'; n_star: number; n_achievable: number; ci_lo: number | null; ci_hi: number | null }
  | { status: 'at_or_below_min'; n_min: number }
  | { status: 'not_reached'; n_approx: number | null }
  | { status: 'non_monotone'; max_violation: number };

export interface SampleSizeResult {
  grid_or_trace: PowerResult[];
  first_achieved: (number | null)[];
  /** Smallest N where >= k targets are jointly significant at target_power,
   *  derived from each grid point's CORRECTED histogram. Index j is k = j+1. */
  first_joint_achieved: (number | null)[];
  target_power: number;
  method: SampleSizeMethod;
  /** Model-based crossing fits, one per target; parallel to first_achieved.
   *  Absent in older payloads — fall back to first_achieved when missing. */
  fitted?: CrossingFit[];
  /** Model-based crossing fits for the joint ≥k targets; parallel to first_joint_achieved.
   *  Index j corresponds to k = j+1. Absent in older payloads. */
  fitted_joint?: CrossingFit[];
  /** Grid-empirical first N at which the overall/omnibus test reaches target_power.
   *  `null`/absent when no overall test was requested (or it never crossed). */
  first_overall_achieved?: number | null;
  /** Model-based crossing fit for the overall/omnibus test (singular — one test).
   *  `null`/absent when no overall test was requested. */
  fitted_overall?: CrossingFit | null;
  /** Cluster size atom (1 = unclustered). Absent in older payloads. */
  cluster_atom?: number;
  /** Warnings emitted by the grid search (e.g. non-monotone power curves, preflight notices). */
  grid_warnings?: string[];
  /** Per-grid-point sparse-exclusion counts: outer index = grid point, inner = factor. Absent in older payloads. */
  factor_exclusion_counts?: number[][];
  /** Per-grid-point GLM separation-fallback drop counts: outer index = grid point, inner = factor. Absent in older payloads. */
  factor_separation_counts?: number[][];
}

/** Mirrors `engine_orchestrator::ScenarioResult<T>`; `scenarios` is `Vec<(String, T)>` */
export interface ScenarioResult<T> {
  scenarios: [string, T][];
}

/** Mirrors `engine_app_spec::PlotBlock` — one named theme-naked Vega-Lite v5 block. */
export interface PlotBlock {
  key: string;
  spec: string;
}

/** Mirrors `engine_app_spec::PlotSpecs` — ordered plot blocks.
 *  Block keys: `"power"` (find_power), `"curve"` (single-scenario sample-size),
 *  `"scenario:<label>"` / `"overlay"` (multi-scenario sample-size),
 *  `"at_least_k"` / `"exactly_k"` (≥2 targets, sample-size). */
export interface PlotSpecs {
  blocks: PlotBlock[];
}

/** Mirrors `engine-app-spec` FormulaParse (Python-bridge shape). */
export interface FormulaParse {
  dependent: string;
  predictors: string[];
  terms: Array<{ kind: 'main'; name: string } | { kind: 'interaction'; vars: string[] }>;
  random_effects: Array<
    | { kind: 'intercept'; group: string; parent: string | null }
    | { kind: 'slope'; group: string; vars: string[] }
  >;
}

/**
 * Mirrors `engine_app_spec::progress::serialize_event` output — `kind` strings
 * come from there. The `run_started` variant is emitted by the Tauri command
 * before the engine starts, so the frontend learns the run_id without racing.
 */
export type ProgressEvent =
  | { kind: 'run_started'; run_id: string }
  | { kind: 'started'; total_sims: number; total_scenarios: number; total_grid_points: number }
  | { kind: 'scenario_started'; label: string; idx: number; total: number }
  | { kind: 'sims_completed'; n: number; completed: number; total: number }
  | { kind: 'n_point_completed'; n: number; power_uncorrected: number[]; power_corrected: number[] }
  | { kind: 'scenario_completed'; label: string; idx: number }
  | { kind: 'cancelled' }
  | { kind: 'completed' };
