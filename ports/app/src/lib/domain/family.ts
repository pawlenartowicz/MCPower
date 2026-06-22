// UI entrypoints, FamilyConfig shape, and default values for all three analysis families (regression, ANOVA, mixed).
// NOTE: the `Entrypoint` string literals are persisted in saved state; renaming them needs a migration.
import type { Component } from 'svelte';
import AnovaIcon from '$lib/icons/AnovaIcon.svelte';
import LinearIcon from '$lib/icons/LinearIcon.svelte';
import MixedIcon from '$lib/icons/MixedIcon.svelte';
import { SIMULATION } from '$lib/configs/app-config';

/**
 * UI entrypoints — task names, not engine solver names.
 * "regression" covers both Continuous (→ Ols) and Binary (→ Glm) outcomes
 * via an outcome toggle; it does NOT imply a single estimator.
 * Internal `FamilyConfig` still uses the internal key 'regression' for
 * persistence; AppSpec::Linear / AppSpec::Logit are set by the adapter.
 *
 * NAMING GUARD: these labels must never imply they are the engine's
 * algorithm set (Ols / Glm / Mle). "Regression" spans two estimators.
 */
export type Entrypoint = 'anova' | 'regression' | 'mixed';

export const FAMILIES: readonly Entrypoint[] = ['anova', 'regression', 'mixed'] as const;

/** Entrypoints the user can actually select. All three families are enabled. */
export const SELECTABLE_FAMILIES: readonly Entrypoint[] = ['regression', 'anova', 'mixed'] as const;

export const FAMILY_LABEL: Record<Entrypoint, string> = {
  anova: 'ANOVA',
  regression: 'Regression',
  mixed: 'Mixed effects',
};

export type FamilyIcon = Component<{ class?: string }>;

export const FAMILY_ICON: Record<Entrypoint, FamilyIcon> = {
  anova: AnovaIcon,
  regression: LinearIcon,
  mixed: MixedIcon,
};

export type VariableKind = 'continuous' | 'binary' | 'factor';

/** Continuous-predictor synthetic distribution (UI copy of the wire's NumericDistribution). */
export type ContinuousDistribution = 'normal' | 'right_skewed' | 'left_skewed' | 'high_kurtosis' | 'uniform';

export interface VariableRow {
  name: string;
  kind: VariableKind;
  levels?: string[];
  referenceLevel?: string;
  binaryProportion?: number;
  nLevels?: number;
  levelProportions?: number[];
  /** Continuous-only: synthetic distribution. Absent = normal. */
  distribution?: ContinuousDistribution;
  /** Continuous-only: true = the user explicitly chose `distribution` (incl. explicit normal);
   *  scenario distribution swaps leave the column alone. False/absent = unpinned default. */
  pinned?: boolean;
  /** Factor-only: draw shares multinomially each run instead of exact allocation. */
  sampledProportions?: boolean;
  /** ANOVA-only: distinguishes a primary factor row from a covariate row.
   *  `undefined` for regression/mixed (formula-driven). */
  role?: 'factor' | 'covariate';
}

export interface EffectRow {
  name: string;
  value: number;
}

/** One extra grouping factor from a secondary random-effect formula term.
 *  `icc`/`n` are user-set; `clusterName` and `relation` are formula-derived
 *  (`(1|a)+(1|b)` → crossed, `(1|a/b)` → nested) and re-synced by the cluster
 *  cards' reconcile. */
export interface ExtraGroupingConfig {
  /** Cluster name from the secondary RE term. Read-only (formula-synced). */
  clusterName: string;
  /** ICC of this grouping; converted to τ² by the adapter (same map as the primary). */
  icc: number;
  /** 'crossed': fixed N levels across all primary clusters; 'nested': N children per primary. */
  relation: 'crossed' | 'nested';
  /** For 'crossed': number of levels. For 'nested': children per primary cluster. */
  n: number;
}

/** One random slope on the primary grouping factor. `predictorName` names a
 *  modeled predictor; the Rust assembler resolves it to a generation column. */
export interface SlopeConfig {
  predictorName: string;
  slopeVariance: number;
  slopeInterceptCorr: number;
}

/** Defaults for an extra grouping freshly declared in the formula, before the
 *  user touches its card. Shared by the cluster-cards reconcile and the engine
 *  adapter so the UI always shows the values a run would use. */
export const EXTRA_GROUPING_DEFAULTS = { icc: 0.05, crossedN: 10, nestedN: 2 } as const;

/** Defaults for a random slope pre-checked from formula syntax (`(1+x|g)`)
 *  before the user sets its parameters. Shared like EXTRA_GROUPING_DEFAULTS. */
export const SLOPE_DEFAULTS = { variance: 0.1, corr: 0 } as const;

export interface ClusterConfig {
  /** Read-only mirror of the formula's `(1|name)` term. */
  clusterName: string;
  icc: number;
  /** Which knob the user fixes; the other is derived from sample size. */
  dimKind: 'n_clusters' | 'cluster_size';
  nClusters: number;
  clusterSize: number;
  /** Predictor names that are constant within cluster (cluster-level covariates). Default []. */
  clusterLevelVars?: string[];
  /** Extra grouping factors from additional `(1|name)` formula terms. Default []. */
  extraGroupings?: ExtraGroupingConfig[];
  /** Random slopes on the primary grouping factor. Default []. */
  slopes?: SlopeConfig[];
  /** Whether the outcome is binary (GLMM). false/absent → Gaussian LME. */
  binaryOutcome?: boolean;
  /** Baseline event probability for binary GLMM. Used only when binaryOutcome=true. */
  baselineProbability?: number;
}

export interface AdvancedConfig {
  simulations: number;
  seed: number;
  correction: 'none' | 'bonferroni' | 'bh' | 'holm' | 'tukey';
  maxFailedSimulations: number;
  testFormulaOverride: string;
}

/** Model "More options" knobs (UI copy of the wire's OutcomeOptions).
 *  Neutral = unpinned default, NOT a specific value. A pinned explicit normal
 *  must be represented with pinnedResidual=true; the adapter sends it on the wire. */
export interface OutcomeOptionsConfig {
  /** Residual distribution shape. Null/absent = unpinned default (normal; scenarios may swap it).
   *  One of the canonical five: 'normal'|'right_skewed'|'left_skewed'|'high_kurtosis'|'uniform'.
   *  Only meaningful when pinnedResidual=true. */
  residualDistribution: ContinuousDistribution | null;
  /** True = the user explicitly chose a residual distribution (incl. explicit normal) → pinned;
   *  scenario swaps leave it alone. False = unpinned default (null residualDistribution on wire). */
  pinnedResidual: boolean;
  /** Driver predictor name for heteroskedasticity. '' = linear predictor Xβ (default).
   *  Variance ratio λ comes from the active scenario. */
  heteroskedasticityDriver: string;
}

export function defaultOutcomeOptions(): OutcomeOptionsConfig {
  return {
    residualDistribution: null,
    pinnedResidual: false,
    heteroskedasticityDriver: '',
  };
}

export type TestSelection =
  | { kind: 'all' }
  | { kind: 'effects'; names: string[] }
  | { kind: 'contrasts'; names: string[] };

export interface FamilyConfig {
  family: Entrypoint;
  formula: string;
  variables: VariableRow[];
  effects: EffectRow[];
  correlations: number[][];
  targetPower: number;
  alpha: number;
  tests: TestSelection;
  reportOverall: boolean;
  contrasts: Array<{ positiveName: string; negativeName: string; enabled: boolean }>;
  findPower: { n: number };
  findSampleSize: { from: number; to: number; by: number | 'auto' };
  advanced: AdvancedConfig;
  cluster?: ClusterConfig;
  baselineProbability?: number;
  /** Model "More options" knobs. Absent (older saved state) = all neutral. */
  outcomeOptions?: OutcomeOptionsConfig;
}

const baseAdvanced = (): AdvancedConfig => ({
  simulations: SIMULATION.n_sims.anova,
  seed: SIMULATION.seed,
  correction: 'none',
  maxFailedSimulations: SIMULATION.max_failed_fraction,
  testFormulaOverride: '',
});

export function defaultFamilyConfig(family: Entrypoint): FamilyConfig {
  const base: FamilyConfig = {
    family,
    formula: '',
    variables: [],
    effects: [],
    correlations: [],
    targetPower: SIMULATION.target_power * 100,
    alpha: SIMULATION.alpha,
    tests: { kind: 'all' },
    reportOverall: true,
    contrasts: [],
    findPower: { n: 100 },
    findSampleSize: {
      from: SIMULATION.sample_size_bounds.from,
      to: SIMULATION.sample_size_bounds.to,
      by: SIMULATION.sample_size_bounds.by,
    },
    advanced: baseAdvanced(),
  };
  if (family === 'mixed') {
    base.cluster = { clusterName: 'cluster', icc: 0.2, dimKind: 'n_clusters', nClusters: 20, clusterSize: 30 };
    base.advanced.simulations = SIMULATION.n_sims.mixed;
  }
  if (family === 'regression') {
    base.baselineProbability = 0.2;
    base.advanced.simulations = SIMULATION.n_sims.ols;
  }
  if (family === 'anova') {
    // ANOVA seeds factors/covariates and contrasts from the structured UI, so it
    // starts with explicit (initially empty) effect-kind targets rather than 'all'.
    // Per-coefficient effects route as targets; pairwise contrasts go through the
    // separate `contrasts` channel.
    base.tests = { kind: 'effects', names: [] };
    base.reportOverall = true;
  }
  return base;
}
