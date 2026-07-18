// TypeScript mirror of the Rust engine_app_spec::AppSpec wire shape; must stay serde-compatible with the Rust type.
// INVARIANT: field names and union tags (family/kind/status/estimator) must match the Rust serde shape (snake_case) exactly.
export type CorrectionMethod = 'none' | 'bonferroni' | 'holm' | 'benjamini_hochberg' | 'tukey_hsd';

/** Mirrors `engine_spec_builder::input::UploadMode` (snake_case serde). */
export type UploadMode = 'none' | 'partial' | 'strict';

/** Mirrors `engine_spec_builder::input::UploadColumnType` (snake_case serde). */
export type UploadColumnType = 'continuous' | 'binary' | 'factor';

/** Mirrors `engine_spec_builder::input::UploadColumn`. */
export interface UploadColumn {
    name: string;
    col_type: UploadColumnType;
    values: number[];
    /** Level labels for factor columns (code i → labels[i]). Empty for continuous/binary. */
    labels: string[];
}

/** Mirrors `engine_app_spec::CsvData` which has the same shape as `UploadInput`. */
export interface CsvData {
    mode: UploadMode;
    n_rows: number;
    columns: UploadColumn[];
}

export interface ParsedFormula {
    outcome: string;
    predictors: string[];
    interaction_terms: string[][];
}

/** Mirrors `engine_app_spec::NumericDistribution` (snake_case serde; absent = normal). */
export type NumericDistribution = 'normal' | 'right_skewed' | 'left_skewed' | 'high_kurtosis' | 'uniform';

export type VarType =
    | { kind: 'numeric'; name: string;
        /** Synthetic distribution; omitted = normal (pre-knob wire shape). */
        distribution?: NumericDistribution;
        /** `true` = user explicitly chose distribution (incl. explicit normal); scenario swaps
         *  leave the column alone. Omitted (= false) = unpinned default. */
        pinned?: boolean }
    | { kind: 'binary';  name: string; binary_proportion: number }
    | { kind: 'factor';  name: string; factor_n_levels: number; factor_proportions: number[];
        /** 0-based index of the baseline level in the port's full label list. Defaults to 0. */
        factor_reference?: number;
        /** Display labels, parallel to factor_proportions. Load-bearing: the engine
         *  derives effect names (`name[label]`) from them, so they must match the
         *  labels used in `effects`. Omitted = legacy "1".."k". */
        factor_labels?: string[];
        /** Per-factor sampled-shares override; omitted inherits the scenario flag. */
        sampled_proportions?: boolean };

/** Mirrors `engine_app_spec::OutcomeOptions` — structural knobs only.
 *  Magnitudes (λ, heterogeneity) are scenario-only and absent here.
 *  Omit the whole object when both fields are absent (neutral default). */
export interface OutcomeOptions {
    /** Canonical residual distribution name. Absent/null = unpinned default
     *  (normal; scenarios may swap it). Present = pinned — an explicit "normal"
     *  must be sent (neutral keys on unpinned-default, not on the value). */
    residual_distribution?: string | null;
    /** Driver predictor name for heteroskedasticity (non-factor).
     *  Absent/null = linear predictor Xβ drives the variance; ratio λ from scenario. */
    heteroskedasticity_driver?: string | null;
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

export interface EffectSize { name: string; value: number; }

/** Result of `getEffectsFromData`: the fitted effects plus the two scalars the
 *  same fit recovers but that are not effects. snake_case keys mirror the Rust
 *  `EffectsFromData` serde shape (matching the AppSpec wire boundary).
 *  `cluster_icc` is non-null only for a converged mixed fit; `baseline_probability`
 *  only for a binary outcome (logit, or mixed + binary). */
export interface EffectsFromData {
    effects: EffectSize[];
    cluster_icc: number | null;
    baseline_probability: number | null;
}

export interface CorrelationMatrix { names: string[]; values: number[][]; }

export type TestSelection =
    | { kind: 'all' }
    | { kind: 'effects'; names: string[] }
    | { kind: 'contrasts'; names: string[] };

/**
 * Mirrors `engine_spec_builder::input::ScenarioInput` (the projected wire shape
 * Python's `_scenario_dict` produces). The two distribution lists are already
 * integer-encoded (`new_distributions` via `_DIST_CODE`, `residual_dists` via
 * `_RESIDUAL_CODE`). RE-perturbation fields are carried flat on the wire:
 * `random_effect_dist`, `random_effect_df`, and `icc_noise_sd` (see below).
 */
export interface ScenarioWire {
    name: string;
    heterogeneity: number;
    heteroskedasticity_ratio: number;
    correlation_noise_sd: number;
    distribution_change_prob: number;
    new_distributions: number[];
    residual_change_prob: number;
    residual_dists: number[];
    residual_df: number;
    sampled_factor_proportions: boolean;
    /** Truth-seeded fitter start (mixed/clustered designs) vs. cold (blind) start. */
    truth_start: boolean;
    /** Random-effect distribution code (normal=0, heavy_tailed=1); see RE_DIST_CODE. */
    random_effect_dist: number;
    /** Degrees of freedom for a heavy-tailed random-effect distribution (0 = unused). */
    random_effect_df: number;
    /** SD of multiplicative log-normal noise on per-cluster ICC (0 = none). */
    icc_noise_sd: number;
}

export interface LinearSpec {
    parsed_formula: ParsedFormula;
    var_types: VarType[];
    effects: EffectSize[];
    correlations: CorrelationMatrix | null;
    alpha: number;
    target_power: number;
    n_sims: number;
    seed: number;
    tests: TestSelection;
    correction: CorrectionMethod;
    /** Wald SE method; only affects clustered-binary GLMM. Optional — mirrors the
     *  Rust `#[serde(default)]` (omitted ⇒ engine default `hessian`). */
    wald_se?: 'hessian' | 'rx';
    scenarios: ScenarioWire[];
    csv: CsvData | null;
    /** v1-parity omnibus: OLS F-test / Logit LRT vs intercept-only. Always false
     *  for the mixed family — the omnibus is undefined for a mixed-effects fit
     *  (see the gate in app-spec-adapter's projectCommon). */
    report_overall: boolean;
    /** Pairwise contrast pairs as [positiveName, negativeName] tuples. */
    contrasts: Array<[string, string]>;
    /** Optional misspecified test model: generate from the full formula but
     *  fit/test only these terms. Omitted/empty → fit the full model. */
    test_formula?: string;
    /** Outcome-level generation knobs; omitted = builder defaults. */
    outcome_options?: OutcomeOptions;
}

export interface LogitSpec extends LinearSpec {
    baseline_probability: number;
    /** Binary link. Omitted when logit (the default; Rust `skip_serializing_if`);
     *  `'probit'` selects the probit link. */
    link?: 'logit' | 'probit';
    /** Adaptive Gauss-Hermite quadrature points for a clustered GLMM fit. Omitted
     *  when 1 (Laplace; Rust `skip_serializing_if` on the default). */
    agq?: number;
}

/** Mirrors `engine_app_spec::PoissonSpec` — LogitSpec field-for-field, but
 *  `baseline_rate` (a log-link intercept) replaces `baseline_probability` and
 *  there is no `link` (Poisson is always log link). */
export interface PoissonSpec extends LinearSpec {
    baseline_rate: number;
    /** Adaptive Gauss-Hermite quadrature points for a clustered GLMM fit. Omitted
     *  when 1 (Laplace). */
    agq?: number;
}

export type ClusterDim =
    | { kind: 'n_clusters'; value: number }
    | { kind: 'cluster_size'; value: number };

/** Mirrors `engine_app_spec::AppGroupingRelation` (internally tagged on `kind`). */
export type AppGroupingRelation =
    | { kind: 'crossed'; n_clusters: number }
    | { kind: 'nested_within'; n_per_parent: number };

/** Mirrors `engine_app_spec::AppGroupingSpec` — one extra grouping factor. */
export interface AppGroupingSpec {
    tau_squared: number;
    relation: AppGroupingRelation;
    /** Grouping-factor name from its formula term (host-side: script gen, labels). */
    cluster_name?: string;
    /** Random slopes on this grouping factor (mirrors the primary's `slopes`).
     *  Omitted/empty → intercept-only RE on this grouping. */
    slopes?: AppSlopeTerm[];
}

/** Mirrors `engine_app_spec::AppSlopeTerm` — one random slope (predictor named,
 *  not yet resolved to a generation column; the Rust assembler resolves it). */
export interface AppSlopeTerm {
    predictor_name: string;
    slope_variance: number;
    slope_intercept_corr: number;
}

export interface MixedSpec extends LinearSpec {
    cluster_name: string;
    icc: number;
    cluster_dim: ClusterDim;
    /** Predictor names constant within cluster (cluster-level covariates). Default []. */
    cluster_level_vars?: string[];
    /** Extra grouping factors (crossed / nested) after the primary. Default []. */
    extra_groupings?: AppGroupingSpec[];
    /** Random slopes on the primary grouping factor. Default []. */
    slopes?: AppSlopeTerm[];
    /** Outcome distribution. Absent / `{ kind: 'gaussian' }` → Continuous+Mle (Rust default).
     *  `{ kind: 'binary', baseline_probability, link? }` → Binary+Glm (logit intercept +
     *  latent τ²; `link` omitted = logit, `'probit'` for the probit link).
     *  `{ kind: 'poisson', baseline_rate, tau_squared }` → Count+Glm with a log intercept
     *  and RAW random-intercept variance (no ICC conversion). */
    outcome?:
        | { kind: 'gaussian' }
        | { kind: 'binary'; baseline_probability: number; link?: 'logit' | 'probit' }
        | { kind: 'poisson'; baseline_rate: number; tau_squared: number };
    /** Adaptive Gauss-Hermite quadrature points for the GLMM fit. Omitted when 1
     *  (Laplace; Rust `skip_serializing_if` on the default). */
    agq?: number;
}

export interface AnovaFactor {
    name: string;
    levels: string[];
    reference_level: string;
    proportions: number[] | null;
}

export interface AnovaCovariate {
    name: string;
}

export interface AnovaSpec {
    outcome: string;
    factors: AnovaFactor[];
    covariates: AnovaCovariate[];
    effects: EffectSize[];
    correlations: CorrelationMatrix | null;
    alpha: number;
    target_power: number;
    n_sims: number;
    seed: number;
    tests: TestSelection;
    correction: CorrectionMethod;
    scenarios: ScenarioWire[];
    csv: null;
    report_overall: boolean;
    contrasts: Array<[string, string]>;
}

export type AppSpec =
    | ({ family: 'linear'  } & LinearSpec)
    | ({ family: 'logit'   } & LogitSpec)
    | ({ family: 'poisson' } & PoissonSpec)
    | ({ family: 'anova'   } & AnovaSpec)
    | ({ family: 'mixed'   } & MixedSpec);
