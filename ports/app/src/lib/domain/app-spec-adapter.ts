// Maps FamilyConfig (UI state) to a serde-compatible AppSpec (Rust wire shape); validates inputs and returns accumulated errors + non-blocking warnings.

import { LIMITS } from '$lib/configs/app-config';
import {
  parsedFormulaStore,
  toClusterTerms,
  toWireParsed,
} from '$lib/stores/parsed-formula.svelte';
import { uploadStore } from '$lib/stores/upload.svelte';
import { scenariosStore } from '$lib/stores/scenarios.svelte';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
import type {
  AppGroupingSpec,
  AppSlopeTerm,
  AppSpec,
  ClusterDim,
  CorrectionMethod,
  CorrelationMatrix,
  CsvData,
  LinearSpec,
  LogitSpec,
  MixedSpec,
  OutcomeOptions,
  ParsedFormula,
  ScenarioWire,
  TestSelection as WireTestSelection,
} from './app-spec';
import { correlatableVariables } from './correlations';
import { expandInteraction, expandMainEffect, factorLevels } from './effect-names';
import { projectScenario } from './scenario-projection';
import {
  EXTRA_GROUPING_DEFAULTS,
  SLOPE_DEFAULTS,
  type Entrypoint,
  type FamilyConfig,
} from './family';

export interface AdaptResult {
  spec: AppSpec | null;
  errors: string[];
  /** Non-blocking soft warnings (e.g. alpha above the usual maximum, extreme
   *  baseline probability). Unlike `errors`, these do not gate the run. */
  warnings: string[];
}

/** Per-family scenario support: 'lme' passes RE-perturbation fields through;
 *  'plain' zeros them regardless of scenario config. */
export type ScenarioSupport = 'plain' | 'lme';

function projectEnabledScenarios(errors: string[], support: ScenarioSupport): ScenarioWire[] {
  if (!sharedPrefs.scenariosEnabled) {
    // Send the optimistic scenario as the single merged entry so the engine's
    // optimistic fast path fires (all perturbation magnitudes neutral = 0/1).
    const optimistic = scenariosStore.scenarios.find((s) => s.name === 'optimistic');
    if (!optimistic) return []; // store not hydrated yet; fall back to empty
    try {
      const wire = projectScenario(optimistic);
      // Always zero the lme fields when scenarios are off — mirrors the 'plain'
      // support path; the merge optimistic is family-agnostic.
      wire.random_effect_dist = 0;
      wire.random_effect_df = 0;
      wire.icc_noise_sd = 0;
      return [wire];
    } catch {
      return []; // guard: if optimistic scenario errors, fall back to empty
    }
  }
  const out: ScenarioWire[] = [];
  for (const cfg of scenariosStore.scenarios) {
    try {
      const wire = projectScenario(cfg);
      // Contract invariant 13: lme perturbations require an Mle (or clustered
      // Glm) estimator. Zero them for non-mixed families — exact mirror of the
      // Python guard in spec_builder.py (`registry._random_effects_parsed`).
      if (support !== 'lme') {
        wire.random_effect_dist = 0;
        wire.random_effect_df = 0;
        wire.icc_noise_sd = 0;
      }
      out.push(wire);
    } catch (e) {
      errors.push(`scenario '${cfg.name}': ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return out;
}

interface CommonProjection {
  parsed_formula: LinearSpec['parsed_formula'];
  var_types: LinearSpec['var_types'];
  effects: LinearSpec['effects'];
  correlations: CorrelationMatrix | null;
  alpha: number;
  target_power: number;
  n_sims: number;
  seed: number;
  tests: WireTestSelection;
  correction: CorrectionMethod;
  scenarios: ScenarioWire[];
  csv: CsvData | null;
  report_overall: boolean;
  contrasts: Array<[string, string]>;
  test_formula?: string;
}

function projectCommon(
  parsed: ParsedFormula,
  config: FamilyConfig,
  projectOpts?: { allowTukey?: boolean; scenarios?: ScenarioSupport },
): { ok: CommonProjection | null; errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Project var_types
  const varMap = new Map(config.variables.map((v) => [v.name, v]));
  const allPredictors = parsed.predictors;

  const var_types = allPredictors.map((name) => {
    const row = varMap.get(name);
    if (!row) {
      return { kind: 'numeric' as const, name };
    }
    if (row.kind === 'continuous') {
      const distribution = row.distribution;
      const pinned = row.pinned === true;
      // Neutral wire: omit distribution (= normal) and omit pinned (= false).
      // Pinned-normal: distribution still omitted (Rust `#[serde(skip_if_normal)]`),
      // but pinned=true must be sent to prevent scenario swaps.
      if (pinned) {
        return distribution && distribution !== 'normal'
          ? { kind: 'numeric' as const, name, distribution, pinned: true }
          : { kind: 'numeric' as const, name, pinned: true };
      }
      return distribution && distribution !== 'normal'
        ? { kind: 'numeric' as const, name, distribution }
        : { kind: 'numeric' as const, name };
    }
    if (row.kind === 'binary') {
      const prop = row.binaryProportion ?? 0.5;
      if (prop < 0 || prop > 1) {
        errors.push(`binary_proportion for ${name} out of range [0,1]`);
      }
      return { kind: 'binary' as const, name, binary_proportion: prop };
    }
    // factor
    const nLevels = row.nLevels ?? 0;
    const weights = row.levelProportions ?? [];
    // Shares are weights: normalize so the engine (which requires Σ = 1) sees
    // proper proportions while the UI keeps the raw weights the user typed.
    let proportions = weights;
    if (weights.length !== nLevels) {
      errors.push(`factor ${name}: proportions length mismatch`);
    } else if (weights.some((w) => !Number.isFinite(w) || w <= 0)) {
      errors.push(`factor ${name}: every share must be > 0`);
    } else {
      const sum = weights.reduce((a, b) => a + b, 0);
      proportions = weights.map((w) => w / sum);
    }
    // Engine level names = resolved display labels (blank slots fall back to
    // "i+1" — same resolution the effect rows use, so effect names match).
    const labels = factorLevels(row).slice(0, nLevels);
    for (let i = 1; i < labels.length; i++) {
      if (labels.indexOf(labels[i]!) < i) {
        errors.push(`factor ${name}: duplicate level label '${labels[i]}'`);
        break;
      }
    }
    // Resolve referenceLevel (a label string) → 0-based index in the full labels list.
    // If unset or blank, default to index 0.
    let factor_reference = 0;
    if (row.referenceLevel && row.referenceLevel.trim() !== '') {
      const idx = labels.indexOf(row.referenceLevel);
      if (idx !== -1) factor_reference = idx;
    }
    return {
      kind: 'factor' as const,
      name,
      factor_n_levels: nLevels,
      factor_proportions: proportions,
      factor_reference,
      factor_labels: labels,
      ...(row.sampledProportions !== undefined
        ? { sampled_proportions: row.sampledProportions }
        : {}),
    };
  });

  // Project effects — factor predictors and factor interactions expand to
  // per-level dummies (reference level dropped, matching the engine's dummy
  // coding). Both expansions are single-sourced from effect-names.ts so the
  // names sent to the engine match what the effects UI shows.
  const interactionEffectNames = parsed.interaction_terms.flatMap((vars) =>
    expandInteraction(vars, varMap),
  );
  const validEffectNames = new Set<string>([
    ...allPredictors.flatMap((name) => expandMainEffect(varMap.get(name), name, false)),
    ...interactionEffectNames,
  ]);

  // Reference-level dummies (e.g. treatment[1]) are legitimate names the
  // effect rows emit — effectNames() lists every level — but they are not
  // targets (expandMainEffect drops the reference, matching the engine's dummy
  // coding). Treat them as known so they don't read as "unknown effect name".
  const knownEffectNames = new Set<string>([
    ...allPredictors.flatMap((name) => expandMainEffect(varMap.get(name), name, true)),
    ...interactionEffectNames,
  ]);

  for (const eff of config.effects) {
    if (!knownEffectNames.has(eff.name)) errors.push(`unknown effect name: ${eff.name}`);
    if (!Number.isFinite(eff.value)) errors.push(`effect '${eff.name}': value must be a number`);
  }

  const userEffectMap = new Map(config.effects.map((e) => [e.name, e.value]));
  const effects = [...validEffectNames].map((name) => ({
    name,
    value: userEffectMap.has(name) ? userEffectMap.get(name)! : 0,
  }));

  // Project correlations — only correlatable variables (continuous, non-uploaded in strict mode).
  let correlations: CorrelationMatrix | null = null;
  const mat = config.correlations;
  if (mat.length > 0) {
    const corrEntries = correlatableVariables(config.variables, uploadStore.csvData, uploadStore.mode);
    if (corrEntries.length >= 2) {
      const subNames = corrEntries.map((e) => e.row.name);
      const subValues = corrEntries.map((ei) =>
        corrEntries.map((ej) => mat[ei.idx]?.[ej.idx] ?? (ei.idx === ej.idx ? 1 : 0)),
      );
      // Only emit if there is at least one non-zero off-diagonal entry in the submatrix.
      let hasNonZeroOffDiag = false;
      for (let i = 0; i < subValues.length; i++) {
        for (let j = 0; j < subValues.length; j++) {
          if (i !== j && (subValues[i]?.[j] ?? 0) !== 0) {
            hasNonZeroOffDiag = true;
            break;
          }
        }
        if (hasNonZeroOffDiag) break;
      }
      if (hasNonZeroOffDiag) {
        correlations = { names: subNames, values: subValues };
      }
    }
  }

  let tests: WireTestSelection;
  const uiTests = config.tests;
  if (uiTests.kind === 'all') {
    tests = { kind: 'all' };
  } else if (uiTests.kind === 'effects') {
    tests = { kind: 'effects', names: uiTests.names };
  } else {
    // contrasts — Linear/Logit-invalid
    errors.push('contrasts test selection is ANOVA-only; Linear supports all/effects');
    tests = { kind: 'all' };
  }

  let correction: CorrectionMethod;
  const uiCorrection = config.advanced.correction;
  if (uiCorrection === 'none' || uiCorrection === 'bonferroni' || uiCorrection === 'holm') {
    correction = uiCorrection;
  } else if (uiCorrection === 'bh') {
    correction = 'benjamini_hochberg';
  } else if (uiCorrection === 'tukey') {
    if (projectOpts?.allowTukey) {
      correction = 'tukey_hsd';
    } else {
      errors.push('tukey correction is ANOVA-only');
      correction = 'none';
    }
  } else {
    correction = 'none';
  }

  // mode is already stamped into csvData; engine handles mode='none' by resampling without measuring correlations.
  const csv: CsvData | null = uploadStore.csvData;

  const scenarios = projectEnabledScenarios(errors, projectOpts?.scenarios ?? 'plain');

  // Stale persisted snapshots can carry null or miss newer fields (undefined →
  // NaN through arithmetic); JSON.stringify writes both as `null`, which the
  // engine's serde rejects with an opaque "invalid type: null, expected f64".
  // Reject non-finite numbers here so they surface as config errors instead.
  if (!Number.isFinite(config.alpha)) errors.push(`alpha must be a number, got: ${config.alpha}`);
  else if (config.alpha > LIMITS.max_alpha) {
    // The hard (0,1) range is the engine's contract invariant; above max_alpha
    // is only a soft warning (power at such a high significance level is rarely
    // meaningful) — mirrors the Python/R hosts.
    warnings.push(
      `alpha ${config.alpha} is above the usual maximum of ${LIMITS.max_alpha}; power at such a high significance level is rarely meaningful.`,
    );
  }
  if (!Number.isFinite(config.targetPower)) {
    errors.push(`target power must be a number, got: ${config.targetPower}`);
  }

  const ok: CommonProjection = {
    parsed_formula: parsed,
    var_types,
    effects,
    correlations,
    alpha: config.alpha,
    target_power: config.targetPower / 100,
    n_sims: config.advanced.simulations,
    seed: config.advanced.seed,
    tests,
    correction,
    scenarios,
    csv,
    // The omnibus (OLS F / GLM LRT) is not defined for a mixed-effects fit
    // (LME or clustered-logistic GLMM), so the mixed family never requests it —
    // mirrors the Python/R host gate (overall_test_available). The UI already
    // hides the omnibus toggle for mixed; forcing false here also overrides any
    // stale persisted reportOverall=true so the engine returns no overall row.
    report_overall: config.family === 'mixed' ? false : (config.reportOverall ?? true),
    contrasts: (config.contrasts ?? [])
      .filter((c) => c.enabled !== false)
      .map((c) => [c.positiveName, c.negativeName] as [string, string]),
    // Empty/whitespace override → omit (Rust #[serde(default)] → None → fit full model).
    test_formula: config.advanced.testFormulaOverride?.trim() || undefined,
  };

  return { ok, errors, warnings };
}

// ---------------------------------------------------------------------------
// Outcome-level options (Model "More options" dialog)
// ---------------------------------------------------------------------------

/**
 * Project `cfg.outcomeOptions` to the wire, omitting every neutral value so an
 * untouched dialog leaves the wire byte-identical to the pre-knob shape.
 *
 * "Neutral" keys on unpinned-default, NOT on the value: a pinned explicit
 * "normal" MUST be sent (pinnedResidual=true with residualDistribution="normal"
 * serialises as `{ residual_distribution: "normal" }`). Residual + hetero-
 * skedasticity driver only apply to continuous outcomes.
 */
function projectOutcomeOptions(
  config: FamilyConfig,
  continuousOutcome: boolean,
  _errors: string[],
): OutcomeOptions | undefined {
  const o = config.outcomeOptions;
  if (!o) return undefined;
  const out: OutcomeOptions = {};
  if (continuousOutcome) {
    // Pinned residual distribution: send even if it's "normal" (explicit choice).
    if (o.pinnedResidual) {
      out.residual_distribution = o.residualDistribution ?? 'normal';
    }
    // Heteroskedasticity driver: non-empty = a specific predictor was chosen.
    if (o.heteroskedasticityDriver.trim() !== '') {
      out.heteroskedasticity_driver = o.heteroskedasticityDriver;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

// ---------------------------------------------------------------------------
// ANOVA projection
// ---------------------------------------------------------------------------

function projectAnova(config: FamilyConfig): AdaptResult {
  const factorRows = config.variables.filter((v) => v.role === 'factor');
  if (factorRows.length === 0) {
    return { spec: null, errors: ['ANOVA needs at least one factor'], warnings: [] };
  }
  const covariateRows = config.variables.filter((v) => v.role === 'covariate');

  // Build the wire ParsedFormula directly from the structured rows — no formula
  // string, no parser. Factors first, then covariates; no interaction terms.
  const parsed: ParsedFormula = {
    outcome: 'y',
    predictors: [...factorRows, ...covariateRows].map((v) => v.name),
    interaction_terms: [],
  };

  const { ok, errors, warnings } = projectCommon(parsed, config, { allowTukey: true });
  if (!ok) return { spec: null, errors, warnings };

  // ANOVA factor effects are expressed only as pairwise contrasts (the input
  // panel surfaces them as contrast chips, not coefficient tests). The wire
  // `tests` list is therefore COVARIATE effects only: `{kind:'all'}` would
  // expand to every effect, re-introducing factor-level marginals, so override
  // it with the covariate effect-name set. An `{kind:'effects'}` selection is
  // intersected down to that set — a snapshot persisted before this change can
  // still hold factor-dummy names (valid effect names projectCommon never
  // filters), and passing them as-is would resurrect the very factor marginals
  // this removes. Factor levels then reach the engine only through contrasts.
  const covariateEffectNames = new Set(
    config.variables
      .filter((v) => v.role === 'covariate')
      .flatMap((v) => expandMainEffect(v, v.name, false)),
  );
  if (ok.tests.kind === 'all') {
    ok.tests = { kind: 'effects', names: [...covariateEffectNames] };
  } else if (ok.tests.kind === 'effects') {
    ok.tests = {
      kind: 'effects',
      names: ok.tests.names.filter((n) => covariateEffectNames.has(n)),
    };
  }

  // Degenerate fallback: report_overall is layered-only, never a standalone
  // target. If neither an explicit (covariate) effect target nor any contrast is
  // selected, fall back to tests:{kind:'all'} so the engine never sees empty
  // targets. (The covariate override above always leaves ok.tests as 'effects',
  // so no need to re-guard against an already-'all' selection.)
  const hasContrasts = ok.contrasts.length > 0;
  const hasEffectTargets = ok.tests.kind === 'effects' && ok.tests.names.length > 0;
  if (!hasContrasts && !hasEffectTargets) {
    ok.tests = { kind: 'all' };
  }

  const spec: AppSpec = { family: 'linear', ...ok };
  return { spec, errors, warnings };
}

/** Push a soft warning if a (valid, in-(0,1)) baseline probability is outside
 *  the `baseline_p_warn` band — mirrors the Python/R hosts. The hard (0,1)
 *  reject stays at each call site; this is non-blocking. */
function warnBaselineProbability(p: number, warnings: string[]): void {
  const [lo, hi] = LIMITS.baseline_p_warn;
  if (p < lo || p > hi) {
    warnings.push(
      `baseline probability ${p} is extreme (outside [${lo}, ${hi}]); expect near-separation and unstable power estimates.`,
    );
  }
}

export function familyConfigToAppSpec(
  entrypoint: Entrypoint,
  config: FamilyConfig,
  outcomeKind: 'continuous' | 'binary' = 'continuous',
): AdaptResult {
  if (entrypoint === 'regression') {
    const st = parsedFormulaStore.get(config.formula);
    if (st.pending) return { spec: null, errors: ['parsing formula…'], warnings: [] };
    if (!st.result || st.error) return { spec: null, errors: [st.error ?? 'formula parse failed'], warnings: [] };
    const parsed = toWireParsed(st.result);
    if (st.result.random_effects.length > 0)
      return { spec: null, errors: ['Random effects are only available in Mixed models'], warnings: [] };
    const { ok, errors, warnings } = projectCommon(parsed, config);
    if (!ok) return { spec: null, errors, warnings };

    if (outcomeKind === 'binary') {
      const p = config.baselineProbability;
      if (typeof p !== 'number' || p <= 0 || p >= 1) {
        return {
          spec: null,
          errors: [...errors, `baseline_probability out of range (0, 1): ${p}`],
          warnings,
        };
      }
      warnBaselineProbability(p, warnings);
      const outcome_options = projectOutcomeOptions(config, false, errors);
      const spec: AppSpec = {
        family: 'logit',
        ...ok,
        baseline_probability: p,
        ...(outcome_options ? { outcome_options } : {}),
      } satisfies { family: 'logit' } & LogitSpec;
      return { spec, errors, warnings };
    }

    const outcome_options = projectOutcomeOptions(config, true, errors);
    const spec: AppSpec = {
      family: 'linear',
      ...ok,
      ...(outcome_options ? { outcome_options } : {}),
    };
    return { spec, errors, warnings };
  }

  if (entrypoint === 'anova') {
    return projectAnova(config);
  }

  if (entrypoint === 'mixed') {
    const st = parsedFormulaStore.get(config.formula);
    if (st.pending) return { spec: null, errors: ['parsing formula…'], warnings: [] };
    if (!st.result || st.error) return { spec: null, errors: [st.error ?? 'formula parse failed'], warnings: [] };
    const parsed = toWireParsed(st.result);
    const clusterTerms = toClusterTerms(st.result);
    // Both mixed outcomes carry the lme perturbations: binary GLMM (Glm +
    // cluster) and Gaussian LMM (Mle) alike.
    const { ok, errors, warnings } = projectCommon(parsed, config, {
      scenarios: 'lme',
    });
    if (!ok) return { spec: null, errors, warnings };
    if (clusterTerms.length === 0) {
      return {
        spec: null,
        errors: [...errors, 'Mixed family needs a (1|cluster) term in the formula'],
        warnings,
      };
    }
    const primary = clusterTerms[0]!;
    const cluster_name = primary.cluster;

    const cl = config.cluster;
    if (!cl) return { spec: null, errors: [...errors, 'cluster configuration missing'], warnings };
    // Number.isFinite also rejects null/undefined from stale persisted snapshots
    // (`null < 0` and `null >= 1` are both false, so the range check alone lets
    // null onto the wire as a serde-fatal `"icc": null`).
    if (!Number.isFinite(cl.icc) || cl.icc < 0 || cl.icc >= 1) {
      return { spec: null, errors: [...errors, `icc out of range [0, 1): ${cl.icc}`], warnings };
    }
    // Stability band: a nonzero ICC outside [icc_stability] is rejected host-side
    // (icc 0 = no clustering, always allowed) — mirrors the Python/R hosts.
    const [iccLo, iccHi] = LIMITS.icc_stability;
    if (cl.icc !== 0 && (cl.icc < iccLo || cl.icc > iccHi)) {
      errors.push(`icc ${cl.icc} outside the stable band [${iccLo}, ${iccHi}] for numerical stability (use 0 for no clustering)`);
    }
    const cluster_dim: ClusterDim =
      cl.dimKind === 'cluster_size'
        ? { kind: 'cluster_size', value: cl.clusterSize }
        : { kind: 'n_clusters', value: cl.nClusters };

    const binaryOutcome = cl.binaryOutcome === true;
    // ICC → τ² for extra groupings, mirroring assemble.rs (`icc_to_tau_squared`
    // / `icc_to_tau_squared_logit`): the primary's ICC converts Rust-side, but
    // the wire takes secondaries as τ² directly.
    const iccToTau = (icc: number): number => {
      const denom = 1 - icc;
      const tau = denom > 0 ? icc / denom : 0;
      return binaryOutcome ? (tau * Math.PI ** 2) / 3 : tau;
    };

    // Extra groupings: structure comes from formula syntax ((1|a)+(1|b) →
    // crossed, (1|a/b) → nested); icc/n come from the per-cluster card config,
    // matched by cluster name, with shared defaults until the user edits.
    const extraTerms = clusterTerms.slice(1);
    const extrasByName = new Map((cl.extraGroupings ?? []).map((g) => [g.clusterName, g]));
    if (extraTerms.filter((t) => t.parent !== null).length > 1) {
      errors.push('at most one nested grouping factor is supported');
    }
    if (extraTerms.length > 0 && cl.dimKind === 'cluster_size') {
      errors.push("extra grouping factors require 'by n clusters' sizing on the primary cluster");
    }
    for (const t of extraTerms) {
      if (t.slopeVars.length > 0) {
        errors.push(`random slopes are only supported on the primary (first) cluster, not '${t.cluster}'`);
      }
    }
    const extra_groupings: AppGroupingSpec[] = extraTerms.map((t) => {
      const g = extrasByName.get(t.cluster);
      const icc = g?.icc ?? EXTRA_GROUPING_DEFAULTS.icc;
      if (!(icc >= 0 && icc < 1)) {
        errors.push(`grouping '${t.cluster}': icc out of range [0, 1): ${icc}`);
      } else if (icc !== 0 && (icc < iccLo || icc > iccHi)) {
        errors.push(`grouping '${t.cluster}': icc ${icc} outside the stable band [${iccLo}, ${iccHi}] (use 0 for no clustering)`);
      }
      if (t.parent !== null) {
        const n = g?.n ?? EXTRA_GROUPING_DEFAULTS.nestedN;
        return {
          tau_squared: iccToTau(icc),
          relation: { kind: 'nested_within' as const, n_per_parent: n },
          cluster_name: t.cluster,
        };
      }
      const n = g?.n ?? EXTRA_GROUPING_DEFAULTS.crossedN;
      if (n < 2) errors.push(`crossed grouping '${t.cluster}' needs at least 2 clusters`);
      return {
        tau_squared: iccToTau(icc),
        relation: { kind: 'crossed' as const, n_clusters: n },
        cluster_name: t.cluster,
      };
    });

    // Slopes are formula-driven: the primary term's `(1+x|g)` vars ARE the
    // slope set; cl.slopes (kept mirrored by ClusterEditor) only contributes
    // the variance/corr parameters. Carried as predictor NAMES; the Rust
    // assembler resolves name → non-factor column — so each var must be a
    // fixed predictor too, or the engine errors with an unfriendly message.
    const slopeCfgByName = new Map((cl.slopes ?? []).map((s) => [s.predictorName, s]));
    const slopeNames = [...new Set(primary.slopeVars)];
    const varKindByName = new Map(config.variables.map((v) => [v.name, v.kind]));
    const predictorSet = new Set(parsed.predictors);
    for (const n of slopeNames) {
      if (!predictorSet.has(n)) {
        errors.push(
          `random slope on '${n}': add it as a fixed predictor too (e.g. y = ${n} + … + (1+${n}|${cluster_name}))`,
        );
      } else if (varKindByName.get(n) === 'factor') {
        errors.push(`random slope on '${n}': slopes need a continuous (non-factor) predictor`);
      }
    }
    const slopes: AppSlopeTerm[] = slopeNames.map((n) => {
      const s = slopeCfgByName.get(n);
      return {
        predictor_name: n,
        slope_variance: s?.slopeVariance ?? SLOPE_DEFAULTS.variance,
        slope_intercept_corr: s?.slopeInterceptCorr ?? SLOPE_DEFAULTS.corr,
      };
    });

    // Absent → Rust #[serde(default)] = Gaussian; UI toggle is user-owned.
    let outcome: MixedSpec['outcome'];
    if (binaryOutcome) {
      const p = cl.baselineProbability;
      if (typeof p !== 'number' || p <= 0 || p >= 1) {
        errors.push(`binary mixed: baseline_probability out of range (0, 1): ${p}`);
      } else {
        warnBaselineProbability(p, warnings);
        outcome = { kind: 'binary', baseline_probability: p };
      }
    }

    const outcome_options = projectOutcomeOptions(config, !binaryOutcome, errors);
    const spec: AppSpec = {
      family: 'mixed',
      ...ok,
      cluster_name,
      icc: cl.icc,
      cluster_dim,
      cluster_level_vars: cl.clusterLevelVars ?? [],
      extra_groupings,
      slopes,
      ...(outcome !== undefined ? { outcome } : {}),
      ...(outcome_options ? { outcome_options } : {}),
    } satisfies { family: 'mixed' } & MixedSpec;
    return { spec, errors, warnings };
  }

  return { spec: null, errors: [`unhandled entrypoint: ${entrypoint}`], warnings: [] };
}
