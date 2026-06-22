// Derives displayable effect names, reference-level rows, grouped predictor
// structure, contrast labels, effect-size presets, and the cfg.effects reconcile
// from a FamilyConfig and its parsed formula.
// NOTE: expandMainEffect/expandInteraction are the single source of effect-name derivation shared by UI rows and the engine adapter; a change here changes both surfaces.

import { BENCHMARKS } from '$lib/configs/app-config';
import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
import type { EffectRow, FamilyConfig, VariableKind, VariableRow } from './family';

export function factorLevels(v: VariableRow | undefined): string[] {
  if (!v || v.kind !== 'factor') return [];
  const k = typeof v.nLevels === 'number' && v.nLevels >= 2 ? Math.floor(v.nLevels) : 0;
  if (Array.isArray(v.levels) && v.levels.length > 0) {
    // Per-slot fallback for blank labels — the single source of label
    // resolution shared by effect rows, the popup, and the engine adapter
    // (which sends these as the engine's level names).
    const n = Math.max(v.levels.length, k);
    return Array.from({ length: n }, (_, i) => {
      const l = v.levels?.[i];
      return l && l.trim() !== '' ? l : `${i + 1}`;
    });
  }
  return Array.from({ length: k }, (_, i) => `${i + 1}`);
}

// Resolve a factor's reference against the canonicalized `levels` (factorLevels
// output). A blank `referenceLevel` — which an uploaded column produces when its
// baseline cell is empty (PredictorCards stores the raw '' label) — is treated as
// unset and defaults to the first level. Mirrors the engine adapter
// (app-spec-adapter.ts), whose `factor_reference` falls back to index 0 for a
// blank reference. Plain `?? levels[0]` keeps the literal '', which matches no
// canonicalized level (blank slots are rewritten to their 1-based index), so the
// filter drops nothing — an extra dummy and an effect-count mismatch against the
// adapter. Change together with app-spec-adapter.ts's blank guard.
function resolveReference(v: VariableRow | undefined, levels: string[]): string {
  const ref = v?.referenceLevel;
  return ref && ref.trim() !== '' ? ref : (levels[0] ?? '1');
}

// Per-level main-effect names for one predictor — the single source of factor
// expansion shared by the effects UI (effect-names.ts) and the engine adapter
// (app-spec-adapter.ts). A factor expands to `name[level]` dummies; a non-factor
// stays its bare name. `includeReference` distinguishes the two callers: UI rows
// list every level, but engine *targets* drop the reference (matching the
// engine's dummy coding, where the reference level has no column).
export function expandMainEffect(
  v: VariableRow | undefined,
  name: string,
  includeReference: boolean,
): string[] {
  const levels = factorLevels(v);
  if (levels.length === 0) return [name];
  const ref = resolveReference(v, levels);
  const kept = includeReference ? levels : levels.filter((lv) => lv !== ref);
  return kept.map((lv) => `${name}[${lv}]`);
}

// Per-level interaction names for one interaction term — the Cartesian product
// of each component's non-reference dummies, mirroring the engine's
// `build_predictor_table` interaction branch (and Python/R `expand_factors`):
// factor components contribute their non-reference levels (`v[level]`),
// non-factor components their bare name, the first var varying slowest.
// Continuous×continuous collapses to the single coarse name (`x1:x2`); a factor
// with no non-reference levels drops the whole term (no column exists).
export function expandInteraction(vars: string[], byName: Map<string, VariableRow>): string[] {
  let combos: string[][] = [[]];
  for (const v of vars) {
    const options = expandMainEffect(byName.get(v), v, false);
    combos = combos.flatMap((prefix) => options.map((opt) => [...prefix, opt]));
  }
  return combos.map((parts) => parts.join(':'));
}

// Both readers use getStable: effect rows must not collapse (and reconcileEffects
// must not wipe values) during the parse round-trip after a formula edit.
function interactionNames(cfg: FamilyConfig, byName: Map<string, VariableRow>): string[] {
  const r = parsedFormulaStore.getStable(cfg.formula).result;
  if (!r) return [];
  return r.terms
    .filter((t) => t.kind === 'interaction')
    .flatMap((t) => expandInteraction((t as { vars: string[] }).vars, byName));
}

function predictorNames(cfg: FamilyConfig): string[] {
  if (cfg.family === 'anova') return cfg.variables.map((v) => v.name);
  return parsedFormulaStore.getStable(cfg.formula).result?.predictors ?? [];
}

export function effectNames(cfg: FamilyConfig): string[] {
  const byName = new Map(cfg.variables.map((v) => [v.name, v]));
  const out: string[] = [];
  for (const name of predictorNames(cfg)) {
    out.push(...expandMainEffect(byName.get(name), name, true));
  }
  if (cfg.family !== 'anova') {
    out.push(...interactionNames(cfg, byName));
  }
  return out;
}

export interface EffectRowMeta {
  name: string;
  isReference: boolean;
}

// A predictor's effect rows: continuous/binary collapse to one bare row; a factor
// lists its reference level first, then the non-reference dummies.
function variableRows(name: string, v: VariableRow | undefined): EffectRowMeta[] {
  const levels = factorLevels(v);
  if (levels.length === 0) return [{ name, isReference: false }];
  const ref = resolveReference(v, levels);
  return [
    { name: `${name}[${ref}]`, isReference: true },
    ...levels
      .filter((lv) => lv !== ref)
      .map((lv) => ({ name: `${name}[${lv}]`, isReference: false })),
  ];
}

export function effectRows(cfg: FamilyConfig): EffectRowMeta[] {
  const g = effectGroups(cfg);
  return [...g.variables.flatMap((v) => v.rows), ...g.interactions.flatMap((i) => i.rows)];
}

// ---------------------------------------------------------------------------
// Grouped projection — one entry per variable, then one per interaction term.
// The card UI (PredictorCards) consumes this instead of the flat effectRows so
// the grouping (and the factor-interaction flag that drives the hint) lives here
// next to the name derivation rather than being string-sniffed in the component.
// ---------------------------------------------------------------------------

export interface VariableGroup {
  name: string;
  kind: VariableKind;
  rows: EffectRowMeta[];
}
export interface InteractionGroup {
  term: string;
  isFactorInteraction: boolean;
  rows: EffectRowMeta[];
}
export interface EffectGroups {
  variables: VariableGroup[];
  interactions: InteractionGroup[];
}

export function effectGroups(cfg: FamilyConfig): EffectGroups {
  const byName = new Map(cfg.variables.map((v) => [v.name, v]));
  const variables: VariableGroup[] = predictorNames(cfg).map((name) => {
    const v = byName.get(name);
    return { name, kind: v?.kind ?? 'continuous', rows: variableRows(name, v) };
  });

  const interactions: InteractionGroup[] = [];
  if (cfg.family !== 'anova') {
    const r = parsedFormulaStore.getStable(cfg.formula).result;
    for (const t of r?.terms ?? []) {
      if (t.kind !== 'interaction') continue;
      const vars = t.vars;
      interactions.push({
        term: vars.join(':'),
        isFactorInteraction: vars.some((vn) => factorLevels(byName.get(vn)).length > 0),
        rows: expandInteraction(vars, byName).map((name) => ({ name, isReference: false })),
      });
    }
  }
  return { variables, interactions };
}

// Reconciles cfg.effects to the desired ordered name list, preserving existing
// values by name and defaulting new names to 0. Shared by every formula-family
// editor (PredictorCards for regression/mixed, AnovaFactorEditor for ANOVA) so
// exactly one copy of the reconcile runs per family. Call inside an untrack().
export function reconcileEffects(cfg: FamilyConfig, names: string[]): void {
  const byName = new Map(cfg.effects.map((e) => [e.name, e]));
  const current = cfg.effects.map((e) => e.name);
  const matches = current.length === names.length && current.every((n, i) => n === names[i]);
  if (matches) return;
  cfg.effects = names.map((name): EffectRow => byName.get(name) ?? { name, value: 0 });
}

// ---------------------------------------------------------------------------
// Effect-size presets (Cohen benchmarks). A `name[level]` row is a factor dummy
// and a factor-involved interaction also carries `[`; both use the binary/factor
// convention. A bare `a:b` is a continuous interaction; a plain name maps to its
// variable's kind.
// ---------------------------------------------------------------------------

export interface Preset {
  short: string;
  long: string;
  value: number;
}

// Continuous predictors (and continuous interactions): small/medium/large.
const CONTINUOUS_PRESETS: readonly Preset[] = [
  { short: 'S', long: 'small', value: BENCHMARKS.continuous[0]! },
  { short: 'M', long: 'medium', value: BENCHMARKS.continuous[1]! },
  { short: 'L', long: 'large', value: BENCHMARKS.continuous[2]! },
];
// Binary / factor predictors (and factor interactions): Cohen's convention.
const COHEN_PRESETS: readonly Preset[] = [
  { short: 'S', long: 'small', value: BENCHMARKS.binary_factor[0]! },
  { short: 'M', long: 'medium', value: BENCHMARKS.binary_factor[1]! },
  { short: 'L', long: 'large', value: BENCHMARKS.binary_factor[2]! },
];

export function presetsFor(name: string, variables: VariableRow[]): readonly Preset[] {
  if (name.includes('[')) return COHEN_PRESETS;
  if (name.includes(':')) return CONTINUOUS_PRESETS;
  const v = variables.find((x) => x.name === name);
  return v && (v.kind === 'binary' || v.kind === 'factor') ? COHEN_PRESETS : CONTINUOUS_PRESETS;
}

export function contrastNames(cfg: FamilyConfig): string[] {
  const byName = new Map(cfg.variables.map((v) => [v.name, v]));
  const out: string[] = [];
  for (const name of predictorNames(cfg)) {
    const v = byName.get(name);
    // ANOVA auto-contrasts only its primary factors, never covariate-factors.
    if (cfg.family === 'anova' && v?.role === 'covariate') continue;
    const levels = factorLevels(v);
    if (levels.length < 2) continue;
    for (let i = 0; i < levels.length; i++) {
      for (let j = i + 1; j < levels.length; j++) {
        out.push(`${name}[${levels[i]}] − ${name}[${levels[j]}]`);
      }
    }
  }
  return out;
}
