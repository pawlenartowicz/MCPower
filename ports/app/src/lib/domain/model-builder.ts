// Pure logic for the Visual Model Builder. Builds/reads the fixed-effects formula
// only (predictors + pairwise interactions). N-way interactions are NOT
// representable here — they are detected and carried through verbatim, never
// dropped (the headline data-loss guard).

import { factorLevels } from '$lib/domain/effect-names';
import type { VariableRow } from '$lib/domain/family';
import type { FormulaParse } from '$lib/domain/result';

export type PredictorKind = 'continuous' | 'binary' | 'factor';

// A fresh factor starts at 3 levels (the screenshot/demo default) so a factor is
// immediately a multi-level contrast, not a relabeled binary.
export const FACTOR_DEFAULT_LEVELS = ['1', '2', '3'] as const;

export interface BuilderPredictor {
  name: string;
  kind: PredictorKind;
  levels: string[]; // factor only; [] otherwise
  dataBacked: boolean; // true => levels owned by uploaded CSV, lock the UI
}
export interface BuilderInteraction {
  a: string;
  b: string;
}
export interface BuilderState {
  dependent: string;
  predictors: BuilderPredictor[];
  interactions: BuilderInteraction[];
  carried: string[]; // ≥3-way terms passed through unchanged
}

// Interaction terms the pairwise builder cannot round-trip (≥3 vars).
export function detectUnrepresentable(parse: FormulaParse | null): string[] {
  if (!parse) return [];
  return parse.terms
    .filter((t): t is { kind: 'interaction'; vars: string[] } => t.kind === 'interaction')
    .filter((t) => t.vars.length >= 3)
    .map((t) => t.vars.join(':'));
}

export function hydrateBuilder(parse: FormulaParse | null, variables: VariableRow[]): BuilderState {
  if (!parse) return { dependent: 'y', predictors: [], interactions: [], carried: [] };
  const byName = new Map(variables.map((v) => [v.name, v]));
  const predictors: BuilderPredictor[] = parse.predictors.map((name) => {
    const v = byName.get(name);
    const kind: PredictorKind = v?.kind ?? 'continuous';
    return { name, kind, levels: kind === 'factor' ? factorLevels(v) : [], dataBacked: false };
  });
  const interactions: BuilderInteraction[] = parse.terms
    .filter((t): t is { kind: 'interaction'; vars: string[] } => t.kind === 'interaction')
    .filter((t) => t.vars.length === 2)
    .map((t) => ({ a: t.vars[0]!, b: t.vars[1]! }));
  return {
    dependent: parse.dependent,
    predictors,
    interactions,
    carried: detectUnrepresentable(parse),
  };
}

// Emits ':' only (normalizes a*b -> a + b + a:b). Round-trip is semantic, not
// textual. Carried ≥3-way terms are appended verbatim.
export function assembleFormula(state: BuilderState): string {
  const rhs: string[] = [
    ...state.predictors.map((p) => p.name),
    ...state.interactions.map((i) => `${i.a}:${i.b}`),
    ...state.carried,
  ];
  return `${state.dependent} ~ ${rhs.length ? rhs.join(' + ') : '1'}`;
}

// Merge builder predictors into the existing variable rows by name, so "Use this
// model" writes the chosen kinds/levels directly instead of waiting on the async
// formula parse (which defaults every variable to continuous). Factor rows get
// levels/nLevels and referenceLevel = first level; non-factor rows clear those.
// All other fields (distribution, proportions, pinned, …) survive for a row whose
// name is unchanged — only kind + factor coding are authoritative from the builder.
export function applyBuilderVariables(state: BuilderState, existing: VariableRow[]): VariableRow[] {
  const byName = new Map(existing.map((v) => [v.name, v]));
  return state.predictors.map((p): VariableRow => {
    const row: VariableRow = { ...byName.get(p.name), name: p.name, kind: p.kind };
    if (p.kind === 'factor') {
      row.levels = [...p.levels];
      row.nLevels = p.levels.length;
      row.referenceLevel = p.levels[0];
    } else {
      delete row.levels;
      delete row.nLevels;
      delete row.referenceLevel;
    }
    return row;
  });
}

const FORMULA_OPERATORS = /[:*+~\s]/;
export function validatePredictorName(name: string, others: string[]): string | null {
  if (name.trim() === '') return 'Name cannot be empty.';
  if (FORMULA_OPERATORS.test(name))
    return 'Name cannot contain formula operators (: * + ~) or whitespace.';
  if (others.includes(name)) return `Duplicate name: ${name}.`;
  return null;
}
