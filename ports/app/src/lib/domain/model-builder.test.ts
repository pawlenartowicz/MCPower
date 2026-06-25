import { describe, it, expect } from 'vitest';
import type { FormulaParse } from '$lib/domain/result';
import type { VariableRow } from '$lib/domain/family';
import {
  detectUnrepresentable,
  hydrateBuilder,
  assembleFormula,
  validatePredictorName,
  applyBuilderVariables,
  FACTOR_DEFAULT_LEVELS,
  type BuilderState,
} from './model-builder';

const state = (predictors: BuilderState['predictors']): BuilderState => ({
  dependent: 'y',
  predictors,
  interactions: [],
  carried: [],
});

const parse = (
  dependent: string,
  predictors: string[],
  terms: FormulaParse['terms'],
): FormulaParse => ({ dependent, predictors, terms, random_effects: [] });

describe('detectUnrepresentable', () => {
  it('flags 3-way interactions, ignores main + pairwise', () => {
    const p = parse('y', ['a', 'b', 'c'], [
      { kind: 'main', name: 'a' },
      { kind: 'interaction', vars: ['a', 'b'] },
      { kind: 'interaction', vars: ['a', 'b', 'c'] },
    ]);
    expect(detectUnrepresentable(p)).toEqual(['a:b:c']);
  });

  it('returns [] for null parse', () => {
    expect(detectUnrepresentable(null)).toEqual([]);
  });
});

describe('hydrate + assemble round-trip (semantic, canonical inputs only)', () => {
  it('main effects only', () => {
    const p = parse('y', ['a', 'b'], [
      { kind: 'main', name: 'a' },
      { kind: 'main', name: 'b' },
    ]);
    const state = hydrateBuilder(p, []);
    expect(state.dependent).toBe('y');
    expect(state.predictors.map((x) => x.name)).toEqual(['a', 'b']);
    expect(assembleFormula(state)).toBe('y ~ a + b');
  });

  it('factor with levels + pairwise interaction', () => {
    const vars: VariableRow[] = [
      { name: 'dose', kind: 'factor', levels: ['lo', 'hi'], nLevels: 2 },
      { name: 'age', kind: 'continuous' },
    ];
    const p = parse('resp', ['dose', 'age'], [
      { kind: 'main', name: 'dose' },
      { kind: 'main', name: 'age' },
      { kind: 'interaction', vars: ['dose', 'age'] },
    ]);
    const state = hydrateBuilder(p, vars);
    expect(state.predictors.find((x) => x.name === 'dose')?.kind).toBe('factor');
    expect(state.predictors.find((x) => x.name === 'dose')?.levels).toEqual(['lo', 'hi']);
    expect(assembleFormula(state)).toBe('resp ~ dose + age + dose:age');
  });

  it('carries un-representable 3-way terms through unchanged', () => {
    const p = parse('y', ['a', 'b', 'c'], [
      { kind: 'main', name: 'a' },
      { kind: 'main', name: 'b' },
      { kind: 'main', name: 'c' },
      { kind: 'interaction', vars: ['a', 'b', 'c'] },
    ]);
    const state = hydrateBuilder(p, []);
    expect(state.carried).toEqual(['a:b:c']);
    expect(assembleFormula(state)).toBe('y ~ a + b + c + a:b:c');
  });

  it('no predictors assembles an intercept-only formula', () => {
    const state = hydrateBuilder(parse('y', [], []), []);
    expect(assembleFormula(state)).toBe('y ~ 1');
  });
});

describe('applyBuilderVariables', () => {
  it('writes kind + factor levels/nLevels/referenceLevel (first level is the reference)', () => {
    const s = state([
      { name: 'g', kind: 'factor', levels: ['x', 'y', 'z'], dataBacked: false },
      { name: 'age', kind: 'continuous', levels: [], dataBacked: false },
    ]);
    const rows = applyBuilderVariables(s, []);
    const g = rows.find((r) => r.name === 'g')!;
    expect(g.kind).toBe('factor');
    expect(g.levels).toEqual(['x', 'y', 'z']);
    expect(g.nLevels).toBe(3);
    expect(g.referenceLevel).toBe('x');
    expect(rows.find((r) => r.name === 'age')?.kind).toBe('continuous');
  });

  it('preserves unrelated fields for an unchanged name and clears factor fields when no longer a factor', () => {
    const existing: VariableRow[] = [
      { name: 'age', kind: 'continuous', distribution: 'right_skewed', pinned: true },
      { name: 'g', kind: 'factor', levels: ['x', 'y'], nLevels: 2, referenceLevel: 'x' },
    ];
    const s = state([
      { name: 'age', kind: 'continuous', levels: [], dataBacked: false },
      { name: 'g', kind: 'binary', levels: [], dataBacked: false },
    ]);
    const rows = applyBuilderVariables(s, existing);
    const age = rows.find((r) => r.name === 'age')!;
    expect(age.distribution).toBe('right_skewed');
    expect(age.pinned).toBe(true);
    const g = rows.find((r) => r.name === 'g')!;
    expect(g.kind).toBe('binary');
    expect(g.levels).toBeUndefined();
    expect(g.nLevels).toBeUndefined();
    expect(g.referenceLevel).toBeUndefined();
  });

  it('exposes a 3-level factor default', () => {
    expect(FACTOR_DEFAULT_LEVELS).toEqual(['1', '2', '3']);
  });
});

describe('validatePredictorName', () => {
  it('rejects empty, operator-containing, and duplicate names; accepts clean', () => {
    expect(validatePredictorName('', [])).toMatch(/empty/i);
    expect(validatePredictorName('a:b', [])).toMatch(/operator|character/i);
    expect(validatePredictorName('a b', [])).toMatch(/operator|character|whitespace/i);
    expect(validatePredictorName('x', ['x'])).toMatch(/duplicate/i);
    expect(validatePredictorName('x', ['y'])).toBeNull();
  });
});
