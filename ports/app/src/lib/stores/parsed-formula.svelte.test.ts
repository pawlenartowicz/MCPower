import { describe, expect, it, vi } from 'vitest';

// Real getStable reaches parseFormula (no Tauri runtime in vitest) — stub it so
// the empty-formula clear can be exercised against the real store logic.
vi.mock('$lib/api/engine', () => ({
  parseFormula: vi.fn(async () => ({
    dependent: 'y',
    predictors: ['x'],
    terms: [{ kind: 'main', name: 'x' }],
    random_effects: [],
  })),
}));

import {
  type FormulaParse,
  parsedFormulaStore,
  toClusterTerms,
  toWireParsed,
} from './parsed-formula.svelte';

const fp: FormulaParse = {
  dependent: 'y',
  predictors: ['x1', 'x2'],
  terms: [
    { kind: 'main', name: 'x1' },
    { kind: 'main', name: 'x2' },
    { kind: 'interaction', vars: ['x1', 'x2'] },
  ],
  random_effects: [{ kind: 'intercept', group: 'g', parent: null }],
};

describe('parsed-formula derivations', () => {
  it('maps to the wire ParsedFormula', () => {
    expect(toWireParsed(fp)).toEqual({
      outcome: 'y',
      predictors: ['x1', 'x2'],
      interaction_terms: [['x1', 'x2']],
    });
  });
  it('extracts an intercept RE as a cluster term', () => {
    expect(toClusterTerms(fp)).toEqual([{ cluster: 'g', parent: null, slopeVars: [] }]);
  });
  it('maps nested and slope REs to structured cluster terms', () => {
    const multi: FormulaParse = {
      ...fp,
      random_effects: [
        { kind: 'intercept', group: 'site', parent: null },
        { kind: 'intercept', group: 'site:class', parent: 'site' },
        { kind: 'slope', group: 'item', vars: ['x1'] },
      ],
    };
    expect(toClusterTerms(multi)).toEqual([
      { cluster: 'site', parent: null, slopeVars: [] },
      { cluster: 'site:class', parent: 'site', slopeVars: [] },
      { cluster: 'item', parent: null, slopeVars: ['x1'] },
    ]);
  });
});

describe('getStable empty-formula clear', () => {
  it('returns the empty state for an empty formula even after a good parse', async () => {
    // Prime: trigger a parse, wait for it to resolve, then re-read so getStable
    // captures it as lastGood.
    parsedFormulaStore.getStable('y ~ x');
    await vi.waitFor(() => expect(parsedFormulaStore.get('y ~ x').result).not.toBeNull());
    expect(parsedFormulaStore.getStable('y ~ x').result).not.toBeNull();

    // Empty / whitespace formula must collapse to empty — NOT revive lastGood
    // (defect #3: the stale parse otherwise survives Reset/clear).
    expect(parsedFormulaStore.getStable('').result).toBeNull();
    expect(parsedFormulaStore.getStable('   ').result).toBeNull();

    // A subsequent non-empty good parse still resolves normally.
    expect(parsedFormulaStore.getStable('y ~ x').result).not.toBeNull();
  });
});
