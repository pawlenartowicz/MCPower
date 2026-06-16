import { describe, expect, it } from 'vitest';
import { toWireParsed, toClusterTerms, type FormulaParse } from './parsed-formula.svelte';

const fp: FormulaParse = {
  dependent: 'y',
  predictors: ['x1', 'x2'],
  terms: [
    { kind: 'main', name: 'x1' }, { kind: 'main', name: 'x2' },
    { kind: 'interaction', vars: ['x1', 'x2'] },
  ],
  random_effects: [{ kind: 'intercept', group: 'g', parent: null }],
};

describe('parsed-formula derivations', () => {
  it('maps to the wire ParsedFormula', () => {
    expect(toWireParsed(fp)).toEqual({ outcome: 'y', predictors: ['x1', 'x2'], interaction_terms: [['x1', 'x2']] });
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
