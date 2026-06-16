import { describe, expect, it, vi } from 'vitest';

vi.mock('$lib/stores/parsed-formula.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/stores/parsed-formula.svelte')>();
  const { stubParseFormula } = await import('../../tests/parse-formula-stub');
  return {
    ...actual,
    parsedFormulaStore: {
      get: (formula: string) => stubParseFormula(formula),
      getStable: (formula: string) => stubParseFormula(formula),
    },
  };
});

import canonical from '$configs/formula-fixtures/canonical-suite.json';
import { toWireParsed } from '$lib/stores/parsed-formula.svelte';
import { contrastNames, effectGroups, effectNames, effectRows } from './effect-names';
import { defaultFamilyConfig } from './family';

describe('effectNames', () => {
  it('generates numeric placeholder level names without L prefix', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ group',
      variables: [{ kind: 'factor' as const, name: 'group', nLevels: 3 }],
    };
    expect(effectNames(cfg)).toEqual(['group[1]', 'group[2]', 'group[3]']);
  });

  it('uses explicit level names when provided', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ origin',
      variables: [{ kind: 'factor' as const, name: 'origin', levels: ['Japan', 'USA'] }],
    };
    expect(effectNames(cfg)).toEqual(['origin[Japan]', 'origin[USA]']);
  });
});

describe('contrastNames', () => {
  it('generates all pairwise contrasts for numeric placeholder levels', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ group',
      variables: [{ kind: 'factor' as const, name: 'group', nLevels: 3 }],
    };
    expect(contrastNames(cfg)).toEqual([
      'group[1] − group[2]',
      'group[1] − group[3]',
      'group[2] − group[3]',
    ]);
  });
});

describe('effectRows', () => {
  it('marks first level as reference, rest as non-reference, reference first', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ treatment',
      variables: [{ kind: 'factor' as const, name: 'treatment', nLevels: 3 }],
    };
    expect(effectRows(cfg)).toEqual([
      { name: 'treatment[1]', isReference: true },
      { name: 'treatment[2]', isReference: false },
      { name: 'treatment[3]', isReference: false },
    ]);
  });

  it('uses explicit referenceLevel when set', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ origin',
      variables: [
        {
          kind: 'factor' as const,
          name: 'origin',
          levels: ['Japan', 'USA', 'Europe'],
          referenceLevel: 'USA',
        },
      ],
    };
    expect(effectRows(cfg)).toEqual([
      { name: 'origin[USA]', isReference: true },
      { name: 'origin[Japan]', isReference: false },
      { name: 'origin[Europe]', isReference: false },
    ]);
  });

  it('marks continuous predictors as non-reference', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ age',
      variables: [{ kind: 'continuous' as const, name: 'age' }],
    };
    expect(effectRows(cfg)).toEqual([{ name: 'age', isReference: false }]);
  });

  it('includes interaction terms as non-reference', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ a + b + a:b',
      variables: [
        { kind: 'continuous' as const, name: 'a' },
        { kind: 'continuous' as const, name: 'b' },
      ],
    };
    const rows = effectRows(cfg);
    expect(rows).toContainEqual({ name: 'a:b', isReference: false });
  });

  it('for anova family, emits all factor levels without interaction terms', () => {
    const cfg = {
      ...defaultFamilyConfig('anova'),
      formula: '',
      variables: [{ kind: 'factor' as const, name: 'group', nLevels: 2 }],
    };
    expect(effectRows(cfg)).toEqual([
      { name: 'group[1]', isReference: true },
      { name: 'group[2]', isReference: false },
    ]);
  });
});

describe('factor interaction expansion', () => {
  it('effectNames: continuous×factor expands to the per-level Cartesian product (non-reference only)', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + group + x1:group',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'factor' as const, name: 'group', nLevels: 3 },
      ],
    };
    expect(effectNames(cfg)).toEqual([
      'x1',
      'group[1]',
      'group[2]',
      'group[3]',
      'x1:group[2]',
      'x1:group[3]',
    ]);
  });

  it('effectNames: factor×factor expands to the Cartesian of non-reference levels', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ f + g + f:g',
      variables: [
        { kind: 'factor' as const, name: 'f', nLevels: 3 },
        { kind: 'factor' as const, name: 'g', nLevels: 3 },
      ],
    };
    expect(effectNames(cfg)).toEqual([
      'f[1]',
      'f[2]',
      'f[3]',
      'g[1]',
      'g[2]',
      'g[3]',
      'f[2]:g[2]',
      'f[2]:g[3]',
      'f[3]:g[2]',
      'f[3]:g[3]',
    ]);
  });

  it('effectNames: continuous×continuous interaction stays a single coarse name', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + x2 + x1:x2',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'continuous' as const, name: 'x2' },
      ],
    };
    expect(effectNames(cfg)).toEqual(['x1', 'x2', 'x1:x2']);
  });

  it('effectRows: factor interaction rows are per-level and non-reference', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + group + x1:group',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'factor' as const, name: 'group', nLevels: 3 },
      ],
    };
    const rows = effectRows(cfg);
    expect(rows).toContainEqual({ name: 'x1:group[2]', isReference: false });
    expect(rows).toContainEqual({ name: 'x1:group[3]', isReference: false });
    expect(rows.some((r) => r.name === 'x1:group')).toBe(false);
  });

  it('effectNames: honours referenceLevel when expanding factor interactions', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + origin + x1:origin',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        {
          kind: 'factor' as const,
          name: 'origin',
          levels: ['Japan', 'USA', 'Europe'],
          referenceLevel: 'USA',
        },
      ],
    };
    expect(effectNames(cfg)).toEqual([
      'x1',
      'origin[Japan]',
      'origin[USA]',
      'origin[Europe]',
      'x1:origin[Japan]',
      'x1:origin[Europe]',
    ]);
  });
});

describe('effectGroups', () => {
  it('groups continuous and binary predictors as single bare non-reference rows', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + x2',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'binary' as const, name: 'x2' },
      ],
    };
    const g = effectGroups(cfg);
    expect(g.variables).toEqual([
      { name: 'x1', kind: 'continuous', rows: [{ name: 'x1', isReference: false }] },
      { name: 'x2', kind: 'binary', rows: [{ name: 'x2', isReference: false }] },
    ]);
    expect(g.interactions).toEqual([]);
  });

  it('groups a factor with its reference level first', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ treatment',
      variables: [{ kind: 'factor' as const, name: 'treatment', nLevels: 3 }],
    };
    const g = effectGroups(cfg);
    expect(g.variables).toEqual([
      {
        name: 'treatment',
        kind: 'factor',
        rows: [
          { name: 'treatment[1]', isReference: true },
          { name: 'treatment[2]', isReference: false },
          { name: 'treatment[3]', isReference: false },
        ],
      },
    ]);
  });

  it('flags a continuous×factor interaction and expands it to per-level rows', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + group + x1:group',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'factor' as const, name: 'group', nLevels: 3 },
      ],
    };
    const g = effectGroups(cfg);
    expect(g.variables.map((v) => v.name)).toEqual(['x1', 'group']);
    expect(g.interactions).toEqual([
      {
        term: 'x1:group',
        isFactorInteraction: true,
        rows: [
          { name: 'x1:group[2]', isReference: false },
          { name: 'x1:group[3]', isReference: false },
        ],
      },
    ]);
  });

  it('keeps a continuous×continuous interaction a single un-flagged row', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1 + x2 + x1:x2',
      variables: [
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'continuous' as const, name: 'x2' },
      ],
    };
    const g = effectGroups(cfg);
    expect(g.interactions).toEqual([
      {
        term: 'x1:x2',
        isFactorInteraction: false,
        rows: [{ name: 'x1:x2', isReference: false }],
      },
    ]);
  });

  it('expands a factor×factor interaction to the non-reference Cartesian product', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ f + g + f:g',
      variables: [
        { kind: 'factor' as const, name: 'f', nLevels: 3 },
        { kind: 'factor' as const, name: 'g', nLevels: 3 },
      ],
    };
    const g = effectGroups(cfg);
    expect(g.interactions).toEqual([
      {
        term: 'f:g',
        isFactorInteraction: true,
        rows: [
          { name: 'f[2]:g[2]', isReference: false },
          { name: 'f[2]:g[3]', isReference: false },
          { name: 'f[3]:g[2]', isReference: false },
          { name: 'f[3]:g[3]', isReference: false },
        ],
      },
    ]);
  });
});

// Only include fixed-effect cases where all main terms precede all interaction
// terms in the canonical fixed_effects list (i.e. no interleaving).
// toWireParsed splits terms into predictors + interaction_terms; the derived
// reconstruction below only matches canonical when that ordering holds.
const fixedCases = (canonical as any).cases.filter((c: any) => {
  if (!c.expected || c.expected.random_effects.length !== 0) return false;
  const fe: string[] = c.expected.fixed_effects;
  const firstInteraction = fe.findIndex((f: string) => f.includes(':'));
  if (firstInteraction === -1) return true; // no interactions — always fine
  return fe.slice(firstInteraction).every((f: string) => f.includes(':')); // mains first
});

describe('canonical fixed-effect derivation', () => {
  for (const c of fixedCases) {
    it(`${c.id}: interaction strings + predictors match canonical`, () => {
      const terms = c.expected.fixed_effects.map((fe: string) =>
        fe.includes(':')
          ? { kind: 'interaction', vars: fe.split(':') }
          : { kind: 'main', name: fe },
      );
      const wire = toWireParsed({
        dependent: c.expected.outcome,
        predictors: [],
        terms,
        random_effects: [],
      } as any);
      const derived = [
        ...c.expected.fixed_effects.filter((f: string) => !f.includes(':')),
        ...wire.interaction_terms.map((v: string[]) => v.join(':')),
      ];
      expect(derived).toEqual(c.expected.fixed_effects);
    });
  }
});
