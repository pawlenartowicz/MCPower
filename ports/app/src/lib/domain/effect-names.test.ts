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
import {
  contrastNames,
  effectGroups,
  effectNames,
  effectRows,
  presetsFor,
  reconcileTestSelection,
} from './effect-names';
import { defaultFamilyConfig, type VariableRow } from './family';
import { BENCHMARKS } from '$lib/configs/app-config';

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

describe('presetsFor', () => {
  const cont = [{ kind: 'continuous' as const, name: 'x' }];
  const fac = [{ kind: 'factor' as const, name: 'g', nLevels: 3 }];

  it('returns the odds (beta) set for every predictor when the outcome is logit', () => {
    const vals = (rows: VariableRow[], name: string) =>
      presetsFor(name, rows, true).map((p) => p.value);
    // Same odds triple regardless of predictor kind — it replaces both rows.
    expect(vals(cont, 'x')).toEqual(BENCHMARKS.odds);
    expect(vals(fac, 'g[2]')).toEqual(BENCHMARKS.odds);
    expect(vals([], 'x1:x2')).toEqual(BENCHMARKS.odds);
  });

  it('keeps the continuous/Cohen split when the outcome is not logit', () => {
    expect(presetsFor('x', cont, false).map((p) => p.value)).toEqual(BENCHMARKS.continuous);
    expect(presetsFor('g[2]', fac, false).map((p) => p.value)).toEqual(BENCHMARKS.binary_factor);
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

  it('resolves a blank reference label the same way factorLevels canonicalizes it', () => {
    // Uploaded factor with an empty-string baseline cell (e.g. a missing `sex`
    // value): the column's first label is '' and PredictorCards sets
    // referenceLevel to that empty string. factorLevels rewrites the blank slot
    // to its 1-based index ('1'), so the reference must resolve to '1' — not be
    // taken literally as '' (which matches no level and drops nothing, yielding
    // an extra dummy and an effect-count mismatch against the engine adapter).
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ sex',
      variables: [
        {
          kind: 'factor' as const,
          name: 'sex',
          nLevels: 3,
          levels: ['', 'f', 'm'],
          referenceLevel: '',
        },
      ],
    };
    expect(effectRows(cfg)).toEqual([
      { name: 'sex[1]', isReference: true },
      { name: 'sex[f]', isReference: false },
      { name: 'sex[m]', isReference: false },
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
      {
        name: 'x1',
        kind: 'continuous',
        interactionOnly: false,
        rows: [{ name: 'x1', isReference: false }],
      },
      {
        name: 'x2',
        kind: 'binary',
        interactionOnly: false,
        rows: [{ name: 'x2', isReference: false }],
      },
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
        interactionOnly: false,
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

describe('empty / cleared formula', () => {
  it('collapses effectNames and effectGroups to empty (Reset/clear)', () => {
    const cfg = { ...defaultFamilyConfig('regression'), formula: '' };
    expect(effectNames(cfg)).toEqual([]);
    const g = effectGroups(cfg);
    expect(g.variables).toEqual([]);
    expect(g.interactions).toEqual([]);
  });
});

describe('interaction-only variables', () => {
  it('excludes a `:`-only var from effects/tests; engine target_indices align (labels.ts:44)', () => {
    // y ~ a + b + x1:x2 — the parser auto-promotes x1/x2 to predictors but emits
    // no `main` term for them, so they must NOT be effects. The engine builds
    // target_indices dense over [a, b, x1:x2] (= [1,2,3]); the app list must
    // match so labels.ts:44 (`effectNames[idx-1]`) resolves correctly.
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ a + b + x1:x2',
      variables: [
        { kind: 'continuous' as const, name: 'a' },
        { kind: 'continuous' as const, name: 'b' },
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'continuous' as const, name: 'x2' },
      ],
    };
    expect(effectNames(cfg)).toEqual(['a', 'b', 'x1:x2']);
  });

  it('still shows x1/x2 as cards but marks them interactionOnly with no rows', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ a + b + x1:x2',
      variables: [
        { kind: 'continuous' as const, name: 'a' },
        { kind: 'continuous' as const, name: 'b' },
        { kind: 'continuous' as const, name: 'x1' },
        { kind: 'continuous' as const, name: 'x2' },
      ],
    };
    const g = effectGroups(cfg);
    expect(g.variables.map((v) => v.name)).toEqual(['a', 'b', 'x1', 'x2']);
    const byName = new Map(g.variables.map((v) => [v.name, v]));
    expect(byName.get('a')!.interactionOnly).toBe(false);
    expect(byName.get('x1')).toMatchObject({ interactionOnly: true, rows: [] });
    expect(byName.get('x2')).toMatchObject({ interactionOnly: true, rows: [] });
  });

  it('a var in both a main term and an interaction (a*b) is NOT interactionOnly', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ a*b',
      variables: [
        { kind: 'continuous' as const, name: 'a' },
        { kind: 'continuous' as const, name: 'b' },
      ],
    };
    expect(effectNames(cfg)).toEqual(['a', 'b', 'a:b']);
    const g = effectGroups(cfg);
    expect(g.variables.every((v) => v.interactionOnly === false)).toBe(true);
    expect(g.variables.flatMap((v) => v.rows).length).toBeGreaterThan(0);
  });

  it('ANOVA guard: every factor keeps effect rows and interactionOnly stays false', () => {
    const cfg = {
      ...defaultFamilyConfig('anova'),
      formula: '',
      variables: [{ kind: 'factor' as const, name: 'group', nLevels: 3, role: 'factor' as const }],
    };
    const g = effectGroups(cfg);
    expect(g.variables).toHaveLength(1);
    expect(g.variables[0]!.interactionOnly).toBe(false);
    expect(g.variables[0]!.rows.length).toBeGreaterThan(0);
  });
});

describe('reconcileTestSelection', () => {
  it('prunes a test name no longer in the candidate effect list', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      tests: { kind: 'effects' as const, names: ['a', 'x1'] },
    };
    reconcileTestSelection(cfg, ['a', 'b', 'x1:x2']);
    expect(cfg.tests).toEqual({ kind: 'effects', names: ['a'] });
  });

  it('leaves an unchanged selection alone (no needless rewrite)', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      tests: { kind: 'effects' as const, names: ['a', 'b'] },
    };
    const before = cfg.tests;
    reconcileTestSelection(cfg, ['a', 'b', 'x1:x2']);
    expect(cfg.tests).toBe(before);
  });

  it('does not touch a kind:"all" selection', () => {
    const cfg = { ...defaultFamilyConfig('regression'), tests: { kind: 'all' as const } };
    reconcileTestSelection(cfg, ['a']);
    expect(cfg.tests).toEqual({ kind: 'all' });
  });

  it('prunes a contrast whose endpoints left the candidate list', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      contrasts: [
        { positiveName: 'g[2]', negativeName: 'g[3]', enabled: true },
        { positiveName: 'gone[2]', negativeName: 'gone[3]', enabled: true },
      ],
    };
    reconcileTestSelection(cfg, ['g[1]', 'g[2]', 'g[3]']);
    expect(cfg.contrasts).toEqual([{ positiveName: 'g[2]', negativeName: 'g[3]', enabled: true }]);
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
