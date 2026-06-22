import { describe, it, expect } from 'vitest';
import { buildRows, baselineContrastLabels, factorReferenceLabels } from './rows';

describe('buildRows', () => {
  it('groups factor levels under a header with continuous effects', () => {
    // target_indices are β̂-column indices: intercept is col 0, first real
    // effect is col 1. effect_names is intercept-excluded (0-based).
    const rows = buildRows(
      [1, 2, 3],
      ['x1', 'condition[treatment]', 'condition[active]'],
      { condition: { baseline: 'control' } },
    );
    expect(rows[0]).toEqual({ kind: 'continuous', label: 'x1', pos: 0 });
    expect(rows[1]).toEqual({ kind: 'factor_header', label: 'condition', baseline: 'control' });
    expect(rows[2]).toEqual({ kind: 'factor_level', label: 'treatment', factor: 'condition', pos: 1 });
    expect(rows[3]).toEqual({ kind: 'factor_level', label: 'active', factor: 'condition', pos: 2 });
  });

  it('handles continuous-only case', () => {
    const rows = buildRows(
      [1, 2],
      ['age', 'income'],
      {},
    );
    expect(rows).toEqual([
      { kind: 'continuous', label: 'age', pos: 0 },
      { kind: 'continuous', label: 'income', pos: 1 },
    ]);
  });

  it('emits factor_header only once per factor', () => {
    const rows = buildRows(
      [1, 2, 3],
      ['grp[b]', 'grp[c]', 'grp[d]'],
      { grp: { baseline: 'a' } },
    );
    const headers = rows.filter((r) => r.kind === 'factor_header');
    expect(headers.length).toBe(1);
    expect(headers[0]!.label).toBe('grp');
    expect(headers[0]!.baseline).toBe('a');
    expect(rows.filter((r) => r.kind === 'factor_level').length).toBe(3);
  });

  it('throws a clear error when idx-1 is out of range', () => {
    expect(() =>
      buildRows([5], ['x1', 'x2'], {}),
    ).toThrowError(/β̂-column/);
  });

  it('appends contrast rows after the marginals with pos past targetIndices', () => {
    // k=3 one-way ANOVA shape: 2 dummy marginals + 1 pairwise contrast (β2 − β1).
    const rows = buildRows(
      [1, 2],
      ['grp[b]', 'grp[c]'],
      { grp: { baseline: 'a' } },
      [[2, 1]],
    );
    expect(rows.at(-1)).toEqual({ kind: 'contrast', label: 'grp[c] vs grp[b]', pos: 2 });
  });

  it('throws a clear error when a contrast pair maps outside effect_names', () => {
    expect(() =>
      buildRows([1, 2], ['x1', 'x2'], {}, [[3, 1]]),
    ).toThrowError(/contrast_pairs/);
  });

  it('renders a baseline-contrast marginal as a contrast row, keeping others grouped', () => {
    // F1[2] is a baseline contrast (F1[1] vs F1[2], collapsed to a marginal);
    // F1[3] is a plain factor marginal → stays a grouped factor row.
    const rows = buildRows(
      [1, 2],
      ['F1[2]', 'F1[3]'],
      { F1: { baseline: '1' } },
      [],
      new Map([[1, 'F1[1] vs F1[2]']]),
    );
    expect(rows[0]).toEqual({ kind: 'contrast', label: 'F1[1] vs F1[2]', pos: 0 });
    expect(rows.some((r) => r.kind === 'factor_header' && r.label === 'F1')).toBe(true);
    expect(rows.some((r) => r.kind === 'factor_level' && r.label === '3')).toBe(true);
  });
});

describe('factorReferenceLabels', () => {
  it('maps a factor name to its reference label; skips when labels are missing', () => {
    expect(
      factorReferenceLabels([
        { kind: 'factor', name: 'F1', factor_reference: 0, factor_labels: ['a', 'b', 'c'] },
        { kind: 'factor', name: 'F2', factor_reference: 1, factor_labels: ['x', 'y'] },
        { kind: 'factor', name: 'F3' }, // no labels → omitted
        { kind: 'numeric', name: 'cov' },
      ]),
    ).toEqual({ F1: 'a', F2: 'y' });
  });
});

describe('baselineContrastLabels', () => {
  it('relabels a marginal whose (ref, level) pair is an enabled contrast', () => {
    const m = baselineContrastLabels([1], ['F1[2]'], { F1: '1' }, [['F1[2]', 'F1[1]']]);
    expect(m.get(1)).toBe('F1[1] vs F1[2]');
  });

  it('matches regardless of which side the reference sits on', () => {
    const m = baselineContrastLabels([1], ['F1[2]'], { F1: '1' }, [['F1[1]', 'F1[2]']]);
    expect(m.get(1)).toBe('F1[1] vs F1[2]');
  });

  it('leaves a marginal alone when its pair is not an enabled contrast', () => {
    expect(baselineContrastLabels([1], ['F1[2]'], { F1: '1' }, []).size).toBe(0);
    expect(
      baselineContrastLabels([1], ['F1[2]'], { F1: '1' }, [['F1[2]', 'F1[3]']]).size,
    ).toBe(0);
  });

  it('skips when the factor reference label is unavailable', () => {
    expect(baselineContrastLabels([1], ['F1[2]'], {}, [['F1[2]', 'F1[1]']]).size).toBe(0);
  });
});
