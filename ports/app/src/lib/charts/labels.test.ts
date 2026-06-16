import { describe, it, expect } from 'vitest';
import { buildTargetLabelMap, relabelTargets } from './labels';

describe('buildTargetLabelMap', () => {
  // Every map carries an `overall` entry for the sample-size curve's omnibus
  // series; it defaults to the OLS "Overall F" label.
  it('maps target_{idx} to effectNames[idx-1]', () => {
    const map = buildTargetLabelMap([1, 2], ['x1', 'x2']);
    expect(map).toEqual({ target_1: 'x1', target_2: 'x2', overall: 'Overall F' });
  });

  it('skips out-of-range indices', () => {
    // effectNames has only 2 entries; index 5 is out-of-range
    const map = buildTargetLabelMap([1, 5], ['x1', 'x2']);
    expect(map).toEqual({ target_1: 'x1', overall: 'Overall F' });
    expect('target_5' in map).toBe(false);
  });

  it('returns only the overall entry for empty target inputs', () => {
    expect(buildTargetLabelMap([], [])).toEqual({ overall: 'Overall F' });
    expect(buildTargetLabelMap([1], [])).toEqual({ overall: 'Overall F' });
  });

  it('labels the overall token per estimator (F-test for OLS, LRT for GLM)', () => {
    expect(buildTargetLabelMap([1], ['x1'], [], 'ols').overall).toBe('Overall F');
    expect(buildTargetLabelMap([1], ['x1'], [], 'glm').overall).toBe('LR χ²');
  });

  it('maps contrast tokens target_{p}_vs_{n} via the table label source', () => {
    const map = buildTargetLabelMap([1, 2], ['grp[b]', 'grp[c]'], [[2, 1]]);
    expect(map['target_2_vs_1']).toBe('grp[c] vs grp[b]');
  });

  it('skips contrast pairs with out-of-range sides', () => {
    const map = buildTargetLabelMap([1], ['x1'], [[2, 1]]);
    expect('target_2_vs_1' in map).toBe(false);
  });

  it('relabels a baseline-contrast target token to "ref vs level" (table/chart parity)', () => {
    // Same map source as buildRows: a marginal collapsed from a baseline
    // contrast carries the "ref vs level" label on its target_{idx} token.
    const map = buildTargetLabelMap([1], ['F1[2]'], [], 'ols', new Map([[1, 'F1[1] vs F1[2]']]));
    expect(map.target_1).toBe('F1[1] vs F1[2]');
  });
});

type RowArr = { target: string }[];

describe('relabelTargets', () => {
  it('rewrites target values in data.values rows', () => {
    const spec: unknown = {
      data: { values: [{ target: 'target_1', v: 0.7 }, { target: 'target_2', v: 0.5 }] },
    };
    relabelTargets(spec, { target_1: 'sleep', target_2: 'age' });
    const values = (spec as { data: { values: RowArr } }).data.values;
    expect(values[0]!.target).toBe('sleep');
    expect(values[1]!.target).toBe('age');
  });

  it('rewrites target in a vconcat child spec', () => {
    type Child = { data: { values: RowArr } };
    const spec: unknown = {
      vconcat: [
        { data: { values: [{ target: 'target_1', v: 0.8 }] } },
        { data: { values: [{ target: 'target_2', v: 0.6 }] } },
      ],
    };
    relabelTargets(spec, { target_1: 'x1', target_2: 'x2' });
    const children = (spec as { vconcat: Child[] }).vconcat;
    expect(children[0]!.data.values[0]!.target).toBe('x1');
    expect(children[1]!.data.values[0]!.target).toBe('x2');
  });

  it("rewrites target in a layered mark's own data", () => {
    type LayerChild = { data: { values: RowArr } };
    const spec: unknown = {
      layer: [
        { data: { values: [{ target: 'target_1', y: 1 }] } },
        { data: { values: [{ target: 'target_1', y: 2 }] } },
      ],
    };
    relabelTargets(spec, { target_1: 'sleep' });
    const layers = (spec as { layer: LayerChild[] }).layer;
    expect(layers[0]!.data.values[0]!.target).toBe('sleep');
    expect(layers[1]!.data.values[0]!.target).toBe('sleep');
  });

  it('leaves rows without a matching token unchanged', () => {
    const spec: unknown = {
      data: { values: [{ target: 'target_3', v: 0.4 }, { target: 'target_1', v: 0.9 }] },
    };
    relabelTargets(spec, { target_1: 'x1' });
    const values = (spec as { data: { values: RowArr } }).data.values;
    expect(values[0]!.target).toBe('target_3'); // not in map → unchanged
    expect(values[1]!.target).toBe('x1');
  });

  it('is a no-op when map is empty', () => {
    const spec: unknown = {
      data: { values: [{ target: 'target_1', v: 0.7 }] },
    };
    relabelTargets(spec, {});
    const values = (spec as { data: { values: RowArr } }).data.values;
    expect(values[0]!.target).toBe('target_1'); // unchanged
  });
});
