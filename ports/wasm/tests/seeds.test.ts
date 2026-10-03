import { describe, it, expect } from 'vitest';
import { classifyCase, poolSize, splitSims, workerSeeds, type CaseClass, type RouteInputs } from '../src/seeds';
import type { AppSpec } from '../src/types';

// A routing input with the "default OLS" baseline; tests override one field.
const base: RouteInputs = {
  caseClass: 'ols',
  scenarioCount: 0,
  n: 200,
  nSims: 1600,
  hardwareConcurrency: 16,
};
const route = (o: Partial<RouteInputs>): number => poolSize({ ...base, ...o });

describe('poolSize routing heuristic', () => {
  // Anchors — the defaults must land where the design intends. Each expectation
  // is the rung the 2026-08-27/28 sweep measured as optimal for that workload.
  it('default OLS (1600 sims, small N, no scenarios) → 4', () => {
    // 200 x 1600 x 1 = 3.2e5
    expect(route({ caseClass: 'ols', nSims: 1600 })).toBe(4);
  });
  it('tiny OLS run stays single-worker', () => {
    // 100 x 500 x 1 = 5e4, below the 2e5 rung
    expect(route({ caseClass: 'ols', n: 100, nSims: 500 })).toBe(1);
  });
  it('default simple LMM (gaussian mixed, 800 sims) → 4', () => {
    // 480 x 800 x 2 = 7.68e5
    expect(route({ caseClass: 'lmm', nSims: 800, n: 480 })).toBe(4);
  });
  it('glmm_intercept at the 800-sim mixed default → 12', () => {
    // 480 x 800 x 32 = 1.23e7, over the top rung
    expect(route({ caseClass: 'glmm_intercept', nSims: 800, n: 480 })).toBe(12);
  });
  it('crossed/nested GLMM saturates the ladder even at 100 sims', () => {
    // 480 x 100 x 256 = 1.23e7
    expect(route({ caseClass: 'glmm_crossed_nested', nSims: 100, n: 480 })).toBe(12);
  });
  it('large N alone lifts a cheap OLS run off the bottom rung', () => {
    // 10000 x 1000 x 1 = 1e7 — the case the old independent-tier rule sent to w2
    expect(route({ caseClass: 'ols', n: 10000, nSims: 1000 })).toBe(8);
  });

  // Class weights: the score is linear in the weight. Pin each weight exactly at
  // the top breakpoint: n x nSims = 1.2e7 / weight scores 1.2e7 (→ 12), one sim
  // fewer scores 1.2e7 − weight (→ 8). Any other weight flips one of the two.
  it.each<[CaseClass, number]>([
    ['ols', 1],
    ['glm', 2],
    ['lmm', 2],
    ['glmm_intercept', 32],
    ['glmm_slopes', 128],
    ['glmm_crossed_nested', 256],
  ])('case %s carries weight %d', (caseClass, weight) => {
    const atRung = 1.2e7 / weight;
    expect(route({ caseClass, n: 1, nSims: atRung })).toBe(12);
    expect(route({ caseClass, n: 1, nSims: atRung - 1 })).toBe(8);
  });

  // Scenarios multiply the score by their count, not a flat x2 — three
  // scenarios is three times the work.
  it('scenarioCount multiplies the score', () => {
    expect(route({ n: 200, nSims: 1000, scenarioCount: 0 })).toBe(4); // 2e5
    expect(route({ n: 200, nSims: 1000, scenarioCount: 1 })).toBe(4); // 2e5, 1 == baseline
    expect(route({ n: 200, nSims: 5000, scenarioCount: 3 })).toBe(8); // 3e6
  });

  // Ladder breakpoints.
  it.each<[number, number]>([
    [199999, 1],
    [200000, 4],
    [2999999, 4],
    [3000000, 8],
    [11999999, 8],
    [12000000, 12],
  ])('score %d → %d workers', (score, expected) => {
    // caseClass ols (weight 1), no scenarios → score == n x nSims.
    expect(route({ caseClass: 'ols', n: 1, nSims: score })).toBe(Math.min(expected, score));
  });

  // Clamps.
  it('clamps to 12 even when the score saturates', () => {
    expect(route({ caseClass: 'glmm_crossed_nested', n: 100000, nSims: 100000 })).toBe(12);
  });
  it('clamps to hardwareConcurrency below 12', () => {
    expect(route({ caseClass: 'glmm_slopes', hardwareConcurrency: 4 })).toBe(4);
  });
  it('clamps to n_sims (no zero-share workers)', () => {
    // Large n keeps the score on the top rung so the nSims clamp is what binds.
    expect(route({ caseClass: 'glmm_slopes', n: 100000, nSims: 2 })).toBe(2);
  });
  it('is at least 1 with zero hardwareConcurrency', () => {
    expect(route({ hardwareConcurrency: 0 })).toBe(1);
  });

  // forced (bench override) bypasses the heuristic, clamped to [1, nSims], uncapped.
  it('forced bypasses routing, clamped to [1, nSims], past the 12-cap', () => {
    expect(route({ forced: 4, caseClass: 'ols' })).toBe(4);
    expect(route({ forced: 16, nSims: 10000 })).toBe(16); // uncapped → sweep w16
    expect(route({ forced: 16, nSims: 8, caseClass: 'glmm_slopes' })).toBe(8); // clamped to nSims
    expect(route({ forced: 0 })).toBe(1); // floor at 1
    expect(route({ forced: 1, caseClass: 'glmm_slopes', scenarioCount: 3 })).toBe(1);
  });
});

// classifyCase schema-drift tripwire: it cracks AppSpec for `family`,
// `outcome.kind`, `slopes`, and `extra_groupings` only. These specs mirror the
// app-spec.ts `family` union — if a renamed family/field reaches the engine, the
// classification here changes and these assertions fail rather than silently
// mis-sizing the pool.
describe('classifyCase', () => {
  const spec = (o: Record<string, unknown>): AppSpec => o as unknown as AppSpec;

  it('linear / anova → ols', () => {
    expect(classifyCase(spec({ family: 'linear' }))).toBe('ols');
    expect(classifyCase(spec({ family: 'anova' }))).toBe('ols');
  });
  it('top-level logit / poisson (unclustered) → glm', () => {
    expect(classifyCase(spec({ family: 'logit' }))).toBe('glm');
    expect(classifyCase(spec({ family: 'poisson' }))).toBe('glm');
  });
  it('mixed gaussian / absent outcome → lmm', () => {
    expect(classifyCase(spec({ family: 'mixed' }))).toBe('lmm');
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'gaussian' } }))).toBe('lmm');
  });
  it('mixed binary, empty/absent slopes → glmm_intercept', () => {
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'binary' } }))).toBe('glmm_intercept');
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'binary' }, slopes: [] }))).toBe('glmm_intercept');
  });
  it('mixed binary, non-empty slopes → glmm_slopes', () => {
    expect(classifyCase(spec({
      family: 'mixed', outcome: { kind: 'binary' },
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: 0 }],
    }))).toBe('glmm_slopes');
  });
  it('mixed binary with extra_groupings → glmm_crossed_nested', () => {
    expect(classifyCase(spec({
      family: 'mixed', outcome: { kind: 'binary' },
      extra_groupings: [{ tau_squared: 0.3, relation: 'crossed' }],
    }))).toBe('glmm_crossed_nested');
  });
  it('extra_groupings wins over slopes (higher weight, never measured together)', () => {
    expect(classifyCase(spec({
      family: 'mixed', outcome: { kind: 'binary' },
      extra_groupings: [{ tau_squared: 0.3, relation: 'nested' }],
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: 0 }],
    }))).toBe('glmm_crossed_nested');
  });
  it('empty extra_groupings does not promote', () => {
    expect(classifyCase(spec({
      family: 'mixed', outcome: { kind: 'binary' }, extra_groupings: [],
    }))).toBe('glmm_intercept');
  });
  it('mixed probit (binary + link) → glmm_intercept', () => {
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'binary', link: 'probit' } }))).toBe('glmm_intercept');
  });
  it('mixed poisson, empty/absent slopes → glmm_intercept', () => {
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'poisson', baseline_rate: 2, tau_squared: 0.5 } }))).toBe('glmm_intercept');
  });
  it('mixed poisson, non-empty slopes → glmm_slopes', () => {
    expect(classifyCase(spec({
      family: 'mixed', outcome: { kind: 'poisson', baseline_rate: 2, tau_squared: 0.5 },
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: 0 }],
    }))).toBe('glmm_slopes');
  });
});

describe('splitSims', () => {
  it('sums to the total and spreads the remainder to the front', () => {
    expect(splitSims(1600, 8)).toEqual([200, 200, 200, 200, 200, 200, 200, 200]);
    const s = splitSims(1603, 8);
    expect(s.reduce((a, b) => a + b, 0)).toBe(1603);
    expect(s).toEqual([201, 201, 201, 200, 200, 200, 200, 200]);
  });
});

describe('workerSeeds', () => {
  it('offsets the master seed by worker index, wrapped to u64', () => {
    expect(workerSeeds(2137n, 3)).toEqual([2137n, 2138n, 2139n]);
  });
  it('wraps past 2^64', () => {
    const max = (1n << 64n) - 1n;
    expect(workerSeeds(max, 2)).toEqual([max, 0n]);
  });
});
