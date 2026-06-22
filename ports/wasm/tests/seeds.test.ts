import { describe, it, expect } from 'vitest';
import { classifyCase, poolSize, splitSims, workerSeeds, type CaseClass, type RouteInputs } from '../src/seeds';
import type { AppSpec } from '../src/types';

// A routing input with the "default OLS" baseline; tests override one field.
const base: RouteInputs = {
  caseClass: 'simple',
  hasScenarios: false,
  n: 200,
  nSims: 1600,
  hardwareConcurrency: 16,
};
const route = (o: Partial<RouteInputs>): number => poolSize({ ...base, ...o });

describe('poolSize routing heuristic', () => {
  // Anchors — the defaults must land where the design intends.
  it('default OLS / GLM / ANOVA (1600 sims, small N, no scenarios) → 1', () => {
    expect(route({ caseClass: 'simple', nSims: 1600 })).toBe(1);
  });
  it('default simple LMM (gaussian mixed, 800 sims) → 1', () => {
    expect(route({ caseClass: 'simple', nSims: 800, n: 480 })).toBe(1);
  });
  it('glmm_intercept at the 800-sim mixed default → 2', () => {
    expect(route({ caseClass: 'glmm_intercept', nSims: 800, n: 480 })).toBe(2);
  });
  it('heavy glmm_slopes with scenarios → 8 (saturates)', () => {
    expect(route({ caseClass: 'glmm_slopes', hasScenarios: true })).toBe(8); // 4×2
  });
  it('large N (>24000) → 8 via N×n_sims', () => {
    expect(route({ n: 24001, nSims: 8001 })).toBe(8); // 4×4 = 16, capped 8
  });
  it('high n_sims (>8000) heavy glmm → 8', () => {
    expect(route({ caseClass: 'glmm_slopes', nSims: 8001 })).toBe(8); // 4×4 = 16, capped 8
  });

  // Case-class multiplier.
  it.each<[CaseClass, number]>([
    ['simple', 1],
    ['glmm_intercept', 2],
    ['glmm_slopes', 4],
  ])('case %s → ×%d', (caseClass, expected) => {
    expect(route({ caseClass })).toBe(expected);
  });

  // Scenarios multiplier.
  it('any scenarios → ×2', () => {
    expect(route({ hasScenarios: true })).toBe(2);
  });

  // N tiers: ≤8000 → ×1, >8000 → ×2, >24000 → ×4.
  it.each<[number, number]>([
    [8000, 1],
    [8001, 2],
    [24000, 2],
    [24001, 4],
  ])('N=%d → %d workers', (n, expected) => {
    expect(route({ n })).toBe(expected);
  });

  // n_sims tiers: ≤2000 → ×1, >2000 → ×2, >8000 → ×4.
  it.each<[number, number]>([
    [2000, 1],
    [2001, 2],
    [8000, 2],
    [8001, 4],
  ])('n_sims=%d → %d workers', (nSims, expected) => {
    expect(route({ nSims })).toBe(expected);
  });

  // Multiplicative composition: every factor stacks before the clamp.
  it('composes factors: glmm_intercept × scenarios × N>8000 = 8', () => {
    expect(route({ caseClass: 'glmm_intercept', hasScenarios: true, n: 8001 })).toBe(8); // 2×2×2
  });

  // Clamps.
  it('clamps to 8 even when factors exceed it', () => {
    // glmm_slopes ×4 × scenarios ×2 × N>8000 ×2 = 16 → 8
    expect(route({ caseClass: 'glmm_slopes', hasScenarios: true, n: 8001 })).toBe(8);
  });
  it('clamps to hardwareConcurrency below 8', () => {
    expect(route({ caseClass: 'glmm_slopes', hasScenarios: true, hardwareConcurrency: 4 })).toBe(4);
  });
  it('clamps to n_sims (no zero-share workers)', () => {
    expect(route({ caseClass: 'glmm_slopes', hasScenarios: true, nSims: 2 })).toBe(2);
  });
  it('is at least 1 with zero hardwareConcurrency', () => {
    expect(route({ hardwareConcurrency: 0 })).toBe(1);
  });

  // forced (bench override) bypasses the heuristic, clamped to [1, nSims], uncapped.
  it('forced bypasses routing, clamped to [1, nSims], past the 8-cap', () => {
    expect(route({ forced: 4, caseClass: 'simple' })).toBe(4);
    expect(route({ forced: 16, nSims: 10000 })).toBe(16); // uncapped → sweep w16
    expect(route({ forced: 16, nSims: 8, caseClass: 'glmm_slopes' })).toBe(8); // clamped to nSims
    expect(route({ forced: 0 })).toBe(1); // floor at 1
    expect(route({ forced: 1, caseClass: 'glmm_slopes', hasScenarios: true })).toBe(1);
  });
});

// classifyCase schema-drift tripwire: it cracks AppSpec for `family`,
// `outcome.kind`, and `slopes` only. These specs mirror the app-spec.ts `family`
// union — if a renamed family/field reaches the engine, the classification here
// changes and these assertions fail rather than silently mis-sizing the pool.
describe('classifyCase', () => {
  const spec = (o: Record<string, unknown>): AppSpec => o as unknown as AppSpec;

  it('linear / logit / anova → simple', () => {
    expect(classifyCase(spec({ family: 'linear' }))).toBe('simple');
    expect(classifyCase(spec({ family: 'logit' }))).toBe('simple');
    expect(classifyCase(spec({ family: 'anova' }))).toBe('simple');
  });
  it('mixed gaussian / absent outcome → simple (lmm)', () => {
    expect(classifyCase(spec({ family: 'mixed' }))).toBe('simple');
    expect(classifyCase(spec({ family: 'mixed', outcome: { kind: 'gaussian' } }))).toBe('simple');
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
