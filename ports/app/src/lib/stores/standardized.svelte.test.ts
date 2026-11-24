import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AppSpec, EffectMoments } from '$lib/domain/app-spec';

const effectMoments = vi.hoisted(() => vi.fn());
vi.mock('$lib/api/engine', () => ({ effectMoments }));

import { evaluateStandardized, standardizedStore } from './standardized.svelte';

const spec = (value: number, alpha = 0.05) =>
  ({ family: 'linear', alpha, effects: [{ name: 'x1', value }], csv: null }) as unknown as AppSpec;

// One standard-normal predictor: std β = β / √(β² + 1), R² = β² / (β² + 1).
const normal: EffectMoments = {
  descriptors: [{ kind: 'continuous', predictor: 'x1' }],
  moments: { noise: 1, sd: [1], cov: [1], groups: [null] },
};

describe('evaluateStandardized', () => {
  it('gives the closed-form values of one normal predictor', () => {
    const v = evaluateStandardized(normal, spec(0.5));
    expect(v.rows[0]!.std_beta).toBeCloseTo(0.5 / Math.sqrt(1.25), 14);
    expect(v.rows[0]!.cohen_d).toBeNull();
    expect(v.r2).toBeCloseTo(0.25 / 1.25, 14);
  });

  it('gives d = β for one balanced binary predictor', () => {
    // x ~ Bernoulli(0.5): SD 0.5, constant within each group.
    const binary: EffectMoments = {
      descriptors: [{ kind: 'continuous', predictor: 'x1' }],
      moments: {
        noise: 1,
        sd: [0.5],
        cov: [0.25],
        groups: [{ w: 0.5, mean_diff: [1], cov_1: [0], cov_0: [0] }],
      },
    };
    const v = evaluateStandardized(binary, spec(0.4));
    expect(v.rows[0]!.cohen_d).toBeCloseTo(0.4, 14);
    expect(v.rows[0]!.std_beta).toBeCloseTo((0.4 * 0.5) / Math.sqrt(0.04 + 1), 14);
  });

  it('gives dashes where the moments are undefined', () => {
    const v = evaluateStandardized({ ...normal, moments: null }, spec(0.5));
    expect(v.rows).toEqual([{ name: 'x1', input: 0.5, std_beta: null, cohen_d: null }]);
    expect(v.r2).toBeNull();
  });
});

describe('standardizedStore', () => {
  afterEach(() => {
    vi.useRealTimers();
    effectMoments.mockReset();
  });

  it('fetches moments once per model and evaluates effect-size edits without the engine', async () => {
    vi.useFakeTimers();
    effectMoments.mockResolvedValue(normal);
    standardizedStore.refresh(spec(0.1));
    standardizedStore.refresh(spec(0.5));
    expect(standardizedStore.current).toBeNull(); // nothing computed yet
    await vi.runAllTimersAsync();
    // Debounced: only the last edit reaches the engine.
    expect(effectMoments).toHaveBeenCalledTimes(1);
    expect(effectMoments).toHaveBeenCalledWith(spec(0.5));
    expect(standardizedStore.current?.rows[0]?.input).toBe(0.5);
    // An effect-size edit is evaluated at once from the cached moments.
    standardizedStore.refresh(spec(0.6));
    expect(standardizedStore.current?.rows[0]?.std_beta).toBeCloseTo(0.6 / Math.sqrt(1.36), 14);
    await vi.runAllTimersAsync();
    expect(effectMoments).toHaveBeenCalledTimes(1);
    // Any other edit fetches new moments; the old values stay until they land.
    standardizedStore.refresh(spec(0.6, 0.01));
    expect(standardizedStore.current?.rows[0]?.input).toBe(0.6);
    await vi.runAllTimersAsync();
    expect(effectMoments).toHaveBeenCalledTimes(2);
    // A spec that cannot be built shows a dash at once.
    standardizedStore.refresh(null);
    expect(standardizedStore.current).toBeNull();
  });
});
