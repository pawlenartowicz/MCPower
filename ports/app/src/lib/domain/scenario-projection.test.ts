import { describe, it, expect } from 'vitest';
import type { ScenarioConfig } from '$lib/configs/scenarios';
import {
  DIST_CODE,
  RESIDUAL_CODE,
  RE_DIST_CODE,
  projectScenario,
} from './scenario-projection';

// These tables are the parity guard: they MUST equal the engine's `DIST_CODES` /
// `RESIDUAL_CODES` (engine-spec-builder `project_contract.rs`, which Python
// reads via `get_dist_codes()` / `get_residual_codes()`) verbatim, or app
// scenario numbers silently diverge from the engine's at equal X.Y.
describe('scenario code tables match the engine tables', () => {
  it('DIST_CODE matches the engine DIST_CODES', () => {
    expect(DIST_CODE).toEqual({
      normal: 0,
      binary: 1,
      right_skewed: 2,
      left_skewed: 3,
      high_kurtosis: 4,
      uniform: 5,
      uploaded_factor: 97,
      uploaded_binary: 98,
      uploaded_data: 99,
    });
  });

  it('RESIDUAL_CODE matches canonical RESIDUAL_CODES in project_contract.rs (aliases dropped)', () => {
    expect(RESIDUAL_CODE).toEqual({
      normal: 0,
      right_skewed: 2,
      left_skewed: 3,
      high_kurtosis: 4,
      uniform: 5,
    });
  });

  it('RE_DIST_CODE encodes the two UI-exposed RE distributions', () => {
    expect(RE_DIST_CODE).toEqual({ normal: 0, heavy_tailed: 1 });
  });
});

function cfg(overrides: Partial<ScenarioConfig> = {}): ScenarioConfig {
  return {
    name: 'realistic',
    heterogeneity: 0.1,
    heteroskedasticity_ratio: 1.5,
    correlation_noise_sd: 0.2,
    distribution_change_prob: 0.4,
    new_distributions: [],
    residual_change_prob: 0.3,
    residual_dists: [],
    residual_df: 5,
    sampled_factor_proportions: false,
    truth_start: false,
    lme: { random_effect_dist: 'normal', random_effect_df: 0, icc_noise_sd: 0 },
    ...overrides,
  };
}

describe('projectScenario (mirrors Python build_scenario_dict)', () => {
  it('carries all wire fields verbatim, hoists lme fields, drops the lme sub-object', () => {
    const wire = projectScenario(cfg());
    expect(wire).toEqual({
      name: 'realistic',
      heterogeneity: 0.1,
      heteroskedasticity_ratio: 1.5,
      correlation_noise_sd: 0.2,
      distribution_change_prob: 0.4,
      new_distributions: [],
      residual_change_prob: 0.3,
      residual_dists: [],
      residual_df: 5,
      sampled_factor_proportions: false,
      truth_start: false,
      // lme fields hoisted onto the wire (default cfg() lme is neutral → all zero)
      random_effect_dist: 0,
      random_effect_df: 0,
      icc_noise_sd: 0,
    });
    expect('lme' in wire).toBe(false); // the sub-object is gone; its fields are hoisted
  });

  it('encodes lme fields: normal→0, heavy_tailed→1, passes df and icc_noise_sd verbatim', () => {
    const wire = projectScenario(
      cfg({ lme: { random_effect_dist: 'heavy_tailed', random_effect_df: 5, icc_noise_sd: 0.1 } }),
    );
    expect(wire.random_effect_dist).toBe(1);
    expect(wire.random_effect_df).toBe(5);
    expect(wire.icc_noise_sd).toBeCloseTo(0.1);
    expect('lme' in wire).toBe(false);
  });

  it('carries sampled_factor_proportions true and defaults missing values to false', () => {
    expect(
      projectScenario(cfg({ sampled_factor_proportions: true })).sampled_factor_proportions,
    ).toBe(true);
    // Persisted pre-knob snapshots lack the field at runtime.
    const legacy = { ...cfg() } as Record<string, unknown>;
    delete legacy.sampled_factor_proportions;
    expect(
      projectScenario(legacy as unknown as ScenarioConfig).sampled_factor_proportions,
    ).toBe(false);
  });

  it('carries truth_start true and defaults missing values to false (optimistic warm-starts, realistic/doomer cold-start)', () => {
    expect(projectScenario(cfg({ name: 'optimistic', truth_start: true })).truth_start).toBe(true);
    expect(projectScenario(cfg({ name: 'realistic', truth_start: false })).truth_start).toBe(false);
    // Persisted pre-knob snapshots lack the field at runtime.
    const legacy = { ...cfg() } as Record<string, unknown>;
    delete legacy.truth_start;
    expect(projectScenario(legacy as unknown as ScenarioConfig).truth_start).toBe(false);
  });

  it('encodes new_distributions through _DIST_CODE', () => {
    const wire = projectScenario(cfg({ new_distributions: ['right_skewed', 'left_skewed', 'uniform'] }));
    expect(wire.new_distributions).toEqual([2, 3, 5]);
  });

  it('encodes all five canonical residual distributions', () => {
    const wire = projectScenario(
      cfg({ residual_dists: ['normal', 'right_skewed', 'left_skewed', 'high_kurtosis', 'uniform'] }),
    );
    expect(wire.residual_dists).toEqual([0, 2, 3, 4, 5]);
  });

  it('throws on a completely unknown residual name', () => {
    const bad = cfg({ residual_dists: ['nonsense'] as unknown as ScenarioConfig['residual_dists'] });
    expect(() => projectScenario(bad)).toThrow(/unknown residual distribution/);
  });
});
