import { describe, it, expect } from 'vitest';
import type { AppSpec } from '../src/types';

// The WASM worker pool treats AppSpec as opaque JSON (`Record<string, unknown>`),
// stringifies it, and posts it to the engine-wasm shell — it never reads the
// spec's internal fields. So the M2/M3 + cluster_level_vars growth fields need no
// type mirror here; they pass through by construction. This guards that a Mixed
// AppSpec carrying all of them survives the JSON.stringify → parse round-trip the
// pool performs (regression tripwire if AppSpec is ever made non-opaque).
describe('WASM AppSpec passthrough — Mixed with growth fields', () => {
  it('round-trips a MixedSpec with cluster_level_vars, extra_groupings, and slopes', () => {
    const spec: AppSpec = {
      family: 'mixed',
      parsed_formula: { outcome: 'y', predictors: ['x'], interaction_terms: [] },
      var_types: [{ kind: 'numeric', name: 'x' }],
      effects: [{ name: 'x', value: 0.3 }],
      correlations: null,
      alpha: 0.05,
      target_power: 0.8,
      n_sims: 100,
      seed: 2137,
      tests: { kind: 'all' },
      correction: 'none',
      scenarios: [
        {
          name: 'realistic',
          heterogeneity: 0.1,
          heteroskedasticity_ratio: 1.5,
          correlation_noise_sd: 0.0,
          distribution_change_prob: 0.0,
          new_distributions: [],
          residual_change_prob: 0.0,
          residual_dists: [],
          residual_df: 5,
          sampled_factor_proportions: false,
          random_effect_dist: 1,
          random_effect_df: 5,
          icc_noise_sd: 0.05,
        },
      ],
      csv: null,
      report_overall: false,
      contrasts: [],
      cluster_name: 'school',
      icc: 0.2,
      cluster_dim: { kind: 'n_clusters', value: 20 },
      cluster_level_vars: ['x'],
      extra_groupings: [{ tau_squared: 0.05, relation: { kind: 'crossed', n_clusters: 8 } }],
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: -0.1 }],
    };

    const parsed = JSON.parse(JSON.stringify(spec)) as Record<string, unknown>;
    expect(parsed.family).toBe('mixed');
    expect(parsed.cluster_level_vars).toEqual(['x']);
    expect(parsed.extra_groupings).toHaveLength(1);
    expect(parsed.slopes).toHaveLength(1);
    const scenarios = parsed.scenarios as Array<Record<string, unknown>>;
    expect(scenarios[0]?.random_effect_dist).toBe(1);
    expect(scenarios[0]?.icc_noise_sd).toBeCloseTo(0.05);
  });

  it('preserves slopes nested on an extra grouping', () => {
    const spec: AppSpec = {
      target_power: 0.8,
      extra_groupings: [
        {
          tau_squared: 0.16,
          relation: { kind: 'crossed', n_clusters: 6 },
          slopes: [{ predictor_name: 'x1', slope_variance: 0.10, slope_intercept_corr: 0.2 }],
        },
      ],
    };

    const parsed = JSON.parse(JSON.stringify(spec)) as Record<string, unknown>;
    expect(parsed.extra_groupings).toHaveLength(1);
    const extraGrouping = (parsed.extra_groupings as Array<Record<string, unknown>>)[0];
    expect(extraGrouping?.slopes).toEqual(
      spec.extra_groupings[0].slopes
    );
  });
});
