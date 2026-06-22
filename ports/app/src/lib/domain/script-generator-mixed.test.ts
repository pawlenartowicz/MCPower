import { describe, it, expect } from 'vitest';
import { generateScript } from './script-generator';
import { generateMixedScript } from './script-generator-mixed';
import type { AppSpec, ClusterDim } from './app-spec';

function mixedSpec(dim: ClusterDim): AppSpec {
  return {
    family: 'mixed',
    parsed_formula: { outcome: 'y', predictors: ['x'], interaction_terms: [] },
    var_types: [{ kind: 'numeric', name: 'x' }],
    effects: [{ name: 'x', value: 0.5 }],
    correlations: null,
    alpha: 0.05,
    target_power: 0.8,
    n_sims: 800,
    seed: 2137,
    tests: { kind: 'all' },
    correction: 'none',
    scenarios: [],
    csv: null,
    report_overall: true,
    contrasts: [],
    cluster_name: 'school',
    icc: 0.2,
    cluster_dim: dim,
  } as AppSpec;
}

describe('generateScript (mixed)', () => {
  it('emits MCPower with the random-effects formula and set_cluster(n_clusters=)', () => {
    const s = generateScript(mixedSpec({ kind: 'n_clusters', value: 20 }), 'find-power', { sample_size: 200 });
    expect(s).toContain('MCPower("y ~ x + (1|school)", family="lme")');
    expect(s).toContain('model.set_cluster("school", n_clusters=20, ICC=0.2)');
    expect(s).toContain('model.find_power(');
  });

  it('uses cluster_size= for cluster_size mode', () => {
    const s = generateScript(mixedSpec({ kind: 'cluster_size', value: 30 }), 'find-power', { sample_size: 200 });
    expect(s).toContain('model.set_cluster("school", cluster_size=30, ICC=0.2)');
  });

  it('emits find_sample_size with grid bounds and step', () => {
    const s = generateScript(mixedSpec({ kind: 'n_clusters', value: 20 }), 'find-sample-size', {
      bounds: [40, 160],
      method: { Grid: { by: { Fixed: 20 }, mode: 'Linear' } },
    });
    expect(s).toContain('model.find_sample_size(');
    expect(s).toContain('from_size=40,');
    expect(s).toContain('to_size=160,');
    expect(s).toContain('by=20,');
  });

  it('omits by= when ByValue is Auto (port default)', () => {
    const s = generateScript(mixedSpec({ kind: 'n_clusters', value: 20 }), 'find-sample-size', {
      bounds: [40, 160],
      method: { Grid: { by: { Auto: { count: 12 } }, mode: 'Linear' } },
    });
    // Auto is the port default — omit from the call rather than emitting by="auto"
    expect(s).not.toContain('by=');
  });

  it('folds the test selection into target_test= when non-default', () => {
    const spec = mixedSpec({ kind: 'n_clusters', value: 20 });
    if (spec.family !== 'mixed') throw new Error('expected mixed');
    spec.report_overall = false; // non-default → enumerate the betas
    const s = generateScript(spec, 'find-power', { sample_size: 200 });
    expect(s).toContain('target_test="x",');
    expect(s).not.toContain('model.set_tests(');
  });

  it('emits crossed and nested groupings as formula terms + extra set_cluster calls', () => {
    const base = mixedSpec({ kind: 'n_clusters', value: 20 });
    const spec = {
      ...base,
      extra_groupings: [
        // gaussian outcome: τ² = icc/(1−icc) → icc 0.1 ↔ τ² 0.1/0.9
        { tau_squared: 0.1 / 0.9, relation: { kind: 'crossed', n_clusters: 8 }, cluster_name: 'item' },
        { tau_squared: 0.05 / 0.95, relation: { kind: 'nested_within', n_per_parent: 4 }, cluster_name: 'school:class' },
      ],
    } as AppSpec;
    const s = generateScript(spec, 'find-power', { sample_size: 200 });
    expect(s).toContain('MCPower("y ~ x + (1|school/class) + (1|item)", family="lme")');
    expect(s).toContain('model.set_cluster("item", n_clusters=8, ICC=0.1)');
    expect(s).toContain('model.set_cluster("school:class", n_per_parent=4, ICC=0.05)');
  });

  it('emits random slopes and cluster-level vars on the primary set_cluster', () => {
    const base = mixedSpec({ kind: 'n_clusters', value: 20 });
    const spec = {
      ...base,
      cluster_level_vars: ['x'],
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: -0.1 }],
    } as AppSpec;
    const s = generateScript(spec, 'find-power', { sample_size: 200 });
    expect(s).toContain(
      'model.set_cluster("school", n_clusters=20, ICC=0.2, random_slopes=["x"], slope_variance=0.05, slope_intercept_corr=-0.1, cluster_level_vars=["x"])',
    );
  });

  it('raises a type-guard error when called with a non-mixed spec', () => {
    const linearSpec = { ...mixedSpec({ kind: 'n_clusters', value: 20 }), family: 'linear' } as unknown as AppSpec;
    expect(() => generateMixedScript(linearSpec, 'find-power', { sample_size: 200 })).toThrow();
  });

  it('passes family="lme" and trims default config', () => {
    const spec = mixedSpec({ kind: 'n_clusters', value: 20 });
    const out = generateMixedScript(spec, 'find-power', { sample_size: 100 }, 'python');
    expect(out).toContain('family="lme"');
    expect(out).not.toContain('print_results');
    // fixture n_sims=800 = mixed default → set_simulations omitted
    expect(out).not.toContain('set_simulations');
    // fixture seed=2137 = default → set_seed omitted
    expect(out).not.toContain('set_seed');
  });

  it('binary outcome → family="logit" + baseline probability', () => {
    const s = {
      ...mixedSpec({ kind: 'n_clusters', value: 20 }),
      outcome: { kind: 'binary' as const, baseline_probability: 0.3 },
    } as AppSpec;
    const out = generateMixedScript(s, 'find-power', { sample_size: 100 }, 'python');
    expect(out).toContain('family="logit"');
    expect(out).toContain('model.set_baseline_probability(0.3)');
  });

  it('R output', () => {
    const spec = mixedSpec({ kind: 'n_clusters', value: 20 });
    const out = generateMixedScript(spec, 'find-power', { sample_size: 100 }, 'r');
    expect(out).toContain('MCPower$new("');
    expect(out).toContain('family = "lme"');
    expect(out).toContain('model$set_cluster(');
  });

  it('R output with slopes and cluster_level_vars uses R list syntax', () => {
    const base = mixedSpec({ kind: 'n_clusters', value: 20 });
    const spec = {
      ...base,
      cluster_level_vars: ['x'],
      slopes: [{ predictor_name: 'x', slope_variance: 0.05, slope_intercept_corr: -0.1 }],
    } as AppSpec;
    const out = generateMixedScript(spec, 'find-power', { sample_size: 200 }, 'r');
    // cluster_level_vars uses R c() vector syntax
    expect(out).toContain('cluster_level_vars = c("x")');
    // random_slopes uses R named-list syntax
    expect(out).toContain('list(predictor = "x", variance = 0.05, corr_with_intercept = -0.1)');
  });

  it('maps non-none corrections to their Python tokens', () => {
    const base = mixedSpec({ kind: 'n_clusters', value: 20 });
    const bh = generateScript({ ...base, correction: 'benjamini_hochberg' }, 'find-power', { sample_size: 200 });
    expect(bh).toContain('correction="bh"');
    expect(bh).not.toContain('benjamini_hochberg');

    const bonf = generateScript({ ...base, correction: 'bonferroni' }, 'find-power', { sample_size: 200 });
    expect(bonf).toContain('correction="bonferroni"');

    const holm = generateScript({ ...base, correction: 'holm' }, 'find-power', { sample_size: 200 });
    expect(holm).toContain('correction="holm"');
  });
});
