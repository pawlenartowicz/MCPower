import { describe, it, expect } from 'vitest';
import { generateLogitScript } from './script-generator-logit';
import type { AppSpec } from './app-spec';

// All-default run config: alpha 0.05, target_power 0.8, n_sims 1600 (ols default), seed 2137.
function sampleLogitSpec(): AppSpec {
  return {
    family: 'logit',
    parsed_formula: { outcome: 'y', predictors: ['x1', 'x2'], interaction_terms: [] },
    var_types: [
      { kind: 'numeric', name: 'x1' },
      { kind: 'binary', name: 'x2', binary_proportion: 0.4 },
    ],
    effects: [
      { name: 'x1', value: 0.3 },
      { name: 'x2', value: 0.2 },
    ],
    correlations: null,
    alpha: 0.05,
    target_power: 0.8,
    n_sims: 1600,
    seed: 2137,
    tests: { kind: 'all' },
    correction: 'none',
    scenarios: [],
    csv: null,
    report_overall: true,
    contrasts: [],
    baseline_probability: 0.3,
  };
}

describe('generateLogitScript', () => {
  it('emits a structurally valid find-power snippet with family="logit" and baseline', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-power', { sample_size: 80 });
    expect(out).toContain('from mcpower import MCPower');
    expect(out).toContain('MCPower("y = x1 + x2", family="logit")');
    expect(out).toContain('model.set_baseline_probability(0.3)');
    expect(out).toContain('model.set_effects(');
    expect(out).toContain('model.find_power(');
    expect(out).toContain('sample_size=80,');
    expect(out).not.toContain('find_sample_size(');
  });

  it('emits a structurally valid find-sample-size snippet with non-default grid bounds', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-sample-size', {
      bounds: [50, 400],
      method: { Grid: { by: { Fixed: 10 }, mode: 'Linear' } },
    });
    expect(out).toContain('MCPower("y = x1 + x2", family="logit")');
    expect(out).toContain('model.find_sample_size(');
    expect(out).toContain('from_size=50,');
    expect(out).toContain('to_size=400,');
    expect(out).toContain('by=10,');
  });

  it('omits by when ByValue is Auto (port default)', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-sample-size', {
      bounds: [30, 200],
      method: { Grid: { by: { Auto: { count: 12 } }, mode: 'Linear' } },
    });
    expect(out).not.toContain('by=');
  });

  it('raises a type-guard error when called with a non-logit spec', () => {
    const linearSpec = { ...sampleLogitSpec(), family: 'linear' } as unknown as AppSpec;
    expect(() => generateLogitScript(linearSpec, 'find-power', { sample_size: 80 })).toThrow();
  });

  it('maps non-none corrections to their Python tokens', () => {
    const bh = generateLogitScript({ ...sampleLogitSpec(), correction: 'benjamini_hochberg' }, 'find-power', { sample_size: 80 });
    expect(bh).toContain('correction="bh"');
    expect(bh).not.toContain('benjamini_hochberg');

    const bonf = generateLogitScript({ ...sampleLogitSpec(), correction: 'bonferroni' }, 'find-power', { sample_size: 80 });
    expect(bonf).toContain('correction="bonferroni"');

    const holm = generateLogitScript({ ...sampleLogitSpec(), correction: 'holm' }, 'find-power', { sample_size: 80 });
    expect(holm).toContain('correction="holm"');
  });

  // --- defaults trimming ---

  it('omits set_alpha/set_power/set_simulations/set_seed at port defaults', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-power', { sample_size: 100 });
    expect(out).not.toContain('set_alpha');
    expect(out).not.toContain('set_power');
    expect(out).not.toContain('set_simulations');
    expect(out).not.toContain('set_seed');
  });

  it('emits non-default config with set_power in percent', () => {
    const s = { ...sampleLogitSpec(), alpha: 0.01, target_power: 0.9, n_sims: 5000, seed: 42 };
    const out = generateLogitScript(s, 'find-power', { sample_size: 100 });
    expect(out).toContain('model.set_alpha(0.01)');
    expect(out).toContain('model.set_power(90)');
    expect(out).toContain('model.set_simulations(5000)');
    expect(out).toContain('model.set_seed(42)');
  });

  it('drops the v1-only call args and the invalid correction none', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-power', { sample_size: 100 });
    expect(out).not.toContain('print_results');
    expect(out).not.toContain('return_results');
    expect(out).not.toContain('correction');   // spec.correction === 'none'
  });

  it('omits default find_sample_size bounds and auto by', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-sample-size', {
      bounds: [30, 200],
      method: { Grid: { by: { Auto: { count: 12 } }, mode: 'Linear' } },
    });
    expect(out).not.toContain('from_size');
    expect(out).not.toContain('to_size');
    expect(out).not.toContain('by=');
  });

  it('emits family="probit" when the link is probit', () => {
    const out = generateLogitScript({ ...sampleLogitSpec(), link: 'probit' } as AppSpec, 'find-power', { sample_size: 80 });
    expect(out).toContain('MCPower("y = x1 + x2", family="probit")');
    expect(out).toContain('model.set_baseline_probability(0.3)');
  });

  it('generates an R script', () => {
    const out = generateLogitScript(sampleLogitSpec(), 'find-power', { sample_size: 100 }, 'r');
    expect(out).toContain('library(mcpower)');
    expect(out).toContain('install.packages("mcpower", repos = "https://r.mcpower.app")');
    expect(out).toContain('model <- MCPower$new("');
    expect(out).toContain('y ~ x1 + x2');              // R formula separator; both predictors present
    expect(out).toContain('family = "logit"');
    expect(out).toContain('model$set_baseline_probability(');
    expect(out).toContain('result <- model$find_power(');
    expect(out).not.toContain('model.');               // no Python syntax leaked
  });
});
