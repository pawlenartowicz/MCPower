import { describe, it, expect } from 'vitest';
import { generateAnovaScript } from './script-generator-anova';
import type { AppSpec } from './app-spec';

const anovaSpec: AppSpec = {
  family: 'anova',
  outcome: 'y',
  factors: [
    {
      name: 'treatment',
      levels: ['A', 'B', 'C'],
      reference_level: 'A',
      proportions: null,
    },
    {
      name: 'group',
      levels: ['old', 'new'],
      reference_level: 'old',
      proportions: null,
    },
  ],
  covariates: [{ name: 'age' }],
  effects: [
    { name: 'treatment[B]', value: 0.5 },
    { name: 'treatment[C]', value: 0.3 },
    { name: 'group[new]', value: 0.4 },
    { name: 'age', value: 0.2 },
  ],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 1600,
  seed: 2137,
  tests: { kind: 'contrasts', names: [] },
  correction: 'tukey_hsd',
  scenarios: [],
  csv: null,
  report_overall: true,
  contrasts: [['treatment[B]', 'treatment[C]']],
};

describe('generateAnovaScript', () => {
  it('lists factors before covariates, sets var-types for factors only, maps tukey', () => {
    const out = generateAnovaScript(anovaSpec, 'find-power', { sample_size: 80 });
    // Factors precede the covariate in the formula.
    const formulaMatch = out.match(/MCPower\("([^"]+)"\)/);
    expect(formulaMatch).not.toBeNull();
    const formula = formulaMatch![1]!;
    expect(formula.indexOf('treatment')).toBeLessThan(formula.indexOf('age'));
    expect(formula.indexOf('group')).toBeLessThan(formula.indexOf('age'));
    // set_variable_type uses the assignment grammar; covariate is omitted.
    const varTypeLine = out.split('\n').find((l) => l.includes('set_variable_type'))!;
    expect(varTypeLine).toContain('treatment=(factor,');
    expect(varTypeLine).toContain('group=(factor,');
    expect(varTypeLine).not.toContain('age');
    // tukey_hsd correction maps to 'tukey'.
    expect(out).toContain('correction="tukey"');
    expect(out).not.toContain('tukey_hsd');
    // No legacy params.
    expect(out).not.toContain('print_results');
    // fixture has alpha=0.05 / target_power=0.8 / seed=2137 / n_sims=1600 (== ols default) → omitted
    expect(out).not.toContain('set_alpha');
    expect(out).not.toContain('set_seed');
    expect(out).not.toContain('set_simulations');
    expect(out).toContain('model.find_power(');
    expect(out).toContain('sample_size=80,');
  });

  it('emits R constructor with ~ separator', () => {
    const out = generateAnovaScript(anovaSpec, 'find-power', { sample_size: 80 }, 'r');
    expect(out).toContain('model <- MCPower$new("y ~ treatment + group + age")');
  });

  it('emits a structurally valid find-sample-size snippet with grid bounds', () => {
    // Use to_size=250 (non-default) to verify both bounds are emitted.
    const out = generateAnovaScript(anovaSpec, 'find-sample-size', {
      bounds: [40, 250],
      method: { Grid: { by: { Fixed: 20 }, mode: 'Linear' } },
    });
    expect(out).toContain('model.find_sample_size(');
    expect(out).toContain('from_size=40,');
    expect(out).toContain('to_size=250,');
    expect(out).toContain('by=20,');
  });

  it('omits by when ByValue is Auto (Auto is the default)', () => {
    const out = generateAnovaScript(anovaSpec, 'find-sample-size', {
      bounds: [40, 200],
      method: { Grid: { by: { Auto: { count: 12 } }, mode: 'Linear' } },
    });
    // Auto is the port default — shared buildFindCallLines omits it
    expect(out).not.toContain('by=');
  });

  it('raises a type-guard error when called with a non-anova spec', () => {
    const linearSpec = { ...anovaSpec, family: 'linear' } as unknown as AppSpec;
    expect(() => generateAnovaScript(linearSpec, 'find-power', { sample_size: 80 })).toThrow();
  });

  it('maps non-none/non-tukey corrections to their port tokens', () => {
    const bh = generateAnovaScript({ ...anovaSpec, correction: 'benjamini_hochberg' }, 'find-power', { sample_size: 80 });
    expect(bh).toContain('correction="bh"');
    expect(bh).not.toContain('benjamini_hochberg');

    const bonf = generateAnovaScript({ ...anovaSpec, correction: 'bonferroni' }, 'find-power', { sample_size: 80 });
    expect(bonf).toContain('correction="bonferroni"');

    const holm = generateAnovaScript({ ...anovaSpec, correction: 'holm' }, 'find-power', { sample_size: 80 });
    expect(holm).toContain('correction="holm"');
  });
});
