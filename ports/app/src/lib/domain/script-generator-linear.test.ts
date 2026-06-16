import { describe, it, expect } from 'vitest';
import {
  generateLinearScript,
  buildTestsArg,
  buildVarTypeString,
  buildCorrelationsString,
} from './script-generator-linear';
import type { AppSpec, LinearSpec } from './app-spec';
import type { ScriptParams } from './script-generator-linear';

// All-default run config: alpha 0.05, target_power 0.8, n_sims 1600, seed 2137, correction 'none'.
const linearSpec: AppSpec = {
  family: 'linear',
  parsed_formula: { outcome: 'y', predictors: ['x1', 'x2'], interaction_terms: [['x1', 'x2']] },
  var_types: [
    { kind: 'numeric', name: 'x1' },
    { kind: 'binary', name: 'x2', binary_proportion: 0.4 },
  ],
  effects: [
    { name: 'x1', value: 0.3 },
    { name: 'x2', value: 0.2 },
    { name: 'x1:x2', value: 0.1 },
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
  report_overall: false,
  contrasts: [],
};

describe('generateLinearScript', () => {
  it('emits a structurally valid find-power snippet (right calls/args)', () => {
    const out = generateLinearScript(linearSpec, 'find-power', { sample_size: 80 });
    // Structural: required import + constructor + setters present, find_power
    // call carries the sample_size, and the binary proportion / interaction
    // effect are reflected — not a frozen whitespace snapshot.
    expect(out).toContain('from mcpower import MCPower');
    expect(out).toContain('MCPower("y = x1 + x2 + x1:x2")');
    expect(out).toContain('model.set_variable_type("x2=(binary,0.4)")');
    expect(out).toContain('model.set_effects(');
    expect(out).toContain('x1:x2=0.1');
    expect(out).toContain('model.find_power(');
    expect(out).toContain('sample_size=80,');
    expect(out).not.toContain('find_sample_size(');
  });

  it('emits a structurally valid find-sample-size snippet with non-default grid bounds', () => {
    const out = generateLinearScript(linearSpec, 'find-sample-size', {
      bounds: [40, 300],
      method: { Grid: { by: { Fixed: 20 }, mode: 'Linear' } },
    });
    expect(out).toContain('model.find_sample_size(');
    expect(out).toContain('from_size=40,');
    expect(out).toContain('to_size=300,');
    expect(out).toContain('by=20,');
    expect(out).not.toContain('find_power(');
  });

  it('omits set_variable_type when all predictors are numeric', () => {
    const noVarTypes: AppSpec = {
      ...linearSpec,
      var_types: [
        { kind: 'numeric', name: 'x1' },
        { kind: 'numeric', name: 'x2' },
      ],
    };
    const out = generateLinearScript(noVarTypes, 'find-power', { sample_size: 80 });
    expect(out).not.toContain('set_variable_type');
  });

  it('raises a type-guard error when called with a non-linear spec', () => {
    const logitSpec = { ...linearSpec, family: 'logit' } as unknown as AppSpec;
    expect(() => generateLinearScript(logitSpec, 'find-power', { sample_size: 80 })).toThrow();
  });

  it('maps non-none corrections to their port tokens', () => {
    const bh = generateLinearScript({ ...linearSpec, correction: 'benjamini_hochberg' }, 'find-power', { sample_size: 80 });
    expect(bh).toContain('correction="bh"');
    expect(bh).not.toContain('benjamini_hochberg');

    const bonf = generateLinearScript({ ...linearSpec, correction: 'bonferroni' }, 'find-power', { sample_size: 80 });
    expect(bonf).toContain('correction="bonferroni"');

    const holm = generateLinearScript({ ...linearSpec, correction: 'holm' }, 'find-power', { sample_size: 80 });
    expect(holm).toContain('correction="holm"');
  });

  // --- defaults trimming ---

  it('omits set_alpha/set_power/set_simulations/set_seed at port defaults', () => {
    const out = generateLinearScript(linearSpec, 'find-power', { sample_size: 100 }, 'python');
    expect(out).not.toContain('set_alpha');
    expect(out).not.toContain('set_power');
    expect(out).not.toContain('set_simulations');
    expect(out).not.toContain('set_seed');
  });

  it('emits non-default config with set_power in percent', () => {
    const s = { ...linearSpec, alpha: 0.01, target_power: 0.9, n_sims: 5000, seed: 42 };
    const out = generateLinearScript(s, 'find-power', { sample_size: 100 }, 'python');
    expect(out).toContain('model.set_alpha(0.01)');
    expect(out).toContain('model.set_power(90)');     // percent, not 0.9
    expect(out).toContain('model.set_simulations(5000)');
    expect(out).toContain('model.set_seed(42)');
  });

  it('drops the v1-only call args and the invalid correction none', () => {
    const out = generateLinearScript(linearSpec, 'find-power', { sample_size: 100 }, 'python');
    expect(out).not.toContain('print_results');
    expect(out).not.toContain('return_results');
    expect(out).not.toContain('correction');          // spec.correction === 'none'
  });

  it('omits default find_sample_size bounds and auto by', () => {
    const out = generateLinearScript(linearSpec, 'find-sample-size', {
      bounds: [30, 200],
      method: { Grid: { by: { Auto: { count: 12 } }, mode: 'Linear' } },
    } as ScriptParams, 'python');
    expect(out).not.toContain('from_size');
    expect(out).not.toContain('to_size');
    expect(out).not.toContain('by=');
    // non-default bounds still appear:
    const out2 = generateLinearScript(linearSpec, 'find-sample-size', {
      bounds: [50, 400],
      method: { Grid: { by: { Fixed: 25 }, mode: 'Linear' } },
    } as ScriptParams, 'python');
    expect(out2).toContain('from_size=50,');
    expect(out2).toContain('to_size=400,');
    expect(out2).toContain('by=25,');
  });

  it('generates an R script', () => {
    const out = generateLinearScript(linearSpec, 'find-power', { sample_size: 100 }, 'r');
    expect(out).toContain('library(mcpower)');
    expect(out).toContain('install.packages("mcpower", repos = "https://pawlenartowicz.r-universe.dev")');
    expect(out).toContain('model <- MCPower$new("');
    expect(out).toContain('y ~ x1');                   // R formula separator
    expect(out).toContain('model$set_effects("');
    expect(out).toContain('result <- model$find_power(');
    expect(out).not.toContain('model.');               // no Python syntax leaked
  });
});

// APIF-32: buildTestsArg default-vs-deviation shape.
describe('buildTestsArg', () => {
  const base = linearSpec as LinearSpec;

  it('returns null for the family default (all + report_overall + no contrasts)', () => {
    const s: LinearSpec = { ...base, tests: { kind: 'all' }, report_overall: true, contrasts: [] };
    expect(buildTestsArg(s)).toBeNull();
  });

  it('returns a non-null string when report_overall is disabled', () => {
    const s: LinearSpec = { ...base, tests: { kind: 'all' }, report_overall: false, contrasts: [] };
    expect(buildTestsArg(s)).not.toBeNull();
  });

  it('returns a non-null string when a contrast pair is present', () => {
    const s: LinearSpec = {
      ...base,
      tests: { kind: 'all' },
      report_overall: true,
      contrasts: [['x1', 'x2']],
    };
    expect(buildTestsArg(s)).not.toBeNull();
  });

  it('returns a non-null string when tests.kind is effects', () => {
    const s: LinearSpec = {
      ...base,
      tests: { kind: 'effects', names: ['x1'] },
      report_overall: true,
      contrasts: [],
    };
    expect(buildTestsArg(s)).not.toBeNull();
  });
});

// APIF-33: buildVarTypeString shape — null when all numeric, one token per non-numeric.
describe('buildVarTypeString', () => {
  it('returns null when every variable is numeric', () => {
    expect(
      buildVarTypeString([
        { kind: 'numeric', name: 'x1' },
        { kind: 'numeric', name: 'x2' },
      ]),
    ).toBeNull();
  });

  it('emits a binary token for a binary variable', () => {
    const out = buildVarTypeString([{ kind: 'binary', name: 'x2', binary_proportion: 0.4 }]);
    expect(out).not.toBeNull();
    expect(out).toContain('x2=(binary,0.4)');
  });

  it('emits one token per non-numeric variable, skipping numerics', () => {
    const out = buildVarTypeString([
      { kind: 'numeric', name: 'x1' },
      { kind: 'binary', name: 'x2', binary_proportion: 0.4 },
      { kind: 'factor', name: 'f', factor_n_levels: 3, factor_proportions: [0.3, 0.3, 0.4] },
    ])!;
    expect(out).not.toBeNull();
    expect(out.split(', ').length).toBe(2); // x2 + f, not x1
    expect(out).toContain('x2=(binary,0.4)');
    expect(out).toContain('f=(factor,0.3,0.3,0.4)');
    expect(out).not.toContain('x1');
  });
});

// APIF-34: buildCorrelationsString shape — null when absent/all-zero, only non-zero upper triangle.
describe('buildCorrelationsString', () => {
  it('returns null when correlations are absent', () => {
    expect(buildCorrelationsString({ ...(linearSpec as LinearSpec), correlations: null })).toBeNull();
  });

  it('returns null when every off-diagonal entry is zero', () => {
    const s: LinearSpec = {
      ...(linearSpec as LinearSpec),
      correlations: { names: ['a', 'b'], values: [[1, 0], [0, 1]] },
    };
    expect(buildCorrelationsString(s)).toBeNull();
  });

  it('emits only non-zero upper-triangle pairs in a,b=v notation', () => {
    const s: LinearSpec = {
      ...(linearSpec as LinearSpec),
      correlations: { names: ['a', 'b', 'c'], values: [[1, 0.2, 0], [0.2, 1, 0], [0, 0, 1]] },
    };
    const out = buildCorrelationsString(s)!;
    expect(out).not.toBeNull();
    expect(out).toContain('a,b=0.2');
    expect(out).not.toContain('c'); // c has no non-zero correlation
    expect(out.split(',').length).toBe(2); // exactly one pair → "a,b=0.2"
  });
});
