import { describe, expect, it, vi } from 'vitest';
import type { CsvData, UploadMode } from './app-spec';
import type { ScenarioConfig } from '$lib/configs/scenarios';

// uploadStoreMock must be created via vi.hoisted so it is available inside the vi.mock factory
// (vi.mock calls are hoisted to the top of the module before other imports run).
const uploadStoreMock = vi.hoisted<{ csvData: CsvData | null; mode: UploadMode }>(() => ({
  csvData: null,
  mode: 'none',
}));
const sharedPrefsMock = vi.hoisted<{ scenariosEnabled: boolean }>(() => ({ scenariosEnabled: false }));
const scenariosStoreMock = vi.hoisted<{ scenarios: ScenarioConfig[] }>(() => ({ scenarios: [] }));

vi.mock('$lib/stores/parsed-formula.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/stores/parsed-formula.svelte')>();
  const { stubParseFormula } = await import('../../tests/parse-formula-stub');
  return {
    ...actual,
    parsedFormulaStore: {
      get: (formula: string) => stubParseFormula(formula),
      getStable: (formula: string) => stubParseFormula(formula),
    },
  };
});

vi.mock('$lib/stores/upload.svelte', () => ({
  uploadStore: uploadStoreMock,
}));

// The adapter reads these two stores when projecting scenarios; mock them so the
// Scenarios toggle and the editable set list are controllable per test (and so the
// real persistence side-effects aren't pulled in).
vi.mock('$lib/stores/shared-prefs.svelte', () => ({ sharedPrefs: sharedPrefsMock }));
vi.mock('$lib/stores/scenarios.svelte', () => ({ scenariosStore: scenariosStoreMock }));

import { familyConfigToAppSpec } from './app-spec-adapter';
import type { FamilyConfig } from './family';
import { defaultFamilyConfig } from './family';

function linearConfig(overrides: Partial<FamilyConfig> = {}): FamilyConfig {
  return { ...defaultFamilyConfig('regression'), ...overrides };
}

describe('familyConfigToAppSpec(linear, ...)', () => {
  it('happy_path_two_predictors: emits a spec with 2 effects and no errors', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1 + x2',
      variables: [
        { name: 'x1', kind: 'continuous' },
        { name: 'x2', kind: 'continuous' },
      ],
      effects: [
        { name: 'x1', value: 0.3 },
        { name: 'x2', value: 0.2 },
      ],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec).not.toBeNull();
    expect(r.spec!.family).toBe('linear');
    expect(r.spec!.effects.length).toBe(2);
  });

  it('interaction_term_included: y ~ x1 + x2 + x1:x2 with named effect', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1 + x2 + x1:x2',
      variables: [
        { name: 'x1', kind: 'continuous' },
        { name: 'x2', kind: 'continuous' },
      ],
      effects: [
        { name: 'x1', value: 0.3 },
        { name: 'x2', value: 0.2 },
        { name: 'x1:x2', value: 0.1 },
      ],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec!.effects.find((e) => e.name === 'x1:x2')!.value).toBe(0.1);
  });

  it('factor_interaction_expands_to_per_level_effects: y ~ x1 + group + x1:group', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1 + group + x1:group',
      variables: [
        { name: 'x1', kind: 'continuous' },
        { name: 'group', kind: 'factor', nLevels: 3, levelProportions: [1 / 3, 1 / 3, 1 / 3] },
      ],
      effects: [
        { name: 'x1', value: 0.3 },
        { name: 'x1:group[2]', value: 0.1 },
        { name: 'x1:group[3]', value: 0.2 },
      ],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    const names = r.spec!.effects.map((e) => e.name);
    // per-level interaction columns the engine expects, with the user's values
    expect(r.spec!.effects.find((e) => e.name === 'x1:group[2]')!.value).toBe(0.1);
    expect(r.spec!.effects.find((e) => e.name === 'x1:group[3]')!.value).toBe(0.2);
    // never the coarse name (the engine has no such column → validate gate would fail)
    expect(names).not.toContain('x1:group');
    // the full target set: continuous + non-reference factor dummies + per-level interaction
    expect(names).toEqual(['x1', 'group[2]', 'group[3]', 'x1:group[2]', 'x1:group[3]']);
  });

  it('unknown_effect_name_errors', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x2', value: 0.5 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors.some((e) => e.includes('x2'))).toBe(true);
  });

  it('binary_proportion_out_of_range_errors', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'binary', binaryProportion: 1.5 }],
      effects: [{ name: 'x1', value: 0.5 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors.some((e) => /binary_proportion|\[0,1\]/.test(e))).toBe(true);
  });

  it('factor_shares_are_weights_normalized_to_proportions', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'factor', nLevels: 3, levelProportions: [0.5, 0.5, 0.5] }],
      effects: [
        { name: 'x1[2]', value: 0.5 },
        { name: 'x1[3]', value: 0.5 },
      ],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    const vt = r.spec.var_types[0]!;
    if (vt.kind !== 'factor') throw new Error('expected factor');
    expect(vt.factor_proportions.every((p) => Math.abs(p - 1 / 3) < 1e-12)).toBe(true);
  });

  it('factor_zero_share_errors', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'factor', nLevels: 3, levelProportions: [0.5, 0, 0.5] }],
      effects: [],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors.some((e) => /share must be > 0/.test(e))).toBe(true);
  });

  it('factor_labels_and_sampled_shares_reach_the_wire', () => {
    const cfg = linearConfig({
      formula: 'y ~ origin',
      variables: [
        {
          name: 'origin',
          kind: 'factor',
          nLevels: 3,
          levelProportions: [0.4, 0.3, 0.3],
          levels: ['Europe', 'Japan', 'USA'],
          referenceLevel: 'Japan',
          sampledProportions: true,
        },
      ],
      effects: [
        { name: 'origin[Europe]', value: 0.4 },
        { name: 'origin[USA]', value: 0.2 },
      ],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    const vt = r.spec.var_types[0]!;
    if (vt.kind !== 'factor') throw new Error('expected factor');
    expect(vt.factor_labels).toEqual(['Europe', 'Japan', 'USA']);
    expect(vt.factor_reference).toBe(1);
    expect(vt.sampled_proportions).toBe(true);
  });

  it('blank_factor_labels_fall_back_per_slot', () => {
    const cfg = linearConfig({
      formula: 'y ~ g',
      variables: [
        {
          name: 'g',
          kind: 'factor',
          nLevels: 3,
          levelProportions: [1 / 3, 1 / 3, 1 / 3],
          levels: ['A', '', ''],
        },
      ],
      effects: [],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    const vt = r.spec.var_types[0]!;
    if (vt.kind !== 'factor') throw new Error('expected factor');
    expect(vt.factor_labels).toEqual(['A', '2', '3']);
  });

  it('duplicate_factor_labels_error', () => {
    const cfg = linearConfig({
      formula: 'y ~ g',
      variables: [
        {
          name: 'g',
          kind: 'factor',
          nLevels: 3,
          levelProportions: [1 / 3, 1 / 3, 1 / 3],
          levels: ['A', 'B', 'A'],
        },
      ],
      effects: [],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors.some((e) => /duplicate level label/.test(e))).toBe(true);
  });

  it('continuous_distribution_reaches_the_wire_and_normal_is_omitted', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1 + x2',
      variables: [
        { name: 'x1', kind: 'continuous', distribution: 'right_skewed' },
        { name: 'x2', kind: 'continuous', distribution: 'normal' },
      ],
      effects: [],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    expect(r.spec.var_types[0]).toEqual({ kind: 'numeric', name: 'x1', distribution: 'right_skewed' });
    expect(r.spec.var_types[1]).toEqual({ kind: 'numeric', name: 'x2' });
  });

  it('missing_effect_defaults_to_zero', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec!.effects[0]?.value).toBe(0);
  });

  it('numeric_default_when_var_type_unspecified', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [], // no entry for x1
      effects: [{ name: 'x1', value: 0.3 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec?.family).toBe('linear');
    if (r.spec?.family === 'linear') {
      expect(r.spec.var_types[0]).toEqual({ kind: 'numeric', name: 'x1' });
    }
  });

  it('formula_parse_error_propagates', () => {
    const cfg = linearConfig({ formula: 'y ~ ' });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.spec).toBeNull();
    expect(r.errors[0]).toMatch(/predictor/i);
  });

  it('projects the optimistic scenario when the Scenarios toggle is off (engine fast path)', () => {
    sharedPrefsMock.scenariosEnabled = false;
    scenariosStoreMock.scenarios = [
      {
        name: 'optimistic',
        heterogeneity: 0.0,
        heteroskedasticity_ratio: 1.0,
        correlation_noise_sd: 0.0,
        distribution_change_prob: 0.0,
        new_distributions: [],
        residual_change_prob: 0.0,
        residual_dists: [],
        residual_df: 10,
        sampled_factor_proportions: false,
        lme: { random_effect_dist: 'normal', random_effect_df: 0, icc_noise_sd: 0 },
      },
    ];
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    // Sends the optimistic scenario (all magnitudes neutral → engine optimistic fast path)
    expect(r.spec!.scenarios).toHaveLength(1);
    expect(r.spec!.scenarios[0]!.name).toBe('optimistic');
    expect(r.spec!.scenarios[0]!.heterogeneity).toBe(0);
    expect(r.spec!.scenarios[0]!.heteroskedasticity_ratio).toBe(1);
    // lme fields zeroed (family-agnostic merge)
    expect(r.spec!.scenarios[0]!.random_effect_dist).toBe(0);
    expect(r.spec!.scenarios[0]!.icc_noise_sd).toBe(0);
    scenariosStoreMock.scenarios = [];
  });

  it('projects enabled scenarios from the store (lme dropped, residual encoded) when the toggle is on', () => {
    sharedPrefsMock.scenariosEnabled = true;
    scenariosStoreMock.scenarios = [
      {
        name: 'realistic',
        heterogeneity: 0.1,
        heteroskedasticity_ratio: 1.5,
        correlation_noise_sd: 0.2,
        distribution_change_prob: 0.0,
        new_distributions: [],
        residual_change_prob: 0.3,
        residual_dists: ['high_kurtosis'], // canonical name (replaces legacy 'heavy_tailed')
        residual_df: 5,
        sampled_factor_proportions: false,
        lme: { random_effect_dist: 'normal', random_effect_df: 0, icc_noise_sd: 0 },
      },
    ];
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec!.scenarios).toHaveLength(1);
    const wire = r.spec!.scenarios[0]!;
    expect(wire.name).toBe('realistic');
    expect(wire.heteroskedasticity_ratio).toBe(1.5);
    expect(wire.residual_dists).toEqual([4]); // "high_kurtosis" → RESIDUAL_CODE 4
    expect('lme' in wire).toBe(false); // lme sub-object gone — its fields are hoisted
    expect(wire.random_effect_dist).toBe(0); // 'normal' → 0
    expect(wire.random_effect_df).toBe(0);
    expect(wire.icc_noise_sd).toBe(0);
    // restore defaults so later tests see the toggle off
    sharedPrefsMock.scenariosEnabled = false;
    scenariosStoreMock.scenarios = [];
  });

  it('projects reportOverall=true through to report_overall', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      reportOverall: true,
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec!.report_overall).toBe(true);
  });

  it('projects reportOverall=false through to report_overall', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      reportOverall: false,
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.spec!.report_overall).toBe(false);
  });
});

describe('familyConfigToAppSpec — Logit', () => {
  function baseLogitConfig() {
    const cfg = defaultFamilyConfig('regression');
    cfg.formula = 'y ~ x1 + x2';
    cfg.effects = [
      { name: 'x1', value: 0.3 },
      { name: 'x2', value: 0.2 },
    ];
    return cfg;
  }
  it('projects a valid logit config with default baseline_probability', () => {
    const cfg = baseLogitConfig();
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(errors).toEqual([]);
    expect(spec).not.toBeNull();
    expect(spec!.family).toBe('logit');
    if (spec!.family === 'logit') {
      expect(spec!.baseline_probability).toBe(0.2);
      expect(spec!.parsed_formula.predictors).toEqual(['x1', 'x2']);
      expect(spec!.effects.find((e) => e.name === 'x1')!.value).toBe(0.3);
    }
  });
  it('reports an error when baseline_probability is out of range (low)', () => {
    const cfg = baseLogitConfig();
    cfg.baselineProbability = 0;
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/baseline/i);
  });
  it('reports an error when baseline_probability is out of range (high)', () => {
    const cfg = baseLogitConfig();
    cfg.baselineProbability = 1;
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/baseline/i);
  });
  it('rejects tukey correction (anova-only) the same way Linear does', () => {
    const cfg = baseLogitConfig();
    cfg.advanced.correction = 'tukey';
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(errors.join(' ')).toMatch(/tukey/i);
    expect(spec).not.toBeNull();
    if (spec!.family === 'logit') {
      expect(spec!.correction).toBe('none');
    }
  });
});

describe('familyConfigToAppSpec — Mixed', () => {
  it('builds a mixed AppSpec with cluster_dim and formula-derived cluster name', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'ignored',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    expect(spec?.family).toBe('mixed');
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.cluster_name).toBe('school'); // from the formula, not cfg
    expect(spec.icc).toBe(0.2);
    expect(spec.cluster_dim).toEqual({ kind: 'n_clusters', value: 20 });
    expect(spec.parsed_formula.predictors).toEqual(['x']); // (1|school) stripped
  });

  it('forces report_overall=false for the mixed family even if reportOverall is set', () => {
    // The omnibus is not defined for a mixed-effects fit; the mixed family must
    // never request it, overriding any stale persisted reportOverall=true.
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.reportOverall = true; // stale / leftover
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    expect(spec?.report_overall).toBe(false);
  });

  it('errors when a Mixed formula lacks a (1|cluster) term', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/\(1\|cluster\)/);
  });

  it('builds cluster_size cluster_dim when dimKind is cluster_size', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|g)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'g',
      icc: 0.1,
      dimKind: 'cluster_size',
      nClusters: 20,
      clusterSize: 25,
    };
    const { spec } = familyConfigToAppSpec('mixed', cfg);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.cluster_dim).toEqual({ kind: 'cluster_size', value: 25 });
  });

  it('rejects an icc outside [0, 1)', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|g)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'g',
      icc: 1.0,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/icc out of range/i);
  });

  it('errors when the cluster configuration is missing', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|g)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = undefined;
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/cluster configuration missing/i);
  });

  // Growth-field plumbing: the adapter forwards cluster_level_vars /
  // extra_groupings / slopes from ClusterConfig to the MixedSpec wire. (The
  // ClusterEditor UI that populates these config fields is owned separately.)

  it('forwards cluster_level_vars from ClusterConfig to MixedSpec wire', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + z + (1|school)';
    cfg.effects = [
      { name: 'x', value: 0.5 },
      { name: 'z', value: 0.2 },
    ];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
      clusterLevelVars: ['z'],
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.cluster_level_vars).toEqual(['z']);
  });

  it('crossed extra grouping comes from a second (1|g) term; icc converts to τ²', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school) + (1|district)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
      extraGroupings: [{ clusterName: 'district', icc: 0.1, relation: 'crossed', n: 8 }],
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.extra_groupings).toHaveLength(1);
    const g = spec.extra_groupings![0]!;
    expect(g.relation).toEqual({ kind: 'crossed', n_clusters: 8 });
    expect(g.tau_squared).toBeCloseTo(0.1 / 0.9, 12); // icc/(1−icc), gaussian scale
  });

  it('nested grouping comes from (1|parent/child) formula syntax', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school/class)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
      extraGroupings: [{ clusterName: 'school:class', icc: 0.05, relation: 'nested', n: 5 }],
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.cluster_name).toBe('school');
    expect(spec.extra_groupings![0]!.relation).toEqual({ kind: 'nested_within', n_per_parent: 5 });
  });

  it('an extra grouping with no card config yet falls back to shared defaults', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school) + (1|district)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    const g = spec.extra_groupings![0]!;
    expect(g.relation).toEqual({ kind: 'crossed', n_clusters: 10 });
    expect(g.tau_squared).toBeCloseTo(0.05 / 0.95, 12);
  });

  it('extra groupings reject cluster-size sizing on the primary', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school) + (1|district)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'cluster_size',
      nClusters: 20,
      clusterSize: 30,
    };
    const { errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors.join(' ')).toMatch(/by n clusters/);
  });

  it('forwards formula slopes with card params (name carried, not yet column-resolved)', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1+x|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
      slopes: [{ predictorName: 'x', slopeVariance: 0.05, slopeInterceptCorr: -0.1 }],
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.slopes).toHaveLength(1);
    expect(spec.slopes![0]).toEqual({
      predictor_name: 'x',
      slope_variance: 0.05,
      slope_intercept_corr: -0.1,
    });
  });

  it('ignores config slopes whose predictor is not a formula slope (formula-driven set)', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
      // Stale entry (e.g. left over from an earlier formula) — must not reach the wire.
      slopes: [{ predictorName: 'stale', slopeVariance: 0.05, slopeInterceptCorr: -0.1 }],
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.slopes).toEqual([]);
  });

  it("errors when a formula slope var is not a fixed predictor (engine can't resolve it)", () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1+w|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors.join(' ')).toMatch(/random slope on 'w'.*fixed predictor/);
  });

  it('a (1+x|g) formula slope is included even without card config', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1+x|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school',
      icc: 0.2,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.cluster_name).toBe('school');
    expect(spec.slopes).toHaveLength(1);
    expect(spec.slopes![0]).toEqual({
      predictor_name: 'x',
      slope_variance: 0.1, // SLOPE_DEFAULTS.variance
      slope_intercept_corr: 0,
    });
  });

  it('binary mixed outcome: adapter maps binary cluster config to outcome.kind === binary', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school', icc: 0.2, dimKind: 'n_clusters',
      nClusters: 20, clusterSize: 30,
      binaryOutcome: true, baselineProbability: 0.3,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.outcome?.kind).toBe('binary');
    expect((spec.outcome as { kind: 'binary'; baseline_probability: number }).baseline_probability).toBeCloseTo(0.3);
  });

  it('gaussian mixed outcome: outcome absent or kind gaussian', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = { clusterName: 'school', icc: 0.2, dimKind: 'n_clusters', nClusters: 20, clusterSize: 30 };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.outcome == null || spec.outcome.kind === 'gaussian').toBe(true);
  });

  it('toggle-back: a lingering baselineProbability stays off the wire when binaryOutcome is false', () => {
    // After flipping Mixed Binary→Continuous the UI clears binaryOutcome but leaves
    // baselineProbability (the seeded 0.2). The adapter reads the baseline only when
    // binaryOutcome is true, so the wire must be a clean Gaussian LME — no stale binary.
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'school', icc: 0.2, dimKind: 'n_clusters',
      nClusters: 20, clusterSize: 30,
      binaryOutcome: false, baselineProbability: 0.2, // lingering
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect(spec.outcome == null || spec.outcome.kind === 'gaussian').toBe(true);
  });
});

describe('familyConfigToAppSpec — ANOVA', () => {
  function anovaCfg(overrides: Partial<FamilyConfig> = {}): FamilyConfig {
    return {
      ...defaultFamilyConfig('anova'),
      variables: [
        {
          name: 'F1',
          kind: 'factor',
          role: 'factor',
          nLevels: 3,
          levelProportions: [1 / 3, 1 / 3, 1 / 3],
        },
        { name: 'cov1', kind: 'continuous', role: 'covariate' },
      ],
      effects: [
        { name: 'F1[2]', value: 0.4 },
        { name: 'F1[3]', value: 0.4 },
        { name: 'cov1', value: 0.3 },
      ],
      tests: { kind: 'effects', names: ['F1[2]', 'F1[3]'] },
      ...overrides,
    };
  }

  it('ANOVA projects to a Linear spec with no parser involvement', () => {
    const { spec, errors } = familyConfigToAppSpec('anova', anovaCfg());
    expect(errors).toEqual([]);
    expect(spec?.family).toBe('linear');
    expect((spec as any)?.parsed_formula).toEqual({
      outcome: 'y',
      predictors: ['F1', 'cov1'],
      interaction_terms: [],
    });
  });

  it('expands factor effects to non-reference dummies (drops F1[1])', () => {
    const { spec } = familyConfigToAppSpec('anova', anovaCfg());
    const names = (spec as any).effects.map((e: any) => e.name).sort();
    expect(names).toEqual(['F1[2]', 'F1[3]', 'cov1'].sort());
  });

  it('allows tukey for ANOVA', () => {
    const { spec, errors } = familyConfigToAppSpec(
      'anova',
      anovaCfg({ advanced: { ...defaultFamilyConfig('anova').advanced, correction: 'tukey' } }),
    );
    expect(errors).toEqual([]);
    expect((spec as any).correction).toBe('tukey_hsd');
  });

  it('rejects tukey for regression', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1',
      effects: [{ name: 'x1', value: 0.3 }],
      advanced: { ...defaultFamilyConfig('regression').advanced, correction: 'tukey' as const },
    };
    const { errors } = familyConfigToAppSpec('regression', cfg);
    expect(errors).toContain('tukey correction is ANOVA-only');
  });

  it('restricts wire tests to covariate effects (factor dummies dropped, contrasts intact)', () => {
    const { spec } = familyConfigToAppSpec(
      'anova',
      anovaCfg({
        tests: { kind: 'effects', names: ['F1[2]', 'F1[3]', 'cov1'] },
        contrasts: [{ positiveName: 'F1[2]', negativeName: 'F1[1]', enabled: true }],
      }),
    );
    // Factor levels reach the engine only through contrasts; the per-coefficient
    // test list is covariates only.
    expect((spec as any).tests).toEqual({ kind: 'effects', names: ['cov1'] });
    expect((spec as any).contrasts).toEqual([['F1[2]', 'F1[1]']]);
  });

  it("maps tests:'all' to the covariate effect set (no factor marginals)", () => {
    const { spec } = familyConfigToAppSpec(
      'anova',
      anovaCfg({
        tests: { kind: 'all' },
        contrasts: [{ positiveName: 'F1[2]', negativeName: 'F1[1]', enabled: true }],
      }),
    );
    expect((spec as any).tests).toEqual({ kind: 'effects', names: ['cov1'] });
  });

  it('falls back to tests:all when no effect target and no contrast', () => {
    const { spec } = familyConfigToAppSpec(
      'anova',
      anovaCfg({ tests: { kind: 'effects', names: [] }, contrasts: [] }),
    );
    expect((spec as any).tests).toEqual({ kind: 'all' });
  });

  it('errors when there are no factors', () => {
    const cfg = anovaCfg({ variables: [{ name: 'cov1', kind: 'continuous', role: 'covariate' }] });
    const { spec, errors } = familyConfigToAppSpec('anova', cfg);
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/factor/i);
  });

  it('regression: factor predictor effects expand to per-level dummies', () => {
    const cfg = {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ g',
      variables: [
        { name: 'g', kind: 'factor' as const, nLevels: 3, levelProportions: [1 / 3, 1 / 3, 1 / 3] },
      ],
      effects: [
        { name: 'g[2]', value: 0.4 },
        { name: 'g[3]', value: 0.4 },
      ],
    };
    const { spec, errors } = familyConfigToAppSpec('regression', cfg);
    expect(errors).toEqual([]);
    const names = (spec as any).effects.map((e: any) => e.name).sort();
    expect(names).toEqual(['g[2]', 'g[3]']);
  });
});

describe('familyConfigToAppSpec — Regression entrypoint + outcome toggle', () => {
  function regressionConfig(overrides: Partial<FamilyConfig> = {}): FamilyConfig {
    return {
      ...defaultFamilyConfig('regression'),
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      ...overrides,
    };
  }

  it('continuous outcome → AppSpec family: linear', () => {
    const cfg = regressionConfig();
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(errors).toEqual([]);
    expect(spec?.family).toBe('linear');
  });

  it('binary outcome + valid baseline → AppSpec family: logit', () => {
    const cfg = regressionConfig({ baselineProbability: 0.3 });
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(errors).toEqual([]);
    expect(spec?.family).toBe('logit');
    if (spec?.family === 'logit') {
      expect(spec.baseline_probability).toBe(0.3);
    }
  });

  it('binary outcome + missing baseline → null spec with baseline error', () => {
    const cfg = regressionConfig({ baselineProbability: undefined });
    const { spec, errors } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/baseline/i);
  });

  it('binary outcome + out-of-range baseline → null spec', () => {
    const cfg = regressionConfig({ baselineProbability: 1 });
    const { spec } = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(spec).toBeNull();
  });
});

describe('familyConfigToAppSpec — lme scenario zeroing (contract invariant 13)', () => {
  function lmeScenario(): ScenarioConfig {
    return {
      name: 'doomer',
      heterogeneity: 0,
      heteroskedasticity_ratio: 1,
      correlation_noise_sd: 0,
      distribution_change_prob: 0,
      new_distributions: [],
      residual_change_prob: 0,
      residual_dists: [],
      residual_df: 0,
      sampled_factor_proportions: false,
      lme: { random_effect_dist: 'heavy_tailed', random_effect_df: 5, icc_noise_sd: 0.2 },
    } as ScenarioConfig;
  }

  it('zeroes lme perturbations for a linear run (no Mle estimator)', () => {
    sharedPrefsMock.scenariosEnabled = true;
    scenariosStoreMock.scenarios = [lmeScenario()];
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    const wire = r.spec!.scenarios[0]!;
    expect(wire.random_effect_dist).toBe(0);
    expect(wire.random_effect_df).toBe(0);
    expect(wire.icc_noise_sd).toBe(0);
    sharedPrefsMock.scenariosEnabled = false;
    scenariosStoreMock.scenarios = [];
  });

  it('keeps lme perturbations for a binary mixed run (Glm + cluster)', () => {
    sharedPrefsMock.scenariosEnabled = true;
    scenariosStoreMock.scenarios = [lmeScenario()];
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      ...cfg.cluster!,
      clusterName: 'school',
      binaryOutcome: true,
      baselineProbability: 0.3,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    const wire = spec!.scenarios[0]!;
    expect(wire.random_effect_dist).toBe(1); // heavy_tailed
    expect(wire.random_effect_df).toBe(5);
    expect(wire.icc_noise_sd).toBe(0.2);
    sharedPrefsMock.scenariosEnabled = false;
    scenariosStoreMock.scenarios = [];
  });

  it('keeps lme perturbations for a Gaussian mixed run (Mle)', () => {
    // Default mixed config is continuous-outcome → estimator=Mle. The engine
    // now accepts scenarios for Mle (RE perturbations apply on the shared
    // cluster path), so the wire must carry the same knobs as the binary GLMM.
    sharedPrefsMock.scenariosEnabled = true;
    scenariosStoreMock.scenarios = [lmeScenario()];
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(errors).toEqual([]);
    const wire = spec!.scenarios[0]!;
    expect(wire.random_effect_dist).toBe(1); // heavy_tailed
    expect(wire.random_effect_df).toBe(5);
    expect(wire.icc_noise_sd).toBe(0.2);
    sharedPrefsMock.scenariosEnabled = false;
    scenariosStoreMock.scenarios = [];
  });
});

describe('familyConfigToAppSpec — outcome options projection', () => {
  it('omits outcome_options entirely when all knobs are neutral (unpinned, no driver)', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      outcomeOptions: {
        residualDistribution: null,
        pinnedResidual: false,
        heteroskedasticityDriver: '',
      },
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    expect(r.spec.outcome_options).toBeUndefined();
  });

  it('pinned non-normal residual reaches the wire', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      outcomeOptions: {
        residualDistribution: 'high_kurtosis',
        pinnedResidual: true,
        heteroskedasticityDriver: '',
      },
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    expect(r.spec.outcome_options).toEqual({ residual_distribution: 'high_kurtosis' });
  });

  it('pinned explicit normal reaches the wire (neutral keys on unpinned-default, not on value)', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      outcomeOptions: {
        residualDistribution: 'normal',
        pinnedResidual: true,
        heteroskedasticityDriver: '',
      },
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    expect(r.spec.outcome_options).toEqual({ residual_distribution: 'normal' });
  });

  it('heteroskedasticity driver reaches the wire when non-empty', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      outcomeOptions: {
        residualDistribution: null,
        pinnedResidual: false,
        heteroskedasticityDriver: 'x1',
      },
    });
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'linear') throw new Error('expected linear');
    expect(r.spec.outcome_options).toEqual({ heteroskedasticity_driver: 'x1' });
  });

  it('binary outcome drops residual and heteroskedasticity driver (not applicable)', () => {
    const cfg = linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
      baselineProbability: 0.3,
      outcomeOptions: {
        residualDistribution: 'right_skewed',
        pinnedResidual: true,
        heteroskedasticityDriver: 'x1',
      },
    });
    const r = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(r.errors).toEqual([]);
    if (r.spec?.family !== 'logit') throw new Error('expected logit');
    // Residual + driver are continuous-only; binary outcome → no outcome_options
    expect(r.spec.outcome_options).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Correlation-filtering invariants (R1/R2 via correlatableVariables)
// ---------------------------------------------------------------------------

describe('familyConfigToAppSpec — correlation projection filtering', () => {
  // Helper: build a 3×3 correlation matrix with an off-diagonal non-zero at [0][1]
  function corrMatrix3(v01: number): number[][] {
    return [
      [1, v01, 0],
      [v01, 1, 0],
      [0, 0, 1],
    ];
  }

  function baseConfig(vars: FamilyConfig['variables'], corr: number[][]): FamilyConfig {
    return {
      ...defaultFamilyConfig('regression'),
      formula: `y ~ ${vars.map((v) => v.name).join(' + ')}`,
      variables: vars,
      effects: vars.map((v) => ({ name: v.name, value: 0 })),
      correlations: corr,
    };
  }

  it('factor predictor never appears in spec.correlations.names', () => {
    uploadStoreMock.csvData = null;
    uploadStoreMock.mode = 'none';
    const vars: FamilyConfig['variables'] = [
      { name: 'x1', kind: 'continuous' },
      { name: 'grp', kind: 'factor', nLevels: 2, levelProportions: [0.5, 0.5] },
      { name: 'x2', kind: 'continuous' },
    ];
    // non-zero correlation stored between x1(0) and x2(2)
    const corr = corrMatrix3(0.4);
    corr[0]![2] = 0.4;
    corr[2]![0] = 0.4;
    const cfg = baseConfig(vars, corr);
    const { spec } = familyConfigToAppSpec('regression', cfg, 'continuous');
    // correlations is emitted because x1↔x2 has a non-zero value
    expect(spec!.correlations).not.toBeNull();
    expect(spec!.correlations!.names).not.toContain('grp');
  });

  it('binary predictor never appears in spec.correlations.names', () => {
    uploadStoreMock.csvData = null;
    uploadStoreMock.mode = 'none';
    const vars: FamilyConfig['variables'] = [
      { name: 'x1', kind: 'continuous' },
      { name: 'treated', kind: 'binary', binaryProportion: 0.5 },
      { name: 'x2', kind: 'continuous' },
    ];
    const corr = corrMatrix3(0);
    corr[0]![2] = 0.3;
    corr[2]![0] = 0.3;
    const cfg = baseConfig(vars, corr);
    const { spec } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(spec!.correlations).not.toBeNull();
    expect(spec!.correlations!.names).not.toContain('treated');
  });

  it('uploaded var in strict mode never appears in spec.correlations.names', () => {
    uploadStoreMock.mode = 'strict';
    uploadStoreMock.csvData = {
      mode: 'strict',
      n_rows: 3,
      columns: [{ name: 'x1', col_type: 'continuous', values: [1, 2, 3], labels: [] }],
    };
    const vars: FamilyConfig['variables'] = [
      { name: 'x1', kind: 'continuous' }, // uploaded
      { name: 'x2', kind: 'continuous' }, // generated
    ];
    // put a non-zero correlation at x1↔x2
    const corr: number[][] = [[1, 0.5], [0.5, 1]];
    const cfg = baseConfig(vars, corr);
    const { spec } = familyConfigToAppSpec('regression', cfg, 'continuous');
    // Only x2 is correlatable in strict mode (x1 is uploaded → excluded)
    // With only 1 correlatable variable the submatrix has no off-diagonal → null
    expect(spec!.correlations).toBeNull();
    // Verify x1 absent if somehow non-null
    if (spec!.correlations !== null) {
      expect((spec!.correlations as { names: string[] }).names).not.toContain('x1');
    }
    // Reset for subsequent tests
    uploadStoreMock.csvData = null;
    uploadStoreMock.mode = 'none';
  });

  it('continuous generated↔generated correlation survives the projection', () => {
    uploadStoreMock.csvData = null;
    uploadStoreMock.mode = 'none';
    const vars: FamilyConfig['variables'] = [
      { name: 'x1', kind: 'continuous' },
      { name: 'x2', kind: 'continuous' },
    ];
    const corr: number[][] = [[1, 0.6], [0.6, 1]];
    const cfg = baseConfig(vars, corr);
    const { spec } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(spec!.correlations).not.toBeNull();
    expect(spec!.correlations!.names).toContain('x1');
    expect(spec!.correlations!.names).toContain('x2');
    const idx1 = spec!.correlations!.names.indexOf('x1');
    const idx2 = spec!.correlations!.names.indexOf('x2');
    expect(spec!.correlations!.values[idx1]![idx2]).toBeCloseTo(0.6, 10);
  });

  it('untouched uploaded↔uploaded pair (stored 0) emits no CorrelationMatrix (respects null guard)', () => {
    uploadStoreMock.mode = 'partial';
    uploadStoreMock.csvData = {
      mode: 'partial',
      n_rows: 3,
      columns: [
        { name: 'x1', col_type: 'continuous', values: [1, 2, 3], labels: [] },
        { name: 'x2', col_type: 'continuous', values: [3, 2, 1], labels: [] },
      ],
    };
    const vars: FamilyConfig['variables'] = [
      { name: 'x1', kind: 'continuous' },
      { name: 'x2', kind: 'continuous' },
    ];
    // Both stored as 0 (untouched) — engine measures the correlation itself
    const corr: number[][] = [[1, 0], [0, 1]];
    const cfg = baseConfig(vars, corr);
    const { spec } = familyConfigToAppSpec('regression', cfg, 'continuous');
    // Stored 0 → null guard → null (engine measures it)
    expect(spec!.correlations).toBeNull();
    // Reset
    uploadStoreMock.csvData = null;
    uploadStoreMock.mode = 'none';
  });
});

// Stale persisted snapshots (per-family IndexedDB/Tauri-store blobs from older
// builds) can carry null or drop newer fields entirely (undefined). Both end up
// as `null` after JSON.stringify (undefined → NaN through arithmetic first),
// which the engine's serde rejects with an opaque "invalid type: null,
// expected f64". The adapter must catch them as config errors instead.
describe('familyConfigToAppSpec — non-finite numeric guards (stale persisted snapshots)', () => {
  const twoPredictorCfg = () =>
    linearConfig({
      formula: 'y ~ x1 + x2',
      variables: [
        { name: 'x1', kind: 'continuous' },
        { name: 'x2', kind: 'continuous' },
      ],
      effects: [
        { name: 'x1', value: 0.3 },
        { name: 'x2', value: 0.2 },
      ],
    });

  it('rejects a null mixed icc (null passes naive range comparisons)', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|g)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = {
      clusterName: 'g',
      icc: null as unknown as number,
      dimKind: 'n_clusters',
      nClusters: 20,
      clusterSize: 30,
    };
    const { spec, errors } = familyConfigToAppSpec('mixed', cfg);
    expect(spec).toBeNull();
    expect(errors.join(' ')).toMatch(/icc/i);
  });

  it('rejects a null alpha', () => {
    const cfg = twoPredictorCfg();
    cfg.alpha = null as unknown as number;
    const { errors } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(errors.join(' ')).toMatch(/alpha/i);
  });

  it('rejects a missing targetPower (undefined → NaN via /100)', () => {
    const cfg = twoPredictorCfg();
    cfg.targetPower = undefined as unknown as number;
    const { errors } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(errors.join(' ')).toMatch(/target power/i);
  });

  it('rejects a null effect value', () => {
    const cfg = twoPredictorCfg();
    cfg.effects[0]!.value = null as unknown as number;
    const { errors } = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(errors.join(' ')).toMatch(/effect 'x1'/i);
  });
});

describe('wald_se adapter', () => {
  it('defaults to hessian and passes rx through for mixed', () => {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    // defaultFamilyConfig('mixed') already seeds a cluster config; use it as-is.
    const r1 = familyConfigToAppSpec('mixed', cfg);
    expect(r1.errors).toEqual([]);
    if (r1.spec?.family !== 'mixed') throw new Error('expected mixed');
    expect((r1.spec as any).wald_se).toBe('hessian');
    // set to rx → must propagate
    cfg.advanced.wald_se = 'rx';
    const r2 = familyConfigToAppSpec('mixed', cfg);
    expect(r2.errors).toEqual([]);
    if (r2.spec?.family !== 'mixed') throw new Error('expected mixed');
    expect((r2.spec as any).wald_se).toBe('rx');
  });
});

describe('familyConfigToAppSpec — validation harmonization (soft warns + ICC band)', () => {
  function oneEffectLinear() {
    return linearConfig({
      formula: 'y ~ x1',
      variables: [{ name: 'x1', kind: 'continuous' }],
      effects: [{ name: 'x1', value: 0.3 }],
    });
  }
  function mixedWithIcc(icc: number): FamilyConfig {
    const cfg = defaultFamilyConfig('mixed');
    cfg.formula = 'y ~ x + (1|school)';
    cfg.effects = [{ name: 'x', value: 0.5 }];
    cfg.cluster = { clusterName: 'school', icc, dimKind: 'n_clusters', nClusters: 20, clusterSize: 30 };
    return cfg;
  }

  it('alpha above max_alpha soft-warns but still builds a spec (no error, run not gated)', () => {
    const cfg = oneEffectLinear();
    cfg.alpha = 0.3;
    const r = familyConfigToAppSpec('regression', cfg, 'continuous');
    expect(r.errors).toEqual([]);
    expect(r.spec).not.toBeNull();
    expect(r.warnings.join(' ')).toMatch(/alpha/i);
  });

  it('alpha at the max_alpha edge (0.25) produces no warning', () => {
    const cfg = oneEffectLinear();
    cfg.alpha = 0.25;
    expect(familyConfigToAppSpec('regression', cfg, 'continuous').warnings).toEqual([]);
  });

  it('extreme (but in-(0,1)) baseline probability soft-warns, still builds', () => {
    const cfg = oneEffectLinear();
    cfg.baselineProbability = 0.02;
    const r = familyConfigToAppSpec('regression', cfg, 'binary');
    expect(r.spec).not.toBeNull();
    expect(r.errors).toEqual([]);
    expect(r.warnings.join(' ')).toMatch(/baseline/i);
  });

  it('baseline probability at the band edge (0.05) produces no warning', () => {
    const cfg = oneEffectLinear();
    cfg.baselineProbability = 0.05;
    expect(familyConfigToAppSpec('regression', cfg, 'binary').warnings).toEqual([]);
  });

  it('rejects a nonzero icc below the stability band (0.04)', () => {
    expect(familyConfigToAppSpec('mixed', mixedWithIcc(0.04)).errors.join(' ')).toMatch(/stable band/i);
  });

  it('rejects a nonzero icc above the stability band (0.96)', () => {
    expect(familyConfigToAppSpec('mixed', mixedWithIcc(0.96)).errors.join(' ')).toMatch(/stable band/i);
  });

  it('accepts icc at the inclusive band edge (0.05)', () => {
    const r = familyConfigToAppSpec('mixed', mixedWithIcc(0.05));
    expect(r.errors).toEqual([]);
    expect(r.spec).not.toBeNull();
  });

  it('accepts icc = 0 (no clustering)', () => {
    const r = familyConfigToAppSpec('mixed', mixedWithIcc(0));
    expect(r.errors).toEqual([]);
    expect(r.spec).not.toBeNull();
  });
});
