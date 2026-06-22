import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/svelte';
import type { RunTab } from '$lib/stores/run.svelte';
import type { PowerResult, SampleSizeResult, PlotSpecs, CrossingFit } from '$lib/domain/result';

// JointDistTab embeds VegaChart (vega-embed needs a real DOM/SVG that jsdom lacks).
// Stub it so this unit test targets the table data, not the chart.
vi.mock('./VegaChart.svelte', async () => ({
  default: (await import('../../../tests/ChartStub.svelte')).default,
}));

import JointDistTab from './JointDistTab.svelte';

function makeTabWithHistogram(histogram: number[]): RunTab {
  const result: PowerResult = {
    n: 80,
    n_sims: 1000,
    target_indices: [1, 2],
    power_uncorrected: [0.72, 0.65],
    power_corrected: [0.72, 0.65],
    ci_uncorrected: [{ lo: 0.62, hi: 0.82 }, { lo: 0.55, hi: 0.75 }],
    ci_corrected: [{ lo: 0.62, hi: 0.82 }, { lo: 0.55, hi: 0.75 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
    success_count_histogram_uncorrected: histogram,
  };

  const spec = {
    family: 'linear' as const,
    parsed_formula: {
      outcome: 'y',
      predictors: ['x1', 'condition[treatment]'],
      interaction_terms: [],
    },
    var_types: [
      { kind: 'numeric' as const, name: 'x1' },
      { kind: 'factor' as const, name: 'condition', factor_n_levels: 2, factor_proportions: [0.5, 0.5] },
    ],
    effects: [
      { name: 'x1', value: 0.3 },
      { name: 'condition[treatment]', value: 0.4 },
    ],
    correlations: null,
    alpha: 0.05,
    target_power: 0.8,
    n_sims: 1000,
    seed: 2137,
    tests: { kind: 'all' as const },
    correction: 'none' as const,
    scenarios: [],
    csv: null,
    report_overall: true,
    contrasts: [],
  };

  return {
    id: 'test-tab',
    label: 'Run 1',
    kind: 'find-power',
    subView: 'joint',
    spec,
    sample_size: 80,
    effect_names: ['x1', 'condition[treatment]'],
    result,
    scenarios: [['default', result]],
  };
}

function makeSampleSizeTab(fitted_joint?: CrossingFit[]): RunTab {
  const gridPoint = (n: number, hist: number[]): PowerResult => ({
    n,
    n_sims: 100,
    target_indices: [1, 2],
    power_uncorrected: [0, 0],
    power_corrected: [0, 0],
    ci_uncorrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
    ci_corrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
    success_count_histogram_corrected: hist,
  });

  const grid = [
    gridPoint(50, [80, 15, 5]),
    gridPoint(100, [40, 40, 20]),
  ];

  const result: SampleSizeResult = {
    grid_or_trace: grid,
    first_achieved: [50, 100],
    first_joint_achieved: [50, 100],
    target_power: 0.8,
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
    fitted_joint,
    cluster_atom: 1,
  };

  const spec = {
    family: 'linear' as const,
    parsed_formula: {
      outcome: 'y',
      predictors: ['a', 'b'],
      interaction_terms: [],
    },
    var_types: [
      { kind: 'numeric' as const, name: 'a' },
      { kind: 'numeric' as const, name: 'b' },
    ],
    effects: [
      { name: 'a', value: 0.3 },
      { name: 'b', value: 0.4 },
    ],
    correlations: null,
    alpha: 0.05,
    target_power: 0.8,
    n_sims: 100,
    seed: 2137,
    tests: { kind: 'all' as const },
    correction: 'none' as const,
    scenarios: [],
    csv: null,
    report_overall: true,
    contrasts: [],
  };

  const plots: PlotSpecs = {
    blocks: [
      { key: 'curve', spec: '{"mark":"line"}' },
      { key: 'at_least_k', spec: '{"mark":"line","data":{"values":[]}}' },
      { key: 'exactly_k', spec: '{"mark":"line","data":{"values":[]}}' },
    ],
  };

  return {
    id: 'test-ss-tab',
    label: 'Run 2',
    kind: 'find-sample-size',
    subView: 'joint',
    spec,
    bounds: [10, 200],
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
    effect_names: ['a', 'b'],
    result,
    scenarios: [['default', result]],
    plots,
  };
}

function makeSampleSizeTabSingleTarget(): RunTab {
  const gridPoint = (n: number): PowerResult => ({
    n,
    n_sims: 100,
    target_indices: [1],
    power_uncorrected: [0],
    power_corrected: [0],
    ci_uncorrected: [{ lo: 0, hi: 0 }],
    ci_corrected: [{ lo: 0, hi: 0 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
    success_count_histogram_corrected: [80, 20],
  });

  const result: SampleSizeResult = {
    grid_or_trace: [gridPoint(50), gridPoint(100)],
    first_achieved: [50],
    first_joint_achieved: [50],
    target_power: 0.8,
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
  };

  const spec = {
    family: 'linear' as const,
    parsed_formula: {
      outcome: 'y',
      predictors: ['a'],
      interaction_terms: [],
    },
    var_types: [{ kind: 'numeric' as const, name: 'a' }],
    effects: [{ name: 'a', value: 0.3 }],
    correlations: null,
    alpha: 0.05,
    target_power: 0.8,
    n_sims: 100,
    seed: 2137,
    tests: { kind: 'all' as const },
    correction: 'none' as const,
    scenarios: [],
    csv: null,
    report_overall: true,
    contrasts: [],
  };

  // Single target → engine emits no at_least_k / exactly_k blocks.
  const plots: PlotSpecs = {
    blocks: [{ key: 'curve', spec: '{"mark":"line"}' }],
  };

  return {
    id: 'test-ss-single',
    label: 'Run 3',
    kind: 'find-sample-size',
    subView: 'joint',
    spec,
    bounds: [10, 200],
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
    effect_names: ['a'],
    result,
    scenarios: [['default', result]],
    plots,
  };
}

describe('JointDistTab', () => {
  it('renders a combined k | Exactly | At least table from the histogram', () => {
    const tab = makeTabWithHistogram([10, 30, 60]);
    const { getByText, getByTestId, getAllByText } = render(JointDistTab, { props: { tab } });
    expect(getByTestId('joint-dist-tab')).toBeTruthy();
    // New design: one table with column headers "k", "Exactly", "At least"
    // (not two separate captioned tables for Exactly-k and At-least-k)
    expect(getByText(/^Exactly$/i)).toBeTruthy();
    expect(getByText(/^At least$/i)).toBeTruthy();
    // k column: rows 0,1,2 should all be present
    expect(getAllByText('0').length).toBeGreaterThan(0);
  });

  it('shows the neutral empty-state message when histogram is empty', () => {
    const tab = makeTabWithHistogram([]);
    const { getByText } = render(JointDistTab, { props: { tab } });
    expect(getByText(/Joint significance distribution is unavailable for this result/i)).toBeTruthy();
  });
});

describe('JointDistTab — find-sample-size', () => {
  it('renders the joint required-N table and no stale analytical message', () => {
    const tab = makeSampleSizeTab();
    const { queryByText, getAllByText, getByText } = render(JointDistTab, { props: { tab } });
    expect(queryByText(/analytical/i)).toBeNull();
    // Exactly two rows: "≥ 2 of 2 tests" and "≥ 1 of 2 tests".
    expect(getAllByText(/of 2 tests/)).toHaveLength(2);
    // first_joint_achieved [50, 100] threaded into cells: k=1 -> 50, k=2 -> 100.
    expect(getByText('100')).toBeTruthy();
    expect(getByText('50')).toBeTruthy();
  });

  it('renders at_least_k chart when block is present (≥2 targets)', () => {
    const tab = makeSampleSizeTab();
    // plots.blocks contains at_least_k → chart stub with testid at-least-k-view renders.
    const { getByTestId } = render(JointDistTab, { props: { tab } });
    expect(getByTestId('at-least-k-view')).toBeTruthy();
  });

  it('renders exactly_k chart when block is present (≥2 targets)', () => {
    const tab = makeSampleSizeTab();
    // plots.blocks contains exactly_k → chart stub with testid exactly-k-view renders.
    const { getByTestId } = render(JointDistTab, { props: { tab } });
    expect(getByTestId('exactly-k-view')).toBeTruthy();
  });

  it('renders neither at_least_k nor exactly_k when blocks absent (single target)', () => {
    const tab = makeSampleSizeTabSingleTarget();
    // plots.blocks has only curve → no joint charts.
    const { queryByTestId } = render(JointDistTab, { props: { tab } });
    expect(queryByTestId('at-least-k-view')).toBeNull();
    expect(queryByTestId('exactly-k-view')).toBeNull();
  });
});

describe('JointDistTab — fitted_joint crossing fits', () => {
  it('fitted: joint headline shows n_achievable (overrides first_joint_achieved)', () => {
    // first_joint_achieved = [50, 100]; fitted_joint n_achievable = [60, 120] — different values prove the swap.
    const fitted_joint: CrossingFit[] = [
      { status: 'fitted', n_star: 58, n_achievable: 60, ci_lo: null, ci_hi: null },
      { status: 'fitted', n_star: 118, n_achievable: 120, ci_lo: null, ci_hi: null },
    ];
    const tab = makeSampleSizeTab(fitted_joint);
    const { getByText, queryByText } = render(JointDistTab, { props: { tab } });
    // k=2 row (j=0): fitted_joint[1] → n_achievable=120; k=1 row (j=1): fitted_joint[0] → n_achievable=60.
    expect(getByText('120')).toBeTruthy();
    expect(getByText('60')).toBeTruthy();
    // Grid values 50 and 100 should NOT appear (overridden).
    expect(queryByText('50')).toBeNull();
    expect(queryByText('100')).toBeNull();
  });

  it('at_or_below_min: joint headline shows ≤ n_min', () => {
    const fitted_joint: CrossingFit[] = [
      { status: 'at_or_below_min', n_min: 50 },
      { status: 'at_or_below_min', n_min: 50 },
    ];
    const tab = makeSampleSizeTab(fitted_joint);
    const { getAllByText } = render(JointDistTab, { props: { tab } });
    expect(getAllByText('≤ 50').length).toBeGreaterThanOrEqual(2);
  });

  it('not_reached: joint headline shows ≥ ceiling', () => {
    const fitted_joint: CrossingFit[] = [
      { status: 'not_reached', n_approx: null },
      { status: 'not_reached', n_approx: null },
    ];
    const tab = makeSampleSizeTab(fitted_joint);
    const { getAllByText } = render(JointDistTab, { props: { tab } });
    // ceiling = max(50, 100) = 100
    expect(getAllByText('≥ 100').length).toBeGreaterThanOrEqual(2);
  });

  it('missing fitted_joint (older payload): falls back silently to first_joint_achieved', () => {
    // No fitted_joint → behaves exactly like pre-fitting payloads (first_joint_achieved).
    const tab = makeSampleSizeTab();
    const { getByText } = render(JointDistTab, { props: { tab } });
    // first_joint_achieved = [50, 100]: k=1→50, k=2→100.
    expect(getByText('100')).toBeTruthy();
    expect(getByText('50')).toBeTruthy();
  });
});
