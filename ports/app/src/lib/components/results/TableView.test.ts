import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/svelte';
import type { RunTab } from '$lib/stores/run.svelte';
import type { PowerResult, SampleSizeResult, CrossingFit } from '$lib/domain/result';
import TableView from './TableView.svelte';

const baseSpec = {
  family: 'linear' as const,
  parsed_formula: { outcome: 'y', predictors: ['a', 'b'], interaction_terms: [] },
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

function gridPoint(n: number, power: [number, number]): PowerResult {
  return {
    n,
    n_sims: 100,
    target_indices: [1, 2],
    power_uncorrected: power,
    power_corrected: power, // correction 'none' ⇒ corrected == uncorrected
    ci_uncorrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
    ci_corrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
  };
}

function makeSampleSizeTab(
  firstAchieved: (number | null)[],
  fitted?: CrossingFit[],
  overall?: { first_overall_achieved?: number | null; fitted_overall?: CrossingFit | null },
): RunTab {
  const grid = [
    gridPoint(50, [0.55, 0.30]),
    gridPoint(100, [0.85, 0.55]),
    gridPoint(150, [0.93, 0.82]),
  ];
  const result: SampleSizeResult = {
    grid_or_trace: grid,
    first_achieved: firstAchieved,
    first_joint_achieved: [100, 150],
    target_power: 0.8,
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
    fitted,
    fitted_joint: undefined,
    cluster_atom: 1,
    first_overall_achieved: overall?.first_overall_achieved ?? null,
    fitted_overall: overall?.fitted_overall ?? null,
  };
  return {
    id: 'ss-tab',
    label: 'Run 1',
    kind: 'find-sample-size',
    subView: 'summary',
    spec: baseSpec,
    bounds: [10, 200],
    method: { Grid: { by: { Fixed: 50 }, mode: 'Linear' } },
    effect_names: ['a', 'b'],
    result,
    scenarios: [['default', result]],
  } as RunTab;
}

describe('TableView — find-sample-size marginal table (A4)', () => {
  it('renders Effect | Required N (no Power at N / Target / ✓✗ columns)', () => {
    const tab = makeSampleSizeTab([100, 150]);
    const { getByTestId, getByText, getAllByText, queryByText } = render(TableView, { props: { tab } });
    const table = getByTestId('sample-size-table');
    expect(table).toBeTruthy();
    // Charter-faithful column: "Required N" is the only metric column.
    expect(getByText('Required N')).toBeTruthy();
    // The old columns must NOT appear.
    expect(queryByText('Power at N')).toBeNull();
    expect(queryByText('Target')).toBeNull();
    expect(queryByText(/✓/)).toBeNull();
    expect(queryByText(/✗/)).toBeNull();
    // Effect names from the mapping.
    expect(getByText('a')).toBeTruthy();
    expect(getByText('b')).toBeTruthy();
    // Required N values from first_achieved (may appear >1 time: table cell + footer)
    expect(getAllByText('100').length).toBeGreaterThanOrEqual(1);
    expect(getAllByText('150').length).toBeGreaterThanOrEqual(1);
  });

  it('adds a First N achieving all targets footer line', () => {
    const tab = makeSampleSizeTab([100, 150]);
    const { getByText } = render(TableView, { props: { tab } });
    expect(getByText(/First N achieving all targets/i)).toBeTruthy();
  });

  it('shows ≥ ceiling for an unachieved target instead of a bare — in the Required N column', () => {
    // Grid is n=50,100,150 → ceiling=150; target b (null) → "≥ 150"
    const tab = makeSampleSizeTab([100, null]);
    const { getAllByText, getByText } = render(TableView, { props: { tab } });
    // "≥ 150" appears in both the table row and the footer — at least one match is expected.
    expect(getAllByText('≥ 150').length).toBeGreaterThanOrEqual(1);
    // Required N for achieved target shows its value, not a dash.
    expect(getByText('100')).toBeTruthy();
  });

  it('does not render the joint required-N table (that lives in JointDistTab)', () => {
    const tab = makeSampleSizeTab([100, 150]);
    const { queryByText } = render(TableView, { props: { tab } });
    expect(queryByText(/Joint detection/i)).toBeNull();
    expect(queryByText(/of 2 tests/)).toBeNull();
  });
});

describe('TableView — find-sample-size overall (omnibus) row', () => {
  it('renders the overall required-N row first when fitted_overall is present', () => {
    const tab = makeSampleSizeTab([100, 150], undefined, {
      first_overall_achieved: 90,
      fitted_overall: { status: 'fitted', n_star: 88, n_achievable: 90, ci_lo: 70, ci_hi: 110 },
    });
    const { getByText, getByTestId } = render(TableView, { props: { tab } });
    // OLS grid points ⇒ "Overall F" label (from overall_label_by_estimator).
    expect(getByText('Overall F')).toBeTruthy();
    // The fitted overall n_achievable (90) shows in the Required N column.
    expect(getByText('90')).toBeTruthy();
    // The overall row is the first body row.
    const rows = getByTestId('sample-size-table').querySelectorAll('tbody tr');
    expect(rows[0]?.textContent).toContain('Overall F');
  });

  it('omits the overall row when both overall fields are null (e.g. mixed family)', () => {
    const tab = makeSampleSizeTab([100, 150]);
    const { queryByText } = render(TableView, { props: { tab } });
    expect(queryByText('Overall F')).toBeNull();
    expect(queryByText('Overall')).toBeNull();
  });
});

describe('TableView — find-power table still works', () => {
  it('renders the per-test power table', () => {
    const result: PowerResult = {
      n: 80,
      n_sims: 100,
      target_indices: [1, 2],
      power_uncorrected: [0.72, 0.65],
      power_corrected: [0.72, 0.65],
      ci_uncorrected: [{ lo: 0.62, hi: 0.82 }, { lo: 0.55, hi: 0.75 }],
      ci_corrected: [{ lo: 0.62, hi: 0.82 }, { lo: 0.55, hi: 0.75 }],
      convergence_rate: 1.0,
      boundary_hit: [],
      estimator_extras: { estimator: 'ols' },
      overall_significant_rate: 0.6,
      overall_significant_ci: { lo: 0.5, hi: 0.7 },
    };
    const tab = {
      id: 'fp-tab',
      label: 'Run 1',
      kind: 'find-power',
      subView: 'summary',
      spec: { ...baseSpec },
      sample_size: 80,
      effect_names: ['a', 'b'],
      result,
      scenarios: [['default', result]],
    } as RunTab;
    const { getByTestId, getByText } = render(TableView, { props: { tab } });
    expect(getByTestId('summary-table')).toBeTruthy();
    expect(getByText('95% CI')).toBeTruthy();
  });
});

describe('TableView — model-based crossing fit display', () => {
  it('fitted: headline shows n_achievable (overrides first_achieved)', () => {
    // first_achieved = [100, 150], fitted n_achievable = [110, 160] — different values prove the swap.
    const fitted: CrossingFit[] = [
      { status: 'fitted', n_star: 108, n_achievable: 110, ci_lo: 95, ci_hi: 125 },
      { status: 'fitted', n_star: 158, n_achievable: 160, ci_lo: 145, ci_hi: 175 },
    ];
    const tab = makeSampleSizeTab([100, 150], fitted);
    const { getByText, queryByText } = render(TableView, { props: { tab } });
    // Fitted n_achievable wins.
    expect(getByText('110')).toBeTruthy();
    expect(getByText('160')).toBeTruthy();
    // Grid-empirical first_achieved values NOT shown in table cells (footer shows 150 = max of first_achieved).
    expect(queryByText('100')).toBeNull(); // the cell shows 110; 100 not shown at all
  });

  it('fitted: single-scenario shows a 95% CI column with rounded integer bounds', () => {
    const fitted: CrossingFit[] = [
      { status: 'fitted', n_star: 108, n_achievable: 110, ci_lo: 94.7, ci_hi: 124.3 },
      { status: 'fitted', n_star: 158, n_achievable: 160, ci_lo: null, ci_hi: 175.1 },
    ];
    const tab = makeSampleSizeTab([100, 150], fitted);
    const { getByText } = render(TableView, { props: { tab } });
    expect(getByText('95% CI')).toBeTruthy();
    // ci_lo 94.7 → floor 94; ci_hi 124.3 → ceil 125.
    expect(getByText('[94, 125]')).toBeTruthy();
    // ci_lo null → ≤ floorN (50); ci_hi 175.1 → ceil 176.
    expect(getByText(`[≤ 50, 176]`)).toBeTruthy();
  });

  it('at_or_below_min: headline shows ≤ n_min', () => {
    const fitted: CrossingFit[] = [
      { status: 'at_or_below_min', n_min: 50 },
      { status: 'fitted', n_star: 108, n_achievable: 110, ci_lo: null, ci_hi: null },
    ];
    const tab = makeSampleSizeTab([null, 110], fitted);
    const { getByText } = render(TableView, { props: { tab } });
    expect(getByText('≤ 50')).toBeTruthy();
    // at_or_below_min CI cell → —
    expect(getByText('—')).toBeTruthy();
  });

  it('not_reached: headline shows ≥ ceiling; CI cell shows appr. n_approx when present', () => {
    const fitted: CrossingFit[] = [
      { status: 'not_reached', n_approx: 200 },
      { status: 'not_reached', n_approx: null },
    ];
    const tab = makeSampleSizeTab([null, null], fitted);
    const { getAllByText, getByText } = render(TableView, { props: { tab } });
    // Headline: ≥ 150 (ceiling of grid [50,100,150]) — appears in both rows and footer.
    expect(getAllByText('≥ 150').length).toBeGreaterThanOrEqual(2);
    // not_reached CI: appr. 200 for first; — for second (n_approx null).
    expect(getByText('appr. 200')).toBeTruthy();
  });

  it('non_monotone: falls back to grid value silently (no warning line)', () => {
    const fitted: CrossingFit[] = [
      { status: 'non_monotone', max_violation: 0.042 },
      { status: 'fitted', n_star: 148, n_achievable: 150, ci_lo: null, ci_hi: null },
    ];
    // first_achieved[0] = 100 → headline for non_monotone shows 100 (grid fallback)
    const tab = makeSampleSizeTab([100, 150], fitted);
    const { getAllByText, queryByTestId } = render(TableView, { props: { tab } });
    expect(getAllByText('100').length).toBeGreaterThanOrEqual(1);
    expect(queryByTestId('non-monotone-warning')).toBeNull();
  });

  it('missing fitted (older payload): falls back silently to first_achieved', () => {
    // No fitted field → behaves exactly like pre-fitting payloads.
    const tab = makeSampleSizeTab([100, 150]);
    const { getAllByText } = render(TableView, { props: { tab } });
    expect(getAllByText('100').length).toBeGreaterThanOrEqual(1);
    expect(getAllByText('150').length).toBeGreaterThanOrEqual(1);
  });
});

describe('TableView — EP-1 contrast-bearing PowerResult (find-power)', () => {
  // EP-1: power_uncorrected/ci_uncorrected are LONGER than target_indices when
  // contrast_pairs is non-empty: length == target_indices.length + contrast_pairs.length
  // (marginals first, then one entry per contrast). The original bug was a consumer
  // indexing the power vector by target_indices.length, missing the contrast tail and
  // reading `undefined`. This test feeds a contrast-bearing PowerResult into TableView
  // and asserts the contrast row renders the 3rd entry (index 2), not undefined.
  it('renders the pairwise contrast row from the contrast tail of power_uncorrected', () => {
    // target_indices: [1, 2] → two marginals (grp[b], grp[c])
    // contrast_pairs: [[2, 1]] → one contrast (grp[c] vs grp[b])
    // power_uncorrected[2] = 0.91 — the contrast entry; indices 0 and 1 are marginals.
    const result: PowerResult = {
      n: 80,
      n_sims: 100,
      target_indices: [1, 2],
      contrast_pairs: [[2, 1]],
      power_uncorrected: [0.72, 0.65, 0.91],
      power_corrected: [0.72, 0.65, 0.91],
      ci_uncorrected: [
        { lo: 0.62, hi: 0.82 },
        { lo: 0.55, hi: 0.75 },
        { lo: 0.83, hi: 0.97 },
      ],
      ci_corrected: [
        { lo: 0.62, hi: 0.82 },
        { lo: 0.55, hi: 0.75 },
        { lo: 0.83, hi: 0.97 },
      ],
      convergence_rate: 1.0,
      boundary_hit: [],
      estimator_extras: { estimator: 'ols' },
    };
    const tab = {
      id: 'ep1-tab',
      label: 'Run EP1',
      kind: 'find-power',
      subView: 'summary',
      spec: { ...baseSpec },
      sample_size: 80,
      // effect_names is intercept-excluded: idx 1 → 'grp[b]', idx 2 → 'grp[c]'
      effect_names: ['grp[b]', 'grp[c]'],
      result,
      scenarios: [['default', result]],
    } as RunTab;
    const { getByText } = render(TableView, { props: { tab } });
    // The contrast label built by contrastLabel(effectNames, 2, 1) = "grp[c] vs grp[b]".
    expect(getByText('grp[c] vs grp[b]')).toBeTruthy();
    // power_uncorrected[2] = 0.91 → formatted as "91.0%" (REPORT_CONFIG power_decimals_short = 1).
    // If the component read index 1 (OOB assumption) it would show "65.0%"; undefined → "0.0%".
    expect(getByText('91.0%')).toBeTruthy();
  });
});

describe('TableView — multi-scenario find-power layout (B4)', () => {
  function powerResult(power: [number, number]): PowerResult {
    return {
      n: 80,
      n_sims: 100,
      target_indices: [1, 2],
      power_uncorrected: power,
      power_corrected: power,
      ci_uncorrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
      ci_corrected: [{ lo: 0, hi: 0 }, { lo: 0, hi: 0 }],
      convergence_rate: 1.0,
      boundary_hit: [],
      estimator_extras: { estimator: 'ols' },
      overall_significant_rate: 0.6,
      overall_significant_ci: { lo: 0.5, hi: 0.7 },
    };
  }

  it('renders per-scenario columns and a Δ column when the run fans out to >1 scenario', () => {
    const optimistic = powerResult([0.85, 0.78]);
    const doomer = powerResult([0.55, 0.40]);
    const tab = {
      id: 'multi-tab',
      label: 'Run 1',
      kind: 'find-power',
      subView: 'summary',
      spec: { ...baseSpec },
      sample_size: 80,
      effect_names: ['a', 'b'],
      result: optimistic, // baseline scenario's result
      scenarios: [
        ['optimistic', optimistic],
        ['doomer', doomer],
      ],
    } as RunTab;
    const { getByTestId, getByText } = render(TableView, { props: { tab } });
    expect(getByTestId('summary-table')).toBeTruthy();
    // Per-scenario column headers (the multi layout) + the Δ column.
    expect(getByText('optimistic')).toBeTruthy();
    expect(getByText('doomer')).toBeTruthy();
    expect(getByText('Δ')).toBeTruthy();
    // The multi layout drops the single-scenario "95% CI" column.
    expect(() => getByText('95% CI')).toThrow();
  });
});
