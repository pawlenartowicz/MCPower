import { describe, it, expect, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import type { RunTab } from '$lib/stores/run.svelte';
import type { PowerResult, SampleSizeResult, PlotSpecs } from '$lib/domain/result';
import type { AppSpec, CsvData } from '$lib/domain/app-spec';

// SummaryTab embeds VegaChart (vega-embed needs a real DOM/SVG that jsdom lacks).
// Stub it so this unit test targets the table / scenario selector / diagnostics.
vi.mock('./VegaChart.svelte', async () => ({
  default: (await import('../../../tests/ChartStub.svelte')).default,
}));

import SummaryTab from './SummaryTab.svelte';

// Minimal find-power RunTab: 2 effects (1 continuous, 1 factor with 1 level).
// target_indices are β̂-column indices: intercept=0, x1=1, condition[treatment]=2.
function makeTab(nScenarios = 1): RunTab {
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
    overall_significant_rate: 0.91,
    overall_significant_ci: { lo: 0.88, hi: 0.94 },
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

  const scenarios: [string, PowerResult][] =
    nScenarios === 1
      ? [['default', result]]
      : [['default', result], ['optimistic', { ...result, n: 120, power_uncorrected: [0.85, 0.78], power_corrected: [0.85, 0.78] }]];

  // Single-block find-power plots (no chips).
  const plots: PlotSpecs = {
    blocks: [{ key: 'power', spec: '{"mark":"bar"}' }],
  };

  // Multi-scenario plots: scenario:<label> blocks + overlay block.
  const multiPlots: PlotSpecs = {
    blocks: [
      { key: 'scenario:default', spec: '{"mark":"line"}' },
      { key: 'scenario:optimistic', spec: '{"mark":"line"}' },
      { key: 'overlay', spec: '{"mark":"line"}' },
    ],
  };

  return {
    id: 'test-tab',
    label: 'Run 1',
    kind: 'find-power',
    subView: 'summary',
    spec,
    sample_size: 80,
    effect_names: ['x1', 'condition[treatment]'],
    result,
    scenarios,
    plots: nScenarios === 1 ? plots : multiPlots,
  };
}

describe('SummaryTab', () => {
  it('renders the table, an overall row, and the diagnostics badge', () => {
    const { getByTestId, getByText } = render(SummaryTab, { props: { tab: makeTab() } });
    expect(getByTestId('summary-table')).toBeTruthy();
    expect(getByText(/Overall/)).toBeTruthy();
    expect(getByTestId('diagnostics-badge')).toBeTruthy();
  });

  it('shows the scenario selector only with >1 scenario', () => {
    const { queryByTestId } = render(SummaryTab, { props: { tab: makeTab(/*1 scenario*/) } });
    expect(queryByTestId('scenario-selector')).toBeNull();
  });

  it('shows the scenario selector with >1 scenario', () => {
    const { getByTestId } = render(SummaryTab, { props: { tab: makeTab(2) } });
    expect(getByTestId('scenario-selector')).toBeTruthy();
  });

  it('renders correct effect labels (idx-1 mapping, not raw idx)', () => {
    // target_indices [1,2] → effectNames[0]='x1', effectNames[1]='condition[treatment]'
    // If the raw-idx bug existed, we'd see undefined/'condition[treatment]' for x1.
    // Use getAllByText since 'x1' also appears in sr-only bar spans from SummaryTab.
    const { getAllByText } = render(SummaryTab, { props: { tab: makeTab() } });
    const matches = getAllByText('x1');
    // At least one td with 'x1' must exist (the table row label)
    const tdMatch = matches.find((el) => el.tagName === 'TD');
    expect(tdMatch).toBeTruthy();
  });

  it('multi-scenario: renders scenario name columns and a Δ column', () => {
    // makeTab(2) → ['default', {power:[0.72,0.65]}], ['optimistic', {power:[0.85,0.78]}]
    // REPORT_CONFIG.baseline_scenario.prefer_label = 'optimistic' → baseline = optimistic
    // For x1 (pos=0): baseline=0.85, default=0.72 → Δ = -13.0pp
    const { getByTestId, getAllByText } = render(SummaryTab, { props: { tab: makeTab(2) } });
    // Table must still have the summary-table testid
    expect(getByTestId('summary-table')).toBeTruthy();
    // Both scenario names appear (table headers + scenario selector buttons) — use getAllByText
    expect(getAllByText('default').length).toBeGreaterThan(0);
    expect(getAllByText('optimistic').length).toBeGreaterThan(0);
    // A Δ column header must be present in the table thead
    const table = getByTestId('summary-table');
    const thCells = table.querySelectorAll('th');
    const hasDelta = Array.from(thCells).some((th) => th.textContent?.trim() === 'Δ');
    expect(hasDelta).toBeTruthy();
    // The Δ for x1 row (pos=0): baseline=optimistic=0.85, default=0.72, diff=-0.13 → -13.0pp
    const deltaMatches = getAllByText('-13.0pp');
    expect(deltaMatches.length).toBeGreaterThan(0);
  });

  // Helper: build a strict-mode CsvData fixture.
  function makeStrictCsv(n_rows: number): CsvData {
    return {
      mode: 'strict',
      n_rows,
      columns: [],
    };
  }

  it('strict mode: reuse-diagnostic panel IS present and shows a reuse line', () => {
    const tab = makeTab();
    const tabWithCsv: RunTab = {
      ...tab,
      spec: { ...tab.spec, csv: makeStrictCsv(100) } as AppSpec,
    };
    const { getByTestId } = render(SummaryTab, { props: { tab: tabWithCsv } });
    const panel = getByTestId('reuse-diagnostic');
    expect(panel).toBeTruthy();
    // Panel must mention a percentage — reuseFraction(100, 80) > 0.
    expect(panel.textContent).toMatch(/%/);
  });

  it('non-strict mode: reuse-diagnostic panel is absent', () => {
    const tab = makeTab();
    // mode='partial' → panel must not render.
    const tabPartial: RunTab = {
      ...tab,
      spec: { ...tab.spec, csv: { mode: 'partial', n_rows: 100, columns: [] } } as AppSpec,
    };
    const { queryByTestId: queryPartial } = render(SummaryTab, { props: { tab: tabPartial } });
    expect(queryPartial('reuse-diagnostic')).toBeNull();

    // csv=null → panel must not render.
    const tabNoCsv: RunTab = { ...tab, spec: { ...tab.spec, csv: null } };
    const { queryByTestId: queryNull } = render(SummaryTab, { props: { tab: tabNoCsv } });
    expect(queryNull('reuse-diagnostic')).toBeNull();
  });

  it('single-block find-power: no scenario selector shown (chips derived from blocks, not scenarios)', () => {
    // plots.blocks = [{ key:'power', ... }] → no scenario: blocks → no chips → no selector.
    const tab = makeTab(1);
    const { queryByTestId } = render(SummaryTab, { props: { tab } });
    expect(queryByTestId('scenario-selector')).toBeNull();
  });

  it('multi-scenario: scenario chips derived from scenario:<label> block keys', () => {
    // plots.blocks = [scenario:default, scenario:optimistic, overlay] → 2 chips + overlay.
    const tab = makeTab(2);
    const { getByTestId, getAllByText } = render(SummaryTab, { props: { tab } });
    expect(getByTestId('scenario-selector')).toBeTruthy();
    // Both scenario labels must appear as chip buttons (getAllByText: may also appear in table).
    expect(getAllByText('default').length).toBeGreaterThan(0);
    expect(getAllByText('optimistic').length).toBeGreaterThan(0);
    // Overlay chip must also appear.
    expect(getAllByText('⧉ Overlay').length).toBeGreaterThan(0);
  });

  // Regression: switching the active run-tab from find-power (blocks=[{key:'power'}]) to
  // find-sample-size (blocks=[{key:'curve'}]) must NOT leave selectedBlockKey stale.
  // Without Fix 1 the $effect was absent, defaultBlockKey() returned '' for the new blocks
  // (stale 'power' key not in [{key:'curve'}]), and currentSpec() returned undefined →
  // no chart element rendered.
  it('tab switch power→curve: curve chart is rendered after rerender', async () => {
    // Build a minimal sample-size RunTab with blocks=[{key:'curve'}].
    const powerResult: PowerResult = {
      n: 80,
      n_sims: 800,
      target_indices: [1],
      power_uncorrected: [0.75],
      power_corrected: [0.75],
      ci_uncorrected: [{ lo: 0.65, hi: 0.85 }],
      ci_corrected: [{ lo: 0.65, hi: 0.85 }],
      convergence_rate: 1.0,
      boundary_hit: [],
      estimator_extras: { estimator: 'ols' },
      overall_significant_rate: 0.75,
      overall_significant_ci: { lo: 0.65, hi: 0.85 },
    };
    const ssResult: SampleSizeResult = {
      grid_or_trace: [powerResult],
      first_achieved: [80],
      first_joint_achieved: [80],
      target_power: 0.8,
      method: { Grid: { by: { Fixed: 10 }, mode: 'Linear' } },
    };
    const baseSpec = makeTab().spec;
    const sampleSizeTab: RunTab = {
      id: 'ss-tab',
      label: 'Run 2',
      kind: 'find-sample-size',
      subView: 'summary',
      spec: baseSpec,
      effect_names: ['x1'],
      result: ssResult,
      scenarios: [['default', ssResult]],
      plots: { blocks: [{ key: 'curve', spec: '{"mark":"line"}' }] },
    };

    // Render initially with a find-power tab (key='power').
    const powerTab = makeTab(1);
    const { queryByTestId, rerender } = render(SummaryTab, { props: { tab: powerTab } });

    // Sanity: find-power chart is present.
    expect(queryByTestId('bars-view')).not.toBeNull();

    // Switch to a find-sample-size tab — rerender with new tab prop.
    await rerender({ tab: sampleSizeTab });

    // After the $effect fires, selectedBlockKey must be re-validated to 'curve'
    // and the curve chart must render. Without Fix 1 this assertion would fail.
    await waitFor(() => {
      expect(queryByTestId('curve-view')).not.toBeNull();
    });
  });
});
