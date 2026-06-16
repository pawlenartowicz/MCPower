import { render, screen, fireEvent } from '@testing-library/svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import { runStore } from '$lib/stores/run.svelte';
import ResultsPane from './ResultsPane.svelte';

const samplePowerResult = {
  n: 80,
  n_sims: 100,
  target_indices: [0, 1],
  power_uncorrected: [0.5, 0.3],
  power_corrected: [0.5, 0.3],
  ci_uncorrected: [{ lo: 0.4, hi: 0.6 }, { lo: 0.2, hi: 0.4 }],
  ci_corrected: [{ lo: 0.4, hi: 0.6 }, { lo: 0.2, hi: 0.4 }],
  convergence_rate: 1.0,
  boundary_hit: [],
  estimator_extras: { estimator: 'ols' },
};

const sampleSpec = {
  family: 'linear' as const,
  parsed_formula: { outcome: 'y', predictors: ['x1', 'x2'], interaction_terms: [] },
  var_types: [{ kind: 'numeric' as const, name: 'x1' }, { kind: 'numeric' as const, name: 'x2' }],
  effects: [{ name: 'x1', value: 0.3 }, { name: 'x2', value: 0.2 }],
  correlations: null,
  alpha: 0.05,
  target_power: 0.8,
  n_sims: 100,
  seed: 1,
  tests: { kind: 'all' as const },
  correction: 'none' as const,
  scenarios: [],
  csv: null,
  report_overall: false,
  contrasts: [],
};

describe('ResultsPane', () => {
  beforeEach(() => {
    runStore.clearTabs();
    runStore.runState = 'idle';
    runStore.lastError = null;
  });

  it('shows the run error card instead of the checklist after a failed first run', () => {
    runStore.runState = 'error';
    runStore.lastError = { severity: 'run', title: 'Run failed', message: 'engine boom' };
    render(ResultsPane);
    expect(screen.getByText('Run failed')).toBeInTheDocument();
    expect(screen.getByText(/engine boom/)).toBeInTheDocument();
    expect(screen.queryByText(/Get started/i)).toBeNull();
  });

  it('shows the run error card above a preserved prior result after a failed later run', () => {
    runStore.pushTab({
      label: 'Run 1',
      kind: 'find-power',
      subView: 'summary',
      spec: sampleSpec,
      sample_size: 80,
      effect_names: ['x1', 'x2'],
      result: samplePowerResult as any,
      scenarios: [['default', samplePowerResult as any]],
    });
    runStore.runState = 'error';
    runStore.lastError = { severity: 'run', title: 'Run failed', message: 'later boom' };
    render(ResultsPane);
    expect(screen.getByText('Run failed')).toBeInTheDocument(); // the error card
    expect(screen.getByText('Run 1')).toBeInTheDocument(); // the preserved prior result tab
  });

  it('renders the get-started checklist when no tabs exist', () => {
    render(ResultsPane);
    expect(screen.getByText(/Get started/i)).toBeInTheDocument();
  });

  it('shows the family tutorial and what-to-find card beside the checklist before any run', () => {
    render(ResultsPane);
    expect(screen.getByText(/Get started/i)).toBeInTheDocument();
    expect(screen.getByTestId('what-to-find')).toBeInTheDocument();
    expect(screen.getByTestId('family-tutorial')).toBeInTheDocument();
  });

  it('switches to tabs + bars view when a Find-power tab is pushed', () => {
    runStore.pushTab({
      label: 'Run 1',
      kind: 'find-power',
      subView: 'script',
      spec: sampleSpec,
      sample_size: 80,
      effect_names: ['x1', 'x2'],
      result: samplePowerResult as any,
      scenarios: [['default', samplePowerResult as any]],
    });
    render(ResultsPane);
    expect(screen.getByText('Run 1')).toBeInTheDocument();
    expect(screen.queryByText(/Get started/i)).toBeNull();
    expect(screen.queryByTestId('what-to-find')).toBeNull();
    expect(screen.queryByTestId('family-tutorial')).toBeNull();
  });

  it('shows the progress strip while running', () => {
    runStore.pushTab({
      label: 'Run 1',
      kind: 'find-power',
      subView: 'script',
      spec: sampleSpec,
      sample_size: 80,
      effect_names: ['x1', 'x2'],
      result: samplePowerResult as any,
      scenarios: [['default', samplePowerResult as any]],
    });
    runStore.runState = 'running';
    runStore.progress.total = 100;
    runStore.progress.completed = 30;
    render(ResultsPane);
    expect(screen.getByLabelText(/Cancel run/i)).toBeInTheDocument();
  });

  it('renders the four top tabs and switches on click', async () => {
    runStore.pushTab({
      label: 'Run 1',
      kind: 'find-power',
      subView: 'summary',
      spec: sampleSpec,
      sample_size: 80,
      effect_names: ['x1', 'x2'],
      result: samplePowerResult as any,
      scenarios: [['default', samplePowerResult as any]],
    });
    render(ResultsPane);
    expect(screen.getByTestId('tab-summary')).toBeInTheDocument();
    expect(screen.getByTestId('tab-joint')).toBeInTheDocument();
    expect(screen.getByTestId('tab-script')).toBeInTheDocument();
    expect(screen.getByTestId('tab-export')).toBeInTheDocument();
    await fireEvent.click(screen.getByTestId('tab-joint'));
    expect(screen.getByTestId('joint-dist-tab')).toBeInTheDocument();
  });

  it('exposes every scenario on the pushed tab', () => {
    const result2 = { ...samplePowerResult, n: 120 };
    const tab = runStore.pushTab({
      label: 'Run Multi',
      kind: 'find-power',
      subView: 'summary',
      spec: sampleSpec,
      sample_size: 80,
      effect_names: ['x1', 'x2'],
      result: samplePowerResult as any,
      scenarios: [
        ['scenario-a', samplePowerResult as any],
        ['scenario-b', result2 as any],
      ],
    });
    expect(tab.scenarios.length).toBe(2);
    expect(tab.result).toBe(tab.scenarios[0]![1]);
  });
});
