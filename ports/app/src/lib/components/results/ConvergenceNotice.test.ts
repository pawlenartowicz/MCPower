import { describe, it, expect } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import type { RunTab } from '$lib/stores/run.svelte';
import type { PowerResult, EstimatorExtras } from '$lib/domain/result';
import type { AppSpec } from '$lib/domain/app-spec';

import ConvergenceNotice from './ConvergenceNotice.svelte';

// Minimal find-power result; callers override the diagnostics-relevant fields.
function makeResult(over: Partial<PowerResult> = {}): PowerResult {
  return {
    n: 80,
    n_sims: 100,
    target_indices: [1],
    power_uncorrected: [0.8],
    power_corrected: [0.8],
    ci_uncorrected: [{ lo: 0.7, hi: 0.9 }],
    ci_corrected: [{ lo: 0.7, hi: 0.9 }],
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras: { estimator: 'ols' },
    ...over,
  };
}

// A minimal LinearSpec body (OLS). Family-specific fixtures spread over it.
const LINEAR_BODY = {
  parsed_formula: { outcome: 'y', predictors: ['x1'], interaction_terms: [] },
  var_types: [{ kind: 'numeric' as const, name: 'x1' }],
  effects: [{ name: 'x1', value: 0.3 }],
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
  contrasts: [] as Array<[string, string]>,
};

function makeTab(spec: AppSpec, result: PowerResult): RunTab {
  return {
    id: 'test-tab',
    label: 'Run 1',
    kind: 'find-power',
    subView: 'summary',
    spec,
    sample_size: result.n,
    effect_names: ['x1'],
    result,
    scenarios: [['default', result]],
    plots: { blocks: [{ key: 'power', spec: '{"mark":"bar"}' }] },
  };
}

const olsSpec: AppSpec = { family: 'linear', ...LINEAR_BODY };

async function expandBadge(getByText: (m: RegExp) => HTMLElement): Promise<void> {
  await fireEvent.click(getByText(/issues detected/));
}

describe('ConvergenceNotice', () => {
  it('shows "all clear" for a clean OLS run (no false positives)', () => {
    const { getByText, queryByText } = render(ConvergenceNotice, {
      props: { tab: makeTab(olsSpec, makeResult()) },
    });
    expect(getByText(/all clear/)).toBeTruthy();
    expect(queryByText(/issues detected/)).toBeNull();
  });

  it('computes high-τ̂ boundary from the boundary_hit array, ignoring benign τ̂=0', async () => {
    // 40 benign τ̂=0 (value 1) + 5 high-τ̂ (value 2); n_sims=100 → rate 5% > 1%.
    // Only the high-τ̂ count drives the warning; the 40 ones must be ignored.
    const result = makeResult({
      n_sims: 100,
      boundary_hit: [...Array(40).fill(1), ...Array(5).fill(2)],
      estimator_extras: { estimator: 'glm', baseline_prob_realized: 0.3 } as EstimatorExtras,
    });
    const { getByText } = render(ConvergenceNotice, { props: { tab: makeTab(olsSpec, result) } });
    await expandBadge(getByText);
    expect(getByText(/High-τ̂ boundary hits 5\.0%/)).toBeTruthy();
  });

  it('surfaces GLMM (glm + cluster) boundary hits — previously skipped for non-mle', async () => {
    // estimator 'glm' would have been skipped by the old estimator==='mle' guard.
    const mixedSpec: AppSpec = {
      family: 'mixed',
      ...LINEAR_BODY,
      cluster_name: 'site',
      icc: 0.1,
      cluster_dim: { kind: 'n_clusters', value: 20 },
      outcome: { kind: 'binary', baseline_probability: 0.3 },
    };
    const result = makeResult({
      n_sims: 100,
      boundary_hit: Array(3).fill(2), // 3% > 1%
      estimator_extras: { estimator: 'glm', baseline_prob_realized: 0.3 } as EstimatorExtras,
    });
    const { getByText } = render(ConvergenceNotice, { props: { tab: makeTab(mixedSpec, result) } });
    await expandBadge(getByText);
    expect(getByText(/High-τ̂ boundary hits 3\.0%/)).toBeTruthy();
  });

  it('computes GLM baseline drift against the requested baseline from the spec', async () => {
    const logitSpec: AppSpec = { family: 'logit', ...LINEAR_BODY, baseline_probability: 0.3 };
    const result = makeResult({
      estimator_extras: { estimator: 'glm', baseline_prob_realized: 0.5 } as EstimatorExtras,
    });
    const { getByText } = render(ConvergenceNotice, { props: { tab: makeTab(logitSpec, result) } });
    await expandBadge(getByText);
    // |0.5 − 0.3| = 0.2 → 20.0% > 5% threshold.
    expect(getByText(/drifted by 20\.0%/)).toBeTruthy();
  });

  it('renders the Laplace-bias line for GLMM with large τ̂² and small clusters', async () => {
    // cluster_size 4 < recommended 10; τ̂² 2.0 > glmm_tau_sq_warn 1.0.
    const mixedSpec: AppSpec = {
      family: 'mixed',
      ...LINEAR_BODY,
      cluster_name: 'site',
      icc: 0.1,
      cluster_dim: { kind: 'cluster_size', value: 4 },
      outcome: { kind: 'binary', baseline_probability: 0.3 },
    };
    const result = makeResult({
      estimator_extras: {
        estimator: 'glm', baseline_prob_realized: 0.3, tau_squared_hat_mean: 2.0,
      } as EstimatorExtras,
    });
    const { getByText } = render(ConvergenceNotice, { props: { tab: makeTab(mixedSpec, result) } });
    await expandBadge(getByText);
    expect(getByText(/Laplace-approximation bias likely/)).toBeTruthy();
  });
});
