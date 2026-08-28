// Tests for ClusterEditor.svelte — the obs/cluster warning (which run card it
// reads and how it names its source), the visible label on the cluster n input,
// and the parent name on a nested extra grouping.
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// parsedFormulaStore stub: synchronous, so the cluster cards render on mount.
vi.mock('$lib/stores/parsed-formula.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/stores/parsed-formula.svelte')>();
  const { stubParseFormula } = await import('../../../tests/parse-formula-stub');
  return {
    ...actual,
    parsedFormulaStore: {
      get: (formula: string) => stubParseFormula(formula),
      getStable: (formula: string) => stubParseFormula(formula),
    },
  };
});

import { familyStore } from '$lib/stores/family.svelte';
import ClusterEditor from './ClusterEditor.svelte';

/** Mixed family with one `(1|school)` cluster in the requested regime. */
function mixed(dimKind: 'n_clusters' | 'cluster_size') {
  familyStore.active = 'mixed';
  const cfg = familyStore.byFamily.mixed;
  cfg.formula = 'y ~ x + (1|school)';
  cfg.cluster!.dimKind = dimKind;
  cfg.cluster!.clusterName = 'school';
  return cfg;
}

/** Two ticks: the first flushes $derived, the second the reconcile $effect. */
async function flush() {
  await tick();
  await tick();
}

const text = (el: HTMLElement | null) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

describe('ClusterEditor — obs/cluster warning', () => {
  beforeEach(() => familyStore.resetAll());

  it('warns from the sample-size low end when it is the smaller case', async () => {
    const cfg = mixed('n_clusters');
    cfg.cluster!.nClusters = 40;
    cfg.findPower.n = 100; // ~2 obs/cluster — silent on its own
    cfg.findSampleSize.from = 50; // 1 obs/cluster — under the engine floor
    const { getByTestId } = render(ClusterEditor);
    await flush();
    const warning = getByTestId('cluster-size-warning');
    expect(text(warning)).toContain('~1 obs/cluster at the low end of the sample-size search');
    expect(text(warning)).toContain('engine hard minimum');
    expect(warning.className).toContain('text-red-600');
  });

  it('warns from the Find-power n when it is the smaller case, and says so', async () => {
    const cfg = mixed('n_clusters');
    cfg.cluster!.nClusters = 30;
    cfg.findPower.n = 100; // ~3 obs/cluster
    cfg.findSampleSize.unit = 'cluster_size'; // bounds are obs per cluster
    cfg.findSampleSize.from = 10; // 10 obs/cluster — fine
    cfg.findSampleSize.to = 50;
    const { getByTestId } = render(ClusterEditor);
    await flush();
    expect(text(getByTestId('cluster-size-warning'))).toContain('~3 obs/cluster at n=100');
  });

  it('is silent when both cards clear the recommended threshold', async () => {
    const cfg = mixed('n_clusters');
    cfg.cluster!.nClusters = 10;
    cfg.findPower.n = 200;
    cfg.findSampleSize.from = 150;
    const { queryByTestId } = render(ClusterEditor);
    await flush();
    expect(queryByTestId('cluster-size-warning')).toBeNull();
  });

  it('reads the fixed size directly, ignoring both run cards', async () => {
    const cfg = mixed('cluster_size');
    cfg.cluster!.clusterSize = 3;
    cfg.findPower.n = 100000;
    cfg.findSampleSize.from = 100000;
    const { getByTestId } = render(ClusterEditor);
    await flush();
    const warning = text(getByTestId('cluster-size-warning'));
    expect(warning).toContain('~3 obs/cluster —');
    expect(warning).not.toContain('at n=');
    expect(warning).not.toContain('sample-size search');
  });
});

describe('ClusterEditor — input label and parent name', () => {
  beforeEach(() => familyStore.resetAll());

  it('labels the cluster n input by the live dimension', async () => {
    const cfg = mixed('n_clusters');
    const { container } = render(ClusterEditor);
    await flush();
    const row = container.querySelector('[aria-label="Cluster dimension"]') as HTMLElement;
    expect(text(row)).toContain('clusters:');
    expect(text(row)).not.toContain('obs per');

    cfg.cluster!.dimKind = 'cluster_size';
    await flush();
    expect(text(row)).toContain('obs per school:');
  });

  it('names the parent cluster on a nested extra grouping', async () => {
    const cfg = mixed('n_clusters');
    cfg.formula = 'y ~ x + (1|school/class)';
    const { getByTestId } = render(ClusterEditor);
    await flush();
    expect(text(getByTestId('cluster-extra-school:class'))).toContain('n per school');
  });
});
