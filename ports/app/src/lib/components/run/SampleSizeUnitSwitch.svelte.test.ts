// Tests for the sample-size unit switch: which chip the cluster dimension
// offers, the selection, and the reset when the dimension changes.
import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import { familyStore } from '$lib/stores/family.svelte';
import SampleSizeUnitSwitch from './SampleSizeUnitSwitch.svelte';

function mixedCluster(dimKind: 'n_clusters' | 'cluster_size') {
  familyStore.active = 'mixed';
  const cfg = familyStore.byFamily.mixed;
  cfg.cluster!.clusterName = 'subject';
  cfg.cluster!.dimKind = dimKind;
  cfg.findSampleSize.unit = 'observations';
  return cfg;
}

describe('SampleSizeUnitSwitch', () => {
  beforeEach(() => familyStore.resetAll());

  it('offers "cluster number" when the cluster size is fixed', () => {
    mixedCluster('cluster_size');
    const { getByTestId } = render(SampleSizeUnitSwitch);
    expect(getByTestId('unit-observations').textContent?.trim()).toBe('observations');
    expect(getByTestId('unit-derived').textContent?.replace(/\s+/g, ' ').trim()).toBe(
      'cluster number',
    );
  });

  it('offers "cluster size" when the cluster count is fixed', () => {
    mixedCluster('n_clusters');
    const { getByTestId } = render(SampleSizeUnitSwitch);
    expect(getByTestId('unit-derived').textContent?.replace(/\s+/g, ' ').trim()).toBe(
      'cluster size',
    );
  });

  it('names no cluster in either chip', () => {
    mixedCluster('cluster_size');
    const { getByTestId } = render(SampleSizeUnitSwitch);
    expect(getByTestId('unit-observations').textContent).not.toContain('subject');
    expect(getByTestId('unit-derived').textContent).not.toContain('subject');
  });

  it('renders no chips for a non-mixed family', () => {
    familyStore.active = 'regression';
    const { queryByTestId } = render(SampleSizeUnitSwitch);
    expect(queryByTestId('unit-observations')).toBeNull();
    expect(queryByTestId('unit-derived')).toBeNull();
  });

  it('selects the derived unit on click and marks it checked', async () => {
    const cfg = mixedCluster('cluster_size');
    const { getByTestId } = render(SampleSizeUnitSwitch);
    await fireEvent.click(getByTestId('unit-derived'));
    expect(cfg.findSampleSize.unit).toBe('clusters');
    expect(getByTestId('unit-derived').getAttribute('aria-checked')).toBe('true');
    expect(getByTestId('unit-observations').getAttribute('aria-checked')).toBe('false');
  });

  it('resets to observations when the cluster dimension changes', async () => {
    const cfg = mixedCluster('cluster_size');
    render(SampleSizeUnitSwitch);
    await fireEvent.click(document.querySelector('[data-testid="unit-derived"]')!);
    expect(cfg.findSampleSize.unit).toBe('clusters');
    cfg.cluster!.dimKind = 'n_clusters';
    await tick();
    expect(cfg.findSampleSize.unit).toBe('observations');
  });
});
