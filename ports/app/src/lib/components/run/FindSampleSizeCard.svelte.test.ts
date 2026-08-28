// Tests for the sample-size unit chips on FindSampleSizeCard: which chip the
// cluster dimension offers, the echo line for each unit, the reset when the
// dimension changes, and the untouched card for non-mixed families.
import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import { familyStore } from '$lib/stores/family.svelte';
import FindSampleSizeCard from './FindSampleSizeCard.svelte';

/** Put the mixed family in the requested cluster regime with a named cluster. */
function mixedCluster(dimKind: 'n_clusters' | 'cluster_size') {
  familyStore.active = 'mixed';
  const cfg = familyStore.byFamily.mixed;
  cfg.cluster!.clusterName = 'subject';
  cfg.cluster!.dimKind = dimKind;
  cfg.cluster!.clusterSize = 8;
  cfg.cluster!.nClusters = 20;
  cfg.findSampleSize.from = 80;
  cfg.findSampleSize.to = 400;
  cfg.findSampleSize.unit = 'observations';
  return cfg;
}

describe('FindSampleSizeCard — unit chips', () => {
  beforeEach(() => familyStore.resetAll());

  it('offers "cluster number" when the cluster size is fixed', () => {
    mixedCluster('cluster_size');
    const { getByTestId } = render(FindSampleSizeCard);
    expect(getByTestId('unit-observations').textContent?.trim()).toBe('observations');
    expect(getByTestId('unit-derived').textContent?.replace(/\s+/g, ' ').trim()).toBe(
      'cluster number',
    );
  });

  it('offers "cluster size" when the cluster count is fixed', () => {
    mixedCluster('n_clusters');
    const { getByTestId } = render(FindSampleSizeCard);
    expect(getByTestId('unit-derived').textContent?.replace(/\s+/g, ' ').trim()).toBe(
      'cluster size',
    );
  });

  it('names no cluster in either chip', () => {
    mixedCluster('cluster_size');
    const { getByTestId } = render(FindSampleSizeCard);
    expect(getByTestId('unit-observations').textContent).not.toContain('subject');
    expect(getByTestId('unit-derived').textContent).not.toContain('subject');
  });

  it('puts the chips on their own row, outside the from/to input row', () => {
    mixedCluster('cluster_size');
    const { getByTestId, getByLabelText } = render(FindSampleSizeCard);
    const chipRow = getByLabelText('Sample-size unit');
    expect(chipRow.contains(getByTestId('unit-derived'))).toBe(true);
    // The chips must not sit inside the row holding the numeric inputs.
    const inputRow = (document.getElementById('find-n-from') as HTMLElement).closest(
      '.flex.flex-wrap.items-end',
    );
    expect(inputRow).not.toBeNull();
    expect(inputRow!.contains(chipRow)).toBe(false);
  });

  it('renders no chips and no echo for a non-mixed family', () => {
    familyStore.active = 'regression';
    const { queryByTestId } = render(FindSampleSizeCard);
    expect(queryByTestId('unit-observations')).toBeNull();
    expect(queryByTestId('unit-derived')).toBeNull();
    expect(queryByTestId('unit-echo')).toBeNull();
  });

  it('selects the derived unit on click and marks it checked', async () => {
    const cfg = mixedCluster('cluster_size');
    const { getByTestId } = render(FindSampleSizeCard);
    await fireEvent.click(getByTestId('unit-derived'));
    expect(cfg.findSampleSize.unit).toBe('clusters');
    expect(getByTestId('unit-derived').getAttribute('aria-checked')).toBe('true');
    expect(getByTestId('unit-observations').getAttribute('aria-checked')).toBe('false');
  });

  it('resets to observations when the cluster dimension changes', async () => {
    const cfg = mixedCluster('cluster_size');
    render(FindSampleSizeCard);
    await fireEvent.click(document.querySelector('[data-testid="unit-derived"]')!);
    expect(cfg.findSampleSize.unit).toBe('clusters');
    cfg.cluster!.dimKind = 'n_clusters';
    await tick();
    expect(cfg.findSampleSize.unit).toBe('observations');
  });
});

describe('FindSampleSizeCard — unit echo line', () => {
  beforeEach(() => familyStore.resetAll());

  const echo = (el: HTMLElement) => el.textContent?.replace(/\s+/g, ' ').trim();

  it('restates observations as clusters under a fixed cluster size', () => {
    mixedCluster('cluster_size'); // 8 per subject, bounds 80..400 observations
    const { getByTestId } = render(FindSampleSizeCard);
    expect(echo(getByTestId('unit-echo'))).toBe('≈ 10 to 50 subject clusters, 8 observations each');
  });

  it('restates clusters as observations', async () => {
    const cfg = mixedCluster('cluster_size');
    cfg.findSampleSize.from = 10;
    cfg.findSampleSize.to = 50;
    cfg.findSampleSize.unit = 'clusters';
    const { getByTestId } = render(FindSampleSizeCard);
    expect(echo(getByTestId('unit-echo'))).toBe('= 80 to 400 observations, 8 per subject');
  });

  it('restates observations-per-cluster as observations', () => {
    const cfg = mixedCluster('n_clusters'); // 20 subjects
    cfg.findSampleSize.from = 12;
    cfg.findSampleSize.to = 60;
    cfg.findSampleSize.unit = 'cluster_size';
    const { getByTestId } = render(FindSampleSizeCard);
    expect(echo(getByTestId('unit-echo'))).toBe('= 240 to 1,200 observations, 20 subject clusters');
  });
});
