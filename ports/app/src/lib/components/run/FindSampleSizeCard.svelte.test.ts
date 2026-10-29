// Tests for the unit echo line on FindSampleSizeCard, one per unit, and the
// untouched card for non-mixed families.
import { render } from '@testing-library/svelte';
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

describe('FindSampleSizeCard — unit echo line', () => {
  beforeEach(() => familyStore.resetAll());

  it('renders no echo for a non-mixed family', () => {
    familyStore.active = 'regression';
    const { queryByTestId } = render(FindSampleSizeCard);
    expect(queryByTestId('unit-echo')).toBeNull();
  });

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
