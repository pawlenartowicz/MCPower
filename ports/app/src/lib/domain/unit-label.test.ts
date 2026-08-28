// Tests for the sample-size unit wording helpers: the observations conversion
// every warning shares, the generic chip nouns, and the four echo branches.
import { describe, expect, it } from 'vitest';
import { boundToObservations, partsText, unitChipParts, unitEchoParts } from './unit-label';

describe('boundToObservations', () => {
  it('leaves an observations bound alone', () => {
    expect(boundToObservations('observations', 300, 20)).toBe(300);
  });

  it('multiplies a cluster count by the fixed cluster size', () => {
    expect(boundToObservations('clusters', 10, 8)).toBe(80);
  });

  it('multiplies a per-cluster bound by the fixed cluster count', () => {
    expect(boundToObservations('cluster_size', 12, 20)).toBe(240);
  });
});

describe('unitChipParts', () => {
  it('names generic nouns, never the cluster name, and uses no mono run', () => {
    expect(partsText(unitChipParts('observations'))).toBe('observations');
    expect(partsText(unitChipParts('clusters'))).toBe('cluster number');
    expect(partsText(unitChipParts('cluster_size'))).toBe('cluster size');
    for (const u of ['observations', 'clusters', 'cluster_size'] as const) {
      expect(unitChipParts(u).some((p) => p.mono)).toBe(false);
    }
  });
});

describe('unitEchoParts', () => {
  const echo = (...args: Parameters<typeof unitEchoParts>) => partsText(unitEchoParts(...args));

  it('restates observations as clusters under a fixed cluster size', () => {
    // 80..400 observations, 8 per subject.
    expect(echo('observations', 'cluster_size', 'subject', 80, 400, 8)).toBe(
      '≈ 10 to 50 subject clusters, 8 observations each',
    );
  });

  it('restates observations as obs-per-cluster under a fixed cluster count', () => {
    expect(echo('observations', 'n_clusters', 'subject', 80, 400, 20)).toBe(
      '≈ 4 to 20 observations per subject, 20 subject clusters',
    );
  });

  it('restates clusters as observations', () => {
    expect(echo('clusters', 'cluster_size', 'subject', 10, 50, 8)).toBe(
      '= 80 to 400 observations, 8 per subject',
    );
  });

  it('restates obs-per-cluster as observations', () => {
    expect(echo('cluster_size', 'n_clusters', 'subject', 12, 60, 20)).toBe(
      '= 240 to 1,200 observations, 20 subject clusters',
    );
  });

  it('returns nothing without a usable fixed value', () => {
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, 400, 0)).toEqual([]);
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, Number.NaN, 20)).toEqual([]);
  });
});
