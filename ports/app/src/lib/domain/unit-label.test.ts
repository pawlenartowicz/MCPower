// Tests for the sample-size unit wording helpers: the observations conversion
// every warning shares, and the echo branch and guard the FindSampleSizeCard
// tests do not reach (the chip nouns and the other three echo branches are
// asserted at the rendered card).
import { describe, expect, it } from 'vitest';
import { boundToObservations, partsText, unitEchoParts } from './unit-label';

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

describe('unitEchoParts', () => {
  const echo = (...args: Parameters<typeof unitEchoParts>) => partsText(unitEchoParts(...args));

  it('restates observations as obs-per-cluster under a fixed cluster count', () => {
    expect(echo('observations', 'n_clusters', 'subject', 80, 400, 20)).toBe(
      '≈ 4 to 20 observations per subject, 20 subject clusters',
    );
  });

  it('returns nothing without a usable fixed value', () => {
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, 400, 0)).toEqual([]);
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, Number.NaN, 20)).toEqual([]);
  });
});
