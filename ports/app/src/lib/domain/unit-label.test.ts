// Tests for the sample-size unit wording helpers: the observations conversion
// every warning shares, Find power's n conversion, and the echo branches and
// guard the FindSampleSizeCard tests do not reach (the chip nouns and the other
// three echo branches are asserted at the rendered components).
import { describe, expect, it } from 'vitest';
import type { AppSpec } from './app-spec';
import { boundToObservations, partsText, powerNToObservations, unitEchoParts } from './unit-label';

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

describe('powerNToObservations', () => {
  const mixed = (unit: 'observations' | 'clusters', size: number) =>
    ({
      family: 'mixed',
      cluster_dim: { kind: 'cluster_size', value: size },
      ...(unit !== 'observations' ? { sample_size_unit: unit } : {}),
    }) as unknown as AppSpec;

  it('multiplies a cluster-number n by the fixed cluster size', () => {
    expect(powerNToObservations(mixed('clusters', 100), 30)).toBe(3000);
  });

  it('leaves an observations n alone', () => {
    expect(powerNToObservations(mixed('observations', 100), 100)).toBe(100);
    expect(powerNToObservations({ family: 'linear' } as unknown as AppSpec, 100)).toBe(100);
  });
});

describe('unitEchoParts', () => {
  const echo = (...args: Parameters<typeof unitEchoParts>) => partsText(unitEchoParts(...args));

  it('restates observations as obs-per-cluster under a fixed cluster count', () => {
    expect(echo('observations', 'n_clusters', 'subject', 80, 400, 20)).toBe(
      '≈ 4 to 20 observations per subject, 20 subject clusters',
    );
  });

  it('names a single value once when from equals to', () => {
    expect(echo('observations', 'cluster_size', 'ID', 100, 100, 100)).toBe(
      '≈ 1 ID clusters, 100 observations each',
    );
    expect(echo('clusters', 'cluster_size', 'ID', 30, 30, 100)).toBe('= 3,000 observations, 100 per ID');
  });

  it('returns nothing without a usable fixed value', () => {
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, 400, 0)).toEqual([]);
    expect(unitEchoParts('observations', 'n_clusters', 'subject', 80, Number.NaN, 20)).toEqual([]);
  });
});
