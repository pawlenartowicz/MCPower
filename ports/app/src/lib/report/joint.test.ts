import { describe, it, expect } from 'vitest';
import { jointDistribution } from './joint';

describe('jointDistribution', () => {
  it('derives exactly-k and at-least-k from the histogram', () => {
    const jd = jointDistribution([10, 30, 60], 100);
    expect(jd).not.toBeNull();
    expect(jd!.exactly).toEqual([0.1, 0.3, 0.6]);
    expect(jd!.atLeast).toEqual([1.0, 0.9, 0.6]);
  });

  it('returns null on empty histogram / n=0', () => {
    expect(jointDistribution([], 0)).toBeNull();
    expect(jointDistribution(undefined, 0)).toBeNull();
  });
});
