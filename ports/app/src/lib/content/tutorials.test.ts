import { describe, it, expect } from 'vitest';
import { TUTORIAL } from './tutorials';
import { FAMILIES } from '$lib/domain/family';

describe('TUTORIAL', () => {
  it('has non-empty markdown for every family', () => {
    for (const f of FAMILIES) {
      expect(typeof TUTORIAL[f]).toBe('string');
      expect(TUTORIAL[f].length).toBeGreaterThan(0);
    }
  });

  it('maps each family to its own page', () => {
    expect(TUTORIAL.regression).toContain('Regression power');
    expect(TUTORIAL.anova).toContain('ANOVA power');
    expect(TUTORIAL.mixed).toContain('Mixed-model power');
  });
});
