import { beforeEach, describe, expect, it } from 'vitest';
import { familyStore } from './family.svelte';

describe('familyStore — per-family isolation', () => {
  beforeEach(() => familyStore.resetAll());

  it('preserves ANOVA config when switching away and back', () => {
    familyStore.active = 'anova';
    familyStore.byFamily.anova.formula = 'y ~ A * B';
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y = x1 + x2';
    familyStore.active = 'anova';
    expect(familyStore.byFamily.anova.formula).toBe('y ~ A * B');
  });

  it('resetActive only resets the active family', () => {
    familyStore.byFamily.anova.formula = 'A1';
    familyStore.byFamily.regression.formula = 'L1';
    familyStore.active = 'regression';
    familyStore.resetActive();
    expect(familyStore.byFamily.regression.formula).toBe('');
    expect(familyStore.byFamily.anova.formula).toBe('A1');
  });
});

describe('familyStore.activeOutcome — resolved per active family', () => {
  beforeEach(() => familyStore.resetAll());

  it('regression reads its store-level outcome toggle', () => {
    familyStore.active = 'regression';
    expect(familyStore.activeOutcome).toBe('continuous');
    familyStore.regressionOutcome = 'binary';
    expect(familyStore.activeOutcome).toBe('binary');
  });

  it('mixed reads its cluster.binaryOutcome flag', () => {
    familyStore.active = 'mixed';
    expect(familyStore.activeOutcome).toBe('continuous'); // flag absent → Gaussian LME
    familyStore.byFamily.mixed.cluster!.binaryOutcome = true;
    expect(familyStore.activeOutcome).toBe('binary');
  });

  it('ignores the other family\'s outcome state', () => {
    // Regression binary must not bleed into mixed, and vice-versa.
    familyStore.regressionOutcome = 'binary';
    familyStore.active = 'mixed';
    expect(familyStore.activeOutcome).toBe('continuous');
  });

  it('anova is always continuous', () => {
    familyStore.active = 'anova';
    expect(familyStore.activeOutcome).toBe('continuous');
  });
});
