// Tests for ModelSection.svelte's shared outcome toggle (regression + mixed).
// We render the real section and click the real Continuous/Binary buttons so the
// inline setOutcome (incl. the binary-GLMM baseline seeding) is exercised end to end.
import { describe, expect, it, beforeEach, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';

// Mirror PredictorCards' mock surface: ModelSection mounts children that would
// otherwise reach the real Rust parser / engine (unavailable under vitest). The
// toggle itself depends on none of these — they only keep the children inert.
vi.mock('$lib/stores/upload.svelte', () => ({
  uploadStore: { csvData: null, mode: 'partial', clear() {} },
}));
vi.mock('$lib/stores/parsed-formula.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/stores/parsed-formula.svelte')>();
  const { stubParseFormula } = await import('../../../tests/parse-formula-stub');
  return {
    ...actual,
    parsedFormulaStore: {
      get: (f: string) => stubParseFormula(f),
      getStable: (f: string) => stubParseFormula(f),
    },
  };
});
vi.mock('$lib/api/engine', () => ({
  getEffectsFromData: vi.fn(async () => []),
  parseFormula: vi.fn(async () => null),
  findPower: vi.fn(async () => null),
  findSampleSize: vi.fn(async () => null),
  cancelRun: vi.fn(async () => true),
  onProgress: vi.fn(async () => () => {}),
}));

import ModelSection from './ModelSection.svelte';
import { familyStore } from '$lib/stores/family.svelte';

function toggleButton(container: HTMLElement, label: 'Continuous' | 'Binary'): HTMLButtonElement {
  const btn = Array.from(container.querySelectorAll('button')).find(
    (b) => b.textContent?.trim() === label,
  );
  if (!btn) throw new Error(`toggle button "${label}" not found`);
  return btn as HTMLButtonElement;
}

describe('ModelSection outcome toggle', () => {
  beforeEach(() => familyStore.resetAll());

  it('mixed → Binary sets cluster.binaryOutcome and seeds the baseline; → Continuous clears it', async () => {
    familyStore.active = 'mixed';
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Binary'));
    await tick();
    const cl = familyStore.byFamily.mixed.cluster!;
    expect(cl.binaryOutcome).toBe(true);
    expect(cl.baselineProbability).toBe(0.2); // seeded for the adapter's (0,1) requirement
    expect(familyStore.activeOutcome).toBe('binary');

    await fireEvent.click(toggleButton(container, 'Continuous'));
    await tick();
    expect(familyStore.byFamily.mixed.cluster!.binaryOutcome).toBe(false);
    expect(familyStore.activeOutcome).toBe('continuous');
  });

  it('regression toggle still writes the store-level outcome and leaves mixed untouched', async () => {
    familyStore.active = 'regression';
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Binary'));
    await tick();
    expect(familyStore.regressionOutcome).toBe('binary');
    expect(familyStore.activeOutcome).toBe('binary');
    // Regression's binary flag lives in the store, never on the mixed cluster.
    expect(familyStore.byFamily.mixed.cluster!.binaryOutcome).toBeUndefined();
  });
});
