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
  getEffectsFromData: vi.fn(async () => ({ effects: [], cluster_icc: null, baseline_probability: null })),
  parseFormula: vi.fn(async () => null),
  findPower: vi.fn(async () => null),
  findSampleSize: vi.fn(async () => null),
  cancelRun: vi.fn(async () => true),
  onProgress: vi.fn(async () => () => {}),
}));

import ModelSection from './ModelSection.svelte';
import { familyStore } from '$lib/stores/family.svelte';

function toggleButton(
  container: HTMLElement,
  label: 'Continuous' | 'Logit' | 'Probit' | 'Poisson',
): HTMLButtonElement {
  const btn = Array.from(container.querySelectorAll('button')).find(
    (b) => b.textContent?.trim() === label,
  );
  if (!btn) throw new Error(`toggle button "${label}" not found`);
  return btn as HTMLButtonElement;
}

describe('ModelSection outcome toggle', () => {
  beforeEach(() => familyStore.resetAll());

  it('mixed → Logit sets cluster.outcomeKind and seeds the baseline; → Continuous clears it', async () => {
    familyStore.active = 'mixed';
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Logit'));
    await tick();
    const cl = familyStore.byFamily.mixed.cluster!;
    expect(cl.outcomeKind).toBe('logit');
    expect(cl.baselineProbability).toBe(0.2); // seeded for the adapter's (0,1) requirement
    expect(familyStore.activeOutcome).toBe('logit');

    await fireEvent.click(toggleButton(container, 'Continuous'));
    await tick();
    expect(familyStore.byFamily.mixed.cluster!.outcomeKind).toBe('linear');
    expect(familyStore.activeOutcome).toBe('linear');
  });

  it('mixed → Poisson seeds baseline rate and raw τ²', async () => {
    familyStore.active = 'mixed';
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Poisson'));
    await tick();
    const cl = familyStore.byFamily.mixed.cluster!;
    expect(cl.outcomeKind).toBe('poisson');
    expect(cl.baselineRate).toBe(2.0);
    expect(cl.tauSquared).toBe(0.5);
    expect(familyStore.activeOutcome).toBe('poisson');
  });

  it('mixed → Poisson: τ² input is editable and replaces the ICC input (adapter ships the edited value)', async () => {
    familyStore.active = 'mixed';
    familyStore.byFamily.mixed.formula = 'y ~ x + (1|school)';
    familyStore.byFamily.mixed.effects = [{ name: 'x', value: 0.5 }];
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Poisson'));
    await tick();

    // ICC input must be gone — it does nothing for a log-link count model.
    expect(container.querySelector('input#cluster-icc')).toBeNull();
    const tauInput = container.querySelector('input#cluster-tau-squared') as HTMLInputElement;
    expect(tauInput).not.toBeNull();
    expect(tauInput.value).toBe('0.5'); // the ModelSection seed

    await fireEvent.change(tauInput, { target: { value: '1.2' } });
    await tick();
    expect(familyStore.byFamily.mixed.cluster!.tauSquared).toBe(1.2);

    // The adapter must carry the edited value, not the stale 0.5 seed.
    const { familyConfigToAppSpec } = await import('$lib/domain/app-spec-adapter');
    const { spec, errors } = familyConfigToAppSpec('mixed', familyStore.byFamily.mixed);
    expect(errors).toEqual([]);
    if (spec?.family !== 'mixed') throw new Error('expected mixed');
    expect((spec.outcome as { tau_squared: number }).tau_squared).toBe(1.2);
  });

  it('regression toggle still writes the store-level outcome and leaves mixed untouched', async () => {
    familyStore.active = 'regression';
    const { container } = render(ModelSection);
    await tick();

    await fireEvent.click(toggleButton(container, 'Probit'));
    await tick();
    expect(familyStore.regressionOutcome).toBe('probit');
    expect(familyStore.activeOutcome).toBe('probit');
    // Regression's outcome lives in the store, never on the mixed cluster.
    expect(familyStore.byFamily.mixed.cluster!.outcomeKind).toBeUndefined();
  });
});
