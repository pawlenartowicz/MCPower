// BaselineProbabilityInput binds to the active family's baseline home:
// mixed → cfg.cluster.baselineProbability (binary GLMM), others → cfg.baselineProbability.
import { describe, expect, it, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import BaselineProbabilityInput from './BaselineProbabilityInput.svelte';
import BaselineRateInput from './BaselineRateInput.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('BaselineProbabilityInput bind target', () => {
  beforeEach(() => familyStore.resetAll());

  it('binds to cfg.baselineProbability under Regression', async () => {
    familyStore.active = 'regression'; // default seeds cfg.baselineProbability = 0.2
    const { getByLabelText } = render(BaselineProbabilityInput);
    const input = getByLabelText('Baseline probability') as HTMLInputElement;
    expect(input.value).toBe('0.2');

    // NumberInput commits on change, not input.
    await fireEvent.change(input, { target: { value: '0.35' } });
    expect(familyStore.byFamily.regression.baselineProbability).toBeCloseTo(0.35);
  });

  it('binds to cfg.cluster.baselineProbability under Mixed', async () => {
    familyStore.active = 'mixed';
    familyStore.byFamily.mixed.cluster!.baselineProbability = 0.3;
    const { getByLabelText } = render(BaselineProbabilityInput);
    const input = getByLabelText('Baseline probability') as HTMLInputElement;
    expect(input.value).toBe('0.3');

    await fireEvent.change(input, { target: { value: '0.45' } });
    expect(familyStore.byFamily.mixed.cluster!.baselineProbability).toBeCloseTo(0.45);
  });

  it('renders nothing when the active family has no baseline target', () => {
    familyStore.active = 'mixed'; // mixed default cluster has no baselineProbability
    const { container } = render(BaselineProbabilityInput);
    expect(container.querySelector('input')).toBeNull();
  });
});

describe('BaselineRateInput bind target', () => {
  beforeEach(() => familyStore.resetAll());

  it('binds to cfg.baselineRate under Regression', async () => {
    familyStore.active = 'regression'; // default seeds cfg.baselineRate = 2.0
    const { getByLabelText } = render(BaselineRateInput);
    const input = getByLabelText('Baseline rate') as HTMLInputElement;
    expect(input.value).toBe('2');

    await fireEvent.change(input, { target: { value: '3.5' } });
    expect(familyStore.byFamily.regression.baselineRate).toBeCloseTo(3.5);
  });

  it('binds to cfg.cluster.baselineRate under Mixed', async () => {
    familyStore.active = 'mixed';
    familyStore.byFamily.mixed.cluster!.baselineRate = 1.5;
    const { getByLabelText } = render(BaselineRateInput);
    const input = getByLabelText('Baseline rate') as HTMLInputElement;
    expect(input.value).toBe('1.5');

    await fireEvent.change(input, { target: { value: '4' } });
    expect(familyStore.byFamily.mixed.cluster!.baselineRate).toBeCloseTo(4);
  });

  it('renders nothing when the active family has no rate target', () => {
    familyStore.active = 'mixed'; // mixed default cluster has no baselineRate
    const { container } = render(BaselineRateInput);
    expect(container.querySelector('input')).toBeNull();
  });
});
