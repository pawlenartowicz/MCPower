import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Force the adapter to report a formula error deterministically, independent of the
// async parser, so we test FormulaInput's gating + rendering, not the parse itself.
vi.mock('$lib/domain/app-spec-adapter', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/domain/app-spec-adapter')>();
  return {
    ...actual,
    familyConfigToAppSpec: vi.fn(() => ({ spec: null, errors: ['Outcome variable is required'], warnings: [] })),
  };
});

import FormulaInput from './FormulaInput.svelte';
import { familyStore } from '$lib/stores/family.svelte';

describe('FormulaInput inline validation', () => {
  beforeEach(() => {
    familyStore.resetActive();
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = '';
  });

  it('shows no inline error while the formula is still empty', () => {
    const { container } = render(FormulaInput);
    expect(container.querySelector('.text-destructive')).toBeNull();
  });

  it('renders the adapter validation errors inline once a formula is entered', () => {
    familyStore.byFamily.regression.formula = 'y =';
    const { getByText } = render(FormulaInput);
    expect(getByText('Outcome variable is required')).toBeTruthy();
  });
});

describe('FormulaInput example formulas follow (entrypoint, outcome)', () => {
  beforeEach(() => familyStore.resetAll()); // active=regression, regressionOutcome=continuous

  it('Regression Continuous shows continuous-outcome examples', () => {
    const { getByText, queryByText } = render(FormulaInput);
    expect(getByText('score = hours_studied + sleep')).toBeTruthy();
    expect(getByText('income = education + experience')).toBeTruthy();
    expect(queryByText('passed = hours_studied + sleep')).toBeNull();
  });

  it('Regression Binary swaps only the outcome word in the examples', () => {
    familyStore.regressionOutcome = 'logit';
    const { getByText, queryByText } = render(FormulaInput);
    expect(getByText('passed = hours_studied + sleep')).toBeTruthy();
    expect(getByText('promoted = education + experience')).toBeTruthy();
    expect(queryByText('score = hours_studied + sleep')).toBeNull();
  });

  it('toggling Mixed to Binary flips its example to the binary form', async () => {
    familyStore.active = 'mixed';
    const { getByText, queryByText } = render(FormulaInput);
    expect(getByText('score = lesson_time + (1 | classroom)')).toBeTruthy();

    familyStore.byFamily.mixed.cluster!.binaryOutcome = true;
    await tick();
    expect(getByText('passed = lesson_time + (1 | classroom)')).toBeTruthy();
    expect(queryByText('score = lesson_time + (1 | classroom)')).toBeNull();
  });
});
