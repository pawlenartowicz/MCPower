import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// Make getStable deterministic for the seeded formula (no engine in jsdom).
vi.mock('$lib/stores/parsed-formula.svelte', () => ({
  parsedFormulaStore: {
    getStable: () => ({
      result: {
        dependent: 'y',
        predictors: ['a', 'b'],
        terms: [
          { kind: 'main', name: 'a' },
          { kind: 'main', name: 'b' },
          { kind: 'interaction', vars: ['a', 'b', 'c'] }, // 3-way -> must be carried
        ],
        random_effects: [],
      },
      error: null,
      pending: false,
    }),
  },
}));

// uploadStore mock: a mutable object so a test can seed csvData and drive the
// "Add from data" chips (null by default → no chips, matching the other suites).
const uploadStoreMock = vi.hoisted(() => ({ csvData: null as unknown, mode: 'partial' }));
vi.mock('$lib/stores/upload.svelte', () => ({ uploadStore: uploadStoreMock }));

import { familyStore } from '$lib/stores/family.svelte';
import ModelBuilderDialog from './ModelBuilderDialog.svelte';

describe('ModelBuilderDialog', () => {
  beforeEach(() => {
    familyStore.resetAll();
    uploadStoreMock.csvData = null;
  });

  it('hydrates rows, keeps a 3-way term, and writes only cfg.formula on Use', async () => {
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y ~ a + b + a:b:c';

    const onOpenChange = vi.fn();
    const { getByText, getByTestId } = render(ModelBuilderDialog, { open: true, onOpenChange });
    await tick();

    // 3-way term is surfaced as carried, not dropped.
    expect(getByTestId('formula-preview').textContent).toContain('a:b:c');

    await fireEvent.click(getByText('Use this model'));
    await tick();

    expect(familyStore.byFamily.regression.formula).toContain('a:b:c');
    expect(familyStore.byFamily.regression.formula).toContain('y ~ a + b');
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('a fresh factor starts at 3 levels with a per-level remove that hides at the 2-level minimum', async () => {
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y ~ a + b';

    const { getAllByText, getByLabelText, queryAllByLabelText } = render(ModelBuilderDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    // Switch predictor 'a' to a factor -> 3 default levels, 3 remove buttons.
    await fireEvent.click(getAllByText('factor')[0]!);
    await tick();
    expect(queryAllByLabelText(/^remove level/).length).toBe(3);

    // Removing one drops to the 2-level minimum and hides the remove buttons.
    await fireEvent.click(getByLabelText('remove level 3'));
    await tick();
    expect(queryAllByLabelText(/^remove level/).length).toBe(0);
  });

  it('useModel writes cfg.variables with the chosen factor kind + levels + referenceLevel', async () => {
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y ~ a + b';

    const { getAllByText, getByText } = render(ModelBuilderDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getAllByText('factor')[0]!); // predictor 'a' -> factor (3 levels)
    await tick();
    await fireEvent.click(getByText('Use this model'));
    await tick();

    const vars = familyStore.byFamily.regression.variables;
    const a = vars.find((v) => v.name === 'a')!;
    expect(a.kind).toBe('factor');
    expect(a.levels).toEqual(['1', '2', '3']);
    expect(a.referenceLevel).toBe('1');
    expect(vars.find((v) => v.name === 'b')?.kind).toBe('continuous');
  });

  it('adds a data-backed predictor from an uploaded-column chip, carrying its detected type + levels', async () => {
    familyStore.active = 'regression';
    familyStore.byFamily.regression.formula = 'y ~ a + b';
    // An uploaded column not already used as the dependent or a predictor.
    uploadStoreMock.csvData = {
      mode: 'partial',
      n_rows: 4,
      columns: [
        { name: 'c', col_type: 'factor', values: [0, 1, 2, 0], labels: ['lo', 'mid', 'hi'] },
      ],
    };

    const { getByText } = render(ModelBuilderDialog, { open: true, onOpenChange: vi.fn() });
    await tick();

    // The chip adds 'c' pre-typed from its detection; Use this model commits it.
    await fireEvent.click(getByText('+ c'));
    await tick();
    await fireEvent.click(getByText('Use this model'));
    await tick();

    const c = familyStore.byFamily.regression.variables.find((v) => v.name === 'c')!;
    expect(c.kind).toBe('factor');
    expect(c.levels).toEqual(['lo', 'mid', 'hi']);
  });
});
