// Tests for CorrelationsEditor.svelte — the editor shows the measured Pearson r for an
// untouched uploaded pair in partial mode (without storing it), and editing a lower-triangle
// cell writes both mirrored entries of the stored matrix.
import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { familyStore } from '$lib/stores/family.svelte';
import { uploadStore } from '$lib/stores/upload.svelte';
import CorrelationsEditor from './CorrelationsEditor.svelte';

/** Two ticks: the first flushes $derived, the second the reconcile $effect. */
async function flush() {
  await tick();
  await tick();
}

function setup() {
  familyStore.active = 'regression';
  const cfg = familyStore.byFamily.regression;
  cfg.variables = [
    { name: 'x1', kind: 'continuous' },
    { name: 'x2', kind: 'continuous' },
    { name: 'x3', kind: 'continuous' },
  ];
  // x1 and x2 are uploaded and perfectly anti-correlated; x3 is generated.
  uploadStore.mode = 'partial';
  uploadStore.load(
    {
      mode: 'partial',
      n_rows: 5,
      columns: [
        { name: 'x1', col_type: 'continuous', values: [1, 2, 3, 4, 5], labels: [] },
        { name: 'x2', col_type: 'continuous', values: [5, 4, 3, 2, 1], labels: [] },
      ],
    },
    'd.csv',
  );
  return cfg;
}

describe('CorrelationsEditor', () => {
  beforeEach(() => familyStore.resetAll());
  afterEach(() => uploadStore.clear());

  it('shows the measured r for an untouched uploaded pair, without storing it', async () => {
    const cfg = setup();
    const { container } = render(CorrelationsEditor);
    await flush();
    // Lower triangle in row-major order: (x2,x1), (x3,x1), (x3,x2).
    const inputs = [...container.querySelectorAll<HTMLInputElement>('input[type="number"]')];
    expect(inputs.map((i) => Number(i.value))).toEqual([-1, 0, 0]);
    expect(cfg.correlations[1]![0]).toBe(0);
  });

  it('editing a lower cell stores both mirrored entries and updates the read-only upper cell', async () => {
    const cfg = setup();
    const { container } = render(CorrelationsEditor);
    await flush();
    const inputs = container.querySelectorAll<HTMLInputElement>('input[type="number"]');
    await fireEvent.change(inputs[1]!, { target: { value: '0.3' } }); // (x3,x1)
    await flush();
    expect(cfg.correlations[2]![0]).toBe(0.3);
    expect(cfg.correlations[0]![2]).toBe(0.3);
    // Row x1 is [1, (x1,x2), (x1,x3)], all read-only.
    const row = container.querySelector('tbody tr')!;
    expect([...row.querySelectorAll('span')].map((s) => s.textContent?.trim())).toEqual(['1', '-1', '0.3']);
  });
});
