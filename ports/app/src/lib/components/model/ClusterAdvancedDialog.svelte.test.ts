// Tests for ClusterAdvancedDialog.svelte — the cluster-level checkboxes refuse
// predictors bound to an uploaded column (the engine resamples those and cannot
// draw them once per cluster).
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// parsedFormulaStore stub: synchronous, so the predictor list renders on mount.
vi.mock('$lib/stores/parsed-formula.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/stores/parsed-formula.svelte')>();
  const { stubParseFormula } = await import('../../../tests/parse-formula-stub');
  return {
    ...actual,
    parsedFormulaStore: {
      get: (formula: string) => stubParseFormula(formula),
      getStable: (formula: string) => stubParseFormula(formula),
    },
  };
});

import { familyStore } from '$lib/stores/family.svelte';
import { uploadStore } from '$lib/stores/upload.svelte';
import ClusterAdvancedDialog from './ClusterAdvancedDialog.svelte';

function mixed() {
  familyStore.active = 'mixed';
  const cfg = familyStore.byFamily.mixed;
  cfg.formula = 'y ~ x1 + x2 + (1|school)';
  cfg.cluster!.clusterName = 'school';
  return cfg;
}

function uploadX1() {
  uploadStore.load(
    {
      mode: 'partial',
      n_rows: 3,
      columns: [{ name: 'x1', col_type: 'continuous', values: [1, 2, 3], labels: [] }],
    },
    'pilot.csv',
  );
}

const box = (id: string) => document.querySelector<HTMLInputElement>(`[data-testid="cluster-level-${id}"]`)!;

describe('ClusterAdvancedDialog — cluster-level variables', () => {
  beforeEach(() => familyStore.resetAll());
  afterEach(() => uploadStore.clear());

  it('disables an uploaded predictor and leaves the synthesized one selectable', async () => {
    mixed();
    uploadX1();
    render(ClusterAdvancedDialog, { open: true });
    await tick();
    expect(box('x1').disabled).toBe(true);
    expect(box('x2').disabled).toBe(false);
  });

  it('keeps an already-checked uploaded predictor clearable', async () => {
    const cfg = mixed();
    cfg.cluster!.clusterLevelVars = ['x1'];
    uploadX1();
    render(ClusterAdvancedDialog, { open: true });
    await tick();
    expect(box('x1').disabled).toBe(false);
    box('x1').click();
    await tick();
    expect(cfg.cluster!.clusterLevelVars).toEqual([]);
  });
});
