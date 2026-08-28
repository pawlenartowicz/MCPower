// Tests for PredictorAdvancedDialog.svelte — the factor "Factor shares" tri-state.
// The three buttons map to VariableRow.sampledProportions (Option<bool> on the
// wire): Default → undefined (inherit each scenario), Exact → false, Sampled → true.
// Dialog content is portaled to <body>, so queries go through `screen`.
import { render, screen, fireEvent } from '@testing-library/svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import type { VariableRow } from '$lib/domain/family';
import { familyStore } from '$lib/stores/family.svelte';
import PredictorAdvancedDialog from './PredictorAdvancedDialog.svelte';

function factorRow(over: Partial<VariableRow> = {}): VariableRow {
  return {
    name: 'experience',
    kind: 'factor',
    levels: ['1', '2', '3'],
    nLevels: 3,
    levelProportions: [1 / 3, 1 / 3, 1 / 3],
    referenceLevel: '1',
    ...over,
  };
}

const checked = (testid: string) =>
  screen.getByTestId(testid).getAttribute('aria-checked');

describe('PredictorAdvancedDialog factor shares tri-state', () => {
  it('renders all three modes and selects Default when sampledProportions is unset', async () => {
    render(PredictorAdvancedDialog, { props: { variable: factorRow(), open: true } });
    await screen.findByTestId('shares-default');
    expect(checked('shares-default')).toBe('true');
    expect(checked('shares-exact')).toBe('false');
    expect(checked('shares-sampled')).toBe('false');
  });

  it('pre-selects Exact when sampledProportions is false', async () => {
    render(PredictorAdvancedDialog, {
      props: { variable: factorRow({ sampledProportions: false }), open: true },
    });
    await screen.findByTestId('shares-exact');
    expect(checked('shares-exact')).toBe('true');
    expect(checked('shares-default')).toBe('false');
    expect(checked('shares-sampled')).toBe('false');
  });

  it('pre-selects Sampled when sampledProportions is true', async () => {
    render(PredictorAdvancedDialog, {
      props: { variable: factorRow({ sampledProportions: true }), open: true },
    });
    await screen.findByTestId('shares-sampled');
    expect(checked('shares-sampled')).toBe('true');
    expect(checked('shares-default')).toBe('false');
    expect(checked('shares-exact')).toBe('false');
  });

  it('clicking each mode writes the matching tri-state value back to the bound row', async () => {
    // Asserts the write-back (the component's actual logic). The bound row here
    // is a plain object, so the post-click `aria-checked` re-render isn't
    // exercised — that selected→DOM mapping is covered by the pre-selection
    // tests above; in the app `variable` is a reactive $state row.
    const v = factorRow();
    render(PredictorAdvancedDialog, { props: { variable: v, open: true } });

    await fireEvent.click(await screen.findByTestId('shares-sampled'));
    expect(v.sampledProportions).toBe(true);

    await fireEvent.click(screen.getByTestId('shares-exact'));
    expect(v.sampledProportions).toBe(false);

    // Default clears the override (undefined → None → inherit scenario default).
    await fireEvent.click(screen.getByTestId('shares-default'));
    expect(v.sampledProportions).toBeUndefined();
  });

  it('disables all three mode buttons when locked', async () => {
    render(PredictorAdvancedDialog, {
      props: { variable: factorRow(), open: true, locked: true },
    });
    await screen.findByTestId('shares-default');
    for (const id of ['shares-default', 'shares-exact', 'shares-sampled']) {
      expect((screen.getByTestId(id) as HTMLButtonElement).disabled).toBe(true);
    }
  });
});

describe('PredictorAdvancedDialog small-cell warning', () => {
  // Both run cards are always rendered, so the check quotes the smaller of the
  // Find-power n and the sample-size low bound, and names which one it used.
  beforeEach(() => familyStore.resetAll());

  const warnings = async () => {
    await screen.findAllByTestId('small-cell-warning');
    return screen
      .getAllByTestId('small-cell-warning')
      .map((el) => el.textContent?.replace(/\s+/g, ' ').trim() ?? '');
  };

  it('uses the sample-size low bound when it is the smaller of the two', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.findPower.n = 300; // 100 obs per level — no warning on its own
    cfg.findSampleSize.from = 30; // 10 obs per level
    render(PredictorAdvancedDialog, { props: { variable: factorRow(), open: true } });
    const msgs = await warnings();
    expect(msgs).toHaveLength(3);
    expect(msgs[0]).toContain('~10 obs in [1] at the low end of the sample-size search');
  });

  it('uses the Find-power n when it is the smaller of the two', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.findPower.n = 30;
    cfg.findSampleSize.from = 1000;
    render(PredictorAdvancedDialog, { props: { variable: factorRow(), open: true } });
    const msgs = await warnings();
    expect(msgs[0]).toContain('~10 obs in [1] at n=30');
  });

  it('converts a clustered sample-size bound to observations first', async () => {
    familyStore.active = 'mixed';
    const cfg = familyStore.byFamily.mixed;
    cfg.cluster!.dimKind = 'n_clusters';
    cfg.cluster!.nClusters = 20;
    cfg.findPower.n = 300;
    cfg.findSampleSize.unit = 'cluster_size';
    cfg.findSampleSize.from = 2; // 2 per cluster × 20 clusters = 40 observations
    render(PredictorAdvancedDialog, { props: { variable: factorRow(), open: true } });
    const msgs = await warnings();
    expect(msgs[0]).toContain('~13 obs in [1] at the low end of the sample-size search');
  });
});
