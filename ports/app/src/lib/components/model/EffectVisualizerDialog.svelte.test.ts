import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { CsvData, UploadMode } from '$lib/domain/app-spec';

// Deterministic parse without the engine. The shared stub provides both `get`
// (used by familyConfigToAppSpec) and `getStable` (used by the rail + the fit
// guard's outcome lookup), so the fit flow can build a real spec. Spread the
// real module so its sibling exports (toWireParsed/toClusterTerms, consumed by
// familyConfigToAppSpec) survive the mock.
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

// uploadStore mock: plain mutable object so each test seeds csvData (drives
// canFitFromData + the outcome guard). Set before render.
const uploadStoreMock = vi.hoisted<{
  csvData: CsvData | null;
  mode: UploadMode;
  clear: () => void;
}>(() => ({
  csvData: null,
  mode: 'partial',
  clear() {
    this.csvData = null;
  },
}));
vi.mock('$lib/stores/upload.svelte', () => ({ uploadStore: uploadStoreMock }));

// Engine API: the fit-flow tests override getEffectsFromData's resolved value.
vi.mock('$lib/api/engine', () => ({
  getEffectsFromData: vi.fn(async () => ({
    effects: [],
    cluster_icc: null,
    baseline_probability: null,
  })),
  parseFormula: vi.fn(async () => null),
}));

import { getEffectsFromData } from '$lib/api/engine';
import { familyStore } from '$lib/stores/family.svelte';
import EffectVisualizerDialog from './EffectVisualizerDialog.svelte';

const mockFit = getEffectsFromData as unknown as ReturnType<typeof vi.fn>;

function makeCsvData(cols: CsvData['columns']): CsvData {
  return { mode: 'partial', n_rows: cols[0]?.values.length ?? 0, columns: cols };
}

describe('EffectVisualizerDialog', () => {
  beforeEach(() => {
    familyStore.resetAll();
    uploadStoreMock.csvData = null;
  });

  it('binds EffectControls to the live cfg.effects entry so edits propagate', async () => {
    familyStore.active = 'regression';
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y ~ a';
    cfg.variables = [{ name: 'a', kind: 'continuous' }];
    cfg.effects = [{ name: 'a', value: 0 }];

    render(EffectVisualizerDialog, { open: true, onOpenChange: vi.fn() });
    await tick();

    // EffectControls renders a NumberInput bound to effect.value; set it and assert
    // the SAME proxy entry on cfg.effects updated. DialogContent portals to
    // document.body, so query the whole document, not the mount container.
    // NumberInput commits on `change`, not `input` (number-input.svelte onchange).
    const input = document.body.querySelector('input[type="number"]') as HTMLInputElement;
    expect(input).toBeTruthy();
    await fireEvent.change(input, { target: { value: '0.4' } });
    await tick();
    expect(cfg.effects.find((e) => e.name === 'a')?.value).toBeCloseTo(0.4);
  });

  it('rails one entry per predictor; a factor shows a locked reference and one control per non-reference level', async () => {
    familyStore.active = 'regression';
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y ~ g';
    cfg.variables = [
      { name: 'g', kind: 'factor', levels: ['a', 'b', 'c'], referenceLevel: 'a', nLevels: 3 },
    ];
    cfg.effects = [
      { name: 'g[b]', value: 0.4 },
      { name: 'g[c]', value: 0.8 },
    ];

    render(EffectVisualizerDialog, { open: true, onOpenChange: vi.fn() });
    await tick();

    // One rail entry for the whole factor (not one per dummy).
    const railButtons = document.body.querySelectorAll('nav[aria-label="effect terms"] button');
    expect(railButtons).toHaveLength(1);

    // Reference is shown as a locked "reference" row (no control).
    expect(document.body.textContent).toContain('reference');

    // One EffectControls per non-reference level => two number inputs.
    const inputs = document.body.querySelectorAll('input[type="number"]');
    expect(inputs).toHaveLength(2);

    // Editing the first level's control writes the live cfg.effects[g[b]] entry.
    await fireEvent.change(inputs[0] as HTMLInputElement, { target: { value: '0.5' } });
    await tick();
    expect(cfg.effects.find((e) => e.name === 'g[b]')?.value).toBeCloseTo(0.5);
  });
});

// ---------------------------------------------------------------------------
// Get effects from data — relocated from PredictorCards. The engine's
// get_effects_from_data dispatches OLS/GLM/MLE by spec family, so the button
// shows for every formula family with data loaded; the fit is a preview that
// only Apply commits; and the outcome must be a real uploaded column.
// ---------------------------------------------------------------------------

describe('EffectVisualizerDialog — get effects from data', () => {
  beforeEach(() => {
    familyStore.resetAll();
    uploadStoreMock.csvData = null;
    mockFit.mockReset();
  });

  // Outcome 'y' is on the LHS, so the upload must carry a 'y' column for the
  // guard to pass; 'x' is the predictor.
  function uploadYandX(yType: 'continuous' | 'binary' = 'continuous') {
    uploadStoreMock.csvData = makeCsvData([
      { name: 'y', col_type: yType, values: [0, 1, 0, 1], labels: [] },
      { name: 'x', col_type: 'continuous', values: [1, 2, 3, 4], labels: [] },
    ]);
  }

  it('hides the button when no data is loaded', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y = x';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.3 }];

    const { queryByText } = render(EffectVisualizerDialog, { open: true, onOpenChange: vi.fn() });
    await tick();
    expect(queryByText('Get effects from data')).toBeNull();
  });

  it('shows the button for the mixed family with data loaded', async () => {
    familyStore.active = 'mixed';
    const cfg = familyStore.byFamily.mixed;
    cfg.formula = 'y = x + (1|g)';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.1 }];
    uploadYandX();

    const { queryByText } = render(EffectVisualizerDialog, { open: true, onOpenChange: vi.fn() });
    await tick();
    expect(queryByText('Get effects from data')).not.toBeNull();
  });

  it('guards: outcome not in the upload shows an inline error and never calls the engine', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y = x';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.3 }];
    // Upload carries only the predictor 'x' — the outcome 'y' is absent.
    uploadStoreMock.csvData = makeCsvData([
      { name: 'x', col_type: 'continuous', values: [1, 2, 3, 4], labels: [] },
    ]);

    const { getByText, queryByText } = render(EffectVisualizerDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getByText('Get effects from data'));
    await tick();
    expect(queryByText(/isn't in your uploaded data/)).not.toBeNull();
    expect(mockFit).not.toHaveBeenCalled();
  });

  it('mixed: preview shows name=value + ICC, and only Apply writes effects + cluster.icc', async () => {
    familyStore.active = 'mixed';
    const cfg = familyStore.byFamily.mixed;
    cfg.formula = 'y = x + (1|g)';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.1 }];
    const iccBefore = cfg.cluster!.icc;
    uploadYandX();
    mockFit.mockResolvedValue({
      effects: [{ name: 'x', value: 0.5 }],
      cluster_icc: 0.27,
      baseline_probability: null,
    });

    const { getByText, queryByText } = render(EffectVisualizerDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getByText('Get effects from data'));
    await tick();
    await tick();

    // Preview is shown; config is NOT yet mutated.
    expect(queryByText('x = 0.5000')).not.toBeNull();
    expect(queryByText(/Estimated ICC: 0\.2700/)).not.toBeNull();
    expect(queryByText(/Baseline probability/)).toBeNull();
    expect(familyStore.byFamily.mixed.effects.find((e) => e.name === 'x')?.value).toBe(0.1);
    expect(familyStore.byFamily.mixed.cluster!.icc).toBe(iccBefore);

    // Apply commits the preview into cfg and clears it.
    await fireEvent.click(getByText('Apply'));
    await tick();
    expect(familyStore.byFamily.mixed.effects.find((e) => e.name === 'x')?.value).toBe(0.5);
    expect(familyStore.byFamily.mixed.cluster!.icc).toBe(0.27);
    expect(queryByText('Apply')).toBeNull();
  });

  it('regression/binary: preview shows a baseline line; Apply writes cfg.baselineProbability', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y = x';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.1 }];
    familyStore.regressionOutcome = 'logit';
    const baselineBefore = cfg.baselineProbability;
    uploadYandX('binary');
    mockFit.mockResolvedValue({
      effects: [{ name: 'x', value: 1.3863 }],
      cluster_icc: null,
      baseline_probability: 0.5,
    });

    const { getByText, queryByText } = render(EffectVisualizerDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getByText('Get effects from data'));
    await tick();
    await tick();
    expect(queryByText(/Baseline probability: 0\.5000/)).not.toBeNull();
    expect(queryByText(/Estimated ICC/)).toBeNull();
    expect(familyStore.byFamily.regression.baselineProbability).toBe(baselineBefore);

    await fireEvent.click(getByText('Apply'));
    await tick();
    expect(familyStore.byFamily.regression.baselineProbability).toBe(0.5);
  });

  it('binary family + non-binary outcome column shows a non-blocking warning', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y = x';
    cfg.variables = [{ name: 'x', kind: 'continuous' }];
    cfg.effects = [{ name: 'x', value: 0.1 }];
    familyStore.regressionOutcome = 'logit';
    // 'y' detected as continuous under a binary family → warning.
    uploadYandX('continuous');

    const { queryByText } = render(EffectVisualizerDialog, { open: true, onOpenChange: vi.fn() });
    await tick();
    expect(queryByText(/detected as continuous, not binary/)).not.toBeNull();
  });

  it('per-variable: drops a modeled predictor absent from the upload and fits the sub-model', async () => {
    const cfg = familyStore.byFamily.regression;
    // 'z' is modeled but not uploaded — the recovery must drop it, not error.
    cfg.formula = 'y = x + z';
    cfg.variables = [
      { name: 'x', kind: 'continuous' },
      { name: 'z', kind: 'continuous' },
    ];
    cfg.effects = [
      { name: 'x', value: 0.1 },
      { name: 'z', value: 0.2 },
    ];
    uploadYandX(); // uploads y and x only
    mockFit.mockResolvedValue({
      effects: [{ name: 'x', value: 0.5 }],
      cluster_icc: null,
      baseline_probability: null,
    });

    const { getByText, queryByText } = render(EffectVisualizerDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getByText('Get effects from data'));
    await tick();
    await tick();

    expect(queryByText(/no matching uploaded column|isn't in your uploaded data/)).toBeNull();
    expect(mockFit).toHaveBeenCalledTimes(1);
    // The spec handed to the engine excludes the un-uploaded predictor 'z' from
    // predictors, var_types AND effects — a stale effect for a dropped predictor
    // trips the engine's effect-count validation ("effect count mismatch").
    const passedSpec = mockFit.mock.calls[0]![0] as {
      parsed_formula: { predictors: string[] };
      var_types: { name: string }[];
      effects: { name: string }[];
    };
    expect(passedSpec.parsed_formula.predictors).toEqual(['x']);
    expect(passedSpec.var_types.map((v) => v.name)).toEqual(['x']);
    expect(passedSpec.effects.map((e) => e.name)).toEqual(['x']);
  });

  it('errors (without calling the engine) when no predictor matches an uploaded column', async () => {
    const cfg = familyStore.byFamily.regression;
    cfg.formula = 'y = z';
    cfg.variables = [{ name: 'z', kind: 'continuous' }];
    cfg.effects = [{ name: 'z', value: 0.2 }];
    uploadYandX(); // y and x — neither is 'z'

    const { getByText, queryByText } = render(EffectVisualizerDialog, {
      open: true,
      onOpenChange: vi.fn(),
    });
    await tick();

    await fireEvent.click(getByText('Get effects from data'));
    await tick();
    expect(queryByText(/None of your predictors match/)).not.toBeNull();
    expect(mockFit).not.toHaveBeenCalled();
  });
});
