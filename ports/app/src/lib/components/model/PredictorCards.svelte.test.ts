// Tests for PredictorCards.svelte — upload-driven type-sync $effect (#1)
// and unlock-on-clear ($lockedNames empties when csvData → null).
//
// Plan requirement I1: type-sync forces kind after upload, covering factor,
// binary, and continuous (continuous leaves sub-type untouched).
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import type { CsvData, UploadMode } from '$lib/domain/app-spec';

// ---------------------------------------------------------------------------
// Store mocks — must be hoisted before imports that transitively import them.
// ---------------------------------------------------------------------------

// uploadStore mock: plain mutable object; PredictorCards reads uploadStore.csvData
// inside $effect, which runs on mount. Set csvData before render to have the
// initial effect see the upload data.
const uploadStoreMock = vi.hoisted<{ csvData: CsvData | null; mode: UploadMode; clear: () => void }>(
    () => ({
        csvData: null,
        mode: 'partial',
        clear() { this.csvData = null; },
    }),
);

vi.mock('$lib/stores/upload.svelte', () => ({
    uploadStore: uploadStoreMock,
}));

// parsedFormulaStore stub: synchronous, covers the formula shapes used here.
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

// Engine API: getEffectsFromData is used by the "Fit from data" button only.
// Default returns an empty EffectsFromData preview (no effects, no ICC/baseline);
// the preview/Apply tests override the resolved value per-case.
vi.mock('$lib/api/engine', () => ({
    getEffectsFromData: vi.fn(async () => ({ effects: [], cluster_icc: null, baseline_probability: null })),
    parseFormula: vi.fn(async () => null),
    runPowerAnalysis: vi.fn(async () => null),
    runSampleSizeSearch: vi.fn(async () => null),
}));

import PredictorCards from './PredictorCards.svelte';
import { familyStore } from '$lib/stores/family.svelte';
import { getEffectsFromData } from '$lib/api/engine';

const mockFit = getEffectsFromData as unknown as ReturnType<typeof vi.fn>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function flushEffects() {
    // Two tick rounds: first flushes $derived; second flushes $effect writes.
    await tick();
    await tick();
}

function makeCsvData(cols: CsvData['columns']): CsvData {
    return { mode: 'partial', n_rows: cols[0]?.values.length ?? 0, columns: cols };
}

// ---------------------------------------------------------------------------
// Test #1 — type-sync $effect forces kind after upload
// ---------------------------------------------------------------------------

describe('PredictorCards — upload-driven type-sync', () => {
    beforeEach(() => {
        familyStore.resetAll();
        uploadStoreMock.csvData = null;
    });

    it('forces kind to "factor" for a predictor matched to an uploaded factor column', async () => {
        // Seed the regression config: one predictor initially declared as continuous.
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ cyl';
        cfg.variables = [{ name: 'cyl', kind: 'continuous' }];
        cfg.effects = [{ name: 'cyl', value: 0.2 }];

        // Upload CSV with cyl as a 3-level factor (codes 0/1/2 → labels '4'/'6'/'8').
        uploadStoreMock.csvData = makeCsvData([
            {
                name: 'cyl',
                col_type: 'factor',
                values: [0, 1, 2, 0, 1, 2],
                labels: ['4', '6', '8'],
            },
        ]);

        render(PredictorCards);
        await flushEffects();

        // The $effect must have forced kind to 'factor'.
        const cyl = familyStore.byFamily.regression.variables.find((v) => v.name === 'cyl');
        expect(cyl?.kind).toBe('factor');
        expect(cyl?.nLevels).toBe(3);
        expect(cyl?.levels).toEqual(['4', '6', '8']);
        // levelProportions: each label appears twice in 6 rows → 1/3 each.
        expect(cyl?.levelProportions).toHaveLength(3);
        cyl?.levelProportions?.forEach((p) => {
            expect(p).toBeCloseTo(1 / 3, 5);
        });
    });

    it('forces kind to "binary" for a predictor matched to an uploaded binary column', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ treated';
        cfg.variables = [{ name: 'treated', kind: 'continuous' }];
        cfg.effects = [{ name: 'treated', value: 0.3 }];

        // 4 rows: three 1s, one 0 → binaryProportion = 0.75.
        uploadStoreMock.csvData = makeCsvData([
            { name: 'treated', col_type: 'binary', values: [1, 1, 1, 0], labels: [] },
        ]);

        render(PredictorCards);
        await flushEffects();

        const v = familyStore.byFamily.regression.variables.find((v) => v.name === 'treated');
        expect(v?.kind).toBe('binary');
        expect(v?.binaryProportion).toBeCloseTo(0.75, 5);
    });

    it('forces kind to "continuous" for a continuous upload but leaves a pre-declared distribution untouched', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ age';
        // Start with kind:'binary' to verify it gets overwritten to 'continuous'.
        // Also set a non-default binaryProportion to confirm the sub-type field
        // is preserved (continuous path only writes kind, nothing else).
        cfg.variables = [{ name: 'age', kind: 'binary', binaryProportion: 0.4 }];
        cfg.effects = [{ name: 'age', value: 0.25 }];

        uploadStoreMock.csvData = makeCsvData([
            { name: 'age', col_type: 'continuous', values: [20, 25, 30, 35, 40], labels: [] },
        ]);

        render(PredictorCards);
        await flushEffects();

        const v = familyStore.byFamily.regression.variables.find((v) => v.name === 'age');
        // Kind is forced to continuous.
        expect(v?.kind).toBe('continuous');
        // binaryProportion is a sub-type field; the continuous branch in PredictorCards
        // only writes v.kind — it does not clear other fields. This confirms the
        // "no-op on declared distribution" invariant.
        expect(v?.binaryProportion).toBe(0.4);
    });
});

// ---------------------------------------------------------------------------
// Test #2 — "Fit from data" gate covers OLS, GLM (binary) and MLE (mixed)
// ---------------------------------------------------------------------------
// The engine's get_effects_from_data dispatches OLS/GLM/MLE by spec family, so the
// button must show for every formula family with data loaded — not OLS-only.

describe('PredictorCards — Fit from data gate', () => {
    beforeEach(() => {
        familyStore.resetAll();
        uploadStoreMock.csvData = null;
    });

    function uploadOneContinuous(name: string) {
        uploadStoreMock.csvData = makeCsvData([
            { name, col_type: 'continuous', values: [1, 2, 3, 4], labels: [] },
        ]);
    }

    it('shows the button for regression/continuous (OLS) with data loaded', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ x';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.3 }];
        familyStore.regressionOutcome = 'continuous';
        uploadOneContinuous('x');

        const { queryByText } = render(PredictorCards);
        await flushEffects();
        expect(queryByText('Fit from data')).not.toBeNull();
    });

    it('shows the button for regression/binary (GLM) with data loaded', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ x';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.3 }];
        familyStore.regressionOutcome = 'binary';
        uploadOneContinuous('x');

        const { queryByText } = render(PredictorCards);
        await flushEffects();
        expect(queryByText('Fit from data')).not.toBeNull();
    });

    it('shows the button for the mixed family (MLE) with data loaded', async () => {
        familyStore.active = 'mixed';
        uploadOneContinuous('x');

        const { queryByText } = render(PredictorCards);
        await flushEffects();
        expect(queryByText('Fit from data')).not.toBeNull();
    });

    it('hides the button when no data is loaded', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ x';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.3 }];
        uploadStoreMock.csvData = null;

        const { queryByText } = render(PredictorCards);
        await flushEffects();
        expect(queryByText('Fit from data')).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// Test #3 — fit produces a preview; Apply (not the fit) writes into cfg
// ---------------------------------------------------------------------------
// The fit is a preview, not a write: it lists each fitted name=value (plus an
// ICC line for mixed and a baseline line for binary outcomes), and leaves cfg
// untouched until the explicit Apply button commits effects/ICC/baseline.

describe('PredictorCards — fit preview + Apply', () => {
    beforeEach(() => {
        familyStore.resetAll();
        uploadStoreMock.csvData = null;
        mockFit.mockReset();
    });

    function uploadOneContinuous(name: string) {
        uploadStoreMock.csvData = makeCsvData([
            { name, col_type: 'continuous', values: [1, 2, 3, 4], labels: [] },
        ]);
    }

    it('mixed: preview shows name=value + ICC, and only Apply writes effects + cluster.icc', async () => {
        familyStore.active = 'mixed';
        const cfg = familyStore.byFamily.mixed;
        cfg.formula = 'y ~ x + (1|g)';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.1 }];
        const iccBefore = cfg.cluster!.icc;
        uploadOneContinuous('x');
        mockFit.mockResolvedValue({
            effects: [{ name: 'x', value: 0.5 }],
            cluster_icc: 0.27,
            baseline_probability: null,
        });

        const { getByText, queryByText } = render(PredictorCards);
        await flushEffects();

        await fireEvent.click(getByText('Fit from data'));
        await flushEffects();

        // Preview is shown; config is NOT yet mutated.
        expect(queryByText('x = 0.5000')).not.toBeNull();
        expect(queryByText(/Estimated ICC: 0\.2700/)).not.toBeNull();
        expect(queryByText(/Baseline probability/)).toBeNull();
        expect(familyStore.byFamily.mixed.effects.find((e) => e.name === 'x')?.value).toBe(0.1);
        expect(familyStore.byFamily.mixed.cluster!.icc).toBe(iccBefore);

        // Apply commits the preview into cfg and clears it.
        await fireEvent.click(getByText('Apply'));
        await flushEffects();
        expect(familyStore.byFamily.mixed.effects.find((e) => e.name === 'x')?.value).toBe(0.5);
        expect(familyStore.byFamily.mixed.cluster!.icc).toBe(0.27);
        expect(queryByText('Apply')).toBeNull();
    });

    it('regression/binary: preview shows a baseline line; Apply writes cfg.baselineProbability', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ x';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.1 }];
        familyStore.regressionOutcome = 'binary';
        const baselineBefore = cfg.baselineProbability;
        uploadOneContinuous('x');
        mockFit.mockResolvedValue({
            effects: [{ name: 'x', value: 1.3863 }],
            cluster_icc: null,
            baseline_probability: 0.5,
        });

        const { getByText, queryByText } = render(PredictorCards);
        await flushEffects();

        await fireEvent.click(getByText('Fit from data'));
        await flushEffects();
        expect(queryByText(/Baseline probability: 0\.5000/)).not.toBeNull();
        expect(queryByText(/Estimated ICC/)).toBeNull();
        expect(familyStore.byFamily.regression.baselineProbability).toBe(baselineBefore);

        await fireEvent.click(getByText('Apply'));
        await flushEffects();
        expect(familyStore.byFamily.regression.baselineProbability).toBe(0.5);
    });

    it('regression/continuous (OLS): preview shows neither ICC nor baseline; Apply writes effects only', async () => {
        const cfg = familyStore.byFamily.regression;
        cfg.formula = 'y ~ x';
        cfg.variables = [{ name: 'x', kind: 'continuous' }];
        cfg.effects = [{ name: 'x', value: 0.1 }];
        familyStore.regressionOutcome = 'continuous';
        uploadOneContinuous('x');
        mockFit.mockResolvedValue({
            effects: [{ name: 'x', value: 0.9 }],
            cluster_icc: null,
            baseline_probability: null,
        });

        const { getByText, queryByText } = render(PredictorCards);
        await flushEffects();

        await fireEvent.click(getByText('Fit from data'));
        await flushEffects();
        expect(queryByText('x = 0.9000')).not.toBeNull();
        expect(queryByText(/Estimated ICC/)).toBeNull();
        expect(queryByText(/Baseline probability/)).toBeNull();

        await fireEvent.click(getByText('Apply'));
        await flushEffects();
        expect(familyStore.byFamily.regression.effects.find((e) => e.name === 'x')?.value).toBe(0.9);
    });
});
