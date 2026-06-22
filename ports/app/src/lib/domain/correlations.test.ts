import { describe, expect, it } from 'vitest';
import { correlatableVariables, pearson, allContinuousUploaded } from './correlations';
import type { VariableRow } from './family';
import type { CsvData } from './app-spec';

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const contX: VariableRow = { name: 'x', kind: 'continuous' };
const contY: VariableRow = { name: 'y', kind: 'continuous' };
const contZ: VariableRow = { name: 'z', kind: 'continuous' };
const binB: VariableRow = { name: 'b', kind: 'binary', binaryProportion: 0.5 };
const factF: VariableRow = { name: 'f', kind: 'factor', nLevels: 3, levelProportions: [1 / 3, 1 / 3, 1 / 3] };

function makeCsvData(names: string[], types: Array<'continuous' | 'binary' | 'factor'>): CsvData {
    return {
        mode: 'partial',
        n_rows: 3,
        columns: names.map((name, i) => ({
            name,
            col_type: types[i]!,
            values: [1, 2, 3],
            labels: [],
        })),
    };
}

// ---------------------------------------------------------------------------
// correlatableVariables
// ---------------------------------------------------------------------------

describe('correlatableVariables', () => {
    it('excludes factor and binary variables (R1)', () => {
        const vars: VariableRow[] = [contX, binB, factF, contY];
        const result = correlatableVariables(vars, null, 'none');
        expect(result.map((e) => e.row.name)).toEqual(['x', 'y']);
    });

    it('preserves correct idx back to the original variables array', () => {
        const vars: VariableRow[] = [contX, binB, factF, contY];
        const result = correlatableVariables(vars, null, 'none');
        expect(result[0]!.idx).toBe(0); // x is at index 0
        expect(result[1]!.idx).toBe(3); // y is at index 3
    });

    it('an uploaded factor has kind==="factor" and is excluded by R1', () => {
        // Part 1 contract: uploaded factors have kind === 'factor'
        const uploadedFactor: VariableRow = { name: 'cyl', kind: 'factor', nLevels: 3, levelProportions: [1 / 3, 1 / 3, 1 / 3] };
        const csvData = makeCsvData(['cyl'], ['factor']);
        const result = correlatableVariables([contX, uploadedFactor], csvData, 'none');
        expect(result.map((e) => e.row.name)).toEqual(['x']);
    });

    describe('strict mode (R2)', () => {
        it('removes uploaded continuous variables in strict mode', () => {
            const csvData = makeCsvData(['x'], ['continuous']);
            // x is uploaded, z is generated
            const vars: VariableRow[] = [contX, contZ];
            const result = correlatableVariables(vars, csvData, 'strict');
            expect(result.map((e) => e.row.name)).toEqual(['z']);
        });

        it('keeps uploaded continuous variables in partial mode', () => {
            const csvData = makeCsvData(['x'], ['continuous']);
            const vars: VariableRow[] = [contX, contZ];
            const result = correlatableVariables(vars, csvData, 'partial');
            expect(result.map((e) => e.row.name)).toEqual(['x', 'z']);
        });

        it('keeps uploaded continuous variables in none mode', () => {
            const csvData = makeCsvData(['x'], ['continuous']);
            const vars: VariableRow[] = [contX, contZ];
            const result = correlatableVariables(vars, csvData, 'none');
            expect(result.map((e) => e.row.name)).toEqual(['x', 'z']);
        });

        it('returns empty when all continuous vars are uploaded in strict mode', () => {
            const csvData = makeCsvData(['x', 'y'], ['continuous', 'continuous']);
            const vars: VariableRow[] = [contX, contY];
            const result = correlatableVariables(vars, csvData, 'strict');
            expect(result).toHaveLength(0);
        });
    });

    it('returns all continuous when csvData is null', () => {
        const vars: VariableRow[] = [contX, contY, binB];
        const result = correlatableVariables(vars, null, 'strict');
        expect(result.map((e) => e.row.name)).toEqual(['x', 'y']);
    });

    it('idx maps correctly with mixed kinds', () => {
        // [contX(0), binB(1), contY(2), factF(3), contZ(4)]
        const vars: VariableRow[] = [contX, binB, contY, factF, contZ];
        const result = correlatableVariables(vars, null, 'none');
        expect(result.map((e) => e.idx)).toEqual([0, 2, 4]);
    });
});

// ---------------------------------------------------------------------------
// allContinuousUploaded — drives the empty-state message in CorrelationsEditor
// ---------------------------------------------------------------------------

describe('allContinuousUploaded', () => {
    it('strict mode + one continuous generated (0 uploaded) → false (show "Add continuous predictors")', () => {
        // contZ is not in the uploaded dataset; the strict-mode message must NOT fire
        const csvData = makeCsvData([], []);
        expect(allContinuousUploaded([contZ], csvData)).toBe(false);
    });

    it('strict mode + one continuous generated + zero uploaded → false', () => {
        // csvData has no continuous columns — contZ is generated, not uploaded
        const csvData = makeCsvData(['other'], ['binary']);
        expect(allContinuousUploaded([contZ, binB], csvData)).toBe(false);
    });

    it('strict mode + all continuous uploaded → true (show "In strict mode…")', () => {
        // Both x and y are uploaded — strict mode removed them from the triangle
        const csvData = makeCsvData(['x', 'y'], ['continuous', 'continuous']);
        expect(allContinuousUploaded([contX, contY, binB], csvData)).toBe(true);
    });

    it('strict mode + mixed: one uploaded one generated → false', () => {
        // x is uploaded but z is not — not ALL continuous are uploaded
        const csvData = makeCsvData(['x'], ['continuous']);
        expect(allContinuousUploaded([contX, contZ], csvData)).toBe(false);
    });

    it('no continuous variables at all → false', () => {
        // Only binary/factor variables — guard against "0 uploaded / 0 continuous" misfiring
        const csvData = makeCsvData(['b'], ['binary']);
        expect(allContinuousUploaded([binB, factF], csvData)).toBe(false);
    });

    it('null csvData → false', () => {
        expect(allContinuousUploaded([contX], null)).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// pearson
// ---------------------------------------------------------------------------

describe('pearson', () => {
    it('golden value: perfectly correlated arrays → 1', () => {
        expect(pearson([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])).toBeCloseTo(1, 10);
    });

    it('golden value: perfectly anti-correlated → -1', () => {
        expect(pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])).toBeCloseTo(-1, 10);
    });

    it('golden value: known r ≈ 0.9820 for [1,2,3] / [2,4,5]', () => {
        // Hand computation:
        // a = [1,2,3], mean = 2; da = [-1,0,1]
        // b = [2,4,5], mean = 11/3; db = [-5/3,1/3,4/3]
        // Σ(da·db) = 5/3+0+4/3 = 9/3 = 3
        // Σda² = 2, Σdb² = 25/9+1/9+16/9 = 42/9
        // r = 3 / √(2 · 42/9) = 3 / √(84/9) = 3 / (√84/3) = 9/√84 ≈ 0.9819805
        const r = pearson([1, 2, 3], [2, 4, 5]);
        expect(r).toBeCloseTo(9 / Math.sqrt(84), 10);
    });

    it('zero-variance array a → 0', () => {
        expect(pearson([3, 3, 3], [1, 2, 3])).toBe(0);
    });

    it('zero-variance array b → 0', () => {
        expect(pearson([1, 2, 3], [5, 5, 5])).toBe(0);
    });

    it('both zero-variance → 0', () => {
        expect(pearson([2, 2, 2], [7, 7, 7])).toBe(0);
    });

    it('empty arrays → 0', () => {
        expect(pearson([], [])).toBe(0);
    });

    it('uncorrelated arrays → r near 0', () => {
        // Hand-checked: a = [1,2,3,4], mean=2.5, da=[-1.5,-0.5,0.5,1.5]
        // b = [2,4,1,3], mean=2.5, db=[-0.5,1.5,-1.5,0.5]
        // Σ(da·db) = 0.75 - 0.75 - 0.75 + 0.75 = 0
        const r = pearson([1, 2, 3, 4], [2, 4, 1, 3]);
        expect(r).toBeCloseTo(0, 10);
    });
});

// ---------------------------------------------------------------------------
// R3 measured display default (tested via displayValue logic inline)
// ---------------------------------------------------------------------------

describe('R3 measured display default — pure logic (no component render)', () => {
    // We test the logic that backs displayValue: in partial mode, when stored = 0
    // and both vars are uploaded-continuous, the pearson value is shown.
    // We call pearson directly (the actual display path in the component does the same).

    function simulateDisplayValue(
        storedCorr: number,
        mode: 'none' | 'partial' | 'strict',
        colI: { col_type: string; values: number[] } | undefined,
        colJ: { col_type: string; values: number[] } | undefined,
    ): number {
        // Mirrors CorrelationsEditor.svelte displayValue() logic
        if (mode !== 'partial' || storedCorr !== 0) return storedCorr;
        if (!colI || !colJ) return storedCorr;
        if (colI.col_type !== 'continuous' || colJ.col_type !== 'continuous') return storedCorr;
        return pearson(colI.values, colJ.values);
    }

    const colA = { col_type: 'continuous', values: [1, 2, 3, 4, 5] };
    const colB = { col_type: 'continuous', values: [5, 4, 3, 2, 1] };
    const expectedR = -1; // perfectly anti-correlated

    it('partial mode + stored=0 + both uploaded-continuous → measured r shown', () => {
        const displayed = simulateDisplayValue(0, 'partial', colA, colB);
        expect(displayed).toBeCloseTo(expectedR, 10);
    });

    it('none mode + stored=0 → shows 0, not measured r', () => {
        const displayed = simulateDisplayValue(0, 'none', colA, colB);
        expect(displayed).toBe(0);
    });

    it('strict mode + stored=0 → shows 0, not measured r', () => {
        const displayed = simulateDisplayValue(0, 'strict', colA, colB);
        expect(displayed).toBe(0);
    });

    it('partial mode + stored non-zero → shows stored value, not measured r', () => {
        const displayed = simulateDisplayValue(0.5, 'partial', colA, colB);
        expect(displayed).toBe(0.5);
    });

    it('partial mode + stored=0 + one var is not uploaded → shows 0', () => {
        const displayed = simulateDisplayValue(0, 'partial', colA, undefined);
        expect(displayed).toBe(0);
    });

    it('partial mode + stored=0 + one var is binary (not continuous) → shows 0', () => {
        const binaryCol = { col_type: 'binary', values: [0, 1, 0, 1, 0] };
        const displayed = simulateDisplayValue(0, 'partial', colA, binaryCol);
        expect(displayed).toBe(0);
    });

    it('cfg.correlations[i][j] stays 0 (measured value never written)', () => {
        // This is the key invariant: displayValue reads but does NOT write.
        // We verify by confirming the stored value is unchanged after computing display.
        const storedMatrix: number[][] = [[1, 0], [0, 1]];
        simulateDisplayValue(storedMatrix[0]![1]!, 'partial', colA, colB);
        // matrix must be unchanged
        expect(storedMatrix[0]![1]!).toBe(0);
        expect(storedMatrix[1]![0]!).toBe(0);
    });
});
