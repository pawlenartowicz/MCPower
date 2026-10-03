import { describe, expect, it } from 'vitest';
import {
    allContinuousUploaded,
    correlatableVariables,
    correlationDisplayValue,
    pearson,
} from './correlations';
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

    describe('strict mode (R2)', () => {
        it('removes uploaded continuous variables in strict mode', () => {
            const csvData = makeCsvData(['x'], ['continuous']);
            // x is uploaded, z is generated
            const vars: VariableRow[] = [contX, contZ];
            const result = correlatableVariables(vars, csvData, 'strict');
            expect(result.map((e) => e.row.name)).toEqual(['z']);
        });

        it('keeps uploaded continuous variables in partial and none modes', () => {
            const csvData = makeCsvData(['x'], ['continuous']);
            const vars: VariableRow[] = [contX, contZ];
            expect(correlatableVariables(vars, csvData, 'partial').map((e) => e.row.name)).toEqual(['x', 'z']);
            expect(correlatableVariables(vars, csvData, 'none').map((e) => e.row.name)).toEqual(['x', 'z']);
        });
    });

    it('returns all continuous when csvData is null', () => {
        const vars: VariableRow[] = [contX, contY, binB];
        const result = correlatableVariables(vars, null, 'strict');
        expect(result.map((e) => e.row.name)).toEqual(['x', 'y']);
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

    it('zero variance in either or both arrays → 0', () => {
        expect(pearson([3, 3, 3], [1, 2, 3])).toBe(0);
        expect(pearson([1, 2, 3], [5, 5, 5])).toBe(0);
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
// correlationDisplayValue — R3 measured display default
// ---------------------------------------------------------------------------

describe('correlationDisplayValue (R3 measured display default)', () => {
    // a and b are perfectly anti-correlated uploaded continuous columns; c is a
    // binary column correlated with a, so a measured r would be non-zero.
    const csvData: CsvData = {
        mode: 'partial',
        n_rows: 5,
        columns: [
            { name: 'a', col_type: 'continuous', values: [1, 2, 3, 4, 5], labels: [] },
            { name: 'b', col_type: 'continuous', values: [5, 4, 3, 2, 1], labels: [] },
            { name: 'c', col_type: 'binary', values: [0, 0, 1, 1, 1], labels: [] },
        ],
    };

    it('partial mode + stored=0 + both uploaded-continuous → measured r shown', () => {
        expect(correlationDisplayValue(0, 'partial', csvData, 'a', 'b')).toBeCloseTo(-1, 10);
    });

    it('none mode + stored=0 → shows 0, not measured r', () => {
        expect(correlationDisplayValue(0, 'none', csvData, 'a', 'b')).toBe(0);
    });

    it('strict mode + stored=0 → shows 0, not measured r', () => {
        expect(correlationDisplayValue(0, 'strict', csvData, 'a', 'b')).toBe(0);
    });

    it('partial mode + stored non-zero → shows stored value, not measured r', () => {
        expect(correlationDisplayValue(0.5, 'partial', csvData, 'a', 'b')).toBe(0.5);
    });

    it('partial mode + stored=0 + one var is not uploaded → shows 0', () => {
        expect(correlationDisplayValue(0, 'partial', csvData, 'a', 'generated')).toBe(0);
    });

    it('partial mode + stored=0 + one var is binary (not continuous) → shows 0', () => {
        expect(correlationDisplayValue(0, 'partial', csvData, 'a', 'c')).toBe(0);
        expect(correlationDisplayValue(0, 'partial', csvData, 'c', 'a')).toBe(0);
    });
});
