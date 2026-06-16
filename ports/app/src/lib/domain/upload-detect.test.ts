// Vitest tests for upload-detect.ts; asserts detectColumnTypes verdicts and factor levels
// against the shared golden fixture — identical to Python's test_upload_type_detection.py
// and R's test-upload-detection.R.
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  detectColumnTypes,
  parseCsvText,
  uploadedColumnByName,
  validateUploadRowCount,
  valueToLabel,
} from './upload-detect';
import type { CsvData } from './app-spec';
import fixture from './__fixtures__/upload_type_detection.json';

// ---------------------------------------------------------------------------
// Golden fixture shape helpers
// ---------------------------------------------------------------------------

interface GoldenColumn {
    name: string;
    expect: 'continuous' | 'binary' | 'factor';
    expect_levels?: string[];
    // columns with explicit values array
    values?: (number | string)[];
    // columns described by a sample key (we generate them here)
    sample?: string;
}

const FIXTURE_N_ROWS: number = (fixture as { n_rows: number }).n_rows;
const MAX_K: number = (fixture as { max_factor_k_soft: number }).max_factor_k_soft;
const MAX_RATIO: number = (fixture as { max_factor_ratio: number }).max_factor_ratio;
const COLUMNS: GoldenColumn[] = (fixture as { columns: GoldenColumn[] }).columns;

// ---------------------------------------------------------------------------
// Build test column arrays from fixture column descriptors
// ---------------------------------------------------------------------------

function buildColumn(col: GoldenColumn, nRows: number): string[] {
    if (col.values !== undefined) {
        // Tile the values array up to nRows
        const out: string[] = [];
        for (let i = 0; i < nRows; i++) {
            out.push(String(col.values[i % col.values.length]!));
        }
        return out;
    }
    if (col.sample === 'continuous_60_distinct') {
        // 60 distinct consecutive integers — clearly continuous
        return Array.from({ length: nRows }, (_, i) => String(i + 1));
    }
    if (col.sample === '20_distinct_over_60_rows_ratio_3') {
        // 20 distinct values over 60 rows → ratio = 3 < 15 → continuous
        return Array.from({ length: nRows }, (_, i) => String(i % 20));
    }
    throw new Error(`Unknown sample key: ${col.sample}`);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('valueToLabel', () => {
    it('renders integer-valued floats without decimal', () => {
        expect(valueToLabel(4.0)).toBe('4');
        expect(valueToLabel(6.0)).toBe('6');
        expect(valueToLabel(8.0)).toBe('8');
    });
    it('renders non-integer floats with decimal', () => {
        expect(valueToLabel(3.5)).toBe('3.5');
    });
    it('passes through strings', () => {
        expect(valueToLabel('Japan')).toBe('Japan');
        expect(valueToLabel('USA')).toBe('USA');
    });
});

describe('parseCsvText — quoted fields', () => {
    it('does not emit a phantom trailing field when a row ends with a quoted field', () => {
        // R-exported CSVs quote every header name, so the header ends with a closing quote.
        const { header } = parseCsvText('"a","b","c"\n1,2,3');
        expect(header).toEqual(['a', 'b', 'c']);
    });

    it('parses an mtcars-style header without producing a duplicate empty column name', () => {
        // mtcars: empty-named row-names column up front + quoted trailing "carb".
        // A phantom trailing "" would collide with the leading "" → duplicate keyed-each key.
        const { header, columns } = parseCsvText('"","mpg","carb"\n"Mazda RX4",21,4');
        expect(header).toEqual(['', 'mpg', 'carb']);
        expect(columns).toHaveLength(3);
        expect(columns[0]).toEqual(['Mazda RX4']);
    });

    it('still parses an unquoted trailing field (regression guard)', () => {
        const { header } = parseCsvText('a,b,c\n1,2,3');
        expect(header).toEqual(['a', 'b', 'c']);
    });
});

// ---------------------------------------------------------------------------
// uploadedColumnByName
// ---------------------------------------------------------------------------

describe('uploadedColumnByName', () => {
    it('returns empty Map for null', () => {
        expect(uploadedColumnByName(null).size).toBe(0);
    });

    it('returns empty Map for undefined', () => {
        expect(uploadedColumnByName(undefined).size).toBe(0);
    });

    it('maps column names to UploadColumn objects', () => {
        const csvData: CsvData = {
            mode: 'partial',
            n_rows: 3,
            columns: [
                { name: 'age', col_type: 'continuous', values: [20, 30, 40], labels: [] },
                { name: 'group', col_type: 'factor', values: [0, 1, 2], labels: ['A', 'B', 'C'] },
                { name: 'treated', col_type: 'binary', values: [0, 1, 1], labels: [] },
            ],
        };
        const m = uploadedColumnByName(csvData);
        expect(m.size).toBe(3);
        expect(m.get('age')?.col_type).toBe('continuous');
        expect(m.get('group')?.col_type).toBe('factor');
        expect(m.get('group')?.labels).toEqual(['A', 'B', 'C']);
        expect(m.get('treated')?.col_type).toBe('binary');
        expect(m.get('treated')?.values).toEqual([0, 1, 1]);
    });

    it('does not include columns for unrecognised names', () => {
        const csvData: CsvData = {
            mode: 'partial',
            n_rows: 1,
            columns: [{ name: 'x', col_type: 'continuous', values: [1], labels: [] }],
        };
        expect(uploadedColumnByName(csvData).has('y')).toBe(false);
    });
});

describe('detectColumnTypes — golden fixture', () => {
    const columnArrays = COLUMNS.map((col) => buildColumn(col, FIXTURE_N_ROWS));
    const names = COLUMNS.map((c) => c.name);
    const { types, labelsList } = detectColumnTypes(columnArrays, names, FIXTURE_N_ROWS, MAX_K, MAX_RATIO);

    for (let i = 0; i < COLUMNS.length; i++) {
        const col = COLUMNS[i]!;
        it(`column "${col.name}" — type = ${col.expect}`, () => {
            expect(types[i]).toBe(col.expect);
        });
        if (col.expect_levels !== undefined) {
            it(`column "${col.name}" — factor levels = ${JSON.stringify(col.expect_levels)}`, () => {
                expect(labelsList[i]).toEqual(col.expect_levels);
            });
        }
    }
});

// ---------------------------------------------------------------------------
// validateUploadRowCount — shared host-side row-count gate (min for both shells;
// upper cap branches by VITE_TARGET: native max_rows vs browser max_rows_wasm)
// ---------------------------------------------------------------------------

describe('validateUploadRowCount', () => {
    afterEach(() => {
        vi.unstubAllEnvs();
    });

    it('rejects below the shared minimum (20) on both shells', () => {
        expect(validateUploadRowCount(19)).toMatch(/at least 20/);
        expect(validateUploadRowCount(20)).toBeNull();
    });

    it('native target: accepts up to 1,000,000 and rejects 1,000,001', () => {
        // Default (non-wasm) build target is native.
        expect(validateUploadRowCount(10_001)).toBeNull(); // fine on native — well under 1M
        expect(validateUploadRowCount(1_000_000)).toBeNull();
        expect(validateUploadRowCount(1_000_001)).toMatch(/maximum is 1000000/);
    });

    it('wasm target: accepts up to 10,000 and rejects 10,001', () => {
        vi.stubEnv('VITE_TARGET', 'wasm');
        expect(validateUploadRowCount(10_000)).toBeNull();
        expect(validateUploadRowCount(10_001)).toMatch(/browser limit is 10000/);
        expect(validateUploadRowCount(19)).toMatch(/at least 20/); // min still applies on wasm
    });
});
