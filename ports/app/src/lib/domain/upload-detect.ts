// Column-type detector + typed-CsvData builder for uploaded data. The type-detection
// rules (continuous/binary/factor) mirror Python upload_data_utils.py and R upload-data.R.
// Raw parsing is handled by upload-parsers.ts (PapaParse for CSV/TSV, plus XLSX and
// SPSS .sav), which normalizes every format into the ParsedCsv shape defined here — so
// this file no longer parses CSV text itself and the app now ingests TSV/XLSX/.sav too,
// unlike Python/R which stay CSV/dataframe-only.
import type { CsvData, UploadColumn, UploadMode } from './app-spec';
export type { UploadColumn };
import { UPLOAD } from '$lib/configs/app-config';

// ---------------------------------------------------------------------------
// value_to_label — single-sourced label renderer; mirrors Python value_to_label
// ---------------------------------------------------------------------------

/** Render a raw factor level value as its canonical string label.
 * Integer-valued floats render without decimal (`4.0` → `"4"`).
 * Non-numeric values pass through as `String(v)`.
 */
export function valueToLabel(v: unknown): string {
    const n = Number(v);
    if (!Number.isNaN(n) && Number.isFinite(n)) {
        return n === Math.floor(n) ? String(Math.floor(n)) : String(n);
    }
    return String(v);
}

// ---------------------------------------------------------------------------
// ParsedCsv — the normalized parse result. Raw parsing (CSV/TSV/XLSX/.sav)
// lives in upload-parsers.ts, which produces this shape.
// ---------------------------------------------------------------------------

export interface ParsedCsv {
    header: string[];
    /** Each element is an array of string values for that column (row-major → col-major transposed). */
    columns: string[][];
    nRows: number;
}

/** Host-side upload row-count gate, shared by the Tauri and WASM shells
 * (host-owned upload frames). Both shells reject fewer than
 * `upload.min_rows`; the upper cap branches by build target — native uses
 * `upload.max_rows`, the browser uses the smaller `upload.max_rows_wasm`.
 * Returns an error message, or `null` when the row count is acceptable.
 */
export function validateUploadRowCount(nRows: number): string | null {
    if (nRows < UPLOAD.min_rows) {
        return `Upload has ${nRows} rows; at least ${UPLOAD.min_rows} are required.`;
    }
    const isWasm = import.meta.env.VITE_TARGET === 'wasm';
    const cap = isWasm ? UPLOAD.max_rows_wasm : UPLOAD.max_rows;
    if (nRows > cap) {
        return isWasm
            ? `Upload has ${nRows} rows; the browser limit is ${cap}. Reduce the dataset or use the desktop app (limit ${UPLOAD.max_rows}).`
            : `Upload has ${nRows} rows; the maximum is ${cap}.`;
    }
    return null;
}

// ---------------------------------------------------------------------------
// detectColumnTypes — mirrors Python detect_column_types + R NA fix
// ---------------------------------------------------------------------------

/** Detect types for all parsed columns.
 * Rules (mirror Python/R):
 *   - string column (any non-parseable cell) → factor
 *   - numeric, exactly 2 distinct (non-missing) → binary
 *   - numeric, n_distinct <= max_k AND n_rows/n_distinct >= max_ratio → factor
 *   - otherwise → continuous
 * Missing values (empty string / "NA" / "NaN") are excluded from distinct count.
 */
export function detectColumnTypes(
    columns: string[][],
    _names: string[],
    nRows: number,
    maxK: number,
    maxRatio: number,
): { types: Array<'continuous' | 'binary' | 'factor'>; labelsList: string[][] } {
    const types: Array<'continuous' | 'binary' | 'factor'> = [];
    const labelsList: string[][] = [];

    for (let ci = 0; ci < columns.length; ci++) {
        const col = columns[ci]!;

        // Try numeric cast — if any cell fails, it's a string column.
        let isNumeric = true;
        const floatVals: number[] = [];
        for (const cell of col) {
            if (cell === '' || cell === 'NA' || cell === 'NaN' || cell === 'nan' || cell === 'na') {
                // missing — skip for distinct count (mirror R NA fix)
                floatVals.push(NaN);
                continue;
            }
            const n = Number(cell);
            if (Number.isNaN(n)) {
                isNumeric = false;
                break;
            }
            floatVals.push(n);
        }

        if (!isNumeric) {
            // String column → factor; gather distinct string labels sorted
            const distinct = [...new Set(col.map((v) => String(v)))].sort();
            types.push('factor');
            labelsList.push(distinct);
            continue;
        }

        // Numeric: count distinct (excluding NaN/missing)
        const nonMissing = floatVals.filter((v) => !Number.isNaN(v));
        const distinctSet = new Set(nonMissing);
        const nDistinct = distinctSet.size;

        if (nDistinct === 2) {
            types.push('binary');
            labelsList.push([]);
            continue;
        }

        // Factor guard: few distinct AND enough rows per level
        if (nDistinct > 0 && nDistinct <= maxK && nRows / nDistinct >= maxRatio) {
            const sortedLabels = [...distinctSet].map(valueToLabel).sort();
            types.push('factor');
            labelsList.push(sortedLabels);
        } else {
            types.push('continuous');
            labelsList.push([]);
        }
    }

    return { types, labelsList };
}

// ---------------------------------------------------------------------------
// csvDataFromParsed — produces the wire CsvData shape from parsed CSV
// ---------------------------------------------------------------------------

/** Build a CsvData from a parsed CSV and a chosen mode.
 * Uses UPLOAD config for max_factor_k_soft and max_factor_ratio.
 */
export function csvDataFromParsed(parsed: ParsedCsv, mode: UploadMode): CsvData {
    const { header, columns, nRows } = parsed;
    const { types, labelsList } = detectColumnTypes(
        columns,
        header,
        nRows,
        UPLOAD.max_factor_k_soft,
        UPLOAD.max_factor_ratio,
    );

    const csvColumns: UploadColumn[] = header.map((name, ci) => {
        const colType = types[ci]!;
        const labels = labelsList[ci]!;
        const rawCol = columns[ci]!;

        if (colType === 'factor') {
            // Encode values as 0-indexed level codes (labels[code] = level string).
            const codeMap = new Map(labels.map((lv, idx) => [lv, idx]));
            const values = rawCol.map((cell) => {
                // For numeric factor columns, valueToLabel normalises the cell first.
                const n = Number(cell);
                const labelKey = (!Number.isNaN(n) && Number.isFinite(n))
                    ? valueToLabel(cell)
                    : String(cell);
                // Phase-1 intentional: missing/unknown cells fall to code 0 (first level), mirroring Python.
                return codeMap.get(labelKey) ?? 0;
            });
            return { name, col_type: 'factor', values, labels };
        }

        if (colType === 'binary') {
            // Encode as 0/1: map the two distinct values to 0 and 1 (lower → 0).
            const floatVals = rawCol.map((cell) => Number(cell));
            const distinct = [...new Set(floatVals.filter((v) => !Number.isNaN(v)))].sort((a, b) => a - b);
            const zeroVal = distinct[0] ?? 0;
            const values = floatVals.map((v) => (v === zeroVal ? 0 : 1));
            return { name, col_type: 'binary', values, labels: [] };
        }

        // continuous — raw numeric values (missing → 0 as a safe sentinel; engine will handle)
        const values = rawCol.map((cell) => {
            const n = Number(cell);
            return Number.isNaN(n) ? 0 : n;
        });
        return { name, col_type: 'continuous', values, labels: [] };
    });

    return { mode, n_rows: nRows, columns: csvColumns };
}

// ---------------------------------------------------------------------------
// uploadedColumnByName — shared helper consumed by PredictorCards (type-lock)
// and the correlation UX task. Returns a Map from column name → UploadColumn
// so callers can check membership and read col_type / labels / values in O(1).
// Returns an empty Map when csvData is null/undefined (no upload active).
// ---------------------------------------------------------------------------

/** Index the columns in a CsvData by name for O(1) lookup.
 * Each value exposes `{ name, col_type, labels, values }` (the full UploadColumn).
 * Returns an empty Map when csvData is null or undefined.
 */
export function uploadedColumnByName(csvData: CsvData | null | undefined): Map<string, UploadColumn> {
    if (!csvData) return new Map();
    return new Map(csvData.columns.map((col) => [col.name, col]));
}
