// Domain helpers for the correlation-triangle UX.
// Single source of truth for which variables are correlatable and the Pearson measure used
// as a display default in partial-upload mode.

import type { CsvData, UploadColumn } from './app-spec';
import { uploadedColumnByName } from './upload-detect';
import type { UploadMode } from './app-spec';
import type { VariableRow } from './family';

export type { UploadColumn };

export interface CorrelatableEntry {
    row: VariableRow;
    /** Original index into `variables` / the full NxN `correlations` matrix. */
    idx: number;
}

/**
 * Return the subset of variables that may appear in the correlation triangle.
 *
 * Filtering rules (R1–R2):
 *   R1. Only variables with `kind === 'continuous'` are included; factor and binary are excluded.
 *   R2. In strict mode, variables whose name appears in the uploaded dataset are excluded
 *       (the kernel overwrites them entirely; user correlations would be discarded).
 *
 * `idx` is preserved so callers can index the full NxN matrix unchanged.
 */
export function correlatableVariables(
    variables: VariableRow[],
    csvData: CsvData | null | undefined,
    mode: UploadMode,
): CorrelatableEntry[] {
    const uploadedMap = uploadedColumnByName(csvData);
    return variables
        .map((row, idx) => ({ row, idx }))
        .filter(({ row }) => {
            // R1: continuous only
            if (row.kind !== 'continuous') return false;
            // R2: strict mode — drop uploaded columns
            if (mode === 'strict' && uploadedMap.has(row.name)) return false;
            return true;
        });
}

/**
 * Returns true when strict mode should own the empty-triangle message:
 * there is at least one continuous variable AND every continuous variable
 * is in the uploaded dataset (so strict mode removed them all from the triangle).
 *
 * Use this to pick between:
 *   true  → "In strict mode, correlations come from the uploaded data."
 *   false → "Add continuous predictors to edit correlations."
 */
export function allContinuousUploaded(
    variables: VariableRow[],
    csvData: CsvData | null | undefined,
): boolean {
    const continuous = variables.filter((v) => v.kind === 'continuous');
    if (continuous.length === 0) return false;
    const uploadedMap = uploadedColumnByName(csvData);
    return continuous.every((v) => uploadedMap.has(v.name));
}

/**
 * Centered Pearson correlation of two numeric arrays.
 * Returns 0 when either array's variance is zero (constant column).
 * Mathematically identical to the engine's correlation measurement on the same column.values.
 */
export function pearson(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    if (n === 0) return 0;

    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < n; i++) {
        sumA += a[i]!;
        sumB += b[i]!;
    }
    const meanA = sumA / n;
    const meanB = sumB / n;

    let num = 0;
    let denA = 0;
    let denB = 0;
    for (let i = 0; i < n; i++) {
        const da = a[i]! - meanA;
        const db = b[i]! - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
    }

    const denom = Math.sqrt(denA * denB);
    return denom === 0 ? 0 : num / denom;
}

/**
 * Value the correlation editor shows for the pair (`nameI`, `nameJ`) whose
 * stored correlation is `stored`. In partial mode an untouched pair (stored 0)
 * of two uploaded continuous columns shows the measured Pearson r instead; every
 * other case shows the stored value. Display only: the caller never writes the
 * measured value back into the config.
 */
export function correlationDisplayValue(
    stored: number,
    mode: UploadMode,
    csvData: CsvData | null | undefined,
    nameI: string | undefined,
    nameJ: string | undefined,
): number {
    if (mode !== 'partial' || stored !== 0) return stored;
    if (nameI === undefined || nameJ === undefined) return stored;
    const uploadedMap = uploadedColumnByName(csvData);
    const colI = uploadedMap.get(nameI);
    const colJ = uploadedMap.get(nameJ);
    if (!colI || !colJ || colI.col_type !== 'continuous' || colJ.col_type !== 'continuous') {
        return stored;
    }
    return pearson(colI.values, colJ.values);
}
