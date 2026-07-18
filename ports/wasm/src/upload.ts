// Host-side upload gate: the engine only ever sees inputs within the shared
// limits (host-owned upload frames). max_rows_wasm lives in the single
// shared config.json (configs/config.json → "upload"), not a per-port copy.
import config from '$configs/config.json';

const MAX_ROWS_WASM: number = config.upload.max_rows_wasm;

/** Thrown by `validateUploadRows` when the row count exceeds the browser limit
 *  (`max_rows_wasm` from `configs/config.json`). The message includes the
 *  actual and allowed row counts, and suggests the desktop app for larger data. */
export class UploadRejected extends Error {}

/** Throws UploadRejected if a browser upload exceeds max_rows_wasm. */
export function validateUploadRows(rows: number): void {
  if (rows > MAX_ROWS_WASM) {
    throw new UploadRejected(
      `Upload has ${rows} rows; the browser limit is ${MAX_ROWS_WASM}. ` +
        `Reduce the dataset or use the desktop app (limit ${config.upload.max_rows}).`,
    );
  }
}
