import { describe, it, expect } from 'vitest';
import config from '$configs/config.json';
import { validateUploadRows, UploadRejected } from '../src/upload';

const MAX = config.upload.max_rows_wasm;

describe('validateUploadRows', () => {
  it('accepts at or below max_rows_wasm', () => {
    expect(() => validateUploadRows(MAX)).not.toThrow();
  });
  it('rejects above max_rows_wasm with the limit in the message', () => {
    expect(() => validateUploadRows(MAX + 1)).toThrow(UploadRejected);
    try {
      validateUploadRows(MAX + 1);
    } catch (e) {
      expect((e as Error).message).toContain(String(MAX));
    }
  });
});
