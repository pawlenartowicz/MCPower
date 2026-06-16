import { describe, it, expect } from 'vitest';
import { validateUploadRows, UploadRejected } from '../src/upload';

describe('validateUploadRows', () => {
  it('accepts at or below max_rows_wasm', () => {
    expect(() => validateUploadRows(10_000)).not.toThrow();
  });
  it('rejects above max_rows_wasm with the limit in the message', () => {
    expect(() => validateUploadRows(10_001)).toThrow(UploadRejected);
    try { validateUploadRows(10_001); } catch (e) { expect((e as Error).message).toContain('10000'); }
  });
});
