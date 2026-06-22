/**
 * Unit tests for reuseFraction and strictReuseWarning — mirrors
 * ports/py/tests/test_reuse_diagnostics.py (golden values must match).
 */
import { describe, it, expect } from 'vitest';
import { reuseFraction, strictReuseWarning } from '../src/reuse';

describe('reuseFraction', () => {
  it('golden: (1000, 1000) ≈ 26', () => {
    expect(reuseFraction(1000, 1000)).toBeCloseTo(26, 0);
  });

  it('golden: (1000, 2000) ≈ 59', () => {
    expect(reuseFraction(1000, 2000)).toBeCloseTo(59, 0);
  });

  it('golden: (1000, 999) >= 0', () => {
    expect(reuseFraction(1000, 999)).toBeGreaterThanOrEqual(0);
  });

  it('guard: U=0 → 0', () => {
    expect(reuseFraction(0, 100)).toBe(0);
  });

  it('guard: U<0 → 0', () => {
    expect(reuseFraction(-5, 100)).toBe(0);
  });

  it('guard: U=1 → 100', () => {
    expect(reuseFraction(1, 100)).toBe(100);
  });
});

describe('strictReuseWarning', () => {
  it('fires when N > ratio*U: (U=100, N=201, ratio=2.0) → non-null', () => {
    // ratio=2.0 is read from config; config.upload.strict_warning_ratio=2.0
    expect(strictReuseWarning(100, 201)).not.toBeNull();
  });

  it('null when N == ratio*U: (U=100, N=200, ratio=2.0) → null', () => {
    expect(strictReuseWarning(100, 200)).toBeNull();
  });

  it('null when N < ratio*U', () => {
    expect(strictReuseWarning(100, 150)).toBeNull();
  });

  it('warning message contains N, U, and ratio', () => {
    const w = strictReuseWarning(50, 200);
    expect(w).not.toBeNull();
    expect(w).toContain('N=200');
    expect(w).toContain('50');
  });
});
