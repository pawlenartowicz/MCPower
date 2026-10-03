import { describe, it, expect } from 'vitest';
import { reuseFraction, strictReuseWarning } from './reuse-diagnostic';

describe('reuseFraction', () => {
    // Two decimals: at 0 decimals an exponent slip (26.39 vs 26.42) passes.
    it('golden: g(1000, 1000) ≈ 26.42', () => {
        expect(reuseFraction(1000, 1000)).toBeCloseTo(26.424, 2);
    });

    it('golden: g(1000, 2000) ≈ 59.41', () => {
        expect(reuseFraction(1000, 2000)).toBeCloseTo(59.413, 2);
    });

    it('golden: g(1000, 999) ≈ 26.39', () => {
        expect(reuseFraction(1000, 999)).toBeCloseTo(26.387, 2);
    });

    it('guard: U<=0 returns 0', () => {
        expect(reuseFraction(0, 100)).toBe(0);
        expect(reuseFraction(-5, 100)).toBe(0);
    });

    it('guard: U==1 returns 100', () => {
        expect(reuseFraction(1, 1)).toBe(100);
        expect(reuseFraction(1, 500)).toBe(100);
    });

    it('golden: g(500, 300) ≈ 12.18', () => {
        expect(reuseFraction(500, 300)).toBeCloseTo(12.177, 2);
    });
});

describe('strictReuseWarning', () => {
    it('golden: (100, 201, 2.0) fires', () => {
        expect(strictReuseWarning(100, 201, 2.0)).not.toBeNull();
    });

    it('golden: (100, 200, 2.0) returns null', () => {
        expect(strictReuseWarning(100, 200, 2.0)).toBeNull();
    });

    it('returns null when U<=0', () => {
        expect(strictReuseWarning(0, 300, 2.0)).toBeNull();
        expect(strictReuseWarning(-1, 300, 2.0)).toBeNull();
    });

    it('warning message includes N and U', () => {
        const msg = strictReuseWarning(100, 250, 2.0);
        expect(msg).toContain('N=250');
        expect(msg).toContain('U=100');
    });
});
