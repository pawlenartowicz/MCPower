import { describe, it, expect } from 'vitest';
import { reuseFraction, strictReuseWarning } from './reuse-diagnostic';

describe('reuseFraction', () => {
    it('golden: g(1000, 1000) ≈ 26', () => {
        expect(reuseFraction(1000, 1000)).toBeCloseTo(26, 0);
    });

    it('golden: g(1000, 2000) ≈ 59', () => {
        expect(reuseFraction(1000, 2000)).toBeCloseTo(59, 0);
    });

    it('golden: g(1000, 999) >= 0', () => {
        expect(reuseFraction(1000, 999)).toBeGreaterThanOrEqual(0);
    });

    it('guard: U<=0 returns 0', () => {
        expect(reuseFraction(0, 100)).toBe(0);
        expect(reuseFraction(-5, 100)).toBe(0);
    });

    it('guard: U==1 returns 100', () => {
        expect(reuseFraction(1, 1)).toBe(100);
        expect(reuseFraction(1, 500)).toBe(100);
    });

    it('returns a number in [0, 100] for typical inputs', () => {
        const g = reuseFraction(500, 300);
        expect(g).toBeGreaterThanOrEqual(0);
        expect(g).toBeLessThanOrEqual(100);
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

    it('returns null when N equals ratio*U exactly', () => {
        expect(strictReuseWarning(50, 100, 2.0)).toBeNull();
    });

    it('fires when N is one above ratio*U', () => {
        expect(strictReuseWarning(50, 101, 2.0)).not.toBeNull();
    });

    it('warning message includes N and U', () => {
        const msg = strictReuseWarning(100, 250, 2.0);
        expect(msg).toContain('N=250');
        expect(msg).toContain('U=100');
    });
});
