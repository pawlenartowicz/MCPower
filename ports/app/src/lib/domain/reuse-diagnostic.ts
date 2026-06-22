// Strict-mode upload reuse diagnostics.
// Mirrors the Python reference: g(U,N) = 100*(1 - (1-1/U)^N - (N/U)*(1-1/U)^(N-1))
// Warning fires when N > ratio*U (strict greater-than, matching Python semantics).

/**
 * Bootstrap-reuse fraction (%) for a strict-mode upload run.
 *
 * U = uploaded rows, N = sample size used in the run.
 * Exact closed form — NOT a Poisson approximation.
 * Guards: U<=0 → 0, U==1 → 100.
 */
export function reuseFraction(U: number, N: number): number {
    if (U <= 0) return 0;
    if (U === 1) return 100;
    const p = 1 - 1 / U;
    const pN = Math.pow(p, N);
    const pN1 = Math.pow(p, N - 1);
    return 100 * (1 - pN - (N / U) * pN1);
}

/**
 * Returns a warning string when N > ratio*U (strict-mode only, U>0).
 * Returns null when no warning is needed.
 *
 * Golden: (U=100, N=201, ratio=2.0) fires; (U=100, N=200, ratio=2.0) is null.
 */
export function strictReuseWarning(U: number, N: number, ratio: number): string | null {
    if (U <= 0) return null;
    if (N > ratio * U) {
        return `Sample size (N=${N}) is more than ${ratio}× the uploaded rows (U=${U}). Bootstrap reuse is high — consider uploading more data or using partial mode.`;
    }
    return null;
}
