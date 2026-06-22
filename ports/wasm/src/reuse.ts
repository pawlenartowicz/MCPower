// Strict-bootstrap reuse diagnostics — mirrors Python's _reuse_fraction /
// _strict_reuse_warning in ports/py/mcpower/model.py.
import config from '$configs/config.json';

/**
 * Expected % of uploaded rows reused within one strict-bootstrap dataset.
 *
 * A dataset of size N is drawn with replacement from U uploaded rows.
 * Closed form: g = 100 * [1 - (1-1/U)^N - (N/U)*(1-1/U)^(N-1)]
 * Guard: U <= 0 → 0; U == 1 → 100.
 */
export function reuseFraction(U: number, N: number): number {
  if (U <= 0) return 0;
  if (U === 1) return 100;
  const p = 1 - 1 / U;
  return 100 * (1 - Math.pow(p, N) - (N / U) * Math.pow(p, N - 1));
}

/**
 * Return a warning string when N > ratio*U (strict `>`), else null.
 * Ratio is read from the shared config (upload.strict_warning_ratio).
 */
export function strictReuseWarning(U: number, N: number): string | null {
  const ratio: number = config.upload.strict_warning_ratio;
  if (N > ratio * U) {
    return (
      `N=${N} is more than ${ratio}x the uploaded rows (${U}). ` +
      "Each strict-bootstrap dataset will reuse many rows; consider mode='partial' " +
      "or mode='none' for a faster and more generalizable simulation."
    );
  }
  return null;
}
