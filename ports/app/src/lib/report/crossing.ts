// Shared display helpers for model-based crossing fits (SampleSizeResult.fitted / fitted_joint).
// Used by TableView (per-target marginal) and JointDistTab (joint ≥k) — keep in one place so
// the fallback chain and CI format are never duplicated.
import type { CrossingFit } from '$lib/domain/result';

/**
 * Headline required-N cell from a CrossingFit, falling back to the grid-empirical value.
 *
 * Chain:
 *   fitted      → String(n_achievable)
 *   at_or_below_min → `≤ ${n_min}`
 *   not_reached → `≥ ${ceiling}` (same text as the grid-empirical fallback)
 *   non_monotone → grid value (n ?? `≥ ceiling`)
 *   no fit      → grid value (n ?? `≥ ceiling`)
 */
export function requiredNHeadline(
  fit: CrossingFit | undefined,
  gridN: number | null,
  ceiling: number,
): string {
  if (!fit) return gridN == null ? `≥ ${ceiling}` : String(gridN);
  switch (fit.status) {
    case 'fitted':       return String(fit.n_achievable);
    case 'at_or_below_min': return `≤ ${fit.n_min}`;
    case 'not_reached':  return `≥ ${ceiling}`;
    // non_monotone: engine suppressed the model fit; fall back to grid value
    case 'non_monotone': return gridN == null ? `≥ ${ceiling}` : String(gridN);
  }
}

/**
 * CI cell for the single-scenario Required-N table.
 *
 * Rules:
 *   fitted, both bounds:  `[⌊ci_lo⌋, ⌈ci_hi⌉]`
 *   fitted, ci_lo null:   `[≤ ${floorN}, ⌈ci_hi⌉]`  (floorN = min grid n)
 *   fitted, ci_hi null:   `[⌊ci_lo⌋, ≥ ${ceiling}]`
 *   fitted, both null:    `[≤ ${floorN}, ≥ ${ceiling}]`
 *   not_reached:          `appr. ${n_approx}` when n_approx != null, else `—`
 *   at_or_below_min / non_monotone / missing: `—`
 */
export function requiredNCI(
  fit: CrossingFit | undefined,
  floorN: number,
  ceiling: number,
): string {
  if (!fit || fit.status === 'at_or_below_min' || fit.status === 'non_monotone') return '—';
  if (fit.status === 'not_reached') {
    return fit.n_approx != null ? `appr. ${fit.n_approx}` : '—';
  }
  // fitted
  const lo = fit.ci_lo != null ? String(Math.floor(fit.ci_lo)) : `≤ ${floorN}`;
  const hi = fit.ci_hi != null ? String(Math.ceil(fit.ci_hi)) : `≥ ${ceiling}`;
  return `[${lo}, ${hi}]`;
}
