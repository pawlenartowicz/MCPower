// Three-way estimation control for GLMM fits, encoded as a (wald_se, agq) pair on
// AdvancedConfig. Mirrors wald-se-options.ts in structure.
//   Fast     → wald_se='rx',      agq=1  (Schur SE, Laplace — the default)
//   Accurate → wald_se='hessian', agq=1  (per-fit Hessian SE, Laplace)
//   AGQ      → wald_se='hessian', agq=k  (adaptive Gauss-Hermite, k odd > 1)
import type { WaldSe } from './wald-se-options';

export type EstimationMode = 'fast' | 'accurate' | 'agq';

export const ESTIMATION_LABEL: Record<EstimationMode, string> = {
  fast: 'Fast (Schur SE, default)',
  accurate: 'Accurate (Hessian SE)',
  agq: 'High accuracy (adaptive quadrature)',
};

export const ESTIMATION_OPTIONS: readonly EstimationMode[] = ['fast', 'accurate', 'agq'] as const;

/** AGQ node count: odd integers > 1, capped at 25 (diminishing returns past it). */
export const AGQ_DEFAULT_NODES = 5;
export const AGQ_MIN_NODES = 3;
export const AGQ_MAX_NODES = 25;

/** Resolve the current (wald_se, agq) pair to an estimation mode. AGQ (agq > 1)
 *  wins regardless of wald_se; otherwise the SE method picks Fast vs Accurate. */
export function estimationModeOf(wald_se: WaldSe, agq: number): EstimationMode {
  if (agq > 1) return 'agq';
  return wald_se === 'hessian' ? 'accurate' : 'fast';
}

/** The (wald_se, agq) pair for a chosen mode. For AGQ, keep the caller's current
 *  node count when it is already a valid AGQ value (> 1); otherwise seed the default. */
export function estimationPair(mode: EstimationMode, currentAgq: number): { wald_se: WaldSe; agq: number } {
  if (mode === 'fast') return { wald_se: 'rx', agq: 1 };
  if (mode === 'accurate') return { wald_se: 'hessian', agq: 1 };
  return { wald_se: 'hessian', agq: currentAgq > 1 ? clampAgqNodes(currentAgq) : AGQ_DEFAULT_NODES };
}

/** Clamp a requested node count into the valid odd [MIN, MAX] band (round up to odd).
 *  A non-finite input (NumberInput emits NaN on an emptied field) falls back to the
 *  default rather than propagating NaN — an unclamped NaN silently reads as `agq > 1
 *  === false` downstream, which ships Laplace while the UI still shows AGQ selected. */
export function clampAgqNodes(n: number): number {
  if (!Number.isFinite(n)) return AGQ_DEFAULT_NODES;
  const rounded = Math.round(n);
  const odd = rounded % 2 === 0 ? rounded + 1 : rounded;
  return Math.min(AGQ_MAX_NODES, Math.max(AGQ_MIN_NODES, odd));
}
