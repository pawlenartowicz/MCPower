// Wald SE (standard error) method for GLMM families. 'rx' is the default
// (Schur speed knob); 'hessian' is the opt-in per-fit FD-Hessian SE
// (lme4 use.hessian = TRUE — the "correct" but slower denominator). Mirrors
// correction-options.ts in structure.
export type WaldSe = 'hessian' | 'rx';

export const WALD_SE_LABEL: Record<WaldSe, string> = {
  hessian: 'Per-fit Hessian (exact)',
  rx: 'Schur (fast, default)',
};

export const WALD_SE_OPTIONS: readonly WaldSe[] = ['hessian', 'rx'] as const;
