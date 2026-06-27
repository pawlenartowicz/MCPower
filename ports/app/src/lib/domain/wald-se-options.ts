// Wald SE (standard error) method for GLMM families. 'hessian' is the default
// (per-fit FD-Hessian SE, lme4 use.hessian = TRUE — the "correct" denominator);
// 'rx' is the opt-in Schur speed knob (faster, anticonservative). Mirrors
// correction-options.ts in structure.
export type WaldSe = 'hessian' | 'rx';

export const WALD_SE_LABEL: Record<WaldSe, string> = {
  hessian: 'Per-fit Hessian (exact, default)',
  rx: 'Schur (fast, approximate)',
};

export const WALD_SE_OPTIONS: readonly WaldSe[] = ['hessian', 'rx'] as const;
