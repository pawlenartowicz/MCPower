// Wald SE (standard error) method for GLMM families. 'hessian' is the default
// per-fit Hessian SE (lme4 use.hessian = TRUE, as glmer's summary reports);
// 'rx' is the faster Schur SE. Mirrors correction-options.ts in structure.
export type WaldSe = 'hessian' | 'rx';

export const WALD_SE_LABEL: Record<WaldSe, string> = {
  hessian: 'Per-fit Hessian (default)',
  rx: 'Schur (fast)',
};

export const WALD_SE_OPTIONS: readonly WaldSe[] = ['hessian', 'rx'] as const;
