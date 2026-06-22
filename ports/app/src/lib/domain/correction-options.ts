// Correction method labels and per-entrypoint option sets; ANOVA additionally offers Tukey HSD, which is invalid for regression and mixed.
import type { AdvancedConfig, Entrypoint } from './family';

export type CorrectionMethod = AdvancedConfig['correction'];

export const CORRECTION_LABEL: Record<CorrectionMethod, string> = {
  none: 'None',
  bonferroni: 'Bonferroni',
  bh: 'Benjamini–Hochberg (FDR)',
  holm: 'Holm (recommended)',
  tukey: 'Tukey HSD',
};

// Regression and mixed share the base correction set; ANOVA adds Tukey HSD.
export const CORRECTION_OPTIONS: Record<Entrypoint, readonly CorrectionMethod[]> = {
  regression: ['none', 'bonferroni', 'holm', 'bh'],
  mixed: ['none', 'bonferroni', 'holm', 'bh'],
  anova: ['none', 'bonferroni', 'holm', 'bh', 'tukey'],
} as const;

export const CORRECTION_DEFAULT: Record<Entrypoint, CorrectionMethod> = {
  regression: 'none',
  mixed: 'none',
  anova: 'none',
};
