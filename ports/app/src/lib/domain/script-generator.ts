// Dispatches AppSpec to the family-specific script generator, producing a reproducible script.
import type { AppSpec } from './app-spec';
import type { ScriptParams, ScriptLanguage } from './script-generator-linear';
import { generateLinearScript } from './script-generator-linear';
import { generateLogitScript } from './script-generator-logit';
import { generateAnovaScript } from './script-generator-anova';
import { generateMixedScript } from './script-generator-mixed';

export type { ScriptParams, ScriptLanguage } from './script-generator-linear';

export function generateScript(
  spec: AppSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage = 'python',
): string {
  switch (spec.family) {
    case 'linear':
      return generateLinearScript(spec, mode, params, language);
    case 'logit':
      return generateLogitScript(spec, mode, params, language);
    case 'anova':
      return generateAnovaScript(spec, mode, params, language);
    case 'mixed':
      return generateMixedScript(spec, mode, params, language);
    default: {
      const _exhaustive: never = spec;
      throw new Error(`unhandled family: ${(_exhaustive as { family: string }).family}`);
    }
  }
}
