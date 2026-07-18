// Generates reproducible MCPower snippets for logit and Poisson family specs
// (Python + R output). The two families share every block below except the
// constructor family token and the baseline-value setter name/arg, so both
// script-generator-poisson.ts and this file's own generateLogitScript delegate
// to the shared generateGlmScript.
import type { AppSpec, LogitSpec, PoissonSpec, LinearSpec } from './app-spec';
import {
  toPortCorrection,
  buildVarTypeString,
  buildCorrelationsString,
  buildOutcomeOptionLines,
  buildConfigLines,
  buildFindCallLines,
  buildTestsArg,
  pyEffectName,
  langFor,
} from './script-generator-linear';
import type { ScriptParams, ScriptLanguage } from './script-generator-linear';
import { SIMULATION } from '$lib/configs/app-config';

/** Reconstruct formula string from ParsedFormula. Separator is `=` for Python, `~` for R. */
function buildFormulaString(spec: LogitSpec | PoissonSpec, sep: string = '='): string {
  const { outcome, predictors, interaction_terms } = spec.parsed_formula;
  const parts: string[] = [...predictors];
  for (const terms of interaction_terms) {
    parts.push(terms.join(':'));
  }
  return `${outcome} ${sep} ${parts.join(' + ')}`;
}

/**
 * Shared body for the logit and Poisson generators. `family` is the value
 * passed to the constructor's `family=`/`family =` arg (also used as the
 * probit override check for logit); `baselineSetter` and `baselineValue`
 * cover the one call that differs between the two families
 * (set_baseline_probability vs set_baseline_rate).
 */
function generateGlmScript(
  spec: LogitSpec | PoissonSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage,
  family: string,
  baselineSetter: string,
  baselineValue: number,
): string {
  const s = spec;

  if (!s.parsed_formula?.outcome) {
    return '# Cannot generate script: missing formula\n';
  }

  const lang = langFor(language);
  const lines: string[] = [];

  for (const h of lang.header) lines.push(h);
  lines.push('');

  for (const imp of lang.importLines) lines.push(imp);
  lines.push('');

  // Constructor — R uses `~`, Python uses `=`; family distinguishes from OLS.
  const formula = buildFormulaString(s, language === 'r' ? '~' : '=');
  lines.push(lang.construct(formula, family));

  // Variable types — only when at least one non-numeric
  const varTypeStr = buildVarTypeString(s.var_types);
  if (varTypeStr !== null) {
    lines.push(lang.call('set_variable_type', `"${varTypeStr}"`));
  }

  // Effects
  const effectsStr = s.effects
    .map((e) => `${pyEffectName(e.name, s.var_types)}=${e.value}`)
    .join(', ');
  lines.push(lang.call('set_effects', `"${effectsStr}"`));

  // Correlations — only when present and non-zero
  const corrStr = buildCorrelationsString(s as LinearSpec);
  if (corrStr !== null) {
    lines.push(lang.call('set_correlations', `"${corrStr}"`));
  }

  // Outcome-level knobs — family-agnostic helper; a no-op for specs that
  // carry no outcome_options (residual_distribution / heteroskedasticity_driver).
  for (const line of buildOutcomeOptionLines(s, lang)) lines.push(line);

  // Baseline value — placed after effects, before run-config
  lines.push(lang.call(baselineSetter, String(baselineValue)));

  // Run-config setters — omit values equal to port defaults; set_power takes percent.
  // Both logit and Poisson resolve to the OLS sims default.
  for (const line of buildConfigLines(s, SIMULATION.n_sims.ols, lang)) lines.push(line);

  lines.push('');

  // find_power / find_sample_size call
  const correction = toPortCorrection(s.correction);
  const testsArg = buildTestsArg(s as LinearSpec);
  // agq is only >1 for a clustered GLMM; unclustered logit/Poisson carries the
  // default (1), so this stays omitted — threaded for symmetry with the mixed generator.
  for (const line of buildFindCallLines(lang, mode, params, testsArg, correction, undefined, s.agq))
    lines.push(line);

  return lines.join('\n') + '\n';
}

export function generateLogitScript(
  spec: AppSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage = 'python',
): string {
  if (spec.family !== 'logit') {
    throw new Error('generateLogitScript called with non-logit spec');
  }

  const s: LogitSpec = spec;
  // A probit link selects family="probit"; logit (default/omitted) stays "logit".
  return generateGlmScript(
    s,
    mode,
    params,
    language,
    s.link === 'probit' ? 'probit' : 'logit',
    'set_baseline_probability',
    s.baseline_probability,
  );
}

export function generatePoissonScript(
  spec: AppSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage = 'python',
): string {
  if (spec.family !== 'poisson') {
    throw new Error('generatePoissonScript called with non-poisson spec');
  }

  const s: PoissonSpec = spec;
  return generateGlmScript(s, mode, params, language, 'poisson', 'set_baseline_rate', s.baseline_rate);
}
