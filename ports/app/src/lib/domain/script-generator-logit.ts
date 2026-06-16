// Generates reproducible MCPower snippets for logit family specs (Python + R output).
import type { AppSpec, LogitSpec, LinearSpec } from './app-spec';
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
function buildFormulaString(spec: LogitSpec, sep: string = '='): string {
  const { outcome, predictors, interaction_terms } = spec.parsed_formula;
  const parts: string[] = [...predictors];
  for (const terms of interaction_terms) {
    parts.push(terms.join(':'));
  }
  return `${outcome} ${sep} ${parts.join(' + ')}`;
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

  if (!s.parsed_formula?.outcome) {
    return '# Cannot generate script: missing formula\n';
  }

  const lang = langFor(language);
  const lines: string[] = [];

  for (const h of lang.header) lines.push(h);
  lines.push('');

  for (const imp of lang.importLines) lines.push(imp);
  lines.push('');

  // Constructor — R uses `~`, Python uses `=`; family="logit" distinguishes from OLS
  const formula = buildFormulaString(s, language === 'r' ? '~' : '=');
  lines.push(lang.construct(formula, 'logit'));

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

  // Outcome-level knobs — family-agnostic helper; a no-op for logit specs that
  // carry no outcome_options (residual_distribution / heteroskedasticity_driver).
  for (const line of buildOutcomeOptionLines(s, lang)) lines.push(line);

  // Baseline probability — logit-specific; placed after effects, before run-config
  lines.push(lang.call('set_baseline_probability', String(s.baseline_probability)));

  // Run-config setters — omit values equal to port defaults; set_power takes percent
  // logit uses the OLS sims default (family="logit" resolves to the OLS kernel)
  for (const line of buildConfigLines(s, SIMULATION.n_sims.ols, lang)) lines.push(line);

  lines.push('');

  // find_power / find_sample_size call
  const correction = toPortCorrection(s.correction);
  const testsArg = buildTestsArg(s as LinearSpec);
  for (const line of buildFindCallLines(lang, mode, params, testsArg, correction)) lines.push(line);

  return lines.join('\n') + '\n';
}
