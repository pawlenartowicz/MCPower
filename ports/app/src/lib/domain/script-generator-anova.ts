// Generates reproducible MCPower snippets for ANOVA family specs (factors + covariates, pairwise contrasts).
// ANOVA specs (family:'anova') map to the linear engine family on the wire.
import type { AppSpec, AnovaSpec, AnovaFactor } from './app-spec';
import {
  type ScriptLanguage,
  type ScriptParams,
  langFor,
  toPortCorrection,
  buildConfigLines,
  buildFindCallLines,
} from './script-generator-linear';
import { SIMULATION } from '$lib/configs/app-config';

/** ANOVA factors in the shared assignment grammar (`name=(factor,p1,…)`);
 *  covariates are numeric → omitted. */
function buildVarTypeString(factors: AnovaFactor[]): string | null {
  if (factors.length === 0) return null;
  return factors
    .map((f) => {
      const n = f.levels.length;
      // toFixed(6) truncates: e.g. 3×0.333333 = 0.999999, not 1. The engine's Rust
      // spec-builder renormalises proportions to their sum, so the 1e-6 shortfall is harmless.
      const proportions =
        f.proportions ?? Array.from({ length: n }, () => parseFloat((1 / n).toFixed(6)));
      return `${f.name}=(factor,${proportions.join(',')})`;
    })
    .join(', ');
}

export function generateAnovaScript(
  spec: AppSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage = 'python',
): string {
  if (spec.family !== 'anova') {
    throw new Error('generateAnovaScript called with non-anova spec');
  }

  const s: AnovaSpec = spec;
  const lang = langFor(language);
  const lines: string[] = [];

  for (const h of lang.header) lines.push(h);
  lines.push('');

  for (const imp of lang.importLines) lines.push(imp);
  lines.push('');

  // Formula: outcome = factor1 + factor2 + ... + cov1 + ... (R uses ~, Python uses =)
  const predictors = [...s.factors.map((f) => f.name), ...s.covariates.map((c) => c.name)];
  const sep = language === 'r' ? '~' : '=';
  const formula = `${s.outcome} ${sep} ${predictors.join(' + ')}`;
  lines.push(lang.construct(formula));

  // Variable types — factors only (covariates are numeric, omitted)
  const varTypeStr = buildVarTypeString(s.factors);
  if (varTypeStr !== null) {
    lines.push(lang.call('set_variable_type', `"${varTypeStr}"`));
  }

  // Effects — ANOVA effect names are already label-form in AnovaSpec
  const effectsStr = s.effects.map((e) => `${e.name}=${e.value}`).join(', ');
  lines.push(lang.call('set_effects', `"${effectsStr}"`));

  // No set_correlations for ANOVA (spec.correlations is null for factor designs)

  // Run-config setters — omit values equal to port defaults; set_power takes percent
  for (const line of buildConfigLines(s, SIMULATION.n_sims.ols, lang)) lines.push(line);

  lines.push('');

  // target_test="..." DSL: omit unless selection deviates from the family default
  const contrasts = s.contrasts ?? [];
  const useTargetTest =
    contrasts.length > 0 || s.tests.kind === 'effects' || s.report_overall === false;

  let targetTestArg: string | null = null;
  if (useTargetTest) {
    const tokens: string[] = [];
    if (s.tests.kind === 'all' && s.report_overall === true) {
      tokens.push('all');
    } else if (s.tests.kind === 'all' && s.report_overall === false) {
      for (const eff of s.effects) {
        tokens.push(eff.name);
      }
    } else if (s.tests.kind === 'effects') {
      if (s.report_overall) tokens.push('overall');
      for (const name of s.tests.names) {
        tokens.push(name);
      }
    } else if (s.tests.kind === 'contrasts') {
      if (s.report_overall) tokens.push('overall');
    }
    for (const [pos, neg] of contrasts) {
      tokens.push(`${pos} vs ${neg}`);
    }
    targetTestArg = tokens.join(', ');
  }

  const correction = toPortCorrection(s.correction);
  for (const line of buildFindCallLines(lang, mode, params, targetTestArg, correction)) {
    lines.push(line);
  }

  return lines.join('\n') + '\n';
}
