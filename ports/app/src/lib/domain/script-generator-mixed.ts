// Generates reproducible MCPower snippets for mixed-effects family specs
// (random intercepts incl. crossed/nested extra groupings, slopes, cluster dim, ICC).
// Supports Python and R output via the shared Lang abstraction.
import type { AppGroupingSpec, AppSpec, MixedSpec } from './app-spec';
import type { ScriptParams, ScriptLanguage } from './script-generator-linear';
import {
  buildVarTypeString,
  buildCorrelationsString,
  buildOutcomeOptionLines,
  buildTestsArg,
  pyEffectName,
  toPortCorrection,
  langFor,
  buildConfigLines,
  buildFindCallLines,
} from './script-generator-linear';
import { SIMULATION } from '$lib/configs/app-config';

function extraName(g: AppGroupingSpec, idx: number): string {
  return g.cluster_name ?? `g${idx + 2}`;
}

/** Mixed formula keeps the random-effects terms and uses `~` in both languages.
 *  A nested grouping named `parent:child` (parent === the primary) folds into
 *  a single `(1|parent/child)` term; crossed extras append `(1|name)`. Slopes
 *  don't appear in the formula — the Python API takes them via
 *  `set_cluster(random_slopes=...)`. */
function buildMixedFormula(s: MixedSpec): string {
  const { outcome, predictors, interaction_terms } = s.parsed_formula;
  const parts: string[] = [...predictors];
  for (const terms of interaction_terms) parts.push(terms.join(':'));

  let primaryTerm = `(1|${s.cluster_name})`;
  const extraTerms: string[] = [];
  (s.extra_groupings ?? []).forEach((g, i) => {
    const name = extraName(g, i);
    if (g.relation.kind === 'nested_within' && name.startsWith(`${s.cluster_name}:`)) {
      primaryTerm = `(1|${s.cluster_name}/${name.slice(s.cluster_name.length + 1)})`;
    } else {
      extraTerms.push(`(1|${name})`);
    }
  });
  parts.push(primaryTerm, ...extraTerms);
  return `${outcome} ~ ${parts.join(' + ')}`;
}

/** Invert the adapter's ICC→τ² map (raw: τ = icc/(1−icc); the logit link
 *  additionally scales τ by π²/3) so the script writes `ICC=` back. `logitScaled`
 *  is true only for the logit link — probit/Poisson/Gaussian use the raw map. */
function tauToIcc(tau: number, logitScaled: boolean): number {
  const t = logitScaled ? tau / (Math.PI ** 2 / 3) : tau;
  const icc = t / (1 + t);
  return Math.round(icc * 1e6) / 1e6;
}

/** Render a string array as a language-appropriate list literal.
 *  Python: `["a", "b"]`; R: `c("a", "b")`. Items must already be quoted. */
function renderList(items: string[], language: ScriptLanguage): string {
  if (language === 'r') return `c(${items.join(', ')})`;
  return `[${items.join(', ')}]`;
}

/** Build the set_cluster call argument string for the primary cluster.
 *  Python uses flat kwargs (random_slopes=list, slope_variance=N, slope_intercept_corr=N).
 *  R wraps each slope in a named list (predictor, variance, corr_with_intercept).
 *  cluster_level_vars uses the language list renderer. */
function buildClusterArgs(
  s: MixedSpec,
  language: ScriptLanguage,
  primaryVariance: number,
  poissonOutcome: boolean,
): string[] {
  const dimArg =
    s.cluster_dim.kind === 'n_clusters'
      ? `n_clusters=${s.cluster_dim.value}`
      : `cluster_size=${s.cluster_dim.value}`;
  const sep = language === 'r' ? ' = ' : '=';

  // Variance arg — `ICC=` for every family EXCEPT Poisson, which has no ICC and
  // requires the raw random-intercept variance via `tau_squared=` (the ports
  // reject `ICC=` for family="poisson"). Same keyword in Python and R.
  const varKw = poissonOutcome ? 'tau_squared' : 'ICC';
  const iccArg = `${varKw}${sep}${primaryVariance}`;

  const args = [`"${s.cluster_name}"`, dimArg, iccArg];

  const slopes = s.slopes ?? [];
  if (slopes.length > 0) {
    const quotedNames = slopes.map((sl) => `"${sl.predictor_name}"`);
    if (language === 'r') {
      // R: random_slopes = list(list(predictor = "x", variance = V, corr_with_intercept = C), ...)
      // mirrors R set_cluster validation: sl$predictor, sl$variance, sl$corr_with_intercept
      const rSlopes = slopes
        .map(
          (sl) =>
            `list(predictor = "${sl.predictor_name}", variance = ${sl.slope_variance}, corr_with_intercept = ${sl.slope_intercept_corr})`,
        )
        .join(', ');
      args.push(`random_slopes = list(${rSlopes})`);
    } else {
      // Python: random_slopes=["x"], slope_variance=V, slope_intercept_corr=C
      // Python API shares one (variance, corr) pair for all slopes.
      args.push(`random_slopes=[${quotedNames.join(', ')}]`);
      args.push(`slope_variance=${slopes[0]!.slope_variance}`);
      args.push(`slope_intercept_corr=${slopes[0]!.slope_intercept_corr}`);
    }
  }

  const clv = s.cluster_level_vars ?? [];
  if (clv.length > 0) {
    const quotedClv = clv.map((n) => `"${n}"`);
    args.push(`cluster_level_vars${sep}${renderList(quotedClv, language)}`);
  }

  return args;
}

export function generateMixedScript(
  spec: AppSpec,
  mode: 'find-power' | 'find-sample-size',
  params: ScriptParams,
  language: ScriptLanguage = 'python',
): string {
  if (spec.family !== 'mixed') throw new Error('generateMixedScript called with non-mixed spec');
  const s: MixedSpec = spec;
  if (!s.parsed_formula?.outcome) return '# Cannot generate script: missing formula\n';

  const binaryOutcome = s.outcome?.kind === 'binary';
  const poissonOutcome = s.outcome?.kind === 'poisson';
  const probit = s.outcome?.kind === 'binary' && s.outcome.link === 'probit';
  const glmm = binaryOutcome || poissonOutcome;
  const family = poissonOutcome ? 'poisson' : binaryOutcome ? (probit ? 'probit' : 'logit') : 'lme';
  // A GLMM outcome (family "logit"/"probit"/"poisson") uses the OLS sim budget
  // (1600); "lme" (Gaussian) uses the mixed budget (800).
  const simsDefault = glmm ? SIMULATION.n_sims.ols : SIMULATION.n_sims.mixed;

  const lang = langFor(language);
  const lines: string[] = [];

  for (const h of lang.header) lines.push(h);
  lines.push('');

  for (const imp of lang.importLines) lines.push(imp);
  lines.push('');

  // Constructor — mixed formula always uses `~`
  lines.push(lang.construct(buildMixedFormula(s), family));

  // Variable types — only when at least one non-numeric
  const varTypeStr = buildVarTypeString(s.var_types);
  if (varTypeStr !== null) lines.push(lang.call('set_variable_type', `"${varTypeStr}"`));

  // Effects
  const effectsStr = s.effects
    .map((e) => `${pyEffectName(e.name, s.var_types)}=${e.value}`)
    .join(', ');
  lines.push(lang.call('set_effects', `"${effectsStr}"`));

  // Correlations — only when present and non-zero
  const corrStr = buildCorrelationsString(s);
  if (corrStr !== null) lines.push(lang.call('set_correlations', `"${corrStr}"`));

  // Outcome-level knobs
  for (const line of buildOutcomeOptionLines(s, lang)) lines.push(line);

  // GLMM baseline: probability for binary (logit/probit), rate λ for Poisson.
  if (s.outcome?.kind === 'binary') {
    lines.push(lang.call('set_baseline_probability', String(s.outcome.baseline_probability)));
  } else if (s.outcome?.kind === 'poisson') {
    lines.push(lang.call('set_baseline_rate', String(s.outcome.baseline_rate)));
  }

  // Run-config setters — omit values equal to port defaults
  for (const line of buildConfigLines(s, simsDefault, lang)) lines.push(line);

  // Poisson's primary variance is the raw τ² in the outcome (no ICC); every other
  // outcome uses the top-level icc field directly.
  const primaryVariance = s.outcome?.kind === 'poisson' ? s.outcome.tau_squared : s.icc;
  const clusterArgs = buildClusterArgs(s, language, primaryVariance, poissonOutcome);
  if (
    language === 'python' &&
    s.slopes &&
    s.slopes.length > 1 &&
    s.slopes.some(
      (sl) =>
        sl.slope_variance !== s.slopes![0]!.slope_variance ||
        sl.slope_intercept_corr !== s.slopes![0]!.slope_intercept_corr,
    )
  ) {
    lines.push('# note: per-slope variances differ in the app; the Python API shares one pair');
  }
  lines.push(lang.call('set_cluster', clusterArgs.join(', ')));

  // Extra groupings: one set_cluster per grouping
  (s.extra_groupings ?? []).forEach((g, i) => {
    const name = extraName(g, i);
    // Poisson extra groupings carry raw τ² via `tau_squared=` (skip the ICC
    // inversion AND the ICC keyword); logit inverts through the π²/3 scaling,
    // probit/Gaussian through the raw map, both emitted as `ICC=`.
    const variance = poissonOutcome ? g.tau_squared : tauToIcc(g.tau_squared, binaryOutcome && !probit);
    const varKw = poissonOutcome ? 'tau_squared' : 'ICC';
    const sep = language === 'r' ? ' = ' : '=';
    if (g.relation.kind === 'nested_within') {
      lines.push(lang.call('set_cluster', `"${name}", n_per_parent${sep}${g.relation.n_per_parent}, ${varKw}${sep}${variance}`));
    } else {
      lines.push(lang.call('set_cluster', `"${name}", n_clusters${sep}${g.relation.n_clusters}, ${varKw}${sep}${variance}`));
    }
  });

  lines.push('');

  // find_power / find_sample_size call
  const correction = toPortCorrection(s.correction);
  const testsArg = buildTestsArg(s);
  // wald_se is the GLMM SE knob and agq the adaptive-quadrature knob; both are
  // emitted only when non-default and only meaningful for a clustered GLMM outcome
  // — harmless (default, omitted) on the Gaussian LME path.
  for (const line of buildFindCallLines(lang, mode, params, testsArg, correction, s.wald_se, s.agq))
    lines.push(line);

  return lines.join('\n') + '\n';
}
