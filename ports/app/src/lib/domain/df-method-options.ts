// Test for the fixed effects of a mixed model. 'auto' picks one per design:
// between-within df for a single grouping without a random slope and for any
// single-grouping binary/count model, Satterthwaite df for every other linear
// mixed model, Wald z for a binary/count model with crossed or nested groupings.
// 'between_within' is the between-cluster df for a cluster-level predictor and
// the within-cluster df otherwise, and needs a single grouping factor;
// 'satterthwaite' is per-fit Satterthwaite df on a linear (Gaussian) mixed model
// only; 'asymptotic' is the Wald z against the standard normal. Mirrors
// wald-se-options.ts in structure; the default lives in configs/config.json.
export type DfMethod = 'auto' | 'between_within' | 'satterthwaite' | 'asymptotic';

export const DF_METHOD_LABEL: Record<DfMethod, string> = {
  auto: 'Automatic (default)',
  between_within: 'Between-within df',
  satterthwaite: 'Satterthwaite df',
  asymptotic: 'Wald z (asymptotic)',
};

export const DF_METHOD_OPTIONS: readonly DfMethod[] = [
  'auto',
  'between_within',
  'satterthwaite',
  'asymptotic',
] as const;

/** The values defined on a design: `linear` for a Gaussian outcome (a linear
 *  mixed model), `singleGrouping` when there is no crossed or nested grouping.
 *  The engine refuses the others. */
export function dfMethodOptionsFor(linear: boolean, singleGrouping: boolean): DfMethod[] {
  return DF_METHOD_OPTIONS.filter(
    (d) => (d !== 'between_within' || singleGrouping) && (d !== 'satterthwaite' || linear),
  );
}

/** Why a stored value (for example from a saved configuration) is not defined
 *  on the design, or `null` when it is. */
export function dfMethodError(
  value: DfMethod,
  linear: boolean,
  singleGrouping: boolean,
): string | null {
  if (dfMethodOptionsFor(linear, singleGrouping).includes(value)) return null;
  const fix = `choose "${DF_METHOD_LABEL.auto}" under Denominator df`;
  return value === 'between_within'
    ? `Denominator df: "${DF_METHOD_LABEL.between_within}" needs a single grouping factor, ` +
        `and this model has crossed or nested groupings; ${fix}`
    : `Denominator df: "${DF_METHOD_LABEL.satterthwaite}" needs a continuous outcome, ` +
        `and a binary or count mixed model has no Satterthwaite df; ${fix}`;
}
