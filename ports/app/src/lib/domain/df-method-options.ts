// Reference distribution for the fixed-effect Wald tests of a mixed model.
// 'between_within' uses the between-within denominator degrees of freedom, which
// is the between-cluster df for a cluster-level predictor and the within-cluster
// df otherwise; 'asymptotic' is the Wald z against the standard normal. Mirrors
// wald-se-options.ts in structure; the default lives in configs/config.json.
export type DfMethod = 'asymptotic' | 'between_within';

export const DF_METHOD_LABEL: Record<DfMethod, string> = {
  between_within: 'Between-within df (default)',
  asymptotic: 'Wald z (asymptotic)',
};

export const DF_METHOD_OPTIONS: readonly DfMethod[] = ['between_within', 'asymptotic'] as const;
