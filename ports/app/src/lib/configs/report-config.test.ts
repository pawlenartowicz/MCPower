import { describe, expect, it } from 'vitest';
import { REPORT_CONFIG, type ReportConfig } from './report-config';

// `config.report as ReportConfig` hides a key renamed or dropped in config.json:
// the read would silently yield `undefined` (e.g. a threshold compare that is
// never true). Walk every field of SHAPE and check config.json carries it with
// the declared type. SHAPE is tied to the ReportConfig interface only by the
// type checker (`pnpm check`), not by this test. Values are config.json's own
// data and are not pinned here.
type Shape<T> = {
  [K in keyof T]: T[K] extends object ? Shape<T[K]> : 'number' | 'string' | 'boolean';
};

const SHAPE: Shape<ReportConfig> = {
  format: {
    power_decimals_short: 'number',
    power_decimals_long: 'number',
    target_decimals: 'number',
    drop_decimals: 'number',
    joint_table_decimals: 'number',
  },
  thresholds: {
    convergence_min: 'number',
    lme_boundary_hit_max: 'number',
    factor_exclusion_max: 'number',
    glmm_tau_sq_warn: 'number',
  },
  baseline_scenario: { prefer_label: 'string', fallback_to_first: 'boolean' },
  // One label per engine estimator the result tables can carry.
  overall_label_by_estimator: { ols: 'string', glm: 'string', mle: 'string' },
  text: {
    long_title: 'string',
    main_caption: 'string',
    ci_caption: 'string',
    sample_size_caption: 'string',
    estimator_extras_caption: 'string',
    vs_token: 'string',
    columns: {
      test: 'string',
      power: 'string',
      target: 'string',
      ci: 'string',
      required_n: 'string',
      uncorrected: 'string',
      corrected: 'string',
    },
  },
};

function leaves(shape: object, prefix = ''): Array<[string, string]> {
  return Object.entries(shape).flatMap(([k, v]) =>
    typeof v === 'string'
      ? [[`${prefix}${k}`, v] as [string, string]]
      : leaves(v, `${prefix}${k}.`),
  );
}

describe('REPORT_CONFIG', () => {
  it.each(leaves(SHAPE))('config.json carries report.%s as a %s', (path, type) => {
    const value = path
      .split('.')
      .reduce<unknown>((o, k) => (o as Record<string, unknown>)?.[k], REPORT_CONFIG);
    expect(typeof value).toBe(type);
  });
});
