import { describe, it, expect } from 'vitest';
import { REPORT_CONFIG } from './report-config';

describe('REPORT_CONFIG', () => {
  it('mirrors the canonical report.json values', () => {
    expect(REPORT_CONFIG.format.power_decimals_long).toBe(1);
    expect(REPORT_CONFIG.thresholds.convergence_min).toBe(0.95);
    expect(REPORT_CONFIG.baseline_scenario.prefer_label).toBe('optimistic');
    expect(REPORT_CONFIG.overall_label_by_estimator.ols).toBe('Overall F');
  });
});
