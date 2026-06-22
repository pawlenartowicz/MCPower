// Typed wrapper around the shared `mcpower/configs/config.json`; selects `.report`; no per-port copy is kept.
import config from '$configs/config.json';

/** Build-time snapshot of the `report` section from `mcpower/configs/config.json`. Single source of truth — no per-port copy. */
export interface ReportConfig {
  format: {
    power_decimals_short: number;
    power_decimals_long: number;
    target_decimals: number;
    drop_decimals: number;
    joint_table_decimals: number;
  };
  thresholds: {
    convergence_min: number;
    lme_boundary_hit_max: number;
    glm_baseline_drift_max: number;
    // Declared here so ConvergenceNotice reads typecheck; values from config.json.
    factor_exclusion_max: number;
    glmm_tau_sq_warn: number;
  };
  baseline_scenario: { prefer_label: string; fallback_to_first: boolean };
  overall_label_by_estimator: Record<string, string>;
  /** Section captions + column labels (shared with Py/R via config.json — no
   *  per-port copy). The GUI reuses these for table captions and figure headings. */
  text: {
    long_title: string;
    main_caption: string;
    ci_caption: string;
    sample_size_caption: string;
    estimator_extras_caption: string;
    vs_token: string;
    columns: {
      test: string;
      power: string;
      target: string;
      ci: string;
      required_n: string;
      uncorrected: string;
      corrected: string;
    };
  };
}

export const REPORT_CONFIG: ReportConfig = config.report as ReportConfig;
