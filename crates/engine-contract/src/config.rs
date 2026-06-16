//! Unified scalar build-constant config — `configs/config.json`.
//! Reuses the former `ReportConfig`/`UploadConfig` shapes (moved here verbatim);
//! the former `ClusterConstraints` fields are absorbed into `Simulation`/`Limits`.

use crate::error::ContractError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Typed mirror of `configs/config.json` — field names match the JSON keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub simulation: Simulation,
    pub benchmarks: Benchmarks,
    pub limits: Limits,
    pub report: ReportConfig,
    pub upload: UploadConfig,
}

/// Simulation defaults: seed, α, target power, per-family sim counts, bounds.
// Not `Copy`: `sample_size_bounds.by` is `ByStep`, which owns a `String` in its `Auto` variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Simulation {
    pub seed: u64,
    pub alpha: f64,
    pub target_power: f64,
    pub n_sims: NSims,
    pub max_failed_fraction: f64,
    pub sample_size_bounds: SampleSizeBounds,
    pub cluster_auto_count: u32,
}

/// Default sim counts per model family.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct NSims {
    pub ols: u32,
    pub mixed: u32,
    pub anova: u32,
}

/// Default `find_sample_size` search range.
// Not `Copy`: `by` is `ByStep`, which owns a `String` in its `Auto` variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleSizeBounds {
    pub from: u32,
    pub to: u32,
    pub by: ByStep,
}

/// `by` is either a fixed integer step or the sentinel string `"auto"`.
/// Untagged: an integer parses as `Fixed`, a string as `Auto`. The config is
/// author-controlled and embedded at build time, so the shape dispatch is the
/// only guard needed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ByStep {
    Fixed(u32),
    Auto(String),
}

/// Effect-size benchmark triples, ordered `[small, medium, large]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Benchmarks {
    pub continuous: [f64; 3],
    pub binary_factor: [f64; 3],
}

/// Validation limits shared by every host.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Limits {
    /// Soft-warn threshold for the significance level: hosts accept any alpha in
    /// `(0, 1)` (the hard reject is contract `invariant_15`) but warn when
    /// `alpha > max_alpha`, since a too-large alpha yields a near-meaningless power.
    pub max_alpha: f64,
    /// `[lo, hi]` stable-ICC band. A nonzero ICC outside this band is rejected
    /// host-side for numerical stability; ICC `0` (no clustering) is always allowed.
    pub icc_stability: [f64; 2],
    /// `[lo, hi]` reasonable baseline-probability band for GLM/GLMM. A baseline `p`
    /// outside `(0, 1)` is rejected; one merely outside this band is soft-warned.
    pub baseline_p_warn: [f64; 2],
    /// `[min, max]` allowed factor levels.
    pub factor_levels: [u32; 2],
    pub min_clusters: u32,
    pub min_rows_per_cluster: u32,
    pub reliable_rows_per_cluster: u32,
    pub recommended_rows_per_cluster: u32,
    /// Minimum observations per factor level for the factor to enter a fit;
    /// 0 disables exclusion. See `SimulationSpec::factor_min_level_count`.
    #[serde(default = "default_factor_min_level_count")]
    pub factor_min_level_count: u32,
}

// ---- moved verbatim from report_config.rs ----
/// Host report rendering: number formats, health thresholds, baseline pick.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportConfig {
    pub format: ReportFormat,
    pub thresholds: ReportThresholds,
    pub baseline_scenario: BaselineScenarioRule,
    pub overall_label_by_estimator: BTreeMap<String, String>,
}

/// Decimal places per report surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportFormat {
    pub power_decimals_short: u8,
    pub power_decimals_long: u8,
    pub target_decimals: u8,
    pub drop_decimals: u8,
    pub joint_table_decimals: u8,
}

/// Result-health warning thresholds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ReportThresholds {
    pub convergence_min: f64,
    pub lme_boundary_hit_max: f64,
    pub glm_baseline_drift_max: f64,
    /// Max tolerated per-factor exclusion rate; 0.0 = warn on any exclusion
    /// ("silent when healthy").
    #[serde(default = "default_factor_exclusion_max")]
    pub factor_exclusion_max: f64,
    /// GLMM Laplace-bias warning trigger: hosts warn when the estimated
    /// random-intercept variance τ̂² exceeds this AND the minimum cluster size is
    /// below `limits.reliable_rows_per_cluster`. Thresholds + copy are host-owned.
    #[serde(default = "default_glmm_tau_sq_warn")]
    pub glmm_tau_sq_warn: f64,
}

/// Which scenario the report treats as baseline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineScenarioRule {
    pub prefer_label: String,
    pub fallback_to_first: bool,
}

// ---- moved verbatim from upload_config.rs ----
/// Upload-validator limits shared by every host (native vs WASM row caps,
/// factor-cardinality guards).
// Note: `Eq` is intentionally absent — `strict_warning_ratio: f64` is not `Eq`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct UploadConfig {
    pub max_rows: u32,
    pub max_rows_wasm: u32,
    /// Minimum uploaded rows accepted by any host (degenerate-fit guard).
    pub min_rows: u32,
    pub max_factor_k_soft: u32,
    pub max_factor_ratio: u32,
    #[serde(default = "default_strict_warning_ratio")]
    pub strict_warning_ratio: f64,
}

fn default_strict_warning_ratio() -> f64 {
    2.0
}

fn default_factor_min_level_count() -> u32 {
    5
}

fn default_factor_exclusion_max() -> f64 {
    0.0
}

fn default_glmm_tau_sq_warn() -> f64 {
    1.0
}

/// Parse + shape-validate config JSON bytes.
///
/// # Errors
/// `InvalidConfig` on malformed JSON or a shape mismatch.
pub fn validate_config(json_bytes: &[u8]) -> Result<Config, ContractError> {
    serde_json::from_slice(json_bytes).map_err(|e| ContractError::InvalidConfig(e.to_string()))
}

/// The workspace `configs/config.json`, embedded at build time. Single source
/// of truth surfaced to every host (via the orchestrator re-export) and read
/// directly by the spec-builder for its business limits — never duplicated.
pub const CONFIG_JSON: &str = include_str!("../../../configs/config.json");

/// Parse the embedded config. Panics only if the embedded JSON is malformed,
/// which the build-time test `embedded_config_validates` prevents from shipping.
pub fn config() -> Config {
    validate_config(CONFIG_JSON.as_bytes()).expect("embedded config.json must validate")
}

#[cfg(test)]
mod embed_tests {
    #[test]
    fn embedded_config_validates() {
        let cfg = super::config();
        assert_eq!(cfg.limits.max_alpha, 0.25);
        assert_eq!(cfg.limits.icc_stability, [0.05, 0.95]);
        assert_eq!(cfg.limits.baseline_p_warn, [0.05, 0.95]);
        assert_eq!(cfg.limits.factor_levels, [2, 20]);
        assert_eq!(cfg.limits.factor_min_level_count, 5);
        assert_eq!(cfg.upload.min_rows, 20);
        assert_eq!(cfg.report.thresholds.factor_exclusion_max, 0.0);
    }
}
