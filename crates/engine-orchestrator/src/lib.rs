//! Engine orchestrator for MCPower2 — host-agnostic.
//!
//! Owns the scenario loop, aggregation, Wilson CIs, progress events, and
//! cancellation. Wraps `engine_core::run_batch` per scenario / per N point.

#![forbid(unsafe_code)]

pub mod aggregation;
pub mod cancel;
pub mod debug;
pub mod find_power;
pub mod find_sample_size;
mod fit;
pub mod grid;
pub mod merge;
pub mod plot;
pub mod progress;
pub mod result;
pub mod result_host;
mod scenario_loop;
pub mod single_core;
pub mod unit;

pub use crate::cancel::CancellationToken;
pub use crate::find_power::find_power;
pub use crate::find_sample_size::find_sample_size;
pub use crate::merge::{merge_power_results, merge_sample_size_results};
pub use crate::plot::{power_plot_set, sample_size_plot_set, PlotOptions, PlotPoint, PlotScenario};
pub use crate::progress::{NoOpSink, ProgressEvent, ProgressSink};
pub use crate::result::{
    ByValue, Ci, CrossingFit, EstimatorExtras, GridMode, OrchestratorError, PosthocPower,
    PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};
pub use crate::result_host::{power_result_to_host, sample_size_result_to_host, HostValue};
pub use crate::single_core::{single_core_find_power, single_core_find_sample_size};
pub use crate::unit::{bounds_to_rows, resolve_unit_divisor, SampleSizeUnit};

pub use engine_contract::{
    config, validate_config, BaselineScenarioRule, Benchmarks, ByStep, Config, Limits, NSims,
    ReportConfig, ReportFormat, ReportThresholds, SampleSizeBounds, Simulation, UploadConfig,
    CONFIG_JSON,
};

/// The workspace `configs/scenarios.json`, embedded at build time. Single source
/// of truth for default scenario configs — never duplicated in a port.
pub const SCENARIOS_CONFIG_JSON: &str = include_str!("../../../configs/scenarios.json");

#[cfg(test)]
mod scenarios_embed_tests {
    use serde_json::Value;

    #[test]
    fn embedded_scenarios_canonical_shape() {
        let v: Value = serde_json::from_str(super::SCENARIOS_CONFIG_JSON)
            .expect("scenarios.json must be valid JSON");
        let obj = v.as_object().expect("scenarios.json must be a JSON object");
        for name in ["optimistic", "realistic", "doomer"] {
            let s = obj.get(name).and_then(|x| x.as_object()).expect(name);
            // flat invariants; random_effect_dist keeps its own
            // normal/heavy_tailed vocabulary — only residual pools are canonical
            assert!(s.contains_key("random_effect_dist"), "{name}: flat lme key");
            assert!(!s.contains_key("lme"), "{name}: must not be nested");
            let rd = s.get("residual_dists").and_then(|x| x.as_array()).unwrap();
            assert!(rd.iter().all(|x| x.as_str().is_some()));
            assert!(
                rd.iter().any(|x| x == "high_kurtosis"),
                "{name}: canonical residual pool"
            );
            assert!(
                !rd.iter()
                    .any(|x| x == "heavy_tailed" || x == "skewed" || x == "t"),
                "{name}: legacy alias must not appear in residual_dists"
            );
        }
    }
}
