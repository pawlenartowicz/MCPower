//! Host-agnostic, label-free contract types (POD + serde); `SimulationContract::validate()` is the sole gate.

#![forbid(unsafe_code)]

pub mod config;
pub mod contract;
pub mod design;
pub mod distribution;
pub mod error;
pub mod estimator;
pub mod generation;
pub mod ids;
pub mod outcome;
pub mod plot_theme;
pub mod scenarios;
pub mod scenarios_config;
pub mod test_spec;
pub mod validate;

#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixtures;

pub use config::{
    config, validate_config, BaselineScenarioRule, Benchmarks, ByStep, Config, Limits, NSims,
    ReportConfig, ReportFormat, ReportThresholds, SampleSizeBounds, Simulation, UploadConfig,
    CONFIG_JSON,
};
pub use contract::SimulationContract;
pub use design::{DesignSpec, DesignTerm};
pub use distribution::Distribution;
pub use error::ContractError;
pub use estimator::EstimatorSpec;
pub use generation::{
    ClusterSizing, ClusterSpec, ColumnSpec, Correlations, GenerationSpec, GroupingRelation,
    GroupingSpec, SlopeTerm, SyntheticKind, UploadedFrame, MAX_EXTRA_GROUPINGS, MAX_PRIMARY_Q,
};
pub use ids::ColumnId;
pub use outcome::{OutcomeKind, OutcomeSpec, ResidualDist, ResidualSpec};
pub use scenarios::{LmeScenarioPerturbations, ScenarioPerturbations};
pub use scenarios_config::validate_scenarios;
pub use test_spec::{CorrectionMethod, PosthocSpec, TestSpec, TestTarget};
pub use validate::validate;
