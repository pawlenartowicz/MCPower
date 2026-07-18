//! engine-app-spec — host-agnostic adapter for the MCPower engine.
//!
//! Wire format: `AppSpec` (tagged enum per family) → `Vec<SimulationContract>`
//! → orchestrator. Progress events flow through the family-agnostic
//! `ProgressEmitter` trait; each host binds it to its own transport
//! (Tauri events, JS callback, …).

#![forbid(unsafe_code)]

pub mod app_spec;
pub mod assemble;
pub mod driver;
pub mod error;
pub mod formula_parse;
pub mod plot;
pub mod progress;

pub use app_spec::{
    AppSpec, BinaryLink, ClusterDim, CorrelationMatrix, CsvData, EffectSize, LinearSpec, LogitSpec,
    MixedOutcome, MixedSpec, NumericDistribution, OutcomeOptions, ParsedFormula, PoissonSpec,
    TestSelection, VarType,
};
// Re-export upload types so crate consumers can build CsvData without an
// extra engine-spec-builder dependency.
pub use assemble::{assemble_spec, assemble_spec_with_skeleton, effect_skeleton_json};
pub use engine_spec_builder::input::{UploadColumn, UploadColumnType, UploadMode};
// Re-export the index-only effect skeleton so hosts can name results from it.
pub use driver::{
    get_effects_from_data, run_find_power, run_find_sample_size, run_single_core_find_power,
    run_single_core_find_sample_size, set_n_threads, EffectsFromData,
};
pub use engine_spec_builder::{EffectDescriptor, EffectSkeleton};
pub use error::AdapterError;
pub use formula_parse::{parse_formula, FormulaParse, RandomEffectJson, TermJson};
pub use plot::{power_plot_specs, sample_size_curve_specs, PlotBlock, PlotSpecs};
pub use progress::{EmitterSink, NullEmitter, ProgressEmitter};
