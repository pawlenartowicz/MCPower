//! Builder: user-friendly inputs → `engine_core::SimulationSpec`.
//!
//! Host-agnostic. Consumed by `engine-py` (via PyO3 wrapper) and
//! `engine-app-spec` (shared by Tauri + WASM); future `engine-r` will use it too.

#![forbid(unsafe_code)]

pub mod assignments;
pub mod correlation;
pub mod error;
pub mod formula;
pub mod input;
pub mod project_contract;
pub mod skeleton;
pub mod targets;
pub mod upload;
pub mod validate;
pub mod variables;

pub use assignments::{
    parse_assignments, split_assignments, AssignmentKey, AssignmentKind, AssignmentValue,
    Assignments, KnownNames, VarTypeParams,
};
pub use error::SpecError;
pub use formula::{parse_formula, ParsedFormula, RandomEffect, Term};
pub use input::{
    EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec, ResidualSpec,
    UploadColumn, UploadColumnType, UploadInput, UploadMode, VarKind,
};
pub use project_contract::{
    build_contract, build_contract_with_skeleton, build_linear_contract,
    build_linear_contract_with_skeleton, dist_codes_json, re_dist_codes_json,
    residual_codes_json,
};
pub use skeleton::{EffectDescriptor, EffectSkeleton};
pub use upload::{build_recovery_design, standardize_continuous, RecoveryDesign};
pub use variables::{build_predictor_table, PredictorTable};
