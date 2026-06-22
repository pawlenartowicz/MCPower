//! Engine core for MCPower2 — pure Rust, host-agnostic.
//!
//! No PyO3, no numpy. The `engine-py` crate provides the PyO3 adapter; an
//! eventual `engine-r` crate will provide the extendr adapter. Both depend
//! on this crate.

#![forbid(unsafe_code)]

// Swap the global allocator for `dhat`'s during tests so we can assert
// allocation counts on the OLS warm path (see ols::tests::alloc_tests).
// `#[global_allocator]` must live at the crate root, so it goes here even
// though only the OLS module exercises it.
#[cfg(test)]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod batch;
pub mod contract_adapter;
pub mod correction;
pub mod correlation;
pub mod critvals;
pub mod data_gen;
pub mod distributions;
pub mod glm;
pub mod glmm;
pub mod introspect;
pub mod lme;
pub mod lmm;
pub mod marginals;
pub mod ols;
pub mod philox;
pub mod posthoc;
pub mod rng;
pub mod scenarios;
mod simd_transcendental;
pub mod spec;
pub mod workspace;

pub use batch::{run_batch, run_batch_st, set_n_threads};
pub use data_gen::{fixed_allocation_counts, min_inclusion_n};
pub use spec::{
    BatchResult, ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution,
    EngineError, EstimatorSpec, LmeScenarioPerturbations, OutcomeKind, PosthocBlockShape,
    PosthocSpec, ProgressSink, ResidualDist, ResultShape, ScenarioPerturbations, SimulationSpec,
};

/// Tiny float guard — magnitudes below this are treated as zero (rank /
/// division-by-zero sentinel). Mirrors v1's FLOAT_NEAR_ZERO; comparing
/// `β̂²/var_diag` against thresholds with NaN propagates as "fail" downstream.
pub(crate) const FLOAT_NEAR_ZERO: f64 = 1e-30;
