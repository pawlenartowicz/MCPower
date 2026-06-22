//! `Distribution` enum — synthetic marginal distribution codes for non-factor columns; append-only.

use serde::{Deserialize, Serialize};

/// Typed discriminator for non-factor column distribution in
/// `engine_core::SimulationSpec::var_types`. Parameter values (e.g. binary
/// probability) live in the parallel `var_params: Vec<f64>` slot, not on the
/// enum itself — `Distribution` stays `Copy` so the kernel hot path can
/// `match` it like the integer code it replaces.
///
/// Mirrors the contract-side `SyntheticKind` for synthetic columns and
/// extends it with the three `Uploaded*` variants the kernel routes through
/// uploaded-frame slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    Normal,
    Binary,
    RightSkewed,
    LeftSkewed,
    HighKurtosis,
    Uniform,
    UploadedFactor,
    UploadedBinary,
    UploadedData,
}
