//! Display helper entries: standardized β, Cohen's d and R² of a contract's
//! generating model, and the β-free moments they are computed from. Not run
//! entries — no simulation, no single-core twin, no merge. The definitions
//! and the moment model live in `engine_core::standardized`.

pub use engine_core::standardized::{
    EffectMoments, GroupMoments, StandardizedEffect, StandardizedEffects,
};

use crate::result::OrchestratorError;
use engine_contract::SimulationContract;

/// Standardized effects of `contract`'s generating model, one per
/// `design_generation.terms` entry (intercept excluded, in that order), plus
/// the model R². Identical for every scenario of a run (scenario perturbations
/// play no part), so a host calls it once per run with any one contract.
///
/// # Errors
/// `InvalidScenarios` when the contract fails lowering (`validate()`).
pub fn standardized_effects(
    contract: &SimulationContract,
) -> Result<StandardizedEffects, OrchestratorError> {
    engine_core::standardized::standardized_effects(contract)
        .map_err(|e| OrchestratorError::InvalidScenarios(e.to_string()))
}

/// The β-free moments behind [`standardized_effects`], for a host that
/// re-evaluates the values on every effect-size edit without another engine
/// call; `None` where the values are undefined. Scenario-independent, like
/// `standardized_effects`.
///
/// # Errors
/// `InvalidScenarios` when the contract fails lowering (`validate()`).
pub fn effect_moments(
    contract: &SimulationContract,
) -> Result<Option<EffectMoments>, OrchestratorError> {
    engine_core::standardized::effect_moments(contract)
        .map_err(|e| OrchestratorError::InvalidScenarios(e.to_string()))
}
