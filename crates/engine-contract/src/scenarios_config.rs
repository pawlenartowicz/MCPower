//! Schema validator for `configs/scenarios.json`; parses the pre-normalized perturbation list without renormalizing.

use crate::error::ContractError;
use crate::scenarios::ScenarioPerturbations;

/// Parse + validate a workspace `configs/scenarios.json` byte buffer.
///
/// Returns the parsed `Vec<ScenarioPerturbations>`; callers reuse the parsed
/// value rather than re-parsing. The JSON literal carries pre-normalized
/// factor proportion values; validation does NOT renormalize.
pub fn validate_scenarios(json_bytes: &[u8]) -> Result<Vec<ScenarioPerturbations>, ContractError> {
    let parsed: Vec<ScenarioPerturbations> = serde_json::from_slice(json_bytes)
        .map_err(|e| ContractError::InvalidScenariosConfig(e.to_string()))?;
    Ok(parsed)
}
