//! Build-time well-formedness guard for the author-controlled plot themes in
//! `configs/plot-themes.json`; enforces root-is-object only.

use crate::error::ContractError;

/// Parse + minimal-validate a Vega-Lite `config` block as JSON. Returns the
/// parsed `serde_json::Value` so callers can merge into a spec without
/// re-parsing. A build-time well-formedness guard for the author-controlled
/// themes in `configs/plot-themes.json` — it enforces only that the root is a
/// JSON object. There is no untrusted-theme ingestion path; if one is ever
/// added, validation gets rewritten and wired to that point.
pub fn validate_plot_theme(json_bytes: &[u8]) -> Result<serde_json::Value, ContractError> {
    let value: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| ContractError::InvalidPlotTheme(e.to_string()))?;
    if !value.is_object() {
        return Err(ContractError::InvalidPlotTheme(
            "plot theme root must be a JSON object".to_string(),
        ));
    }
    Ok(value)
}
