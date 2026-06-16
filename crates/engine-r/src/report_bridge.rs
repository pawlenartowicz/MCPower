//! Exposes `config` to R: returns the embedded `configs/config.json` string so the R port shares one source of truth for all scalar build constants.
//! Mirror: `engine-py/src/report_bridge.rs` — change together.
use extendr_api::prelude::*;

/// Return the embedded `configs/config.json` as a JSON string. R parses it
/// with jsonlite and caches it in `.onLoad` (`zzz.R`). Single source of truth.
#[extendr]
pub fn config() -> String {
    engine_orchestrator::CONFIG_JSON.to_string()
}

/// Return the embedded `configs/scenarios.json` (alias/flat/object-keyed) as a
/// JSON string. R parses it with jsonlite and caches it (`zzz.R` `.scenario_defaults()`).
/// Single source of truth for default scenario configs; the R port keeps no copy.
#[extendr]
pub fn scenarios() -> String {
    engine_orchestrator::SCENARIOS_CONFIG_JSON.to_string()
}

extendr_module! {
    mod report_bridge;
    fn config;
    fn scenarios;
}
