//! Expose the embedded `configs/config.json` to Python.
//! Mirror: `engine-r/src/report_bridge.rs` — change together.

use pyo3::prelude::*;

/// Return the embedded `configs/config.json` as a JSON string. Python parses
/// and caches it (`mcpower/config.py`). Single source of truth; the Python port
/// keeps no copy.
#[pyfunction]
pub fn config() -> String {
    engine_orchestrator::CONFIG_JSON.to_string()
}

/// Return the embedded `configs/scenarios.json` (alias/flat/object-keyed) as a
/// JSON string. Python parses and caches it (`mcpower/scenario_config.py`).
/// Single source of truth; the Python port keeps no copy.
#[pyfunction]
pub fn scenarios() -> String {
    engine_orchestrator::SCENARIOS_CONFIG_JSON.to_string()
}
