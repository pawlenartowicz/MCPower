//! Translate `engine_core::EngineError` to PyO3 exception types.

use engine_core::EngineError;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Report-this suffix for genuine engine bugs (caught Rust panics) — never added
/// to the validation (`ValueError`) or cancellation (`KeyboardInterrupt`) paths.
/// `port=py`; version is the engine crate version. Mirrors engine-r's hint
/// (progress.rs `REPORT_HINT`) — change the two together.
const REPORT_HINT: &str = concat!(
    " — this looks like an internal MCPower error; please report it at https://mcpower.app/report?port=py&version=",
    env!("CARGO_PKG_VERSION")
);

/// Message for a caught engine panic, with the report hint appended.
pub fn internal_error_message(panic_msg: &str) -> String {
    format!("internal engine error: {panic_msg}{REPORT_HINT}")
}

/// Convert a caught panic payload into a `RuntimeError` carrying the report hint.
/// Without this the panic would surface as a bare PyO3 `PanicException`; here it
/// becomes an actionable "please report" message. Distinct from the
/// validation/cancellation mappings below — those user paths keep their types.
pub fn panic_to_py(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "engine panicked".to_string());
    PyRuntimeError::new_err(internal_error_message(&msg))
}

/// Map a Rust engine error onto the appropriate Python exception.
///
/// - `InvalidSpec` / `CorrelationNotPSD` / `RankDeficient` → `ValueError`
/// - `Cancelled` → `KeyboardInterrupt`
pub fn engine_err_to_py(e: EngineError) -> PyErr {
    match e {
        EngineError::InvalidSpec(s) => PyValueError::new_err(s),
        EngineError::CorrelationNotPSD => {
            PyValueError::new_err("correlation matrix is not positive semi-definite")
        }
        EngineError::RankDeficient(n) => {
            PyValueError::new_err(format!("rank-deficient design at N={n}"))
        }
        EngineError::Cancelled => PyKeyboardInterrupt::new_err("cancelled"),
    }
}
