//! Translate `engine_core::EngineError` to PyO3 exception types.

use engine_core::EngineError;
use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::PyErr;

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
