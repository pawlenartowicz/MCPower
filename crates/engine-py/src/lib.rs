//! PyO3 adapter for the MCPower2 Rust engine.
//!
//! Mirror crate: `engine-r` exposes the same bridge surface for R — keep the
//! bridge files in step (engine-r cites this crate file-by-file). File sets
//! differ deliberately: `errors.rs` (exception mapping) is Python-only, and
//! the debug surface lives in `orchestrator_bridge.rs` here but in
//! `debug_bridge.rs` on the R side.
//!
//! Translation rules:
//!   - `SimulationSpec` arrives as a msgpack-encoded byte buffer (the Python
//!     frontend assembles the dict and calls `msgpack.packb(spec)`).
//!   - Per-scenario simulation specs are bundled into a msgpack
//!     `[(label, spec_dict, base_seed), ...]` payload and dispatched to the
//!     Rust orchestrator via `find_power` / `find_sample_size`.
//!   - Progress / cancel flows through a Python callable
//!     `fn(current: int, total: int) -> bool`.
//!
//! All heavy work happens inside `Python::detach` so other Python
//! threads can run in parallel during engine calls.

use pyo3::prelude::*;

mod errors;
mod orchestrator_bridge;
mod plot_bridge;
mod report_bridge;
mod spec_builder_bridge;

use crate::errors::engine_err_to_py;
use crate::orchestrator_bridge::{
    decode_contracts, power_result_to_pydict, sample_size_result_to_pydict,
    OrchestratorProgressSink,
};

/// Configure the rayon thread pool. Must be called *before* any engine
/// invocation; subsequent calls raise `ValueError`.
#[pyfunction]
fn set_n_threads(n: usize) -> PyResult<()> {
    engine_core::set_n_threads(n).map_err(engine_err_to_py)
}

/// Run a power simulation for every scenario in `contracts_bytes` (a msgpack
/// `Vec<SimulationContract>` blob) at one `sample_size`. `progress` is an
/// optional Python callable `(current, total) -> bool` (return `False` to
/// cancel). Returns the power-result dict (see `power_result_to_pydict`).
#[pyfunction]
#[pyo3(signature = (contracts_bytes, sample_size, n_sims, base_seed, progress=None))]
fn find_power(
    py: Python<'_>,
    contracts_bytes: &[u8],
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let contracts = decode_contracts(contracts_bytes)?;
    let cancel = engine_orchestrator::CancellationToken::new();
    let mut sink = OrchestratorProgressSink::new(progress, &cancel);

    // catch_unwind turns a Rust engine panic into an actionable RuntimeError
    // (with the report hint) instead of a bare PanicException — mirrors
    // engine-r's `run_engine` (progress.rs), change together.
    let outcome = py.detach(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine_orchestrator::find_power(
                &contracts,
                sample_size,
                n_sims,
                base_seed,
                Some(&mut sink as &mut dyn engine_orchestrator::ProgressSink),
                &cancel,
            )
        }))
    });
    let result = match outcome {
        Ok(r) => r.map_err(orchestrator_err_to_py)?,
        Err(payload) => return Err(crate::errors::panic_to_py(payload)),
    };
    let dict = power_result_to_pydict(py, &result)?;
    Ok(dict)
}

/// Search for the sample size reaching `target_power` over `[lo, hi]` for every
/// scenario in `contracts_bytes`. `method` + (`by`, `by_kind`, `mode`) select the
/// `SampleSizeMethod`; `progress` is an optional `(current, total) -> bool` Python
/// callable. Returns the sample-size-result dict (see `sample_size_result_to_pydict`).
#[pyfunction]
#[pyo3(signature = (contracts_bytes, target_power, lo, hi, n_sims, base_seed, method, by=None, by_kind=None, mode=None, tol_n=None, progress=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 entry point; signature mirrors the Python keyword API"
)]
fn find_sample_size(
    py: Python<'_>,
    contracts_bytes: &[u8],
    target_power: f64,
    lo: usize,
    hi: usize,
    n_sims: usize,
    base_seed: u64,
    method: &str,
    by: Option<usize>,
    by_kind: Option<&str>,
    mode: Option<&str>,
    tol_n: Option<usize>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let contracts = decode_contracts(contracts_bytes)?;
    let cancel = engine_orchestrator::CancellationToken::new();
    let mut sink = OrchestratorProgressSink::new(progress, &cancel);
    let _ = tol_n; // bisection-only; retained for FFI signature stability.

    let method = engine_orchestrator::SampleSizeMethod::from_host_args(method, by, by_kind, mode)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Panic boundary — see find_power (mirrors engine-r's run_engine).
    let outcome = py.detach(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine_orchestrator::find_sample_size(
                &contracts,
                target_power,
                (lo, hi),
                n_sims,
                method,
                base_seed,
                Some(&mut sink as &mut dyn engine_orchestrator::ProgressSink),
                &cancel,
            )
        }))
    });
    let result = match outcome {
        Ok(r) => r.map_err(orchestrator_err_to_py)?,
        Err(payload) => return Err(crate::errors::panic_to_py(payload)),
    };
    let dict = sample_size_result_to_pydict(py, &result)?;
    Ok(dict)
}

// Mirrors engine-r's `orchestrator_err` (progress.rs) — keep the error→host
// message mapping in step (both collapse cancellation to the host's interrupt).
fn orchestrator_err_to_py(e: engine_orchestrator::OrchestratorError) -> PyErr {
    use engine_orchestrator::OrchestratorError as E;
    match e {
        E::Engine(inner) => crate::errors::engine_err_to_py(inner),
        E::InvalidScenarios(s) => pyo3::exceptions::PyValueError::new_err(s),
        E::InvalidGridBounds { from, to, .. } => pyo3::exceptions::PyValueError::new_err(format!(
            "invalid grid bounds: from={from}, to={to}"
        )),
        E::Cancelled { .. } => pyo3::exceptions::PyKeyboardInterrupt::new_err("cancelled"),
        // Merge-only error — not reachable from the Python adapter today
        // (single_core_find_power / merge_power_results aren't bridged yet).
        // Mapped to PyValueError so a future bridge surfaces a clear message.
        E::IncompatibleMerge(msg) => pyo3::exceptions::PyValueError::new_err(format!(
            "cannot merge incompatible results: {msg}"
        )),
        // New cluster-guard variants — map to PyValueError with the Display message.
        other @ (E::InvalidClusterAtom
        | E::ClusterGridEmpty { .. }
        | E::ClusterGridSinglePoint { .. }
        | E::MixedClusterAtoms { .. }
        | E::ClusterSizeTooSmall { .. }
        | E::ClusterTooFewAtN { .. }) => pyo3::exceptions::PyValueError::new_err(other.to_string()),
    }
}

/// Test-only seam: panic inside the engine boundary so tests can assert the
/// internal-error path raises `RuntimeError` with the report hint. Gated on
/// `test-bridge` (stripped from release wheels), mirroring the
/// `find_power_precancelled` seam in engine-r. Surfaced as
/// `mcpower._engine.panic_for_test`.
#[cfg(feature = "test-bridge")]
#[pyfunction]
fn panic_for_test() -> PyResult<()> {
    match std::panic::catch_unwind(|| panic!("simulated internal engine failure")) {
        Ok(()) => Ok(()),
        Err(payload) => Err(crate::errors::panic_to_py(payload)),
    }
}

#[pymodule]
fn _engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_power, m)?)?;
    m.add_function(wrap_pyfunction!(find_sample_size, m)?)?;
    m.add_function(wrap_pyfunction!(set_n_threads, m)?)?;
    #[cfg(feature = "test-bridge")]
    m.add_function(wrap_pyfunction!(
        spec_builder_bridge::build_contract_from_json,
        m
    )?)?;
    #[cfg(feature = "test-bridge")]
    m.add_function(wrap_pyfunction!(panic_for_test, m)?)?;
    m.add_function(wrap_pyfunction!(
        spec_builder_bridge::build_contract_from_spec,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::parse_formula, m)?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::parse_assignments, m)?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::dist_codes, m)?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::residual_codes, m)?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::re_dist_codes, m)?)?;
    m.add_function(wrap_pyfunction!(
        spec_builder_bridge::build_recovery_design,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spec_builder_bridge::standardize_continuous,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spec_builder_bridge::split_assignments, m)?)?;
    m.add_function(wrap_pyfunction!(plot_bridge::power_plot_set_json, m)?)?;
    m.add_function(wrap_pyfunction!(plot_bridge::sample_size_plot_set_json, m)?)?;
    m.add_function(wrap_pyfunction!(plot_bridge::plot_theme, m)?)?;
    m.add_function(wrap_pyfunction!(plot_bridge::list_plot_themes, m)?)?;
    m.add_function(wrap_pyfunction!(plot_bridge::plot_html_template, m)?)?;
    m.add_function(wrap_pyfunction!(report_bridge::config, m)?)?;
    m.add_function(wrap_pyfunction!(report_bridge::scenarios, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrator_bridge::fit_uploaded_data, m)?)?;
    Ok(())
}
