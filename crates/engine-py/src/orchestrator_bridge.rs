//! Glue between PyO3 and engine-orchestrator. Owns:
//!   - contract decode (msgpack Vec<SimulationContract> blob),
//!   - PyO3 ProgressSink (translates ProgressEvent → progress_callback(int,int))
//!     and the background-thread runner that feeds it and polls for Ctrl-C,
//!   - the generic `HostValue` → Python walker that realizes the
//!     single-sourced result shape (`engine_orchestrator::result_host`).
//!
//! Mirror: `engine-r/src/orchestrator_bridge.rs` — change together.

use engine_contract::SimulationContract;
use engine_orchestrator::{
    Ci, HostValue, PowerResult, ProgressEvent, ProgressSink, SampleSizeResult, ScenarioResult,
    power_result_to_host, sample_size_result_to_host,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::Duration;

/// Decode the msgpack `Vec<SimulationContract>` blob produced by
/// `_engine.build_contract_from_spec`. The Python frontend forwards this
/// blob verbatim into `find_power` / `find_sample_size`; per-scenario seed
/// derivation lives in the orchestrator.
pub fn decode_contracts(bytes: &[u8]) -> PyResult<Vec<SimulationContract>> {
    let contracts: Vec<SimulationContract> = rmp_serde::from_slice(bytes)
        .map_err(|e| PyValueError::new_err(format!("malformed contracts bytes: {e}")))?;
    if contracts.is_empty() {
        return Err(PyValueError::new_err("contracts list cannot be empty"));
    }
    Ok(contracts)
}

/// Bridges the orchestrator's structured event stream to the host progress
/// callable `progress(current: int, total: int) -> bool`. Fed on the calling
/// (Python) thread by [`run_with_progress`], never on engine workers.
///
/// Translation: `SimsCompleted { completed, total }` is already cumulative
/// across the whole call (fit-based, denominator = `Started.total_sims`), so
/// it forwards verbatim as `progress(completed, total)`. Other variants
/// ignored.
///
/// Cancellation: a `False` return (or an exception) cancels the run via
/// the shared `CancellationToken` passed in.
///
/// Mirrors engine-r's `CallbackState` (progress.rs) — keep the two in step.
pub struct OrchestratorProgressSink<'cancel> {
    py_callback: Option<Py<PyAny>>,
    cancel: &'cancel engine_orchestrator::CancellationToken,
}

impl<'cancel> OrchestratorProgressSink<'cancel> {
    pub fn new(
        py_callback: Option<Py<PyAny>>,
        cancel: &'cancel engine_orchestrator::CancellationToken,
    ) -> Self {
        Self {
            py_callback,
            cancel,
        }
    }
}

impl<'cancel> ProgressSink for OrchestratorProgressSink<'cancel> {
    fn on_event(&mut self, event: ProgressEvent) {
        let cb = match self.py_callback.as_ref() {
            Some(cb) => cb,
            None => return,
        };
        let (current, total) = match event {
            ProgressEvent::SimsCompleted {
                completed, total, ..
            } => (completed, total),
            _ => return,
        };
        let should_continue = Python::attach(|py| match cb.call1(py, (current, total)) {
            Ok(ret) => ret.extract::<bool>(py).unwrap_or(true),
            Err(_) => false,
        });
        if !should_continue {
            self.cancel.cancel();
        }
    }
}

/// Off-thread sink: forwards each event to the calling thread, which owns the
/// Python callback. Runs on engine (rayon) threads, so it touches no Python.
struct ForwarderSink {
    tx: Sender<ProgressEvent>,
}

impl ProgressSink for ForwarderSink {
    fn on_event(&mut self, event: ProgressEvent) {
        // A closed receiver means the caller is already tearing the run down.
        let _ = self.tx.send(event);
    }
}

/// Upper bound on how long the calling thread waits between Ctrl-C checks when
/// no events arrive. Events and engine completion wake it immediately.
const POLL_INTERVAL_MS: u64 = 50;

/// Run an orchestrator entry with Python progress delivery and Ctrl-C
/// cancellation, returning whatever `run` produced.
///
/// CPython runs signal handlers only on the main thread, and only when Python
/// code gets to run there, so the engine cannot be left to notice a
/// `KeyboardInterrupt` on its own (with progress off no Python runs at all).
/// Instead `run` executes on a scoped background thread while the calling
/// thread, with the GIL released between polls, waits on the event queue: it
/// feeds each event to `progress` and calls `check_signals`. A raised signal
/// flips `cancel`; the engine stops at its next checkpoint and the signal's
/// exception (normally `KeyboardInterrupt`) is returned in place of the result.
/// A cancel from the callback returns `KeyboardInterrupt("cancelled")`.
///
/// Mirrors engine-r's `run_with_progress` (progress.rs); change together.
pub fn run_with_progress<F, T>(
    py: Python<'_>,
    progress: Option<Py<PyAny>>,
    cancel: &engine_orchestrator::CancellationToken,
    run: F,
) -> PyResult<T>
where
    F: FnOnce(&mut dyn ProgressSink, &engine_orchestrator::CancellationToken) -> T + Send,
    T: Send,
{
    let mut sink = OrchestratorProgressSink::new(progress, cancel);
    let (tx, rx) = mpsc::channel();
    let mut signal_err: Option<PyErr> = None;
    // `move` with explicit borrows: the Receiver must move in (it is not Sync).
    let (sink, signal_err_slot) = (&mut sink, &mut signal_err);
    let out = py.detach(move || {
        std::thread::scope(|scope| {
            let handle = scope.spawn(move || run(&mut ForwarderSink { tx }, cancel));
            loop {
                // The engine thread owns the only Sender, so its exit wakes
                // this wait at once with `Disconnected`.
                let done = match rx.recv_timeout(Duration::from_millis(POLL_INTERVAL_MS)) {
                    Ok(event) => {
                        sink.on_event(event);
                        while let Ok(event) = rx.try_recv() {
                            sink.on_event(event);
                        }
                        false
                    }
                    Err(RecvTimeoutError::Timeout) => false,
                    Err(RecvTimeoutError::Disconnected) => true,
                };
                if signal_err_slot.is_none() {
                    if let Err(e) = Python::attach(|py| py.check_signals()) {
                        *signal_err_slot = Some(e);
                        cancel.cancel();
                    }
                }
                if done {
                    return handle.join().expect("orchestrator thread panicked");
                }
            }
        })
    });
    if let Some(e) = signal_err {
        return Err(e);
    }
    if cancel.is_cancelled() {
        // The callback runs behind the engine, so its cancel can land after
        // the engine's last checkpoint; it still cancels the call.
        return Err(PyKeyboardInterrupt::new_err("cancelled"));
    }
    Ok(out)
}

/// Convert a `ScenarioResult<PowerResult>` into the multi-scenario envelope.
///
/// Always returns `{"scenarios": {...}, "comparison": {...}}`. Single-scenario
/// callers unwrap the first entry in Python. The per-scenario shape is
/// single-sourced via `power_result_to_host`; this only walks the tree into
/// Python objects and wraps it in the label-keyed envelope.
pub fn power_result_to_pydict<'py>(
    py: Python<'py>,
    result: &ScenarioResult<PowerResult>,
) -> PyResult<Py<PyAny>> {
    scenarios_envelope(py, &result.scenarios, |pr, label| {
        host_value_to_py(py, &power_result_to_host(pr, label))
    })
}

/// Convert a `ScenarioResult<SampleSizeResult>` into the multi-scenario
/// envelope. Per-scenario shape single-sourced via `sample_size_result_to_host`;
/// this only walks the tree into Python objects and wraps it in the label-keyed
/// envelope.
pub fn sample_size_result_to_pydict<'py>(
    py: Python<'py>,
    result: &ScenarioResult<SampleSizeResult>,
) -> PyResult<Py<PyAny>> {
    scenarios_envelope(py, &result.scenarios, |ssr, label| {
        host_value_to_py(py, &sample_size_result_to_host(ssr, label))
    })
}

/// Build `{"scenarios": {label: <rendered>}, "comparison": {}}` from the
/// per-scenario `(label, T)` pairs, rendering each value via `render`.
fn scenarios_envelope<'py, T>(
    py: Python<'py>,
    scenarios: &[(String, T)],
    mut render: impl FnMut(&T, &str) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (label, value) in scenarios {
        dict.set_item(label, render(value, label)?)?;
    }
    let outer = PyDict::new(py);
    outer.set_item("scenarios", dict)?;
    outer.set_item("comparison", PyDict::new(py))?;
    Ok(outer.into_any().unbind())
}

/// Walk a `HostValue` into the Python object the host result shape expects.
fn host_value_to_py(py: Python<'_>, hv: &HostValue) -> PyResult<Py<PyAny>> {
    match hv {
        HostValue::F64(x) => (*x).into_py_any(py),
        HostValue::OptF64(o) => match o {
            Some(x) => (*x).into_py_any(py),
            None => Ok(py.None()),
        },
        HostValue::Usize(n) => (*n).into_py_any(py),
        HostValue::OptUsize(o) => match o {
            Some(n) => (*n).into_py_any(py),
            None => Ok(py.None()),
        },
        HostValue::VecF64(v) => v.clone().into_py_any(py),
        HostValue::VecU64(v) => v.clone().into_py_any(py),
        HostValue::VecStr(v) => v.clone().into_py_any(py),
        HostValue::VecBool(v) => v.clone().into_py_any(py),
        HostValue::VecCi(cis) => ci_slice_to_pylist(py, cis)?.into_py_any(py),
        HostValue::OptCi(o) => match o {
            Some(ci) => PyTuple::new(py, [ci.lo, ci.hi])?.into_py_any(py),
            None => Ok(py.None()),
        },
        HostValue::Str(s) => s.as_str().into_py_any(py),
        HostValue::Null => Ok(py.None()),
        HostValue::Map(pairs) => {
            let d = PyDict::new(py);
            for (k, v) in pairs {
                d.set_item(*k, host_value_to_py(py, v)?)?;
            }
            d.into_py_any(py)
        }
        HostValue::Seq(items) => {
            let list = PyList::empty(py);
            for it in items {
                list.append(host_value_to_py(py, it)?)?;
            }
            list.into_py_any(py)
        }
        HostValue::IndexMap(items) => {
            let d = PyDict::new(py);
            for (i, it) in items.iter().enumerate() {
                d.set_item(i, host_value_to_py(py, it)?)?;
            }
            d.into_py_any(py)
        }
        HostValue::BoundaryHit { flat, rows, cols } => {
            // Nested list-of-rows (row-major), not a numpy array: keeps numpy an
            // optional *input* convenience rather than a hard runtime dependency.
            let outer = PyList::empty(py);
            for r in 0..*rows {
                let row = PyList::new(py, &flat[r * *cols..(r + 1) * *cols])?;
                outer.append(row)?;
            }
            outer.into_py_any(py)
        }
    }
}

fn ci_slice_to_pylist<'py>(py: Python<'py>, cis: &[Ci]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for ci in cis {
        list.append(PyTuple::new(py, [ci.lo, ci.hi])?)?;
    }
    Ok(list)
}

/// Fit a provided dataset and return a dict matching the R `debug_load_data` shape:
/// `{betas, design_columns, converged, targets}` where each target is a dict with
/// `{target_index, target_label, beta, se, statistic, statistic_kind,
///  critical_value, alpha, df, two_sided}`.
///
/// `design` is column-major (col 0 first, then col 1, ...), length `nrow * ncol`.
/// Pass `cluster_ids=None` for non-clustered designs.
/// Parse the validation-only `wald_se` override string into an optional kernel
/// SE mode. `""`/`"default"` ⇒ `None` (keep the contract's configured mode);
/// `"rx"`/`"hessian"` force that mode. Reachable only via this raw bridge arg,
/// for the Oracle harness; no production path passes a non-empty value.
fn parse_wald_se_override(s: &str) -> PyResult<Option<engine_contract::WaldSe>> {
    match s {
        "" | "default" => Ok(None),
        "rx" => Ok(Some(engine_contract::WaldSe::Rx)),
        "hessian" => Ok(Some(engine_contract::WaldSe::Hessian)),
        other => Err(PyValueError::new_err(format!(
            "unknown wald_se override {other:?}; expected \"\", \"default\", \"rx\", or \"hessian\""
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (contracts, scenario_index, seed, design, nrow, ncol, outcome, cluster_ids, wald_se=""))]
#[allow(clippy::too_many_arguments)]
pub fn fit_uploaded_data(
    py: Python<'_>,
    contracts: &[u8],
    scenario_index: i32,
    seed: i64,
    design: Vec<f64>,
    nrow: i32,
    ncol: i32,
    outcome: Vec<f64>,
    cluster_ids: Option<Vec<u32>>,
    wald_se: &str,
) -> PyResult<Py<PyDict>> {
    let wald_se_override = parse_wald_se_override(wald_se)?;
    let all = decode_contracts(contracts)?;
    let c = all.get(scenario_index as usize).ok_or_else(|| {
        PyValueError::new_err(format!("scenario_index {scenario_index} out of range"))
    })?;

    // Validate the shape before the `as usize` casts: a negative value would
    // wrap to a huge usize and panic (out-of-bounds index) inside the fitter,
    // which is undefined behaviour across the FFI boundary. Reject cleanly.
    if nrow < 0 || ncol < 0 {
        return Err(PyValueError::new_err(format!(
            "nrow and ncol must be non-negative, got nrow={nrow} ncol={ncol}"
        )));
    }
    let (nrow, ncol) = (nrow as usize, ncol as usize);
    if design.len() != nrow * ncol {
        return Err(PyValueError::new_err(format!(
            "design has {} elements, expected nrow*ncol = {}*{} = {}",
            design.len(),
            nrow,
            ncol,
            nrow * ncol
        )));
    }
    if outcome.len() != nrow {
        return Err(PyValueError::new_err(format!(
            "outcome has {} elements, expected nrow = {nrow}",
            outcome.len()
        )));
    }

    let result = engine_orchestrator::debug::debug_load_data(
        c,
        seed as u64,
        &design,
        nrow,
        ncol,
        &outcome,
        cluster_ids.as_deref(),
        wald_se_override,
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;

    let d = PyDict::new(py);
    d.set_item("betas", result.betas.clone())?;
    d.set_item("design_columns", result.design_columns.clone())?;
    d.set_item("converged", result.converged)?;

    let targets_list = PyList::empty(py);
    for t in &result.targets {
        let td = PyDict::new(py);
        td.set_item("target_index", t.target_index)?;
        td.set_item("target_label", t.target_label.clone())?;
        td.set_item("beta", t.beta)?;
        td.set_item("se", t.se)?;
        td.set_item("statistic", t.statistic)?;
        td.set_item("statistic_kind", t.statistic_kind.as_str())?;
        td.set_item("critical_value", t.critical_value)?;
        td.set_item("alpha", t.alpha)?;
        td.set_item("df", t.df.clone())?;
        td.set_item("two_sided", t.two_sided)?;
        targets_list.append(td)?;
    }
    d.set_item("targets", targets_list)?;
    d.set_item("variance_components", result.variance_components.clone())?;
    // NaN when the estimator does not surface σ̂² (OLS/GLM paths).
    d.set_item("sigma_sq_hat", result.sigma_sq_hat)?;
    d.set_item("re_corr", result.re_corr.clone())?;

    Ok(d.into())
}
