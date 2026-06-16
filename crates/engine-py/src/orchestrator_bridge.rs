//! Glue between PyO3 and engine-orchestrator. Owns:
//!   - contract decode (msgpack Vec<SimulationContract> blob),
//!   - PyO3 ProgressSink (translates ProgressEvent → progress_callback(int,int)),
//!   - the generic `HostValue` → Python walker that realizes the
//!     single-sourced result shape (`engine_orchestrator::result_host`).
//!
//! Mirror: `engine-r/src/orchestrator_bridge.rs` — change together.

use engine_contract::SimulationContract;
use engine_orchestrator::{
    power_result_to_host, sample_size_result_to_host, Ci, HostValue, PowerResult, ProgressEvent,
    ProgressSink, SampleSizeResult, ScenarioResult,
};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

/// Bridges the orchestrator's structured event stream to the Python callable
/// `progress(int, int) -> bool` that v1's `progress_callback` expects.
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
    py_callback: Option<PyObject>,
    cancel: &'cancel engine_orchestrator::CancellationToken,
}

impl<'cancel> OrchestratorProgressSink<'cancel> {
    pub fn new(
        py_callback: Option<PyObject>,
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
        let should_continue = Python::with_gil(|py| match cb.call1(py, (current, total)) {
            Ok(ret) => ret.extract::<bool>(py).unwrap_or(true),
            Err(_) => false,
        });
        if !should_continue {
            self.cancel.cancel();
        }
    }
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
    mut render: impl FnMut(&T, &str) -> PyResult<PyObject>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new_bound(py);
    for (label, value) in scenarios {
        dict.set_item(label, render(value, label)?)?;
    }
    let outer = PyDict::new_bound(py);
    outer.set_item("scenarios", dict)?;
    outer.set_item("comparison", PyDict::new_bound(py))?;
    Ok(outer.into())
}

/// Walk a `HostValue` into the Python object the host result shape expects.
fn host_value_to_py(py: Python<'_>, hv: &HostValue) -> PyResult<PyObject> {
    Ok(match hv {
        HostValue::F64(x) => (*x).into_py(py),
        HostValue::OptF64(o) => match o {
            Some(x) => (*x).into_py(py),
            None => py.None(),
        },
        HostValue::Usize(n) => (*n).into_py(py),
        HostValue::OptUsize(o) => match o {
            Some(n) => (*n).into_py(py),
            None => py.None(),
        },
        HostValue::VecF64(v) => v.clone().into_py(py),
        HostValue::VecU64(v) => v.clone().into_py(py),
        HostValue::VecStr(v) => v.clone().into_py(py),
        HostValue::VecCi(cis) => ci_slice_to_pylist(py, cis)?.into_py(py),
        HostValue::OptCi(o) => match o {
            Some(ci) => PyTuple::new_bound(py, [ci.lo, ci.hi]).into_py(py),
            None => py.None(),
        },
        HostValue::Str(s) => s.as_str().into_py(py),
        HostValue::Map(pairs) => {
            let d = PyDict::new_bound(py);
            for (k, v) in pairs {
                d.set_item(*k, host_value_to_py(py, v)?)?;
            }
            d.into_py(py)
        }
        HostValue::Seq(items) => {
            let list = PyList::empty_bound(py);
            for it in items {
                list.append(host_value_to_py(py, it)?)?;
            }
            list.into_py(py)
        }
        HostValue::IndexMap(items) => {
            let d = PyDict::new_bound(py);
            for (i, it) in items.iter().enumerate() {
                d.set_item(i, host_value_to_py(py, it)?)?;
            }
            d.into_py(py)
        }
        HostValue::BoundaryHit { flat, rows, cols } => {
            let arr: Py<PyArray2<u8>> = flat
                .clone()
                .into_pyarray_bound(py)
                .reshape([*rows, *cols])?
                .into();
            arr.into_py(py)
        }
    })
}

fn ci_slice_to_pylist<'py>(py: Python<'py>, cis: &[Ci]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for ci in cis {
        list.append(PyTuple::new_bound(py, [ci.lo, ci.hi]))?;
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
#[pyfunction]
#[pyo3(signature = (contracts, scenario_index, seed, design, nrow, ncol, outcome, cluster_ids))]
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
) -> PyResult<Py<PyDict>> {
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
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;

    let d = PyDict::new_bound(py);
    d.set_item("betas", result.betas.clone())?;
    d.set_item("design_columns", result.design_columns.clone())?;
    d.set_item("converged", result.converged)?;

    let targets_list = PyList::empty_bound(py);
    for t in &result.targets {
        let td = PyDict::new_bound(py);
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
