//! Glue between extendr and engine-orchestrator. Owns:
//!   - contract decode (msgpack Vec<SimulationContract> blob),
//!   - the `run_with_progress` dispatch (background-thread engine run + R
//!     main-thread progress/interrupt poll — see `progress.rs`),
//!   - the generic `HostValue` → R walker that realizes the single-sourced
//!     result shape (`engine_orchestrator::result_host`).
//!
//! Mirror: `engine-py/src/orchestrator_bridge.rs` + `engine-py/src/lib.rs`
//! (progress callback + cancellation semantics) — change together.

use engine_contract::SimulationContract;
use engine_orchestrator::{
    power_result_to_host, sample_size_result_to_host, HostValue, PowerResult, SampleSizeMethod,
    SampleSizeResult, ScenarioResult,
};
use extendr_api::error::Result;
use extendr_api::prelude::*;

use crate::progress::{orchestrator_err, run_engine, run_with_progress};

// ── Contract decoder ──────────────────────────────────────────────────────────

/// Decode the msgpack `Vec<SimulationContract>` blob produced by
/// `build_contract_from_spec`. The R frontend forwards the `raw` vector
/// verbatim into `find_power`; seed derivation lives in the orchestrator.
///
/// Exposed as `pub` so `debug_bridge` can reuse it directly.
pub fn decode_contracts(bytes: &[u8]) -> Result<Vec<SimulationContract>> {
    let v: Vec<SimulationContract> = rmp_serde::from_slice(bytes)
        .map_err(|e| Error::Other(format!("malformed contracts bytes: {e}")))?;
    if v.is_empty() {
        return Err(Error::Other("contracts list cannot be empty".into()));
    }
    Ok(v)
}

// ── find_power binding ────────────────────────────────────────────────────────

/// Run a power simulation for all scenarios encoded in `contracts` (a msgpack
/// `raw` vector produced by `build_contract_from_spec`).
///
/// * `sample_size` — sample size to evaluate.
/// * `n_sims`      — number of Monte-Carlo replications.
/// * `base_seed`   — f64 in R (all R numerics are f64); cast to u64.
/// * `progress`    — an R callback `(current, total)`, or `NULL` for silent.
///   Events are delivered on the R main thread; Ctrl-C / Escape cancels the run
///   (see `progress.rs`).
///
/// Returns a named list mirroring the Python dict shape:
/// `list(scenarios = <named list>, comparison = list())`.
#[extendr]
pub fn find_power(
    contracts: &[u8],
    sample_size: i32,
    n_sims: i32,
    base_seed: f64,
    progress: Robj,
) -> Result<List> {
    // Decode to plain Rust types before the background thread starts — no Robj
    // may cross the thread boundary (`Robj` is `!Send`).
    let contracts = decode_contracts(contracts)?;
    let cancel = engine_orchestrator::CancellationToken::new();
    let result = run_engine(progress, &cancel, |sink, cancel| {
        engine_orchestrator::find_power(
            &contracts,
            sample_size as usize,
            n_sims as usize,
            base_seed as u64,
            Some(sink),
            cancel,
        )
    })?;
    Ok(power_result_to_list(&result))
}

/// Test-only seam for the cancellation path. Identical to [`find_power`] but
/// flips the `CancellationToken` *before* the engine starts, so the run returns
/// `OrchestratorError::Cancelled` at its first checkpoint and `orchestrator_err`
/// maps it to the "cancelled by user" R error. Lets testthat verify the
/// cancel→error wiring without an interactive SIGINT (the live Ctrl-C path,
/// which flips the same token via `R_CheckUserInterrupt`, is exercised
/// manually). Surfaced as `mcpower:::find_power_precancelled`.
#[extendr]
pub fn find_power_precancelled(
    contracts: &[u8],
    sample_size: i32,
    n_sims: i32,
    base_seed: f64,
) -> Result<List> {
    let contracts = decode_contracts(contracts)?;
    let cancel = engine_orchestrator::CancellationToken::new();
    cancel.cancel();
    let result = run_with_progress(r!(NULL), &cancel, |sink, cancel| {
        engine_orchestrator::find_power(
            &contracts,
            sample_size as usize,
            n_sims as usize,
            base_seed as u64,
            Some(sink),
            cancel,
        )
    })
    .map_err(orchestrator_err)?;
    Ok(power_result_to_list(&result))
}

/// Test-only seam for the internal-error path. Panics inside the engine
/// boundary so testthat can assert the caught panic surfaces as an R error
/// carrying the report hint (mirrors engine-py's `panic_for_test`). Surfaced as
/// `mcpower:::panic_for_test`.
#[extendr]
pub fn panic_for_test() -> Result<()> {
    let cancel = engine_orchestrator::CancellationToken::new();
    run_engine(r!(NULL), &cancel, |_sink, _cancel| {
        panic!("simulated internal engine failure")
    })
}

// ── Result decoders ───────────────────────────────────────────────────────────

/// Convert a `ScenarioResult<PowerResult>` into the multi-scenario envelope.
///
/// Always returns `list(scenarios = <named list>, comparison = list())`. The
/// per-scenario shape is single-sourced via `power_result_to_host`; this only
/// walks the tree into R objects and wraps it in the label-keyed envelope.
///
/// Exposed as `pub` so `debug_bridge` can wrap single-scenario debug results.
pub fn power_result_to_list(result: &ScenarioResult<PowerResult>) -> List {
    scenarios_envelope(&result.scenarios, |pr, label| {
        host_value_to_robj(&power_result_to_host(pr, label))
    })
}

/// Convert a `ScenarioResult<SampleSizeResult>` into the multi-scenario
/// envelope. Per-scenario shape single-sourced via `sample_size_result_to_host`.
fn sample_size_result_to_list(result: &ScenarioResult<SampleSizeResult>) -> List {
    scenarios_envelope(&result.scenarios, |ssr, label| {
        host_value_to_robj(&sample_size_result_to_host(ssr, label))
    })
}

/// Build `list(scenarios = <named list keyed by label>, comparison = list())`
/// from the per-scenario `(label, T)` pairs, rendering each value via `render`.
fn scenarios_envelope<T>(
    scenarios: &[(String, T)],
    mut render: impl FnMut(&T, &str) -> Robj,
) -> List {
    let names: Vec<String> = scenarios.iter().map(|(n, _)| n.clone()).collect();
    let objs: Vec<Robj> = scenarios
        .iter()
        .map(|(label, value)| render(value, label))
        .collect();
    // names.len() == objs.len() — both mapped from `scenarios` → constructor can't fail
    let scenarios_list = List::from_names_and_values(names, objs).unwrap();
    list!(scenarios = scenarios_list, comparison = List::new(0))
}

/// Walk a `HostValue` into the R object the host result shape expects.
/// Integer-valued leaves become R integer vectors; `IndexMap` keys are
/// stringified (R named lists key by character — `fja[[as.character(j)]]`
/// in the report layer depends on this).
fn host_value_to_robj(hv: &HostValue) -> Robj {
    match hv {
        HostValue::F64(x) => r!(*x),
        HostValue::OptF64(o) => match o {
            Some(x) => r!(*x),
            None => r!(NULL),
        },
        HostValue::Usize(n) => r!(*n as i32),
        HostValue::OptUsize(o) => match o {
            Some(n) => r!(*n as i32),
            None => r!(NA_INTEGER),
        },
        HostValue::VecF64(v) => r!(v.clone()),
        HostValue::VecU64(v) => r!(v.iter().map(|&x| x as i32).collect::<Vec<i32>>()),
        HostValue::VecStr(v) => r!(v.clone()),
        HostValue::VecCi(cis) => {
            List::from_values(cis.iter().map(|c| r!([c.lo, c.hi]))).into_robj()
        }
        HostValue::OptCi(o) => match o {
            Some(c) => r!([c.lo, c.hi]),
            None => r!(NULL),
        },
        HostValue::Str(s) => r!(s.clone()),
        HostValue::Map(pairs) => {
            let names: Vec<String> = pairs.iter().map(|(k, _)| (*k).to_string()).collect();
            let vals: Vec<Robj> = pairs.iter().map(|(_, v)| host_value_to_robj(v)).collect();
            // names.len() == vals.len() — both mapped from `pairs` → constructor can't fail
            List::from_names_and_values(names, vals)
                .unwrap()
                .into_robj()
        }
        HostValue::Seq(items) => {
            List::from_values(items.iter().map(host_value_to_robj)).into_robj()
        }
        HostValue::IndexMap(items) => {
            let names: Vec<String> = (0..items.len()).map(|i| i.to_string()).collect();
            let vals: Vec<Robj> = items.iter().map(host_value_to_robj).collect();
            // names.len() == vals.len() — both length items.len() → constructor can't fail
            List::from_names_and_values(names, vals)
                .unwrap()
                .into_robj()
        }
        HostValue::BoundaryHit { flat, .. } => {
            // Flat integer vector (row-major); the R caller reshapes with
            // `matrix(...)`. Shape deliberately differs from the Python port's
            // nested list-of-rows form.
            r!(flat.iter().map(|&b| b as i32).collect::<Vec<i32>>())
        }
    }
}

// ── find_sample_size binding ──────────────────────────────────────────────────

/// Run a sample-size search (grid) for all scenarios encoded in `contracts`.
///
/// * `target_power`      — desired power threshold (0..1).
/// * `lo` / `hi`         — search bounds (sample sizes).
/// * `n_sims`            — Monte-Carlo replications per N point.
/// * `base_seed`         — f64 in R; cast to u64.
/// * `method`            — `"grid"` (the only supported method).
/// * `by`                — grid step size / point count (required).
/// * `by_kind`           — `"fixed"` (step) or `"auto"` (point count); default `"fixed"`.
/// * `mode`              — `"linear"` or `"log"` (default `"linear"`).
/// * `tol_n`             — accepted for API compatibility; unused.
/// * `progress`          — an R callback `(current, total)`, or `NULL` for
///   silent. Events are delivered on the R main thread; Ctrl-C / Escape cancels
///   the run (see `progress.rs`).
///
/// Returns a named list mirroring the Python dict shape:
/// `list(scenarios = <named list>, comparison = list())`.
#[extendr]
#[expect(
    clippy::too_many_arguments,
    reason = "extendr entry point; signature mirrors the R argument list"
)]
pub fn find_sample_size(
    contracts: &[u8],
    target_power: f64,
    lo: i32,
    hi: i32,
    n_sims: i32,
    base_seed: f64,
    method: &str,
    by: Option<i32>,
    by_kind: Option<&str>,
    mode: Option<&str>,
    tol_n: Option<i32>,
    progress: Robj,
) -> Result<List> {
    let _ = tol_n; // bisection-only; retained for FFI signature stability.

    let method = SampleSizeMethod::from_host_args(method, by.map(|v| v as usize), by_kind, mode)
        .map_err(Error::Other)?;

    // Decode to plain Rust types before the background thread starts — no Robj
    // may cross the thread boundary (`Robj` is `!Send`).
    let contracts = decode_contracts(contracts)?;
    let cancel = engine_orchestrator::CancellationToken::new();
    let result = run_engine(progress, &cancel, |sink, cancel| {
        engine_orchestrator::find_sample_size(
            &contracts,
            target_power,
            (lo as usize, hi as usize),
            n_sims as usize,
            method,
            base_seed as u64,
            Some(sink),
            cancel,
        )
    })?;

    Ok(sample_size_result_to_list(&result))
}

// ── extendr module registration ───────────────────────────────────────────────

extendr_module! {
    mod orchestrator_bridge;
    fn find_power;
    fn find_power_precancelled;
    fn find_sample_size;
    fn panic_for_test;
}
