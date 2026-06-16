//! wasm-bindgen marshalling shell: JSON in → call an engine-app-spec twin /
//! orchestrator primitive → JSON out. No business logic. Mirrors the thinness
//! of engine-py / engine-r.

use engine_app_spec::{
    parse_formula as app_parse_formula, power_plot_specs, run_single_core_find_power,
    run_single_core_find_sample_size, sample_size_curve_specs, AdapterError, AppSpec, NullEmitter,
    ProgressEmitter,
};
use engine_orchestrator::{
    merge_power_results as orch_merge_power, merge_sample_size_results as orch_merge_sample_size,
    CancellationToken, PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};
use js_sys::Function;
use serde_json::Value;
use wasm_bindgen::prelude::*;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Bridges the host-agnostic `ProgressEmitter` to a JS callback. The callback
/// receives one arg: the serialized `ProgressEvent` JSON string.
struct JsEmitter {
    cb: Function,
}
impl ProgressEmitter for JsEmitter {
    fn emit(&self, event: Value) {
        let _ = self
            .cb
            .call1(&JsValue::NULL, &JsValue::from_str(&event.to_string()));
    }
}

// ---- run twins (one worker's share) ----
#[wasm_bindgen]
pub fn find_power(
    spec_json: &str,
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    on_progress: Option<Function>,
) -> Result<String, JsError> {
    let spec: AppSpec = serde_json::from_str(spec_json).map_err(err)?;
    let cancel = CancellationToken::new(); // cosmetic in WASM; real killswitch is worker.terminate()
    let result = run_with_emitter(on_progress, |emitter| {
        run_single_core_find_power(&spec, sample_size, n_sims, base_seed, emitter, &cancel)
    })?;
    serde_json::to_string(&result).map_err(err)
}

#[wasm_bindgen]
pub fn find_sample_size(
    spec_json: &str,
    bounds_json: &str,
    method_json: &str,
    n_sims: usize,
    base_seed: u64,
    on_progress: Option<Function>,
) -> Result<String, JsError> {
    let spec: AppSpec = serde_json::from_str(spec_json).map_err(err)?;
    let bounds: (usize, usize) = serde_json::from_str(bounds_json).map_err(err)?;
    let method: SampleSizeMethod = serde_json::from_str(method_json).map_err(err)?;
    let cancel = CancellationToken::new();
    let result = run_with_emitter(on_progress, |emitter| {
        run_single_core_find_sample_size(&spec, bounds, method, n_sims, base_seed, emitter, &cancel)
    })?;
    serde_json::to_string(&result).map_err(err)
}

// ---- merge (main thread pools worker parts) ----
#[wasm_bindgen]
pub fn merge_power_results(parts_json: &str) -> Result<String, JsError> {
    let parts: Vec<ScenarioResult<PowerResult>> = serde_json::from_str(parts_json).map_err(err)?;
    serde_json::to_string(&orch_merge_power(&parts).map_err(err)?).map_err(err)
}

#[wasm_bindgen]
pub fn merge_sample_size_results(parts_json: &str) -> Result<String, JsError> {
    let parts: Vec<ScenarioResult<SampleSizeResult>> =
        serde_json::from_str(parts_json).map_err(err)?;
    serde_json::to_string(&orch_merge_sample_size(&parts).map_err(err)?).map_err(err)
}

// ---- plots (main thread, after merge) ----

const PLOT_THEMES_JSON: &str = include_str!("../../../configs/plot-themes.json");

/// Return the embedded theme JSON for `name`, or a JsError listing valid names.
#[wasm_bindgen]
pub fn plot_theme(name: &str) -> Result<String, JsError> {
    let m: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(PLOT_THEMES_JSON).map_err(err)?;
    m.get(name)
        .map(|v| v.to_string())
        .ok_or_else(|| {
            let names: Vec<&String> = m.keys().collect();
            JsError::new(&format!("unknown theme {name:?}; valid themes: {names:?}"))
        })
}

/// Theme-naked power plot set for a `ScenarioResult<PowerResult>` (JSON in/out).
#[wasm_bindgen]
pub fn power_plot_specs_json(
    result_json: &str,
    target_power: f64,
    corrected: bool,
) -> Result<String, JsError> {
    let result: ScenarioResult<PowerResult> = serde_json::from_str(result_json).map_err(err)?;
    serde_json::to_string(&power_plot_specs(&result, target_power, corrected)).map_err(err)
}

/// Theme-naked sample-size curve set for a `ScenarioResult<SampleSizeResult>` (JSON in/out).
#[wasm_bindgen]
pub fn sample_size_plot_specs_json(
    result_json: &str,
    target_power: f64,
    corrected: bool,
) -> Result<String, JsError> {
    let result: ScenarioResult<SampleSizeResult> =
        serde_json::from_str(result_json).map_err(err)?;
    serde_json::to_string(&sample_size_curve_specs(&result, target_power, corrected)).map_err(err)
}

// ---- formula parse ----
/// Parse a formula string; returns the `ParsedFormula` JSON the TS layer consumes.
#[wasm_bindgen]
pub fn parse_formula(formula: &str) -> Result<String, JsError> {
    serde_json::to_string(&app_parse_formula(formula).map_err(err)?).map_err(err)
}

// ---- effect skeleton: index-only result-naming layout ----
/// Deserialize `AppSpec` from JSON and return the index-only `EffectSkeleton`
/// (β-column aligned) as JSON. Consumers render result names from this + their
/// own factor-label store — the same single-sourced layout the Py/R bridges
/// return, so no host re-derives factor expansion. A run's `target_indices`
/// index into the returned array directly (index 0 = intercept).
#[wasm_bindgen]
pub fn effect_skeleton(spec_json: &str) -> Result<String, JsError> {
    let spec: AppSpec = serde_json::from_str(spec_json).map_err(err)?;
    engine_app_spec::effect_skeleton_json(&spec).map_err(err)
}

// ---- upload: recover effects from data ----
/// Deserialize `AppSpec` from JSON, call `get_effects_from_data`, serialize the
/// resulting `Vec<EffectSize>` to JSON. The fitted estimator follows the spec
/// family (OLS/GLM/MLE); returns a `JsError` on invalid input.
#[wasm_bindgen]
pub fn get_effects_from_data(spec_json: &str) -> Result<String, JsError> {
    use engine_app_spec::get_effects_from_data as app_get_effects;
    let spec: engine_app_spec::AppSpec = serde_json::from_str(spec_json).map_err(err)?;
    let effects = app_get_effects(&spec).map_err(err)?;
    serde_json::to_string(&effects).map_err(err)
}

// ---- helper ----
fn run_with_emitter<T>(
    on_progress: Option<Function>,
    run: impl FnOnce(&dyn ProgressEmitter) -> Result<T, AdapterError>,
) -> Result<T, JsError> {
    match on_progress {
        Some(cb) => run(&JsEmitter { cb }).map_err(err),
        None => run(&NullEmitter).map_err(err),
    }
}

// ---- dev-only bench entry (wasm throughput bench) ----

/// FNV-1a 64 over `bytes`, continuing from `h`. Mirrors `fnv1a` in
/// `crates/engine-core/src/bin/throughput.rs` — change together.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Run one bench pass and return control counts + the chained digest. Bench
/// tooling only — NOT contract surface; mirrors the native throughput bin's
/// per-pass protocol (a dumped `SimulationSpec`, single-threaded
/// `run_batch_st`, the bin's call-level seed) — change together with
/// `crates/engine-core/src/bin/throughput.rs`.
///
/// Inputs: a `--dump-cases` pass spec (JSON), the row's `n`/`n_sims`, the
/// dump's `seed` (BigInt on the JS side), and the previous pass's hex
/// `hash_state` (FNV-1a offset basis for the first pass). Output JSON:
/// `{ k_unc, k_conv, hash_state }` — first-target success count, converged
/// count, and the digest folded over this pass's `uncorrected` then
/// `converged` bitstreams. Counts are ≤ n_sims, safe as JS numbers; the
/// digest stays a string (u64 doesn't survive JS JSON).
#[wasm_bindgen]
pub fn bench_run_pass(
    spec_json: &str,
    n: u32,
    n_sims: u32,
    seed: u64,
    hash_state: &str,
) -> Result<String, JsError> {
    let spec: engine_core::spec::SimulationSpec = serde_json::from_str(spec_json).map_err(err)?;
    let batch =
        engine_core::batch::run_batch_st(&spec, &[n], n_sims, seed, None).map_err(err)?;
    let n_sims_out = batch.shape.n_sims as usize;
    let n_targets = batch.shape.n_targets as usize;
    let k_unc: u64 = if n_targets == 0 {
        0
    } else {
        (0..n_sims_out)
            .map(|sim| batch.uncorrected[sim * n_targets] as u64)
            .sum()
    };
    let k_conv: u64 = (0..n_sims_out).map(|sim| batch.converged[sim] as u64).sum();
    let mut h = u64::from_str_radix(hash_state, 16).map_err(err)?;
    h = fnv1a(h, &batch.uncorrected);
    h = fnv1a(h, &batch.converged);
    serde_json::to_string(&serde_json::json!({
        "k_unc": k_unc,
        "k_conv": k_conv,
        "hash_state": format!("{h:016x}"),
    }))
    .map_err(err)
}
