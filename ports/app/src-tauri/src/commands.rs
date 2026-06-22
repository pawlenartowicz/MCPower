//! Tauri command handlers for `find_power`, `find_sample_size`, `cancel_run`, `parse_formula`, and `set_n_threads`; each spawns a blocking task and emits progress events to the window.

use engine_app_spec::{
    effect_skeleton_json, power_plot_specs, run_find_power, run_find_sample_size,
    sample_size_curve_specs, AppSpec, EffectDescriptor, PlotSpecs,
};
use engine_orchestrator::{PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult};
use serde::Serialize;
use serde_json::json;
use tauri::{Emitter, State, Window};

use crate::{run_registry::RunRegistry, window_emitter::WindowEmitter};

/// Structured error payload for the two run commands. `kind` lets the frontend
/// route to a dedicated surface — `"cluster_setup"` for the cluster-vs-sample-
/// size-grid configuration errors, `"generic"` otherwise — while `message`
/// carries the engine's own Display text (single-sourced wording + fix hint).
/// Mirrors RunErrorCard.svelte's branch on `kind`; see `AdapterError::host_kind`.
#[derive(Serialize)]
pub struct RunError {
    pub kind: String,
    pub message: String,
}

impl From<engine_app_spec::AdapterError> for RunError {
    fn from(e: engine_app_spec::AdapterError) -> Self {
        RunError {
            kind: e.host_kind().to_string(),
            message: e.to_string(),
        }
    }
}

/// Response payload for `find_power_cmd`.
#[derive(Serialize)]
pub struct FindPowerResponse {
    pub run_id: String,
    pub result: ScenarioResult<PowerResult>,
    pub plots: PlotSpecs,
}

/// Run a power analysis for `spec` at `sample_size`, emitting progress events to
/// `window`. Registers a cancellable run; returns `FindPowerResponse`.
#[tauri::command]
pub async fn find_power_cmd(
    window: Window,
    registry: State<'_, RunRegistry>,
    spec: AppSpec,
    sample_size: usize,
) -> Result<FindPowerResponse, RunError> {
    let (run_id, cancel) = registry.register();
    let _ = window.emit(
        "progress",
        &json!({ "kind": "run_started", "run_id": run_id }),
    );

    let win = window.clone();
    let id_for_join = run_id.clone();

    let joined = tauri::async_runtime::spawn_blocking(move || {
        let emitter = WindowEmitter::new(win, "progress");
        let target_power = spec.target_power();
        let corrected = spec.is_corrected();
        let result = run_find_power(&spec, sample_size, &emitter, &cancel)?;
        let plots = power_plot_specs(&result, target_power, corrected);
        Ok::<_, engine_app_spec::AdapterError>((result, plots))
    })
    .await;
    // Unconditional drop, before the ?-unwraps below: a join or engine error
    // (cancellation included) must not leave a dead token registered.
    registry.drop_run(&id_for_join);
    let (result, plots) = joined
        .map_err(|e| RunError {
            kind: "generic".to_string(),
            message: format!("join error: {e}"),
        })?
        .map_err(RunError::from)?;
    Ok(FindPowerResponse {
        run_id,
        result,
        plots,
    })
}

/// Response payload for `find_sample_size_cmd`.
#[derive(Serialize)]
pub struct FindSampleSizeResponse {
    pub run_id: String,
    pub result: ScenarioResult<SampleSizeResult>,
    pub plots: PlotSpecs,
}

/// Run a sample-size search for `spec` over `bounds`, emitting progress events to
/// `window`. Registers a cancellable run; returns `FindSampleSizeResponse`.
#[tauri::command]
pub async fn find_sample_size_cmd(
    window: Window,
    registry: State<'_, RunRegistry>,
    spec: AppSpec,
    bounds: (usize, usize),
    // Coupled to the TS sender: serde's external tagging means `by` arrives
    // as {Fixed:n}|{Auto:{count:n}} — change together with the frontend call.
    method: SampleSizeMethod,
) -> Result<FindSampleSizeResponse, RunError> {
    let (run_id, cancel) = registry.register();
    let _ = window.emit(
        "progress",
        &json!({ "kind": "run_started", "run_id": run_id }),
    );

    let win = window.clone();
    let id_for_join = run_id.clone();

    let joined = tauri::async_runtime::spawn_blocking(move || {
        let emitter = WindowEmitter::new(win, "progress");
        let target_power = spec.target_power();
        let corrected = spec.is_corrected();
        let result = run_find_sample_size(&spec, bounds, method, &emitter, &cancel)?;
        let plots = sample_size_curve_specs(&result, target_power, corrected);
        Ok::<_, engine_app_spec::AdapterError>((result, plots))
    })
    .await;
    // Mirrors find_power_cmd's unconditional drop — change together.
    registry.drop_run(&id_for_join);
    let (result, plots) = joined
        .map_err(|e| RunError {
            kind: "generic".to_string(),
            message: format!("join error: {e}"),
        })?
        .map_err(RunError::from)?;
    Ok(FindSampleSizeResponse {
        run_id,
        result,
        plots,
    })
}

/// Cancel the run identified by `run_id`; returns `true` when a live run was
/// found and signalled.
#[tauri::command]
pub fn cancel_run_cmd(registry: State<'_, RunRegistry>, run_id: String) -> bool {
    registry.cancel(&run_id)
}

/// Parse a model formula to the host-agnostic `FormulaParse` shape for the UI.
/// Always parses random effects; callers decide whether REs are allowed.
#[tauri::command]
pub fn parse_formula_cmd(formula: String) -> Result<engine_app_spec::FormulaParse, String> {
    engine_app_spec::parse_formula(&formula).map_err(|e| e.to_string())
}

/// Recover standardized effect sizes from uploaded data attached to the spec.
/// The fitted estimator follows the spec family (OLS/GLM/MLE); returns an error
/// string on invalid input (e.g. missing csv or grouping column). The result
/// also carries the estimated cluster ICC (mixed) and binary-outcome baseline
/// probability for the host's Apply flow.
#[tauri::command]
pub fn get_effects_from_data_cmd(
    spec: engine_app_spec::AppSpec,
) -> Result<engine_app_spec::EffectsFromData, String> {
    engine_app_spec::get_effects_from_data(&spec).map_err(|e| e.to_string())
}

/// Return the index-only effect skeleton (β-column aligned) for `spec` as a
/// JSON-parsed `Vec<EffectDescriptor>`. The JS side renders result names by
/// looking up `skeleton[target_indices[i]]` and resolving factor levels from
/// the port's own label store (`VariableRow.levels`). Mirrors the WASM
/// `effect_skeleton` export.
#[tauri::command]
pub fn effect_skeleton_cmd(spec: AppSpec) -> Result<Vec<EffectDescriptor>, String> {
    let json = effect_skeleton_json(&spec).map_err(|e| e.to_string())?;
    serde_json::from_str::<Vec<EffectDescriptor>>(&json).map_err(|e| e.to_string())
}

/// Configure the rayon thread pool before the first `find_power` or
/// `find_sample_size` call. A second call after the pool is initialised
/// returns an error (pool is a process-global `OnceLock`). Called once from
/// the frontend on startup when a non-default thread count is stored in
/// settings. n = 0 is rejected by the engine.
#[tauri::command]
pub fn set_n_threads_cmd(n: usize) -> Result<(), String> {
    engine_app_spec::set_n_threads(n)
}
