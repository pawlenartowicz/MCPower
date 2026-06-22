//! `ProgressEmitter` trait, `EmitterSink` adapter, and `serialize_event`; one trait, host transports (Tauri `WindowEmitter`, WASM `postMessage`) plug in.

use engine_orchestrator::{ProgressEvent, ProgressSink};
use serde_json::{json, Value};

/// Host-transport seam: each shell (Tauri `WindowEmitter`, WASM `postMessage`) implements this
/// to receive serialised progress events from the engine via [`EmitterSink`].
pub trait ProgressEmitter: Send + Sync {
    fn emit(&self, event: Value);
}

/// No-op [`ProgressEmitter`] for callers that don't need progress events.
pub struct NullEmitter;
impl ProgressEmitter for NullEmitter {
    fn emit(&self, _: Value) {}
}

/// Adapts `engine_orchestrator::ProgressSink` to a `ProgressEmitter` — bridges the engine's
/// pull-style trait to the host's push-style transport.
pub struct EmitterSink<'a> {
    emitter: &'a dyn ProgressEmitter,
}

impl<'a> EmitterSink<'a> {
    pub fn new(emitter: &'a dyn ProgressEmitter) -> Self {
        Self { emitter }
    }
}

impl<'a> ProgressSink for EmitterSink<'a> {
    fn on_event(&mut self, event: ProgressEvent) {
        self.emitter.emit(serialize_event(event));
    }
}

/// Serialize a `ProgressEvent` to the `kind`-tagged JSON shape matched by the TS
/// `ProgressEvent` union in `ports/wasm/src/types.ts` (WASM worker pool) and
/// `ports/app/src/lib/domain/result.ts` (Tauri/app listener). Any rename or new
/// `kind` value must be mirrored in both.
pub fn serialize_event(event: ProgressEvent) -> Value {
    match event {
        ProgressEvent::Started {
            total_sims,
            total_scenarios,
            total_grid_points,
        } => json!({
            "kind": "started",
            "total_sims": total_sims,
            "total_scenarios": total_scenarios,
            "total_grid_points": total_grid_points,
        }),
        ProgressEvent::ScenarioStarted { label, idx, total } => json!({
            "kind": "scenario_started",
            "label": label,
            "idx": idx,
            "total": total,
        }),
        ProgressEvent::SimsCompleted {
            n,
            completed,
            total,
        } => json!({
            "kind": "sims_completed",
            "n": n,
            "completed": completed,
            "total": total,
        }),
        ProgressEvent::NPointCompleted {
            n,
            power_uncorrected,
            power_corrected,
        } => json!({
            "kind": "n_point_completed",
            "n": n,
            "power_uncorrected": power_uncorrected,
            "power_corrected": power_corrected,
        }),
        ProgressEvent::ScenarioCompleted { label, idx } => json!({
            "kind": "scenario_completed",
            "label": label,
            "idx": idx,
        }),
        ProgressEvent::Cancelled => json!({ "kind": "cancelled" }),
        ProgressEvent::Completed => json!({ "kind": "completed" }),
    }
}
