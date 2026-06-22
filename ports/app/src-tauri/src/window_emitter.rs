//! `ProgressEmitter` implementation that forwards engine progress events to the Tauri window via a named JS event channel.

use engine_app_spec::ProgressEmitter;
use serde_json::Value;
use tauri::{Emitter, Window};

/// Tauri binding of `ProgressEmitter`. Construct once per run inside the spawned
/// blocking task and pass `&emitter` into the engine driver.
pub struct WindowEmitter<R: tauri::Runtime> {
    window: Window<R>,
    channel: String,
}

impl<R: tauri::Runtime> WindowEmitter<R> {
    /// Bind the emitter to `window` and the named JS event `channel`.
    pub fn new(window: Window<R>, channel: impl Into<String>) -> Self {
        Self {
            window,
            channel: channel.into(),
        }
    }
}

// `ProgressEmitter` is bound by every host — keep this `emit` in step with the
// sibling forwarders: `JsEmitter` (engine-wasm), `EmitterSink` (engine-app-spec
// progress.rs), and the `ProgressSink` twins `OrchestratorProgressSink`
// (engine-py) and the R callback path (engine-r progress.rs). Change together.
impl<R: tauri::Runtime> ProgressEmitter for WindowEmitter<R> {
    fn emit(&self, event: Value) {
        if let Err(err) = self.window.emit(&self.channel, &event) {
            log::warn!("failed to emit progress event on {}: {err}", self.channel);
        }
    }
}
