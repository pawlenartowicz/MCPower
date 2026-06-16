//! `ProgressEmitter` implementation that forwards engine progress events to the Tauri window via a named JS event channel.

use engine_app_spec::ProgressEmitter;
use serde_json::Value;
use tauri::{Emitter, Window};

/// `ProgressEmitter` impl that forwards each event to a Tauri JS listener
/// on the configured channel. Construct once per run inside the spawned
/// blocking task and pass `&emitter` into the engine driver.
pub struct WindowEmitter<R: tauri::Runtime> {
    window: Window<R>,
    channel: String,
}

impl<R: tauri::Runtime> WindowEmitter<R> {
    pub fn new(window: Window<R>, channel: impl Into<String>) -> Self {
        Self {
            window,
            channel: channel.into(),
        }
    }
}

impl<R: tauri::Runtime> ProgressEmitter for WindowEmitter<R> {
    fn emit(&self, event: Value) {
        if let Err(err) = self.window.emit(&self.channel, &event) {
            log::warn!("failed to emit progress event on {}: {err}", self.channel);
        }
    }
}
