//! Lightweight cooperative cancellation. Checked at every N-point boundary
//! and inside the `engine_core::ProgressSink` shim `EngineSinkAdapter`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared cancellation flag (`Arc<AtomicBool>`); clones observe the same
/// flag. Hosts flip it (Python's `KeyboardInterrupt`, a Tauri cancel command,
/// a JS worker message); the orchestrator only reads.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Fresh, un-cancelled token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Flip the flag (Release — pairs with the Acquire load in `is_cancelled`).
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// True once any clone has cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}
