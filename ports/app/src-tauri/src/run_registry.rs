//! In-process registry that maps live run IDs to their `CancellationToken`s, enabling the UI to cancel in-flight engine calls.

use engine_orchestrator::CancellationToken;
use parking_lot::Mutex;
use std::collections::HashMap;
use uuid::Uuid;

pub type RunId = String;

/// Live-run table: run id → its cancellation token.
#[derive(Default)]
pub struct RunRegistry {
    inner: Mutex<HashMap<RunId, CancellationToken>>,
}

impl RunRegistry {
    /// Register a new run; returns the assigned id and a fresh, registered token.
    pub fn register(&self) -> (RunId, CancellationToken) {
        let id = Uuid::new_v4().to_string();
        let tok = CancellationToken::new();
        self.inner.lock().insert(id.clone(), tok.clone());
        (id, tok)
    }

    /// Flip the cancellation flag for `id`. Returns true if the id was known.
    pub fn cancel(&self, id: &str) -> bool {
        match self.inner.lock().get(id) {
            Some(tok) => {
                tok.cancel();
                true
            }
            None => false,
        }
    }

    /// Drop a finished run's entry. No-op if the id is not registered.
    pub fn drop_run(&self, id: &str) {
        self.inner.lock().remove(id);
    }

    /// Test helper.
    #[cfg(test)]
    pub fn contains(&self, id: &str) -> bool {
        self.inner.lock().contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_cancel_drop_lifecycle() {
        let reg = RunRegistry::default();
        let (id, tok) = reg.register();
        assert!(reg.contains(&id));
        assert!(!tok.is_cancelled());
        assert!(reg.cancel(&id));
        assert!(tok.is_cancelled());
        reg.drop_run(&id);
        assert!(!reg.contains(&id));
    }

    #[test]
    fn cancel_unknown_id_returns_false() {
        let reg = RunRegistry::default();
        assert!(!reg.cancel("not-a-real-id"));
    }

    #[test]
    fn drop_unknown_id_is_a_noop() {
        let reg = RunRegistry::default();
        reg.drop_run("not-a-real-id"); // must not panic
    }
}
