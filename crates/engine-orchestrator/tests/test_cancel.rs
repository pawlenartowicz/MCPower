use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine_contract::fixtures::example1_simple_ols;
use engine_orchestrator::{CancellationToken, OrchestratorError, ProgressEvent, ProgressSink};

/// Held for the whole body of each test that runs a 100,000-sim `find_power`.
/// Cargo runs one binary's tests in parallel and both runs share the engine's
/// rayon pool, so without it the latency test also times the other run.
static ENGINE_RUN: Mutex<()> = Mutex::new(());

/// Takes `ENGINE_RUN`, ignoring poison so one failing test does not fail the other.
fn serialise() -> std::sync::MutexGuard<'static, ()> {
    ENGINE_RUN.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn cancel_is_visible_to_clones() {
    let t = CancellationToken::new();
    let t2 = t.clone();
    assert!(!t2.is_cancelled());
    t.cancel();
    assert!(t.is_cancelled());
    assert!(t2.is_cancelled());
    // Idempotent: a second cancel keeps the flag set (not a toggle).
    t2.cancel();
    assert!(t.is_cancelled());
}

/// Verifies that cancel terminates find_power within ~250ms even when
/// n_sims is large (100_000). A separate thread fires cancel after 5ms;
/// the elapsed time must stay below 250ms to accommodate CI noise.
#[test]
fn cancel_terminates_within_reasonable_time() {
    let _serial = serialise();
    let c = example1_simple_ols();
    let cancel = Arc::new(CancellationToken::new());
    let cancel_clone = cancel.clone();

    // Spawn a thread that cancels after 5ms.
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(5));
        cancel_clone.cancel();
    });

    let t0 = Instant::now();
    let res = engine_orchestrator::find_power(&[c], 100, 100_000, 2137, None, &cancel);
    let elapsed = t0.elapsed();

    match &res {
        // Single-scenario fixture: the only in-flight scenario must be idx 0.
        Err(OrchestratorError::Cancelled { scenario_idx, .. }) => assert_eq!(*scenario_idx, 0),
        other => panic!("expected OrchestratorError::Cancelled, got {other:?}"),
    }
    // OLS sim ~1ms; allow 250ms slack for CI noise.
    assert!(
        elapsed < Duration::from_millis(250),
        "cancel latency {elapsed:?} too high",
    );
}

/// Sink that cancels the run on its first `SimsCompleted` tick, so the cancel
/// lands inside the engine batch rather than between scenarios.
struct CancelOnFirstTick {
    cancel: CancellationToken,
    saw_cancelled: bool,
}

impl ProgressSink for CancelOnFirstTick {
    fn on_event(&mut self, event: ProgressEvent) {
        match event {
            ProgressEvent::SimsCompleted { .. } => self.cancel.cancel(),
            ProgressEvent::Cancelled => self.saw_cancelled = true,
            _ => {}
        }
    }
}

#[test]
fn mid_run_cancel_emits_cancelled_event_and_error() {
    let _serial = serialise();
    let c = example1_simple_ols();
    let cancel = CancellationToken::new();
    let mut sink = CancelOnFirstTick {
        cancel: cancel.clone(),
        saw_cancelled: false,
    };
    let res = engine_orchestrator::find_power(&[c], 100, 100_000, 2137, Some(&mut sink), &cancel);
    assert!(
        matches!(
            res,
            Err(OrchestratorError::Cancelled {
                scenario_idx: 0,
                n: Some(100)
            })
        ),
        "expected OrchestratorError::Cancelled, got {res:?}"
    );
    assert!(
        sink.saw_cancelled,
        "sink never received ProgressEvent::Cancelled"
    );
}
