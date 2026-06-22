use std::sync::Arc;
use std::time::{Duration, Instant};

use engine_contract::fixtures::example1_simple_ols;
use engine_orchestrator::{CancellationToken, OrchestratorError};

#[test]
fn token_starts_uncancelled() {
    let t = CancellationToken::new();
    assert!(!t.is_cancelled());
}

#[test]
fn cancel_is_visible_to_clones() {
    let t = CancellationToken::new();
    let t2 = t.clone();
    assert!(!t2.is_cancelled());
    t.cancel();
    assert!(t.is_cancelled());
    assert!(t2.is_cancelled());
}

#[test]
fn cancel_is_idempotent() {
    let t = CancellationToken::new();
    t.cancel();
    t.cancel();
    assert!(t.is_cancelled());
}

/// Verifies that cancel terminates find_power within ~250ms even when
/// n_sims is large (100_000). A separate thread fires cancel after 5ms;
/// the elapsed time must stay below 250ms to accommodate CI noise.
#[test]
fn cancel_terminates_within_reasonable_time() {
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

    // Cancellation surfaces as either OrchestratorError::Cancelled (cancel fired
    // between scenarios) or OrchestratorError::Engine(EngineError::Cancelled)
    // (cancel fired mid-batch via the progress sink). Both are valid; any other
    // error variant is a bug.
    let is_cancel_err = match &res {
        Err(OrchestratorError::Cancelled { scenario_idx, .. }) => {
            // Single-scenario fixture: the only in-flight scenario must be idx 0.
            assert_eq!(
                *scenario_idx, 0,
                "single-scenario cancel: scenario_idx must be 0, got {scenario_idx}"
            );
            true
        }
        Err(OrchestratorError::Engine(e)) => {
            matches!(e, engine_core::EngineError::Cancelled)
        }
        _ => false,
    };
    assert!(is_cancel_err, "expected a cancellation error, got {res:?}");
    // OLS sim ~1ms; allow 250ms slack for CI noise.
    assert!(
        elapsed < Duration::from_millis(250),
        "cancel latency {elapsed:?} too high",
    );
}
