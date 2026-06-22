//! Main-thread progress + user-cancellation bridge for the R port.
//!
//! `ProgressEvent`s fire on rayon worker threads, but R's C API (the user's
//! progress callback, `R_CheckUserInterrupt`) is main-thread-only. So this
//! module inverts the call: the orchestrator runs on a *background* thread and
//! the R main thread stays inside the `#[extendr]` call as a poller.
//!
//! Mechanism (`run_with_progress`):
//!   1. A [`ForwarderSink`] (owned by the background thread, invoked on rayon
//!      workers) does nothing but `mpsc::Sender::send` each event into a queue —
//!      it NEVER touches an `Robj` or any R C-API symbol, so it is sound to call
//!      off the main thread.
//!   2. The orchestrator call runs inside `std::thread::scope`, letting the
//!      background thread borrow `&contracts` / `&cancel` without `'static`.
//!   3. The main thread polls the queue between checkpoints: each drained event
//!      is translated to a `(current, total)` pair and the user's R callback is
//!      invoked (safe — we are on the R main thread).
//!   4. The main thread also polls `R_CheckUserInterrupt` for Ctrl-C / Escape.
//!      That symbol **longjmps** on a pending interrupt; a longjmp across Rust
//!      frames is undefined behaviour. We contain it with `R_ToplevelExec`,
//!      which converts the longjmp into a `did_jump` boolean. On interrupt we
//!      flip the shared `CancellationToken` and keep polling — the engine
//!      acknowledges at its next checkpoint and the call returns
//!      `OrchestratorError::Cancelled`, which the bridge maps to an R error
//!      (mirrors engine-py's `KeyboardInterrupt`).
//!
//! The `(current, total)` translation mirrors engine-py's `OrchestratorProgressSink`
//! exactly (change together): `SimsCompleted { completed, total }` is already
//! cumulative across the whole call (fit-based), so it forwards verbatim.

use engine_core::EngineError;
use engine_orchestrator::{CancellationToken, OrchestratorError, ProgressEvent, ProgressSink};
use extendr_api::prelude::*;
use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

// `R_CheckUserInterrupt` and `R_ToplevelExec` are stable public R C-API symbols
// exported by libR, but extendr-ffi 0.9 does not re-export them. Declare them
// directly; they link against the same libR the rest of extendr binds to.
extern "C" {
    /// Checks for a pending user interrupt; **longjmps** (never returns
    /// normally on interrupt). MUST only be called inside an `R_ToplevelExec`
    /// containment — a bare call unwinds across Rust frames (UB).
    fn R_CheckUserInterrupt();
    /// Runs `fun(data)` with a fresh top-level context. Returns `TRUE` (non-zero)
    /// on normal return, `FALSE` (zero) if `fun` longjmped (error/interrupt),
    /// converting the jump into a boolean instead of unwinding the caller.
    fn R_ToplevelExec(fun: Option<unsafe extern "C" fn(*mut c_void)>, data: *mut c_void) -> i32;
}

/// Drives `R_CheckUserInterrupt` under it so the interrupt longjmp is caught.
#[expect(
    unsafe_code,
    reason = "interrupt containment — the only hand-written unsafe in the R adapter; bare R_CheckUserInterrupt longjmps and must run inside R_ToplevelExec"
)]
unsafe extern "C" fn check_interrupt_trampoline(_: *mut c_void) {
    R_CheckUserInterrupt();
}

/// `true` if the user pressed Ctrl-C / Escape since the last poll.
///
/// `R_CheckUserInterrupt` longjmps on a pending interrupt; `R_ToplevelExec`
/// converts that jump into a `FALSE` return (it returns `TRUE` only when the
/// inner function returns normally). Inverting that gives "an interrupt fired".
/// Must run on the R main thread.
#[expect(
    unsafe_code,
    reason = "R_ToplevelExec FFI — converts the interrupt longjmp into a boolean so it never unwinds across Rust frames"
)]
fn user_interrupt_pending() -> bool {
    let returned_normally =
        unsafe { R_ToplevelExec(Some(check_interrupt_trampoline), std::ptr::null_mut()) };
    returned_normally == 0
}

/// Off-main-thread sink: forwards each orchestrator event into an `mpsc` queue.
/// Runs on rayon workers, so it must touch nothing in the R C-API — it only
/// clones the event and sends it. A closed receiver (main thread gone) is
/// ignored: the run is being torn down anyway.
struct ForwarderSink {
    tx: Sender<ProgressEvent>,
}

impl ProgressSink for ForwarderSink {
    fn on_event(&mut self, event: ProgressEvent) {
        let _ = self.tx.send(event);
    }
}

/// Translates the structured event stream to the R callback's `(current, total)`
/// contract. Mirrors engine-py's `OrchestratorProgressSink` — keep the two in
/// step. `SimsCompleted` is already cumulative across the call, so the fold is
/// a plain projection.
#[derive(Default)]
struct CallbackState;

impl CallbackState {
    /// Project one event into `(current, total)` to report, or `None` for
    /// events the R callback contract doesn't surface.
    fn fold(&mut self, event: ProgressEvent) -> Option<(u64, u64)> {
        match event {
            ProgressEvent::SimsCompleted {
                completed, total, ..
            } => Some((completed, total)),
            _ => None,
        }
    }
}

/// Run an orchestrator entry point with R progress reporting and user
/// cancellation, returning whatever the entry produced.
///
/// `progress` is the user-facing callback (an R function `(current, total)`),
/// or `NULL` for silent operation. `run` is the orchestrator call: it receives
/// the `ProgressSink` to pass through and the `CancellationToken` to honour,
/// and runs on a background thread while this function polls on the R main
/// thread. The `run` closure must convert all of its inputs to plain Rust types
/// *before* being handed here — no `Robj` may cross into the background thread
/// (`Robj` is `!Send`).
pub fn run_with_progress<F, T>(progress: Robj, cancel: &CancellationToken, run: F) -> T
where
    F: FnOnce(&mut dyn ProgressSink, &CancellationToken) -> T + Send,
    T: Send,
{
    let callback: Option<Function> = progress.as_function();
    let (tx, rx): (Sender<ProgressEvent>, Receiver<ProgressEvent>) = mpsc::channel();

    // Scoped threads let the background closure borrow `cancel` (and, via the
    // caller's closure, the decoded contracts) without a `'static` bound.
    std::thread::scope(|scope| {
        let handle = scope.spawn(move || {
            let mut sink = ForwarderSink { tx };
            run(&mut sink, cancel)
        });

        let mut state = CallbackState;
        // Poll until the engine thread finishes. Draining the queue and the
        // interrupt check both happen here on the R main thread.
        loop {
            drain_queue(&rx, &mut state, callback.as_ref());

            if !cancel.is_cancelled() && user_interrupt_pending() {
                // Engine acknowledges at its next checkpoint and returns
                // OrchestratorError::Cancelled; we keep draining until then.
                cancel.cancel();
            }

            if handle.is_finished() {
                // Flush any events the worker sent between our last drain and
                // its exit, then collect the result.
                drain_queue(&rx, &mut state, callback.as_ref());
                return handle.join().expect("orchestrator thread panicked");
            }

            std::thread::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS));
        }
    })
}

/// Poll interval for the queue/interrupt loop. Short enough that Ctrl-C feels
/// responsive, long enough that an idle poll costs nothing measurable.
const POLL_INTERVAL_MS: u64 = 50;

/// Drain every currently-queued event, invoking `callback` for each that folds
/// to a `(current, total)` report. A user callback that errors is ignored — its
/// return value is advisory only (cancellation is via Ctrl-C, mirroring the R
/// idiom), so a throwing callback must not abort the run.
fn drain_queue(
    rx: &Receiver<ProgressEvent>,
    state: &mut CallbackState,
    callback: Option<&Function>,
) {
    loop {
        match rx.try_recv() {
            Ok(event) => {
                if let Some((current, total)) = state.fold(event) {
                    if let Some(cb) = callback {
                        let _ = cb.call(pairlist!(current as i32, total as i32));
                    }
                }
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return,
        }
    }
}

/// Map an `OrchestratorError` to an R error, surfacing cancellation with a
/// stable "cancelled by user" message (mirrors engine-py raising
/// `KeyboardInterrupt`). All other variants use the curated `Display` text.
///
/// User cancellation reaches here on two distinct paths and **both** must map
/// to the same message (engine-py maps both to `KeyboardInterrupt`): a
/// scenario-boundary check yields `OrchestratorError::Cancelled`, while a
/// mid-batch sink callback returning false yields
/// `OrchestratorError::Engine(EngineError::Cancelled)`. The live Ctrl-C path
/// usually lands mid-batch, so omitting the second arm would surface the raw
/// "engine error: cancelled by host" text instead.
pub fn orchestrator_err(e: OrchestratorError) -> Error {
    match e {
        OrchestratorError::Cancelled { .. } | OrchestratorError::Engine(EngineError::Cancelled) => {
            Error::Other("cancelled by user".into())
        }
        // Display, not Debug: thiserror's #[error] messages are the curated,
        // host-facing text (mirrors engine-py's exception messages).
        other => Error::Other(format!("{other}")),
    }
}

/// Report-this suffix for genuine engine bugs (caught Rust panics) — never added
/// to the validation or cancellation paths handled by `orchestrator_err`. `port=r`;
/// version is the engine crate version. Mirrors engine-py's `REPORT_HINT`
/// (errors.rs) — change the two together.
const REPORT_HINT: &str = concat!(
    " — this looks like an internal MCPower error; please report it at https://mcpower.app/report?port=r&version=",
    env!("CARGO_PKG_VERSION")
);

/// Message for a caught engine panic, with the report hint appended.
fn internal_error_message(panic_msg: &str) -> String {
    format!("internal engine error: {panic_msg}{REPORT_HINT}")
}

/// Convert a caught panic payload into an R error carrying the report hint.
fn panic_to_r_error(payload: Box<dyn std::any::Any + Send>) -> Error {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "engine panicked".to_string());
    Error::Other(internal_error_message(&msg))
}

/// Run an orchestrator entry (with R progress + cancellation) and map its
/// outcome to an R error: an `OrchestratorError` via `orchestrator_err`
/// (validation/cancellation), a caught Rust panic via `panic_to_r_error` (an
/// internal engine bug, with the report hint). The panic is caught on the
/// background thread so it never reaches `run_with_progress`'s join. Mirrors
/// engine-py's panic boundary in `lib.rs` — change together.
pub fn run_engine<F, T>(
    progress: Robj,
    cancel: &CancellationToken,
    run: F,
) -> std::result::Result<T, Error>
where
    F: FnOnce(
            &mut dyn ProgressSink,
            &CancellationToken,
        ) -> std::result::Result<T, OrchestratorError>
        + Send,
    T: Send,
{
    let outcome = run_with_progress(progress, cancel, |sink, cancel| {
        std::panic::catch_unwind(AssertUnwindSafe(|| run(sink, cancel)))
    });
    match outcome {
        Ok(r) => r.map_err(orchestrator_err),
        Err(payload) => Err(panic_to_r_error(payload)),
    }
}
