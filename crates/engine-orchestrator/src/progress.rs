//! Orchestrator-level event channel.
//!
//! Distinct from `engine_core::ProgressSink` (which is `report(cur, tot) -> bool`,
//! called inside `run_batch`). The orchestrator emits structured events; hosts
//! that only care about `(cur, tot)` consume `SimsCompleted` and ignore the rest.
//!
//! **Variant evolution rule:** append-only — new variants may be added without
//! a major bump. Match consumers must use `_ => {}`.

/// Sink for orchestrator events. Host adapters (engine-py, engine-app-spec,
/// engine-r) implement this; pure-Rust callers can use [`NoOpSink`].
///
/// `Send + Sync` is required so `EngineSinkAdapter` can hold an outer
/// sink across `engine_core::ProgressSink::report`, which has its own
/// `Send + Sync` bound. Sinks with mutable state should use interior
/// mutability (e.g. `Mutex<Vec<...>>` in tests).
pub trait ProgressSink: Send + Sync {
    fn on_event(&mut self, event: ProgressEvent);
}

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Orchestrator started; totals known. Emitted exactly once per call.
    Started {
        /// Total model fits this call will perform:
        /// `n_sims × n_scenarios × grid_points` (grid_points = 1 for
        /// `find_power`). One fit is the progress unit everywhere — draws,
        /// grids, and scenario loops are implementation detail.
        total_sims: u64,
        /// Number of scenarios (i.e. contracts) in this call. Always populated.
        total_scenarios: usize,
        /// Number of N-points evaluated. Populated by `find_sample_size`
        /// (grid); `0` from `find_power` (single N).
        total_grid_points: usize,
    },
    /// One scenario started. Always emitted, even for single-scenario calls.
    ScenarioStarted {
        label: String,
        idx: usize,
        total: usize,
    },
    /// Cumulative fit progress across the whole call. `completed` counts model
    /// fits over all scenarios and grid points completed so far (monotone
    /// within a call); `total` equals `Started.total_sims`. Hosts can render
    /// `completed / total` directly — no per-scenario reconstruction needed.
    SimsCompleted {
        /// Sample size of the running batch; `0` when ticks span a grid
        /// (sample-size search) and no single N applies.
        n: usize,
        completed: u64,
        total: u64,
    },
    /// One grid point finished; aggregated power available.
    NPointCompleted {
        n: usize,
        power_uncorrected: Vec<f64>,
        power_corrected: Vec<f64>,
    },
    /// One scenario finished.
    ScenarioCompleted { label: String, idx: usize },
    /// Cancellation acknowledged; no more work will be performed.
    Cancelled,
    /// Orchestrator finished successfully. Emitted exactly once per call.
    Completed,
}

/// No-op sink for callers that don't need events. `Option::None` is also
/// accepted by the entry points; this exists so library code can hold a
/// concrete sink without an `Option` ceremony.
pub struct NoOpSink;

impl ProgressSink for NoOpSink {
    fn on_event(&mut self, _: ProgressEvent) {}
}
