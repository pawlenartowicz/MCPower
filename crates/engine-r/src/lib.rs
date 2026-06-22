//! extendr adapter for the MCPower engine. Stateless functions mirroring
//! engine-py; the R6 host holds the spec.
//!
//! Mirror crate: `engine-py` — keep the bridge files in step. File sets differ
//! deliberately: `errors.rs` (exception mapping) is Python-only with no R
//! equivalent, and the debug surface lives in `debug_bridge.rs` here but in
//! `orchestrator_bridge.rs` on the Python side.
use extendr_api::error::Result;
use extendr_api::prelude::*;

mod debug_bridge;
mod orchestrator_bridge;
mod plot_bridge;
mod progress;
mod report_bridge;
mod spec_builder_bridge;

/// Spike probe — returns the engine's view of the thread count machinery is wired.
#[extendr]
fn engine_r_ping() -> String {
    "engine-r-ok".to_string()
}

/// Configure the rayon thread pool used by `find_power` / `find_sample_size`.
/// Must be called before any engine invocation; a second call raises an error
/// because the pool is already initialised. Mirrors `mcpower._engine.set_n_threads(n)`.
#[extendr]
fn set_n_threads(n: i32) -> Result<()> {
    if n < 1 {
        return Err(Error::Other("n must be >= 1".into()));
    }
    engine_core::set_n_threads(n as usize).map_err(|e| Error::Other(format!("{e}")))
}

extendr_module! {
    mod engine_r;
    fn engine_r_ping;
    fn set_n_threads;
    use debug_bridge;
    use spec_builder_bridge;
    use orchestrator_bridge;
    use report_bridge;
    use plot_bridge;
}
