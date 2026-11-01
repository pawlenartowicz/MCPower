//! extendr adapter for the MCPower engine. Stateless functions mirroring
//! engine-py; the R6 host holds the spec.
//!
//! Mirror crate: `engine-py` — keep the bridge files in step. File sets differ
//! deliberately: `errors.rs` (exception mapping) is Python-only with no R
//! equivalent, and the debug surface lives in `debug_bridge.rs` here but in
//! `orchestrator_bridge.rs` on the Python side.
use extendr_api::error::Result;
use extendr_api::prelude::*;

/// The R error for a message, verbatim. extendr 0.9 hands the error text to
/// `Rf_error` as its format string, so a '%' (say, in a user's column or
/// grouping name) would be read as a conversion and can abort the R session;
/// doubling it makes R print the text exactly. Every R error this crate
/// raises goes through here.
pub(crate) fn r_error(msg: impl std::fmt::Display) -> Error {
    Error::Other(msg.to_string().replace('%', "%%"))
}

mod debug_bridge;
mod orchestrator_bridge;
mod plot_bridge;
mod progress;
mod report_bridge;
mod spec_builder_bridge;

/// Configure the rayon thread pool used by `find_power` / `find_sample_size`.
/// Must be called before any engine invocation; a second call raises an error
/// because the pool is already initialised. Mirrors `mcpower._engine.set_n_threads(n)`.
#[extendr]
fn set_n_threads(n: i32) -> Result<()> {
    if n < 1 {
        return Err(r_error("n must be >= 1"));
    }
    engine_core::set_n_threads(n as usize).map_err(r_error)
}

extendr_module! {
    mod engine_r;
    fn set_n_threads;
    use debug_bridge;
    use spec_builder_bridge;
    use orchestrator_bridge;
    use report_bridge;
    use plot_bridge;
}
