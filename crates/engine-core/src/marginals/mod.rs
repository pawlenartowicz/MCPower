//! Marginal-transform helpers used by `data_gen` row-loop.
//!
//! Split out of `data_gen` because the t(3) PPF needs a shared lookup table
//! (parameter-free across the run, threaded as `Arc<T3PpfTable>` in the spec).

pub mod t3;

pub use t3::T3PpfTable;
