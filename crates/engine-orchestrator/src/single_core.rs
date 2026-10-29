//! Single-threaded variants of `find_power` / `find_sample_size`.
//!
//! Identical to the multi-core entries in input/output shape; both share the
//! entry bodies (`find_power_with` / `find_sample_size_with`), passing
//! `engine_core::run_batch_st` instead of `run_batch`. Hosts
//! that own their own worker pool (WASM Web Workers, Python multiprocessing,
//! Slurm) call these and merge the results with `merge_power_results`.
//!
//! The twin pairing is a contract: any parameter, field, or behaviour added
//! to a multi-core entry must be mirrored here (and vice versa), or hosts
//! diverge.

use crate::cancel::CancellationToken;
use crate::find_power::lower_contracts;
use crate::find_sample_size::sample_size_grid;
use crate::progress::ProgressSink;
use crate::result::{
    OrchestratorError, PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};
use engine_contract::{CurvatureHintParts, SimulationContract};

/// Single-threaded twin of `find_power` (uses `run_batch_st`). Hosts call
/// this once per worker and pool the parts with `merge_power_results`. Same
/// parameters, same output shape — any change to `find_power` must land here
/// too. `curvature_hints_power` starts as this does (snap N, then lower) —
/// change together.
///
/// # Errors
/// Same failure set as `find_power`.
pub fn single_core_find_power(
    contracts: &[SimulationContract],
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    progress: Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<PowerResult>, OrchestratorError> {
    crate::find_power::find_power_with(
        engine_core::run_batch_st,
        contracts,
        sample_size,
        n_sims,
        base_seed,
        progress,
        cancel,
    )
}

/// Single-threaded twin of `find_sample_size` (uses `run_batch_st`). Same
/// parameters, same output shape — any change to `find_sample_size` must land
/// here too; pool per-worker parts with `merge_sample_size_results`.
/// `curvature_hints_sample_size` starts as this does (lower, then build the
/// grid) — change together.
///
/// # Errors
/// Same failure set as `find_sample_size`.
#[allow(clippy::too_many_arguments)]
pub fn single_core_find_sample_size(
    contracts: &[SimulationContract],
    target_power: f64,
    bounds: (usize, usize),
    n_sims: usize,
    method: SampleSizeMethod,
    unit_divisor: usize,
    base_seed: u64,
    progress: Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<SampleSizeResult>, OrchestratorError> {
    crate::find_sample_size::find_sample_size_with(
        engine_core::run_batch_st,
        contracts,
        target_power,
        bounds,
        n_sims,
        method,
        unit_divisor,
        base_seed,
        progress,
        cancel,
    )
}

/// Curvature hints of a power run, one vector per contract in contract order
/// (one entry, at the snapped N). A WASM-only building block, like
/// `merge_*_results`, not a run entry: the main thread computes the hints
/// once from the master seed and puts them in every worker's contracts
/// (`SimulationContract.curvature_hints`), so all workers fit with the same
/// hints. Python, R and Tauri do not expose it.
///
/// Starts as `single_core_find_power` does (snap N, then lower) — change
/// together.
///
/// # Errors
/// Same failure set as `single_core_find_power` before its sim loop.
pub fn curvature_hints_power(
    contracts: &[SimulationContract],
    sample_size: usize,
    base_seed: u64,
) -> Result<Vec<Vec<Option<CurvatureHintParts>>>, OrchestratorError> {
    let (sample_size, _) = crate::find_power::snap_single_n(contracts, sample_size)?;
    let scenarios = lower_contracts(contracts, base_seed)?;
    scenarios
        .iter()
        .map(|s| {
            engine_core::batch::curvature_hint_parts(&s.spec, &[sample_size as u32], s.base_seed)
                .map_err(OrchestratorError::from)
        })
        .collect()
}

/// Curvature hints of a sample-size run, one vector per contract in contract
/// order (one entry per grid point). The sample-size counterpart of
/// `curvature_hints_power`; same WASM-only role.
///
/// Starts as `single_core_find_sample_size` does (lower, then build the grid)
/// — change together.
///
/// # Errors
/// Same failure set as `single_core_find_sample_size` before its sim loop.
pub fn curvature_hints_sample_size(
    contracts: &[SimulationContract],
    bounds: (usize, usize),
    method: SampleSizeMethod,
    base_seed: u64,
) -> Result<Vec<Vec<Option<CurvatureHintParts>>>, OrchestratorError> {
    let scenarios = lower_contracts(contracts, base_seed)?;
    let (grid, _, _) = sample_size_grid(contracts, bounds, method)?;
    let grid: Vec<u32> = grid.iter().map(|&n| n as u32).collect();
    scenarios
        .iter()
        .map(|s| {
            engine_core::batch::curvature_hint_parts(&s.spec, &grid, s.base_seed)
                .map_err(OrchestratorError::from)
        })
        .collect()
}
