//! Single-threaded variants of `find_power` / `find_sample_size`.
//!
//! Identical to the multi-core entries in input/output shape; the only
//! difference is `engine_core::run_batch_st` instead of `run_batch`. Hosts
//! that own their own worker pool (WASM Web Workers, Python multiprocessing,
//! Slurm) call these and merge the results with `merge_power_results`.
//!
//! The twin pairing is a contract: any parameter, field, or behaviour added
//! to a multi-core entry must be mirrored here (and vice versa), or hosts
//! diverge.

use crate::aggregation::aggregate_batch;
use crate::cancel::CancellationToken;
use crate::find_power::{lower_contracts, EngineSinkAdapter};
use crate::grid::{build_grid, derive_sample_size_outputs, SampleSizeDerivations};
use crate::progress::{ProgressEvent, ProgressSink};
use crate::result::{
    OrchestratorError, PowerResult, SampleSizeMethod, SampleSizeResult, Scenario, ScenarioResult,
};
use crate::scenario_loop::for_each_scenario;
use engine_contract::SimulationContract;
use engine_core::ProgressSink as EngineProgressSink;

/// Single-threaded twin of `find_power` (uses `run_batch_st`). Hosts call
/// this once per worker and pool the parts with `merge_power_results`. Same
/// parameters, same output shape — any change to `find_power` must land here
/// too.
///
/// # Errors
/// Same failure set as `find_power`.
pub fn single_core_find_power(
    contracts: &[SimulationContract],
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    mut progress: Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<PowerResult>, OrchestratorError> {
    // Snap the requested N to the cluster atom (no-op for unclustered or atom ≤ 1).
    let (sample_size, snap_warning) = crate::find_power::snap_single_n(contracts, sample_size)?;

    let scenarios = lower_contracts(contracts, base_seed)?;

    let total_sims = (n_sims as u64) * (scenarios.len() as u64);
    if let Some(p) = progress.as_deref_mut() {
        p.on_event(ProgressEvent::Started {
            total_sims,
            total_scenarios: scenarios.len(),
            total_grid_points: 0,
        });
    }

    let out = for_each_scenario(
        &scenarios,
        &mut progress,
        cancel,
        Some(sample_size),
        |idx, scenario, progress| {
            // Per-sim ticks mirror the multi-core twin: hosts that pool
            // single-core workers (WASM Web Workers) sum the per-worker
            // cumulative counts to drive one smooth progress bar.
            let taken: Option<&mut dyn ProgressSink> = progress.take();
            let offset = (idx as u64) * (n_sims as u64);
            let adapter =
                EngineSinkAdapter::new(sample_size, offset, 1, total_sims, taken, cancel);
            let batch_res = engine_core::run_batch_st(
                &scenario.spec,
                &[sample_size as u32],
                n_sims as u32,
                scenario.base_seed,
                Some(&adapter as &dyn EngineProgressSink),
            );
            // Reclaim the sink (even on error) before propagating.
            *progress = adapter.outer.into_inner().unwrap_or(None);
            let batch = batch_res?;
            // Report (design_test term) space — skeleton-aligned host indices;
            // the kernel read spec.target_indices (generation-kernel) above.
            let mut aggs = aggregate_batch(
                &batch,
                &scenario.report_target_indices,
                &scenario.report_contrast_pairs,
                &scenario.spec.estimator,
            );
            debug_assert_eq!(aggs.len(), 1);
            let mut pr = aggs.remove(0);
            pr.n = sample_size;
            // Attach the snap warning (if any) to this scenario's result.
            if let Some(w) = &snap_warning {
                pr.grid_warnings.push(w.clone());
            }
            // Pre-run sparse-factor advisory — mirrors find_power (twin mandate).
            pr.grid_warnings
                .extend(crate::find_power::factor_preflight_warnings(
                    &scenario.spec,
                    sample_size,
                ));
            Ok(pr)
        },
    )?;

    if let Some(p) = progress {
        p.on_event(ProgressEvent::Completed);
    }
    Ok(ScenarioResult { scenarios: out })
}

/// Single-threaded twin of `find_sample_size` (uses `run_batch_st`). Same
/// parameters, same output shape — any change to `find_sample_size` must land
/// here too; pool per-worker parts with `merge_sample_size_results`.
///
/// # Errors
/// Same failure set as `find_sample_size`.
pub fn single_core_find_sample_size(
    contracts: &[SimulationContract],
    target_power: f64,
    bounds: (usize, usize),
    n_sims: usize,
    method: SampleSizeMethod,
    base_seed: u64,
    mut progress: Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<SampleSizeResult>, OrchestratorError> {
    let scenarios = lower_contracts(contracts, base_seed)?;

    let (from, to) = bounds;
    let SampleSizeMethod::Grid { by, mode } = method;
    let (atom, hard_min, mut grid_warnings) =
        crate::find_sample_size::resolve_cluster_grid_params(contracts)?;
    let (grid, build_warnings) = build_grid(from, to, by, mode, atom, hard_min)?;
    grid_warnings.extend(build_warnings);

    let total_scenarios = scenarios.len();
    let total_grid_points = grid.len();
    // Progress unit is one model fit — mirrors find_sample_size (twin mandate):
    // `n_sims * n_scenarios * grid.len()` (one draw evaluates all N values,
    // each evaluation is a fit).
    let total_sims = (n_sims as u64) * (total_scenarios as u64) * (total_grid_points as u64);

    if let Some(p) = progress.as_deref_mut() {
        p.on_event(ProgressEvent::Started {
            total_sims,
            total_scenarios,
            total_grid_points,
        });
    }

    let out = for_each_scenario(
        &scenarios,
        &mut progress,
        cancel,
        None,
        |idx, scenario, progress| {
            let (grid_or_trace, d) = run_grid_for_scenario_st(
                scenario,
                &grid,
                n_sims,
                target_power,
                atom,
                idx,
                total_sims,
                progress,
                cancel,
            )?;
            // Per-scenario preflight — mirrors find_sample_size (twin mandate).
            let mut scenario_warnings = grid_warnings.clone();
            scenario_warnings.extend(
                crate::find_power::factor_preflight_sample_size(&scenario.spec, &grid),
            );
            Ok(SampleSizeResult {
                grid_or_trace,
                first_achieved: d.first_achieved,
                first_joint_achieved: d.first_joint_achieved,
                fitted: d.fitted,
                fitted_joint: d.fitted_joint,
                first_overall_achieved: d.first_overall_achieved,
                fitted_overall: d.fitted_overall,
                cluster_atom: atom,
                target_power,
                method,
                grid_warnings: scenario_warnings,
            })
        },
    )?;

    if let Some(p) = progress {
        p.on_event(ProgressEvent::Completed);
    }
    Ok(ScenarioResult { scenarios: out })
}

fn run_grid_for_scenario_st(
    scenario: &Scenario,
    grid: &[usize],
    n_sims: usize,
    target_power: f64,
    atom: usize,
    scenario_idx: usize,
    call_total: u64,
    progress: &mut Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<(Vec<PowerResult>, SampleSizeDerivations), OrchestratorError> {
    let sample_sizes_u32: Vec<u32> = grid.iter().map(|&n| n as u32).collect();

    // Per-sim ticks mirror run_grid_for_scenario (twin mandate); `n: 0`
    // sentinel for the same reason — ticks span the whole grid.
    let fits_per_sim = grid.len() as u64;
    let offset = (scenario_idx as u64) * (n_sims as u64) * fits_per_sim;
    let taken: Option<&mut dyn ProgressSink> = progress.take();
    let adapter = EngineSinkAdapter::new(0, offset, fits_per_sim, call_total, taken, cancel);
    let batch_res = engine_core::run_batch_st(
        &scenario.spec,
        &sample_sizes_u32,
        n_sims as u32,
        scenario.base_seed,
        Some(&adapter as &dyn EngineProgressSink),
    );
    // Reclaim the sink (even on error) before propagating.
    *progress = adapter.outer.into_inner().unwrap_or(None);
    let batch = batch_res?;

    // Report (design_test term) space — skeleton-aligned host indices; the
    // kernel read spec.target_indices (generation-kernel) above.
    let mut aggs = aggregate_batch(
        &batch,
        &scenario.report_target_indices,
        &scenario.report_contrast_pairs,
        &scenario.spec.estimator,
    );
    for (i, pr) in aggs.iter_mut().enumerate() {
        pr.n = grid[i];
        if let Some(p) = progress.as_deref_mut() {
            p.on_event(ProgressEvent::NPointCompleted {
                n: pr.n,
                power_uncorrected: pr.power_uncorrected.clone(),
                power_corrected: pr.power_corrected.clone(),
            });
        }
    }
    // Same shared helper as run_grid_for_scenario in find_sample_size.rs —
    // the twin mandate is satisfied structurally, not by parallel maintenance.
    let derivations = derive_sample_size_outputs(&aggs, grid, target_power, atom);

    Ok((aggs, derivations))
}
