//! `find_sample_size` — sweep N over a grid per scenario.
//!
//! One `engine_core::run_batch` per scenario, all N values in one call
//! (engine kernel accepts an ascending `sample_sizes` slice).

use crate::aggregation::aggregate_batch;
use crate::cancel::CancellationToken;
use crate::find_power::{factor_preflight_sample_size, lower_contracts, EngineSinkAdapter};
use crate::grid::{build_grid, derive_sample_size_outputs, SampleSizeDerivations};
use crate::progress::{ProgressEvent, ProgressSink};
use crate::result::{
    OrchestratorError, PowerResult, SampleSizeMethod, SampleSizeResult, Scenario, ScenarioResult,
};
use crate::scenario_loop::for_each_scenario;
use engine_contract::{ClusterSizing, ClusterSpec, SimulationContract};
use engine_core::ProgressSink as EngineProgressSink;

/// Resolve the shared cluster atom + grid floor + pre-grid warnings for a set
/// of contracts. Returns `(atom, hard_min, warnings)`. `atom == 1, hard_min == 1`
/// when no cluster is configured. Errors on the cluster guards that must fail
/// (mixed atoms, FixedSize cluster_size below the floor).
pub(crate) fn resolve_cluster_grid_params(
    contracts: &[SimulationContract],
) -> Result<(usize, usize, Vec<String>), OrchestratorError> {
    let cc = crate::config().limits;
    let min_rows = cc.min_rows_per_cluster as usize;
    let min_clusters = cc.min_clusters as usize;

    let mut cluster: Option<&ClusterSpec> = None;
    for c in contracts {
        if let Some(cs) = c.generation.cluster.as_ref() {
            match cluster {
                None => cluster = Some(cs),
                Some(prev) if prev.atom() != cs.atom() => {
                    return Err(OrchestratorError::MixedClusterAtoms {
                        a: prev.atom(),
                        b: cs.atom(),
                    })
                }
                Some(_) => {}
            }
        }
    }

    let Some(cluster) = cluster else {
        return Ok((1, 1, Vec::new())); // unclustered
    };
    let sizing = &cluster.sizing;

    let mut warnings = Vec::new();
    let (atom, hard_min) = match sizing {
        ClusterSizing::FixedClusters { n_clusters } => {
            let n = (*n_clusters).max(1) as usize;
            if n < min_clusters {
                warnings.push(format!(
                    "only {n} clusters (< {min_clusters}); τ² estimates may be unstable"
                ));
            }
            (cluster.atom(), n * min_rows)
        }
        ClusterSizing::FixedSize { cluster_size } => {
            let cs = *cluster_size as usize;
            if cs < min_rows {
                return Err(OrchestratorError::ClusterSizeTooSmall {
                    got: cs,
                    min: min_rows,
                });
            }
            (cluster.atom(), min_clusters * cs)
        }
    };
    Ok((atom, hard_min, warnings))
}

/// Multi-core sample-size search: builds the shared N grid (Grid is the only
/// method), runs one `run_batch` per scenario across all grid points, and
/// derives `first_achieved` / `first_joint_achieved` from the per-N results.
///
/// Dispatch twin: `single_core_find_sample_size` — any parameter or behaviour
/// added here must land there too, or hosts diverge.
///
/// # Errors
/// Contract lowering, cluster/grid guards (`InvalidGridBounds`,
/// `ClusterGridEmpty`, `MixedClusterAtoms`, …), and engine failures all
/// surface as `OrchestratorError`.
#[allow(clippy::too_many_arguments)]
pub fn find_sample_size(
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
    let (atom, hard_min, mut grid_warnings) = resolve_cluster_grid_params(contracts)?;
    let (grid, build_warnings) = build_grid(from, to, by, mode, atom, hard_min)?;
    grid_warnings.extend(build_warnings);

    let total_scenarios = scenarios.len();
    let total_grid_points = grid.len();
    // Progress unit is one model fit: each of the `n_sims` draws per scenario
    // is fitted at every grid N, so the call total is
    // `n_sims * n_scenarios * grid.len()`. (The *draw* budget is still
    // `n_sims * n_scenarios` — one draw evaluates all N values.)
    let total_sims = (n_sims as u64) * (total_scenarios as u64) * (total_grid_points as u64);

    if let Some(p) = progress.as_deref_mut() {
        p.on_event(ProgressEvent::Started {
            total_sims,
            total_scenarios,
            total_grid_points,
        });
    }

    // NOTE: per-scenario dispatch is sequential today; mirrors find_power's
    // dispatch shape. Do NOT replace the inner per-sim loop in `run_batch`
    // with `par_chunks_mut` chunked dispatch.
    let out = for_each_scenario(
        &scenarios,
        &mut progress,
        cancel,
        None,
        |idx, scenario, progress| {
            let (grid_or_trace, d) = run_grid_for_scenario(
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
            // Per-scenario preflight: factor allocation mode is a scenario knob,
            // so sparse-factor warnings are scenario-specific even though the
            // grid itself is shared.
            let mut scenario_warnings = grid_warnings.clone();
            scenario_warnings.extend(factor_preflight_sample_size(&scenario.spec, &grid));
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

#[allow(clippy::too_many_arguments)]
fn run_grid_for_scenario(
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

    // Dispatch twin: `run_grid_for_scenario_st` must mirror the progress
    // sentinel (`n: 0`), offset, fits_per_sim, and aggregation path — change
    // together.
    // Move sink into the adapter for the duration of the engine call.
    // Use `n: 0` as sentinel — the engine emits aggregated progress across
    // all N values in this single batch call; we can't disambiguate which N
    // a tick is for. One engine draw = `grid.len()` fits, and earlier
    // scenarios contributed `idx * n_sims * grid.len()` fits already.
    let fits_per_sim = grid.len() as u64;
    let offset = (scenario_idx as u64) * (n_sims as u64) * fits_per_sim;
    let taken: Option<&mut dyn ProgressSink> = progress.take();
    let adapter = EngineSinkAdapter::new(0, offset, fits_per_sim, call_total, taken, cancel);
    let batch_res = engine_core::run_batch(
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
    // kernel read used spec.target_indices (generation-kernel) in run_batch.
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
    // Grid-empirical first-N lookups + model-based crossing fits, all four
    // from the one shared helper (also used by the single-core twin and the
    // merge recompute).
    let derivations = derive_sample_size_outputs(&aggs, grid, target_power, atom);

    Ok((aggs, derivations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_contract::{GroupingRelation, GroupingSpec};

    fn clustered(sizing: ClusterSizing, tau: f64) -> SimulationContract {
        let mut c = engine_contract::fixtures::example1_simple_ols();
        c.generation.cluster = Some(ClusterSpec::intercept_only(sizing, tau));
        c
    }

    #[test]
    fn resolve_cluster_grid_params_uses_full_grouping_atom() {
        // Primary 20 clusters × crossed 12 ⇒ atom 240; hard_min = 20·min_rows.
        let mut c = clustered(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.25);
        c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 12 },
            tau_squared: 0.1,
            slopes: vec![],
        }];
        let (atom, hard_min, _w) = resolve_cluster_grid_params(std::slice::from_ref(&c)).unwrap();
        assert_eq!(atom, 240);
        let min_rows = crate::config().limits.min_rows_per_cluster as usize;
        assert_eq!(hard_min, 20 * min_rows);
    }
}
