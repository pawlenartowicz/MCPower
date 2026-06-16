//! `find_power` — single-N, multi-scenario entry point.
//!
//! Runs one `engine_core::run_batch` per scenario at the requested
//! `sample_size`, aggregates into `PowerResult`, and emits lifecycle events.

use crate::aggregation::aggregate_batch;
use crate::cancel::CancellationToken;
use crate::progress::{ProgressEvent, ProgressSink};
use crate::result::{OrchestratorError, PowerResult, Scenario, ScenarioResult};
use crate::scenario_loop::for_each_scenario;
use engine_contract::{ClusterSizing, SimulationContract};
use engine_core::contract_adapter::contract_to_simulation_spec;
use engine_core::rng::splitmix64_finalize;
use engine_core::SimulationSpec;
use engine_core::ProgressSink as EngineProgressSink;

/// Floor a single requested N to the cluster atom (so the nested-prefix
/// dataset is a valid balanced / complete-cluster design), returning the
/// snapped N and an optional advisory warning. Unclustered ⇒ unchanged.
///
/// FixedSize also enforces the same `min_clusters` floor as the search path
/// (`find_power` and `find_sample_size` share this minimum): errors via
/// `ClusterTooFewAtN` when the snapped N yields fewer than `min_clusters`
/// complete clusters.
pub(crate) fn snap_single_n(
    contracts: &[SimulationContract],
    requested: usize,
) -> Result<(usize, Option<String>), OrchestratorError> {
    let Some(cluster_spec) = contracts.iter().find_map(|c| c.generation.cluster.as_ref()) else {
        return Ok((requested, None)); // unclustered
    };
    let sizing = &cluster_spec.sizing;
    let a = cluster_spec.atom();
    if a <= 1 {
        return Ok((requested, None));
    }
    let mut snapped = (requested / a) * a;
    if snapped == 0 {
        snapped = a; // never a zero-row fit
    }
    if let ClusterSizing::FixedSize { cluster_size } = sizing {
        let min_clusters = crate::config().limits.min_clusters as usize;
        let cs = (*cluster_size).max(1) as usize;
        let got = snapped / cs;
        if got < min_clusters {
            return Err(OrchestratorError::ClusterTooFewAtN {
                n: snapped,
                cluster_size: cs,
                got,
                min: min_clusters,
            });
        }
    }
    let warning = (snapped != requested).then(|| {
        format!("sample_size {requested} not a multiple of the cluster atom {a}; using {snapped}")
    });
    Ok((snapped, warning))
}

/// Pre-run advisory warnings for factor levels too sparse at `n`. Exact under
/// fixed allocation (the deterministic walk run via
/// engine_core::fixed_allocation_counts). Fixed-allocation designs only —
/// under sampled allocation level counts are stochastic, so preflight is
/// skipped and the post-run per-factor counters carry the signal. Factors are
/// identified by 1-based spec index — the engine is label-free;
/// named diagnostics come from the per-factor counters rendered port-side.
pub(crate) fn factor_preflight_warnings(spec: &SimulationSpec, n: usize) -> Vec<String> {
    let k_min = spec.factor_min_level_count;
    if k_min == 0 {
        return Vec::new();
    }
    // Uploaded-data designs resample rows from the frame — the fixed-allocation
    // walk does not describe their level counts. Skip preflight; the post-run
    // per-factor counters still report any exclusion.
    if !spec.upload_data.is_empty() {
        return Vec::new();
    }
    let scenario_sampled = spec.scenario.sampled_factor_proportions;
    let mut out = Vec::new();
    let mut off = 0usize;
    for (f, &nl) in spec.factor_n_levels.iter().enumerate() {
        let l = nl.max(0) as usize;
        // Sampled factors have stochastic level counts; the deterministic
        // fixed-allocation preflight does not describe them — skip (mirrors the
        // per-factor resolution in data_gen.rs). None inherits the scenario default.
        let sampled = spec.factor_sampled.get(f).copied().flatten().unwrap_or(scenario_sampled);
        if sampled {
            off += l;
            continue;
        }
        let probs = &spec.factor_proportions[off..off + l];
        let counts = engine_core::fixed_allocation_counts(probs, n);
        if let Some((lvl, &c)) = counts.iter().enumerate().min_by_key(|&(_, c)| *c) {
            if c < k_min {
                out.push(format!(
                    "factor {}: level {} receives {} of {} observations (minimum {}); \
                     the factor is excluded from every simulation at N={} and its \
                     effects report power 0 — increase N or this level's proportion",
                    f + 1, lvl + 1, c, n, k_min, n
                ));
            }
        }
        off += l;
    }
    out
}

/// Sample-size variant: warn per factor that is sparse at the grid's lower
/// bound, and name the smallest N (within the grid ceiling) at which all its
/// levels clear the minimum. Fixed-allocation only, like
/// factor_preflight_warnings.
pub(crate) fn factor_preflight_sample_size(spec: &SimulationSpec, grid: &[usize]) -> Vec<String> {
    let k_min = spec.factor_min_level_count;
    let (Some(&lo), Some(&hi)) = (grid.first(), grid.last()) else { return Vec::new(); };
    if k_min == 0 {
        return Vec::new();
    }
    if !spec.upload_data.is_empty() {
        return Vec::new(); // uploaded-data designs: see factor_preflight_warnings
    }
    let scenario_sampled = spec.scenario.sampled_factor_proportions;
    let mut out = Vec::new();
    let mut off = 0usize;
    for (f, &nl) in spec.factor_n_levels.iter().enumerate() {
        let l = nl.max(0) as usize;
        // Sampled factors have stochastic level counts; the deterministic
        // fixed-allocation preflight does not describe them — skip (mirrors the
        // per-factor resolution in data_gen.rs). None inherits the scenario default.
        let sampled = spec.factor_sampled.get(f).copied().flatten().unwrap_or(scenario_sampled);
        if sampled {
            off += l;
            continue;
        }
        let probs = &spec.factor_proportions[off..off + l];
        let counts = engine_core::fixed_allocation_counts(probs, lo);
        if counts.iter().any(|&c| c < k_min) {
            match engine_core::min_inclusion_n(probs, k_min, hi) {
                Some(n_inc) => out.push(format!(
                    "factor {}: levels reach the {}-observation minimum only at N >= {}; \
                     grid points below that exclude the factor and report power 0 for its effects",
                    f + 1, k_min, n_inc
                )),
                None => out.push(format!(
                    "factor {}: a level stays under the {}-observation minimum across the whole \
                     [{}, {}] range; the factor is excluded everywhere and its effects report power 0",
                    f + 1, k_min, lo, hi
                )),
            }
        }
        off += l;
    }
    out
}

/// Adapter wrapping the orchestrator's outer sink + cancellation token into
/// the `(current, total) -> bool` contract `engine_core::run_batch` expects.
///
/// Rescales the engine's per-batch draw counter into the call-level cumulative
/// fit count `SimsCompleted` promises: the engine reports `current` completed
/// draws within one scenario's batch; one draw is `fits_per_sim` model fits
/// (the grid length — 1 for a single-N power run), and `offset` is the fit
/// count already completed by earlier scenarios in this call.
///
/// `Send + Sync`: the inner `Mutex<Option<&mut dyn ProgressSink>>` provides
/// interior mutability; `Send + Sync` are inherited because
/// `dyn ProgressSink: Send + Sync` (declared in `progress.rs`).
pub(crate) struct EngineSinkAdapter<'s, 'c> {
    n: usize,
    /// Fits completed by earlier scenarios in this call.
    offset: u64,
    /// Model fits per engine-reported draw (= grid length; 1 for find_power).
    fits_per_sim: u64,
    /// Call-level denominator; equals `Started.total_sims`.
    call_total: u64,
    pub(crate) outer: std::sync::Mutex<Option<&'s mut (dyn ProgressSink + 's)>>,
    cancel: &'c CancellationToken,
}

impl<'s, 'c> EngineSinkAdapter<'s, 'c> {
    pub(crate) fn new(
        n: usize,
        offset: u64,
        fits_per_sim: u64,
        call_total: u64,
        outer: Option<&'s mut dyn ProgressSink>,
        cancel: &'c CancellationToken,
    ) -> Self {
        Self {
            n,
            offset,
            fits_per_sim,
            call_total,
            outer: std::sync::Mutex::new(outer),
            cancel,
        }
    }
}

impl<'s, 'c> EngineProgressSink for EngineSinkAdapter<'s, 'c> {
    fn report(&self, current: u64, _batch_total: u64) -> bool {
        if self.cancel.is_cancelled() {
            return false;
        }
        if let Ok(mut guard) = self.outer.lock() {
            if let Some(sink) = guard.as_mut() {
                sink.on_event(ProgressEvent::SimsCompleted {
                    n: self.n,
                    completed: self.offset + current * self.fits_per_sim,
                    total: self.call_total,
                });
            }
        }
        true
    }
}

/// Lower `contracts` to internal `Scenario` values, validating each through
/// `contract_to_simulation_spec`. Every scenario receives the same call-level
/// seed, `splitmix64_finalize(base_seed)`: scenarios in one call are *paired*
/// runs — identical X/residual draws wherever knobs don't perturb them — so
/// cross-scenario power deltas are attributable to the knobs, not RNG
/// re-seeding. The finalizer (rather than the raw `base_seed`) keeps the
/// avalanche between consecutive host-supplied base seeds (per-worker
/// `master + i`) and matches the historical index-0 derivation, so
/// single-scenario results are unchanged. Shared with
/// `single_core_find_power` and the sample-size variants so all four entries
/// agree on lowering.
pub(crate) fn lower_contracts(
    contracts: &[SimulationContract],
    base_seed: u64,
) -> Result<Vec<Scenario>, OrchestratorError> {
    if contracts.is_empty() {
        return Err(OrchestratorError::InvalidScenarios(
            "contracts slice is empty".into(),
        ));
    }
    let call_seed = splitmix64_finalize(base_seed);
    contracts
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let mut spec = contract_to_simulation_spec(c).map_err(|e| {
                OrchestratorError::InvalidScenarios(format!(
                    "contract {idx} ({label:?}): {e}",
                    label = c.scenario.name
                ))
            })?;
            // Single source: configs limits.factor_min_level_count (embedded at build).
            spec.factor_min_level_count = crate::config().limits.factor_min_level_count;
            // Host-report (design_test term) twins of the kernel target indices,
            // for skeleton-aligned naming. Safe to compute here: the line above
            // already validated `c` via contract_to_simulation_spec.
            let (report_target_indices, report_contrast_pairs) =
                engine_core::contract_adapter::report_targets_and_contrasts(c);
            Ok(Scenario {
                label: c.scenario.name.clone(),
                spec,
                base_seed: call_seed,
                report_target_indices,
                report_contrast_pairs,
            })
        })
        .collect()
}

/// Multi-core power at a single N: one `run_batch` per contract, folded into
/// one `PowerResult` per scenario (`ScenarioResult` keys = scenario labels,
/// input order preserved).
///
/// Dispatch twin: `single_core_find_power` — any parameter or behaviour added
/// here must land there too, or hosts diverge.
///
/// # Errors
/// Contract lowering, cluster-snap guards (`ClusterTooFewAtN`), and engine
/// failures (including cancellation) all surface as `OrchestratorError`.
pub fn find_power(
    contracts: &[SimulationContract],
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    mut progress: Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<PowerResult>, OrchestratorError> {
    // Snap the requested N to the cluster atom (no-op for unclustered or atom ≤ 1).
    let (sample_size, snap_warning) = snap_single_n(contracts, sample_size)?;

    let scenarios = lower_contracts(contracts, base_seed)?;

    let total_scenarios = scenarios.len();
    let total_sims = (n_sims as u64) * (total_scenarios as u64);
    if let Some(p) = progress.as_deref_mut() {
        p.on_event(ProgressEvent::Started {
            total_sims,
            total_scenarios,
            total_grid_points: 0,
        });
    }

    // NOTE: per-scenario dispatch is sequential today; the inner per-sim
    // parallelism lives inside `run_batch`. Do NOT replace that inner loop
    // with `par_chunks_mut` chunked dispatch (measured −7%/−6.5% throughput
    // at p<10, n<1000).
    let out = for_each_scenario(
        &scenarios,
        &mut progress,
        cancel,
        Some(sample_size),
        |idx, scenario, progress| {
            // Move the sink into the adapter for the duration of the engine
            // call, then put it back. Avoids fighting the borrow checker over
            // mutable reborrows of the sink across iterations.
            let taken: Option<&mut dyn ProgressSink> = progress.take();
            let offset = (idx as u64) * (n_sims as u64);
            let adapter =
                EngineSinkAdapter::new(sample_size, offset, 1, total_sims, taken, cancel);
            let batch_res = engine_core::run_batch(
                &scenario.spec,
                &[sample_size as u32],
                n_sims as u32,
                scenario.base_seed,
                Some(&adapter as &dyn EngineProgressSink),
            );
            // Reclaim the sink before returning on error.
            *progress = adapter.outer.into_inner().unwrap_or(None);
            let batch = batch_res?;

            // Report (design_test term) space, so PowerResult.{target_indices,
            // contrast_pairs} index the host effect skeleton; the kernel already
            // read spec.target_indices (generation-kernel) in run_batch above.
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
            // Pre-run sparse-factor advisory (fixed-allocation only; uploaded
            // and sampled-allocation designs are silent here).
            pr.grid_warnings
                .extend(factor_preflight_warnings(&scenario.spec, sample_size));

            if let Some(p) = progress.as_deref_mut() {
                p.on_event(ProgressEvent::NPointCompleted {
                    n: sample_size,
                    power_uncorrected: pr.power_uncorrected.clone(),
                    power_corrected: pr.power_corrected.clone(),
                });
            }
            Ok(pr)
        },
    )?;

    if let Some(p) = progress {
        p.on_event(ProgressEvent::Completed);
    }
    Ok(ScenarioResult { scenarios: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_contract::{ClusterSpec, GroupingRelation, GroupingSpec};
    use engine_core::{
        CorrectionMethod, CritValues, EstimatorSpec, OutcomeKind,
        ResidualDist, ScenarioPerturbations,
    };
    use engine_core::spec::HeteroskedasticityCoeffs;

    fn clustered(sizing: ClusterSizing, tau: f64) -> SimulationContract {
        let mut c = engine_contract::fixtures::example1_simple_ols();
        c.generation.cluster = Some(ClusterSpec::intercept_only(sizing, tau));
        c
    }

    /// Minimal two-factor SimulationSpec with no cluster/upload/interaction.
    /// factor 0: 2 levels, proportions `p0`; factor 1: 2 levels, proportions `p1`.
    /// `factor_min_level_count` is left at 0 so callers must set it explicitly.
    fn two_factor_spec(p0: [f64; 2], p1: [f64; 2]) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 0,
            n_factor_dummies: 2, // 1 dummy per factor (2-level factors)
            correlation: vec![1.0],
            var_types: vec![],
            var_pinned: vec![],
            var_params: vec![],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![2, 2],
            factor_proportions: vec![p0[0], p0[1], p1[0], p1[1]],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.3, 0.3], // intercept + 2 dummies
            target_indices: vec![1, 2],
            fit_columns: vec![],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues { alpha: 0.05, posthoc_alpha: None },
            heteroskedasticity_driver: None,
            cluster_slope_design_cols: vec![],
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
        }
    }

    /// scenario sampled (old code: preflight skipped entirely), but factor 0 is
    /// forced EXACT via `factor_sampled[0] = Some(false)` and deterministically
    /// sparse at N=100 ([0.98, 0.02] → level 1 gets 2 rows < k_min=5) — must warn.
    #[test]
    fn preflight_warns_for_overridden_exact_factor_under_sampled_scenario() {
        let mut spec = two_factor_spec([0.98, 0.02], [0.5, 0.5]);
        spec.scenario.sampled_factor_proportions = true;
        spec.factor_sampled = vec![Some(false), None];
        spec.factor_min_level_count = 5;
        let w = factor_preflight_warnings(&spec, 100);
        assert_eq!(w.len(), 1, "exact-overridden sparse factor must warn; got: {:?}", w);
        assert!(
            w[0].contains("factor 1"),
            "warning must identify factor 1 (1-based); got: {:?}",
            w[0]
        );
    }

    /// scenario exact (old code: warns for both sparse factors), but factor 1 is
    /// forced SAMPLED via `factor_sampled[1] = Some(true)` — stochastic counts,
    /// must NOT get a deterministic warning. Factor 0 (exact, sparse) still warns.
    #[test]
    fn preflight_skips_overridden_sampled_factor_under_exact_scenario() {
        let mut spec = two_factor_spec([0.98, 0.02], [0.98, 0.02]);
        spec.scenario.sampled_factor_proportions = false;
        spec.factor_sampled = vec![None, Some(true)];
        spec.factor_min_level_count = 5;
        let w = factor_preflight_warnings(&spec, 100);
        assert!(
            w.iter().all(|m| !m.contains("factor 2")),
            "sampled-overridden factor 2 must not warn; got: {:?}",
            w
        );
        // Factor 0 (exact, sparse) still warns — validates the loop continues.
        assert!(
            w.iter().any(|m| m.contains("factor 1")),
            "exact-overridden sparse factor 1 must still warn; got: {:?}",
            w
        );
    }

    #[test]
    fn snap_single_n_uses_full_grouping_atom() {
        // Primary 20 clusters × crossed 12 ⇒ atom 240.
        let mut c = clustered(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.25);
        c.generation.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 12 },
            tau_squared: 0.1,
            slopes: vec![],
        }];
        let (snapped, warning) = snap_single_n(std::slice::from_ref(&c), 500).unwrap();
        assert_eq!(snapped, 480); // 500 floored to the 240-multiple
        assert!(warning.unwrap().contains("240"));
    }
}
