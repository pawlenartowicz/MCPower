//! `run_batch` — orchestrates N sims across a sample-size grid; host-agnostic entry point for all estimators (OLS/GLM/LME).
//!
//! Takes a borrowed `SimulationSpec`, returns an owned `BatchResult`. The PyO3 / extendr layer
//! is responsible for translating the result to numpy / R arrays. Progress + cancellation flow
//! through the `ProgressSink` trait.
//!
//! Dispatch layout (naive per-sim parallel):
//!
//! 1. Validate inputs (`sample_sizes` ordering, target indices in range,
//!    correlation matrix dimensions, scenario self-consistency).
//! 2. Build the `CritValueTable` once.
//! 3. Allocate result tensors `(n_sims × n_sample_sizes × n_targets)`.
//! 4. Collect one `&mut [u8]` slice per sim, then drive `try_for_each_init`
//!    over the sims. Each rayon worker owns a single `SimWorkspace`.
//! 5. Report progress ~50 times per run; cancel if sink returns `false` or
//!    `is_cancelled()` (polled every sim) goes true.
//!
//! TARGET- vs PREDICTOR-indexing (owning explanation — other sites cite this):
//! OLS/GLM fits return `t_sq`/`var_diag` compact at target rank (length =
//! n_targets, ordered like `spec.target_indices`); the LME fit returns them at
//! predictor index (length = p). Consumers of an MLE fit must gather through
//! `spec.target_indices` first; OLS/GLM consumers must not.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::correction::apply_correction;
use crate::critvals::CritValueTable;
use crate::data_gen::generate_sim_data;
use crate::ols::{fit_suff_stats_t_sq, ols_contrast_t_sq, OlsScratch, OlsSuffStats};
use crate::posthoc::evaluate_posthoc;
use crate::spec::{
    BatchResult, CorrectionMethod, EngineError, EstimatorSpec, PosthocBlockShape, ProgressSink,
    ResultShape, SimulationSpec,
};
use crate::workspace::{ReducedCritEntry, SimWorkspace};

/// Rank-deficiency epsilon for the Cholesky pivot ratio (min|L_diag|/max|L_diag|).
pub(crate) const EPS_RANK: f64 = 1e-12;

/// Process-wide BOBYQA objective-eval counters — bench diagnostics only
/// (`bin/throughput.rs` reads before/after deltas around a row's warm-up run
/// to report evals/fit). Fed once per LMM/GLMM fit at the `fit_lmm`/`fit_glmm`
/// call sites below; relaxed atomics, one add per multi-millisecond fit, so
/// the hot loop is unaffected and results cannot move.
pub mod optim_diag {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

    static EVALS: AtomicU64 = AtomicU64::new(0);
    static FITS: AtomicU64 = AtomicU64::new(0);
    static EVALS_CONV: AtomicU64 = AtomicU64::new(0);
    static FITS_CONV: AtomicU64 = AtomicU64::new(0);
    static FITS_PINNED: AtomicU64 = AtomicU64::new(0);

    /// One BOBYQA-backed fit: its objective-eval count, converged flag, and
    /// whether a variance component pinned at the boundary (`boundary_hit == 1`).
    #[inline]
    pub fn record_fit(n_eval: usize, converged: bool, pinned: bool) {
        EVALS.fetch_add(n_eval as u64, Relaxed);
        FITS.fetch_add(1, Relaxed);
        if converged {
            EVALS_CONV.fetch_add(n_eval as u64, Relaxed);
            FITS_CONV.fetch_add(1, Relaxed);
        }
        if pinned {
            FITS_PINNED.fetch_add(1, Relaxed);
        }
    }

    /// `[evals, fits, evals_converged, fits_converged, fits_pinned]` since
    /// process start; callers diff two snapshots — there is deliberately no reset.
    pub fn snapshot() -> [u64; 5] {
        [
            EVALS.load(Relaxed),
            FITS.load(Relaxed),
            EVALS_CONV.load(Relaxed),
            FITS_CONV.load(Relaxed),
            FITS_PINNED.load(Relaxed),
        ]
    }
}

// ---------------------------------------------------------------------------
// Thread pool — single global, set once.
// ---------------------------------------------------------------------------

static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Configure the rayon pool used by `run_batch`. Must be called before any
/// `run_batch` invocation; subsequent calls return `EngineError::InvalidSpec`.
pub fn set_n_threads(n: usize) -> Result<(), EngineError> {
    if n == 0 {
        return Err(EngineError::InvalidSpec(
            "set_n_threads: n must be >= 1".into(),
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| EngineError::InvalidSpec(format!("thread pool build failed: {e}")))?;
    POOL.set(pool).map_err(|_| {
        EngineError::InvalidSpec(
            "thread pool already initialized — set_n_threads must be called before run_batch"
                .into(),
        )
    })
}

/// Returns a reference to the global rayon pool, initializing a default
/// (all-cores) pool on first access.
fn pool() -> &'static rayon::ThreadPool {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .build()
            .expect("default rayon pool")
    })
}

// ---------------------------------------------------------------------------
// run_batch
// ---------------------------------------------------------------------------

/// Run `n_sims` simulations evaluated at every entry of `sample_sizes`,
/// dispatching across rayon's thread pool.
///
/// On `EngineError::Cancelled` the partially-populated `BatchResult` is
/// discarded; the host should treat the run as aborted.
pub fn run_batch(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_sims: u32,
    base_seed: u64,
    sink: Option<&dyn ProgressSink>,
) -> Result<BatchResult, EngineError> {
    run_batch_impl(
        spec,
        sample_sizes,
        n_sims,
        base_seed,
        sink,
        /* sequential */ false,
        None,
    )
}

/// Sequential (single-thread) variant of [`run_batch`]. Same inputs, same
/// `BatchResult` output, and — given identical `(spec, base_seed, n_sims)` —
/// bit-equal results to `run_batch`: both walk identical `pcg_mix64(base_seed,
/// sim_id)` per-sim seed paths, only the dispatch differs.
///
/// Exists for hosts that own their own worker pool (WASM Web Workers, Python
/// multiprocessing) and don't want rayon to fight with them, plus for
/// targets that can't link rayon at all. The orchestrator's
/// `single_core_find_power` is built on top of this.
pub fn run_batch_st(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_sims: u32,
    base_seed: u64,
    sink: Option<&dyn ProgressSink>,
) -> Result<BatchResult, EngineError> {
    run_batch_impl(
        spec,
        sample_sizes,
        n_sims,
        base_seed,
        sink,
        /* sequential */ true,
        None,
    )
}

/// Single-threaded run that also returns the per-sim squared decision statistic,
/// `(n_sims, n_targets)` row-major (marginals then contrasts; `NaN` where a sim
/// did not converge). Captures sample-size index 0 — debug always passes one
/// sample size. The `BatchResult` is bit-identical to `run_batch_st` on the same
/// `(spec, base_seed, n_sims)`.
pub fn run_batch_st_capture(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_sims: u32,
    base_seed: u64,
) -> Result<(BatchResult, Vec<f64>), EngineError> {
    let mut stat = Vec::new();
    let res = run_batch_impl(
        spec,
        sample_sizes,
        n_sims,
        base_seed,
        None,
        true,
        Some(&mut stat),
    )?;
    Ok((res, stat))
}

fn run_batch_impl(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_sims: u32,
    base_seed: u64,
    sink: Option<&dyn ProgressSink>,
    sequential: bool,
    mut stat_capture: Option<&mut Vec<f64>>, // (n_sims, n_targets) row-major; sequential only
) -> Result<BatchResult, EngineError> {
    // ---------------- 1. Validation ----------------
    // exhaustiveness guard: adding a new EstimatorSpec variant forces a compile error here.
    match spec.estimator {
        EstimatorSpec::Ols | EstimatorSpec::Glm | EstimatorSpec::Mle => {}
    }
    // Mle requires a ClusterSpec.
    if spec.estimator == EstimatorSpec::Mle && spec.cluster.is_none() {
        return Err(EngineError::InvalidSpec(
            "estimator Mle requires ClusterSpec".into(),
        ));
    }
    // cluster=Some is valid with Ols (generate-clustered/solve-OLS) and with
    // Glm — data_gen keys cluster draws on cluster.is_some() — so no family
    // guard rejects a clustered Ols/Glm spec here.

    // Posthoc is Ols-only (preserves both old Logit and LME posthoc gates).
    if !spec.posthoc.is_empty() && spec.estimator != EstimatorSpec::Ols {
        return Err(EngineError::InvalidSpec(
            "posthoc tests are only supported for estimator=Ols".into(),
        ));
    }
    if sample_sizes.is_empty() {
        return Err(EngineError::InvalidSpec(
            "sample_sizes must be non-empty".into(),
        ));
    }
    for w in sample_sizes.windows(2) {
        if w[1] <= w[0] {
            return Err(EngineError::InvalidSpec(format!(
                "sample_sizes must be strictly increasing; got {} then {}",
                w[0], w[1]
            )));
        }
    }
    if sample_sizes[0] == 0 {
        return Err(EngineError::InvalidSpec(
            "sample_sizes[0] must be > 0".into(),
        ));
    }

    let n_non_factor = spec.n_non_factor as usize;
    let n_factor_dummies = spec.n_factor_dummies as usize;
    let n_predictors_total = 1 + n_non_factor + n_factor_dummies + spec.interactions.len();
    if spec.correlation.len() != n_non_factor * n_non_factor {
        return Err(EngineError::InvalidSpec(format!(
            "correlation length {} does not match n_non_factor² = {}",
            spec.correlation.len(),
            n_non_factor * n_non_factor
        )));
    }
    if spec.var_types.len() != n_non_factor {
        return Err(EngineError::InvalidSpec(format!(
            "var_types length {} does not match n_non_factor {}",
            spec.var_types.len(),
            n_non_factor
        )));
    }
    if spec.effect_sizes.len() != n_predictors_total {
        return Err(EngineError::InvalidSpec(format!(
            "effect_sizes length {} does not match n_predictors {} (1 intercept + {} non_factor + {} factor dummies)",
            spec.effect_sizes.len(),
            n_predictors_total,
            n_non_factor,
            n_factor_dummies
        )));
    }
    // Factor proportions must sum across factor_n_levels.
    let mut expected_props = 0usize;
    for &lv in &spec.factor_n_levels {
        if lv < 2 {
            return Err(EngineError::InvalidSpec(format!(
                "factor_n_levels entries must be >= 2; got {lv}"
            )));
        }
        expected_props = expected_props.saturating_add(lv as usize);
    }
    if spec.factor_proportions.len() != expected_props {
        return Err(EngineError::InvalidSpec(format!(
            "factor_proportions length {} does not match Σfactor_n_levels {}",
            spec.factor_proportions.len(),
            expected_props
        )));
    }
    let n_marginal_targets = spec.target_indices.len();
    for &ti in &spec.target_indices {
        if (ti as usize) >= n_predictors_total {
            return Err(EngineError::InvalidSpec(format!(
                "target_indices entry {ti} out of range (n_predictors = {n_predictors_total})"
            )));
        }
    }
    for &(p, n) in &spec.contrast_pairs {
        if (p as usize) >= n_predictors_total {
            return Err(EngineError::InvalidSpec(format!(
                "contrast_pairs positive {p} out of range (n_predictors = {n_predictors_total})"
            )));
        }
        if (n as usize) >= n_predictors_total {
            return Err(EngineError::InvalidSpec(format!(
                "contrast_pairs negative {n} out of range (n_predictors = {n_predictors_total})"
            )));
        }
    }
    let n_targets = n_marginal_targets + spec.contrast_pairs.len();

    // PSD validation of correlation: a trial Cholesky of an owned copy.
    if n_non_factor > 0 {
        let mut tmp = faer::Mat::<f64>::zeros(n_non_factor, n_non_factor);
        for j in 0..n_non_factor {
            for i in 0..n_non_factor {
                tmp[(i, j)] = spec.correlation[j * n_non_factor + i];
            }
        }
        crate::correlation::factor_only(tmp.as_mut())
            .map_err(|_| EngineError::CorrelationNotPSD)?;
    }

    // Build the t(3) lookup table once per run. Cheap (RESOLUTION = 2048 f64s)
    // and required only by DIST_HIGH_KURTOSIS, but always-on so a scenario
    // perturbation that swaps to DIST_HIGH_KURTOSIS still has the table.
    // Mirrored by introspect::capture_sim0_data — change together.
    let mut spec_with_tables = spec.clone();
    spec_with_tables.t3_table = Some(crate::marginals::T3PpfTable::build_default());
    spec_with_tables.het_coeffs = spec_with_tables.compute_het_coeffs();
    let spec = &spec_with_tables;

    let n_sample_sizes = sample_sizes.len();
    let n_targets_u32 = n_targets as u32;
    let n_sample_sizes_u32 = n_sample_sizes as u32;

    // ---------------- 2. CritValueTable ----------------
    // One authority for every df-dependent threshold: the same builder also
    // produces the reduced-model tables for sparse-exclusion rounds (see
    // `reduced_crit_entry`), so reduced thresholds cannot drift from these.
    let (crit, posthoc_correction_crit_sq) =
        build_crit_tables(spec, sample_sizes, n_predictors_total as u32)?;

    // Posthoc support — Σ C(kᵢ, 2) across blocks, where kᵢ = dummies+1.
    let posthoc_blocks: Vec<PosthocBlockShape> = posthoc_block_shapes(spec);
    let posthoc_n_contrasts: usize = posthoc_blocks.iter().map(|b| b.n_contrasts as usize).sum();
    let posthoc_contrasts: Vec<Vec<f64>> = build_posthoc_contrasts(spec)?;

    // ---------------- 3. Result allocation ----------------
    let n_sims_usize = n_sims as usize;
    let main_cell_count = n_sims_usize * n_sample_sizes * n_targets;
    let posthoc_cell_count = n_sims_usize * n_sample_sizes * posthoc_n_contrasts;
    let converged_count = n_sims_usize * n_sample_sizes;

    let mut uncorrected = vec![0u8; main_cell_count];
    let mut corrected = vec![0u8; main_cell_count];
    let mut posthoc_unc = vec![0u8; posthoc_cell_count];
    let mut posthoc_cor = vec![0u8; posthoc_cell_count];
    let mut converged = vec![0u8; converged_count];
    let mut boundary_hit = vec![0u8; converged_count];
    // Parallel to boundary_hit: per-component pinned bitmask (general lmm path);
    // Brent path writes bit 0 = boundary_hit == 1; OLS/Glm leave 0.
    let mut pinned_components = vec![0u32; converged_count];
    // Parallel to converged: estimated random-intercept variance τ̂² (GLMM branch
    // only). NaN-initialised; only the Glm+cluster path writes it, so every other
    // estimator leaves NaN (mirrors the contract — finite ⇒ a GLMM fit ran).
    let mut tau_squared_hat = vec![f64::NAN; converged_count];
    // Joint Wald-χ² significance buffers — `(n_sims × n_sample_sizes)`,
    // sim-major.
    let mut joint_unc = vec![0u8; converged_count];
    let mut joint_cor = vec![0u8; converged_count];
    let overall_count = if spec.report_overall {
        n_sims_usize * n_sample_sizes
    } else {
        0
    };
    let mut overall = vec![0u8; overall_count];

    // n_factors used both in the early-exit guard and in the parallel loop.
    let n_factors = spec.factor_n_levels.len();

    if n_sims == 0 {
        return Ok(BatchResult {
            uncorrected,
            corrected,
            posthoc_unc,
            posthoc_cor,
            converged,
            boundary_hit,
            pinned_components,
            joint_unc,
            joint_cor,
            overall,
            factor_excluded: vec![],
            tau_squared_hat: vec![f64::NAN; n_sims_usize * n_sample_sizes],
            shape: ResultShape {
                n_sims: 0,
                n_sample_sizes: n_sample_sizes_u32,
                n_targets: n_targets_u32,
                posthoc_blocks: posthoc_blocks.clone(),
                n_factors: n_factors as u32,
                n_variance_components: if spec.estimator == EstimatorSpec::Mle
                    || (spec.estimator == EstimatorSpec::Glm && spec.cluster.is_some())
                {
                    spec.cluster
                        .as_ref()
                        .map_or(0, |c| c.n_variance_components()) as u32
                } else {
                    0
                },
            },
        });
    }

    // ---------------- 4. Parallel sim loop ----------------
    let max_n = *sample_sizes.last().unwrap() as usize;
    // factor_excluded: (n_sims × n_sample_sizes × n_factors), sim-major.
    // Empty when the spec has no factors — mirrors the posthoc/overall gate pattern.
    // (n_factors declared above the n_sims == 0 guard, so it's in scope here.)
    let factor_excl_stride = n_sample_sizes * n_factors;
    let mut factor_excluded = vec![0u8; n_sims_usize * n_sample_sizes * n_factors];

    let progress_counter = AtomicU64::new(0);
    let cancelled = std::sync::atomic::AtomicBool::new(false);

    let main_row = n_targets;
    // Debug stat capture buffer: (n_sims, n_targets) row-major, NaN-initialised.
    // Populated only on the sequential path (sample-size index 0 per sim).
    if let Some(cap) = stat_capture.as_deref_mut() {
        cap.clear();
        cap.resize(n_sims as usize * main_row, f64::NAN);
    }
    let posthoc_row = posthoc_n_contrasts;
    let sim_main_stride = n_sample_sizes * main_row;
    let sim_posthoc_stride = n_sample_sizes * posthoc_row;
    let sim_converged_stride = n_sample_sizes;

    let cancelled_ref = &cancelled;
    let counter_ref = &progress_counter;

    // Report ~50 times per run, regardless of n_sims.
    let progress_step = (n_sims as u64 / 50).max(1);

    // One mutable slice per sim, collected serially. Posthoc is Option so the
    // no-contrasts case (empty backing Vec) still yields n_sims iterations
    // rather than truncating the zipped parallel iterator to length 0.
    //
    // Main buffers: when sim_main_stride == 0 (n_targets == 0), the backing
    // Vec is empty and chunks_exact_mut(0) would panic. Mirror the posthoc
    // empty-buffer pattern: produce n_sims empty &mut [] slices instead.
    let unc_per_sim: Vec<&mut [u8]> = if sim_main_stride == 0 {
        (0..n_sims_usize).map(|_| &mut [][..]).collect()
    } else {
        uncorrected.chunks_exact_mut(sim_main_stride).collect()
    };
    let cor_per_sim: Vec<&mut [u8]> = if sim_main_stride == 0 {
        (0..n_sims_usize).map(|_| &mut [][..]).collect()
    } else {
        corrected.chunks_exact_mut(sim_main_stride).collect()
    };
    let conv_per_sim: Vec<&mut [u8]> = converged.chunks_exact_mut(sim_converged_stride).collect();
    let bh_per_sim: Vec<&mut [u8]> = boundary_hit
        .chunks_exact_mut(sim_converged_stride)
        .collect();
    let pc_per_sim: Vec<&mut [u32]> = pinned_components
        .chunks_exact_mut(sim_converged_stride)
        .collect();
    let tau_per_sim: Vec<&mut [f64]> = tau_squared_hat
        .chunks_exact_mut(sim_converged_stride)
        .collect();
    let joint_unc_per_sim: Vec<&mut [u8]> =
        joint_unc.chunks_exact_mut(sim_converged_stride).collect();
    let joint_cor_per_sim: Vec<&mut [u8]> =
        joint_cor.chunks_exact_mut(sim_converged_stride).collect();
    let post_unc_per_sim: Vec<Option<&mut [u8]>> = if posthoc_n_contrasts == 0 {
        (0..n_sims_usize).map(|_| None).collect()
    } else {
        posthoc_unc
            .chunks_exact_mut(sim_posthoc_stride)
            .map(Some)
            .collect()
    };
    let post_cor_per_sim: Vec<Option<&mut [u8]>> = if posthoc_n_contrasts == 0 {
        (0..n_sims_usize).map(|_| None).collect()
    } else {
        posthoc_cor
            .chunks_exact_mut(sim_posthoc_stride)
            .map(Some)
            .collect()
    };
    let overall_per_sim: Vec<Option<&mut [u8]>> = if spec.report_overall {
        overall
            .chunks_exact_mut(sim_converged_stride)
            .map(Some)
            .collect()
    } else {
        (0..n_sims_usize).map(|_| None).collect()
    };
    // factor_excluded per-sim slices: empty-stride guard mirrors the unc_per_sim pattern
    // (chunks_exact_mut(0) panics; n_factors == 0 → produce n_sims empty &mut [] slices).
    let fx_per_sim: Vec<&mut [u8]> = if factor_excl_stride == 0 {
        (0..n_sims_usize).map(|_| &mut [][..]).collect()
    } else {
        factor_excluded
            .chunks_exact_mut(factor_excl_stride)
            .collect()
    };

    // The per-sim body (`run_one_sim`) is family-agnostic at the call site.
    // Dispatch differs only in `for` vs `try_for_each_init`; the body, the
    // cancellation check, and the progress reporting are identical between
    // sequential and parallel paths.
    let result: Result<(), EngineError> = if sequential {
        // Sequential dispatch — one workspace, no rayon. Used by `run_batch_st`
        // and the WASM-friendly orchestrator entry `single_core_find_power`.
        let mut ws = SimWorkspace::new(
            max_n,
            n_predictors_total,
            n_non_factor,
            n_factors,
            spec.cluster.as_ref(),
        );
        // The OLS branch reuses lme_joint_rhs as t² scratch over ALL test
        // targets (marginals + contrasts), which can exceed P when a factor's
        // full pairwise contrast set is requested — grow it up front so the
        // hot loop stays allocation-free. LME consumers only take [..P] views,
        // so the extra tail is inert there. Mirrors the rayon init below —
        // change together.
        ws.lme_joint_rhs
            .resize(n_predictors_total.max(n_targets), 0.0);
        ws.lmm = crate::lmm::build_lmm_workspace(spec, max_n, n_predictors_total);
        // Mutually exclusive with lmm: build_lmm_workspace requires Mle,
        // build_glmm_workspace requires Glm+cluster (returns None otherwise).
        ws.glmm = crate::glmm::build_glmm_workspace(spec, max_n, n_predictors_total);
        let zipped = unc_per_sim
            .into_iter()
            .zip(cor_per_sim)
            .zip(conv_per_sim)
            .zip(bh_per_sim)
            .zip(pc_per_sim)
            .zip(tau_per_sim)
            .zip(post_unc_per_sim)
            .zip(post_cor_per_sim)
            .zip(joint_unc_per_sim)
            .zip(joint_cor_per_sim)
            .zip(overall_per_sim)
            .zip(fx_per_sim)
            .enumerate();
        let mut step_result: Result<(), EngineError> = Ok(());
        for (
            sim_id,
            (
                (
                    (
                        (
                            (
                                ((((((unc, cor), conv), bh), pc), tau), post_unc_slice),
                                post_cor_slice,
                            ),
                            joint_unc_slice,
                        ),
                        joint_cor_slice,
                    ),
                    overall_slice,
                ),
                fx,
            ),
        ) in zipped
        {
            let stat_row = stat_capture
                .as_deref_mut()
                .map(|v| &mut v[sim_id * main_row..(sim_id + 1) * main_row]);
            if let Err(e) = run_one_sim(
                &mut ws,
                sim_id,
                spec,
                sample_sizes,
                base_seed,
                &crit,
                &posthoc_contrasts,
                &posthoc_correction_crit_sq,
                n_predictors_total,
                main_row,
                posthoc_row,
                n_factors,
                unc,
                cor,
                conv,
                bh,
                pc,
                tau,
                joint_unc_slice,
                joint_cor_slice,
                post_unc_slice,
                post_cor_slice,
                overall_slice,
                fx,
                cancelled_ref,
                stat_row,
            ) {
                step_result = Err(e);
                break;
            }
            if let Err(e) = report_progress(
                counter_ref,
                cancelled_ref,
                progress_step,
                n_sims as u64,
                sink,
            ) {
                step_result = Err(e);
                break;
            }
        }
        step_result
    } else {
        pool().install(|| {
            unc_per_sim
                .into_par_iter()
                .zip(cor_per_sim)
                .zip(conv_per_sim)
                .zip(bh_per_sim)
                .zip(pc_per_sim)
                .zip(tau_per_sim)
                .zip(post_unc_per_sim)
                .zip(post_cor_per_sim)
                .zip(joint_unc_per_sim)
                .zip(joint_cor_per_sim)
                .zip(overall_per_sim)
                .zip(fx_per_sim)
                .enumerate()
                .try_for_each_init(
                    || {
                        let mut ws = SimWorkspace::new(
                            max_n,
                            n_predictors_total,
                            n_non_factor,
                            n_factors,
                            spec.cluster.as_ref(),
                        );
                        // OLS t² scratch over all test targets — mirrors the
                        // sequential site above, change together.
                        ws.lme_joint_rhs
                            .resize(n_predictors_total.max(n_targets), 0.0);
                        ws.lmm = crate::lmm::build_lmm_workspace(spec, max_n, n_predictors_total);
                        // Mutually exclusive with lmm (see sequential site).
                        ws.glmm =
                            crate::glmm::build_glmm_workspace(spec, max_n, n_predictors_total);
                        ws
                    },
                    |ws,
                     (
                        sim_id,
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (((((unc, cor), conv), bh), pc), tau),
                                                mut post_unc_slice,
                                            ),
                                            mut post_cor_slice,
                                        ),
                                        joint_unc_slice,
                                    ),
                                    joint_cor_slice,
                                ),
                                mut overall_slice,
                            ),
                            fx,
                        ),
                    )| {
                        run_one_sim(
                            ws,
                            sim_id,
                            spec,
                            sample_sizes,
                            base_seed,
                            &crit,
                            &posthoc_contrasts,
                            &posthoc_correction_crit_sq,
                            n_predictors_total,
                            main_row,
                            posthoc_row,
                            n_factors,
                            unc,
                            cor,
                            conv,
                            bh,
                            pc,
                            tau,
                            joint_unc_slice,
                            joint_cor_slice,
                            post_unc_slice.as_deref_mut(),
                            post_cor_slice.as_deref_mut(),
                            overall_slice.as_deref_mut(),
                            fx,
                            cancelled_ref,
                            None, // stat capture is sequential-only
                        )?;
                        report_progress(
                            counter_ref,
                            cancelled_ref,
                            progress_step,
                            n_sims as u64,
                            sink,
                        )?;
                        Ok(())
                    },
                )
        })
    };
    result?;

    Ok(BatchResult {
        uncorrected,
        corrected,
        posthoc_unc,
        posthoc_cor,
        converged,
        boundary_hit,
        pinned_components,
        joint_unc,
        joint_cor,
        overall,
        factor_excluded,
        tau_squared_hat,
        shape: ResultShape {
            n_sims,
            n_sample_sizes: n_sample_sizes_u32,
            n_targets: n_targets_u32,
            posthoc_blocks,
            n_factors: n_factors as u32,
            n_variance_components: if spec.estimator == EstimatorSpec::Mle
                || (spec.estimator == EstimatorSpec::Glm && spec.cluster.is_some())
            {
                spec.cluster
                    .as_ref()
                    .map_or(0, |c| c.n_variance_components()) as u32
            } else {
                0
            },
        },
    })
}

/// Per-sim body shared by `run_batch` (parallel) and `run_batch_st` (serial).
///
/// Cancellation is checked at entry. Progress reporting happens in the caller
/// via `report_progress` after this returns so the rayon and serial paths
/// share the exact same checkpoint cadence.
///
/// Mirror site: `introspect::fit_provided_data` tracks the three estimator
/// arms here arm-for-arm (same kernels, same injected bytes) — change them
/// together.
///
/// `introspect::fit_provided_data` deliberately does NOT mirror the sparse-factor
/// exclusion branch — debug fits provided bytes as-is so the L3 same-bytes
/// contract holds.
#[expect(
    clippy::too_many_arguments,
    reason = "per-sim body threads the full sim state explicitly so the rayon and serial paths share one checkpoint cadence"
)]
fn run_one_sim(
    ws: &mut SimWorkspace,
    sim_id: usize,
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    base_seed: u64,
    crit: &CritValueTable,
    posthoc_contrasts: &[Vec<f64>],
    posthoc_correction_crit_sq: &[Vec<f64>],
    n_predictors_total: usize,
    main_row: usize,
    posthoc_row: usize,
    n_factors: usize,
    unc: &mut [u8],
    cor: &mut [u8],
    conv: &mut [u8],
    bh: &mut [u8],
    pc: &mut [u32],
    tau_hat: &mut [f64], // per-N τ̂² row; NaN except where the GLMM branch writes it
    joint_unc_slice: &mut [u8],
    joint_cor_slice: &mut [u8],
    mut post_unc_slice: Option<&mut [u8]>,
    mut post_cor_slice: Option<&mut [u8]>,
    mut overall_slice: Option<&mut [u8]>,
    fx: &mut [u8], // (n_sample_sizes × n_factors) exclusion codes for this sim
    cancelled_ref: &std::sync::atomic::AtomicBool,
    mut stat_sink: Option<&mut [f64]>, // nt-wide row for sample-size index 0, or None
) -> Result<(), EngineError> {
    if cancelled_ref.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    generate_sim_data(spec, sim_id as u64, base_seed, ws)?;

    // Sparse-factor exclusion state — reset per sim, grown with the grid.
    // factor_prefix_counts is in factor_proportions layout (Σ factor_n_levels
    // entries); factor_excluded_flags is one flag per factor.
    ws.factor_prefix_counts.clear();
    ws.factor_prefix_counts
        .resize(spec.factor_proportions.len(), 0);
    ws.factor_excluded_flags.clear();
    ws.factor_excluded_flags.resize(n_factors, 0);

    // `test_formula` reduced fit: a non-empty `fit_columns` that drops at least
    // one kernel column means the fitted (test) design is a strict subset of the
    // generation design, so the reduced-fit path must run even when no factor is
    // sparse-excluded. Spec-level — constant across the N-grid. Each arm ORs this
    // into its `needs_reduced_fit` trigger; the factor-only `any_excluded` still
    // drives `update_factor_exclusions` and the GLM separation fallback.
    let test_reduces = !spec.fit_columns.is_empty() && spec.fit_columns.len() < n_predictors_total;

    match spec.estimator {
        EstimatorSpec::Ols => {
            // Reset the suff-stats accumulator at the top of every
            // sim, then grow it monotonically through the sample-size
            // grid. `sample_sizes` is validated strictly increasing
            // above, so the segment slice math below is safe.
            ws.reset_suff_stats();
            let mut last_n_added: usize = 0;

            for (n_idx, &n) in sample_sizes.iter().enumerate() {
                let n_usize = n as usize;

                // Grow the accumulator from last_n_added..n_usize.
                if n_usize > last_n_added {
                    let x_seg = ws
                        .x_full
                        .as_ref()
                        .subrows(last_n_added, n_usize - last_n_added);
                    let y_seg = &ws.y_full[last_n_added..n_usize];
                    let mut suff = OlsSuffStats {
                        xtx: ws.suff_xtx.as_mut(),
                        xty: &mut ws.suff_xty,
                        yty: &mut ws.suff_yty,
                        sum_y: &mut ws.suff_sum_y,
                        n_rows: &mut ws.suff_n_rows,
                        panel_x: &mut ws.panel_x,
                        panel_y: &mut ws.panel_y,
                    };
                    suff.add_rows(x_seg, y_seg);
                    // Update exclusion counts for the new rows
                    // (from = last_n_added before the update).
                    // min_count == 0 ⇒ feature disabled — skip the O(rows) scan;
                    // flags stay zero from the per-sim reset.
                    if n_factors > 0 && spec.factor_min_level_count > 0 {
                        let from = last_n_added;
                        update_factor_exclusions(
                            ws.x_full.as_ref(),
                            &spec.factor_n_levels,
                            1 + spec.n_non_factor as usize,
                            from,
                            n_usize,
                            spec.factor_min_level_count,
                            &mut ws.factor_prefix_counts,
                            &mut ws.factor_excluded_flags,
                        );
                    }
                    last_n_added = n_usize;
                }

                // Inline split-borrow of `ws`. A `&mut self`
                // helper would borrow the whole struct and conflict
                // with the shared `x_full`/`y_full` reads above;
                // direct field accesses let NLL prove the storage
                // is disjoint.
                let any_excluded = ws.factor_excluded_flags.iter().any(|&c| c != 0);
                let needs_reduced_fit = any_excluded || test_reduces;

                // Exclusion scratch — round-scope locals, not workspace fields (cold path,
                // consistent with keep_dummy/reduced_contrasts). Declarations are zero-alloc;
                // real allocation happens only inside the excluded branch. Step 6.2's
                // scatter-back and the contrast/posthoc remaps read them after the fit.
                // The unused_assignments suppression covers the initial zero-alloc values
                // for excl_xtx and excl_xty: they are assigned real data in the excluded
                // branch before being passed to fit_suff_stats_t_sq; on the hot path (!needs_reduced_fit)
                // the declarations are never touched, hence the lint.
                #[allow(unused_assignments)]
                let mut excl_kept_cols: Vec<u32> = Vec::new();
                let mut excl_col_remap: Vec<i32> = Vec::new();
                let mut excl_targets: Vec<u32> = Vec::new();
                let mut excl_target_slots: Vec<usize> = Vec::new();
                #[allow(unused_assignments)]
                let mut excl_xtx = faer::Mat::<f64>::zeros(0, 0);
                #[allow(unused_assignments)]
                let mut excl_xty: Vec<f64> = Vec::new();
                // p_red: reduced predictor count (= n_predictors_total on the hot path).
                // Hoisted out of the fit branch so the overall-F block can read it (step 6.4).
                let p_red: usize;

                let scratch = OlsScratch {
                    fit_betas: &mut ws.fit_betas,
                    fit_var_diag: &mut ws.fit_var_diag,
                    fit_t_sq: &mut ws.fit_t_sq,
                    fit_u_scratch: &mut ws.fit_u_scratch,
                    fit_factor: ws.fit_factor.as_mut(),
                    fit_rhs: ws.fit_rhs.as_mut(),
                };
                let fit = if !needs_reduced_fit {
                    p_red = n_predictors_total;
                    fit_suff_stats_t_sq(
                        ws.suff_xtx.as_ref(),
                        &ws.suff_xty,
                        ws.suff_yty,
                        ws.suff_sum_y,
                        ws.suff_n_rows,
                        &spec.target_indices,
                        EPS_RANK,
                        ws.suff_xtx_work.as_mut(),
                        scratch,
                    )
                } else {
                    // --- reduced model: gather kept rows/cols of the suff stats ---
                    excl_col_remap.resize(n_predictors_total, -1);
                    p_red = build_exclusion_remap(
                        spec,
                        &ws.factor_excluded_flags,
                        &mut excl_kept_cols,
                        &mut excl_col_remap,
                    );
                    excl_xtx = faer::Mat::<f64>::zeros(p_red, p_red);
                    excl_xty = vec![0.0; p_red];
                    // Lower-triangle gather: excl_xtx[(ri, rj)] = suff_xtx[(ci, cj)], ri ≥ rj.
                    // Validity: kept_cols is built ascending, so ri ≥ rj ⇒ ci ≥ cj — every
                    // read stays in suff_xtx's meaningful lower triangle (add_rows only
                    // writes i ≥ j). Do not reorder kept_cols.
                    for rj in 0..p_red {
                        let cj = excl_kept_cols[rj] as usize;
                        excl_xty[rj] = ws.suff_xty[cj];
                        for ri in rj..p_red {
                            let ci = excl_kept_cols[ri] as usize;
                            excl_xtx[(ri, rj)] = ws.suff_xtx[(ci, cj)];
                        }
                    }
                    // Remap marginal targets; dropped targets keep NaN t² (slot list drives scatter-back).
                    for (slot, &ti) in spec.target_indices.iter().enumerate() {
                        let r = excl_col_remap[ti as usize];
                        if r >= 0 {
                            excl_targets.push(r as u32);
                            excl_target_slots.push(slot);
                        }
                    }
                    fit_suff_stats_t_sq(
                        excl_xtx.as_ref(),
                        &excl_xty,
                        ws.suff_yty,
                        ws.suff_sum_y,
                        ws.suff_n_rows,
                        &excl_targets,
                        EPS_RANK,
                        // The p_red×p_red top-left submatrix view satisfies the exact-size
                        // requirement (fit_suff_stats_t_sq asserts its xtx_work is p×p).
                        ws.suff_xtx_work.as_mut().submatrix_mut(0, 0, p_red, p_red),
                        scratch,
                    )
                };

                conv[n_idx] = u8::from(fit.converged);
                bh[n_idx] = 0; // OLS: never a boundary condition

                // Every df-dependent threshold for this round, from one source.
                // Hot path: the precomputed full-model table at n_idx. Excluded
                // round: a reduced-model single-N table built by the same
                // constructor (build_crit_tables at p_red columns — marginal t²,
                // correction row, post-hoc crits, and overall F all at the
                // reduced df), memoized per (N, p_red) in the workspace.
                // Disjoint-field borrow of ws (cache vs the fit_* fields held
                // by `fit`) — same split-borrow pattern as the scratch above.
                let (crit_src, crit_idx, posthoc_corr_row): (&CritValueTable, usize, &[f64]) =
                    if !needs_reduced_fit {
                        (crit, n_idx, &posthoc_correction_crit_sq[n_idx])
                    } else {
                        let e =
                            reduced_crit_entry(&mut ws.reduced_crit_cache, n, p_red as u32, spec)?;
                        (&e.crit, 0, &e.posthoc_corr_row)
                    };

                let row_start = n_idx * main_row;
                let row_end = row_start + main_row;
                let unc_slice = &mut unc[row_start..row_end];
                unc_slice.fill(0);
                let t_crit_unc = crit_src.t_crit_sq_uncorrected[crit_idx];

                // Build combined t² array: marginals followed
                // by contrast pairs. Use lme_joint_rhs as a
                // scratch buffer (length P ≥ n_targets; not
                // consumed by the OLS path).
                let n_marginals = spec.target_indices.len();
                let n_contrasts = spec.contrast_pairs.len();
                debug_assert!(
                    ws.lme_joint_rhs.len() >= n_marginals + n_contrasts,
                    "lme_joint_rhs (len={}) must be >= n_marginals({}) + \
                                         n_contrasts({}) — workspace must be sized to P ≥ \
                                         total number of OLS test targets",
                    ws.lme_joint_rhs.len(),
                    n_marginals,
                    n_contrasts,
                );
                // NaN-initialise every slot so dropped-target positions carry NaN
                // (never significant) rather than stale values from the previous n_idx.
                let all_t_sq = &mut ws.lme_joint_rhs[..n_marginals + n_contrasts];
                all_t_sq.fill(f64::NAN);
                if fit.converged {
                    if !needs_reduced_fit {
                        // Hot path: copy marginal t² values directly.
                        all_t_sq[..n_marginals].copy_from_slice(fit.t_sq);
                    } else {
                        // fit.t_sq is compact over the remapped (kept) targets.
                        for (k, &slot) in excl_target_slots.iter().enumerate() {
                            all_t_sq[slot] = fit.t_sq[k];
                        }
                    }
                    // Contrasts: remap both columns; either side dropped → NaN (non-significant).
                    // lme_u_scratch is not borrowed by fit (fit only borrows fit_betas,
                    // fit_var_diag, fit_t_sq, fit_factor, fit_rhs, fit_u_scratch — but
                    // lme_u_scratch is separate), so field-disjoint access is sound under NLL.
                    for (ci, &(p_col, n_col)) in spec.contrast_pairs.iter().enumerate() {
                        all_t_sq[n_marginals + ci] = if !needs_reduced_fit {
                            ols_contrast_t_sq(&fit, p_col, n_col, &mut ws.lme_u_scratch)
                        } else {
                            let (rp, rn) = (
                                excl_col_remap[p_col as usize],
                                excl_col_remap[n_col as usize],
                            );
                            if rp >= 0 && rn >= 0 {
                                ols_contrast_t_sq(&fit, rp as u32, rn as u32, &mut ws.lme_u_scratch)
                            } else {
                                f64::NAN
                            }
                        };
                    }
                    // Write uncorrected significance.
                    for (j, &t_sq) in all_t_sq.iter().enumerate() {
                        if !t_sq.is_nan() && t_sq > t_crit_unc {
                            unc_slice[j] = 1;
                        }
                    }
                }
                // Debug stat capture (sample-size index 0 only).
                // `all_t_sq` is the marginals+contrasts squared-
                // statistic slice (NaN where not converged or where the
                // target's factor was sparse-excluded) — the
                // same array `unc` is derived from.
                if n_idx == 0 {
                    if let Some(row) = stat_sink.as_deref_mut() {
                        row.copy_from_slice(all_t_sq);
                    }
                }
                let cor_slice = &mut cor[row_start..row_end];
                if fit.converged {
                    apply_correction(
                        spec.correction_method,
                        all_t_sq,
                        t_crit_unc,
                        &crit_src.correction_t_crit_sq[crit_idx],
                        cor_slice,
                    );
                } else {
                    cor_slice.fill(0);
                }
                if let Some(ref mut overall) = overall_slice.as_deref_mut() {
                    // crit_src already carries the reduced-df F crit on excluded
                    // rounds: F(p_red − 1, N − p_red), built by build_crit_tables —
                    // the precomputed full-P crit would be anti-conservative there.
                    let crit_overall = crit_src.overall_crit[crit_idx];
                    let mut sig: u8 = 0;
                    if fit.converged {
                        // Under exclusion, the regression has p_red predictors, not
                        // n_predictors_total — use p_red as the numerator df (step 6.4).
                        let dfn = (p_red as i64 - 1) as f64;
                        let dfd = fit.df_resid as f64;
                        if dfn >= 1.0
                            && fit.sst > 1e-10
                            && fit.rss > 0.0
                            && fit.rss.is_finite()
                            && fit.sst.is_finite()
                        {
                            let f = ((fit.sst - fit.rss) / dfn) / (fit.rss / dfd);
                            if f.is_finite() && f > crit_overall {
                                sig = 1;
                            }
                        }
                    }
                    overall[n_idx] = sig;
                }
                if let (Some(post_unc), Some(post_cor)) =
                    (post_unc_slice.as_deref_mut(), post_cor_slice.as_deref_mut())
                {
                    let post_row_start = n_idx * posthoc_row;
                    let post_row_end = post_row_start + posthoc_row;
                    if !needs_reduced_fit {
                        evaluate_posthoc(
                            &fit,
                            posthoc_contrasts,
                            crit_src.posthoc_t_crit_sq[crit_idx],
                            spec.correction_method,
                            posthoc_corr_row,
                            &mut ws.posthoc_t_sq_scratch,
                            // `lme_u_scratch` is the OLS-contrast
                            // forward-solve scratch (length P, not
                            // borrowed by `fit`); reused here.
                            &mut ws.lme_u_scratch,
                            &mut post_unc[post_row_start..post_row_end],
                            &mut post_cor[post_row_start..post_row_end],
                        );
                    } else {
                        // Under exclusion, remap each contrast vector to the reduced
                        // predictor set. If any nonzero component maps to a dropped column,
                        // zero the whole contrast — a zero contrast produces t²≈0/NaN
                        // (se_sq guard at posthoc.rs:97) and is never significant.
                        // Cold-path allocation acceptable here (excluded rounds are rare).
                        let reduced_contrasts: Vec<Vec<f64>> = posthoc_contrasts
                            .iter()
                            .map(|c| {
                                let mut rc = vec![0.0; p_red];
                                let mut dropped = false;
                                for (col, &v) in c.iter().enumerate() {
                                    if v != 0.0 {
                                        match excl_col_remap[col] {
                                            r if r >= 0 => rc[r as usize] = v,
                                            _ => {
                                                dropped = true;
                                            }
                                        }
                                    }
                                }
                                if dropped {
                                    rc.fill(0.0); // zero contrast → se_sq≈0 → not significant
                                }
                                rc
                            })
                            .collect();
                        // crit_src carries the reduced-df post-hoc crits here
                        // (uncorrected and per-contrast correction row alike).
                        evaluate_posthoc(
                            &fit,
                            &reduced_contrasts,
                            crit_src.posthoc_t_crit_sq[crit_idx],
                            spec.correction_method,
                            posthoc_corr_row,
                            &mut ws.posthoc_t_sq_scratch,
                            &mut ws.lme_u_scratch,
                            &mut post_unc[post_row_start..post_row_end],
                            &mut post_cor[post_row_start..post_row_end],
                        );
                    }
                }
                // Write exclusion flags after the fit (uniform position
                // across all three arms so a future code-2 GLM upgrade
                // can append after the fit in a consistent spot).
                if n_factors > 0 {
                    let fx_row = &mut fx[n_idx * n_factors..(n_idx + 1) * n_factors];
                    fx_row.copy_from_slice(&ws.factor_excluded_flags);
                }
            }
        }
        EstimatorSpec::Mle => {
            // No cross-N reuse: cluster_ids depend on N because
            // cluster_size = N/n_clusters changes, so the
            // suff-stats accumulator is rebuilt per (sim, N).
            // Posthoc is rejected at batch entry above.
            use faer::reborrow::IntoConst;
            for (n_idx, &n) in sample_sizes.iter().enumerate() {
                let n_usize = n as usize;

                // Update exclusion flags BEFORE the fit — mirrors the GLM arm's
                // placement and gating. MLE rebuilds suff-stats from scratch per
                // (sim, N), so recount from 0..n_usize on every grid point; the
                // flags always reflect the exact N rows used.
                // min_count == 0 ⇒ feature disabled — flags stay zero from the
                // per-sim reset.
                if n_factors > 0 && spec.factor_min_level_count > 0 {
                    ws.factor_prefix_counts.fill(0);
                    update_factor_exclusions(
                        ws.x_full.as_ref(),
                        &spec.factor_n_levels,
                        1 + spec.n_non_factor as usize,
                        0,
                        n_usize,
                        spec.factor_min_level_count,
                        &mut ws.factor_prefix_counts,
                        &mut ws.factor_excluded_flags,
                    );
                }

                let any_excluded = ws.factor_excluded_flags.iter().any(|&c| c != 0);
                let needs_reduced_fit = any_excluded || test_reduces;

                // Exclusion scratch — round-scope locals, cold path (mirrors
                // the GLM arm). Zero-alloc until an excluded round fires.
                let mut excl_kept_cols: Vec<u32> = Vec::new();
                let mut excl_col_remap: Vec<i32> = Vec::new();
                let mut excl_targets: Vec<u32> = Vec::new();
                // excl_target_slots: maps position in excl_targets back to
                // position in spec.target_indices (for scatter-back gather).
                let mut excl_target_slots: Vec<usize> = Vec::new();
                // excl_x: reduced design matrix (n_usize × p_red); sized on first use.
                // f32 to match the data plane (copied from x_full, fed to f32 fits).
                let mut excl_x = faer::Mat::<f32>::zeros(0, 0);
                // General-path cold-path workspace (exclusion rounds rebuild
                // reduced-p suff stats; cold path only — mirrors red_lme_*).
                // Stays `None` on the hot/Brent paths; the cold general branch
                // overwrites it before the LmeFitView borrows it.
                #[allow(unused_assignments)]
                let mut excl_lmm: Option<Box<crate::lmm::LmmWorkspace>> = None;

                // p_red: reduced predictor count — n_predictors_total on the hot path.
                let p_red: usize;

                // Reduced suff-stats buffers — cold-path allocations only on the
                // excluded branch. Declared here so they stay alive for the borrow
                // taken by LmeSuffStats and LmeScratch below.
                // (Cold path; mirrors the GLM arm's excl_x allocation.)
                let mut red_lme_xtx;
                let mut red_lme_xty;
                let mut red_lme_yty;
                let mut red_lme_sum_xc;
                // Reduced-scratch p-sized output buffers; subslices of workspace
                // on the hot path, local vecs on the cold path.
                let mut red_lme_xtvix;
                let mut red_lme_xtviy;
                let mut red_lme_xtvix_factor;
                let mut red_lme_betas;
                let mut red_lme_var_diag;
                let mut red_lme_t_sq;
                let mut red_lme_u_scratch;
                let mut red_lme_joint_sigma_t_chol;
                let mut red_lme_joint_rhs;
                let mut red_lme_joint_k_inv;

                if !needs_reduced_fit {
                    p_red = n_predictors_total;
                    // Hot path: point the reduced aliases at the workspace buffers
                    // (same as the pre-exclusion code, just renamed for uniformity).
                    // These are not used — the hot path builds LmeScratch directly
                    // from workspace fields below.
                    // Suppress unused-variable lints with a unit assignment.
                    red_lme_xtx = faer::Mat::<f64>::zeros(0, 0);
                    red_lme_xty = Vec::<f64>::new();
                    red_lme_yty = 0.0f64;
                    red_lme_sum_xc = faer::Mat::<f64>::zeros(0, 0);
                    red_lme_xtvix = faer::Mat::<f64>::zeros(0, 0);
                    red_lme_xtviy = Vec::<f64>::new();
                    red_lme_xtvix_factor = faer::Mat::<f64>::zeros(0, 0);
                    red_lme_betas = Vec::<f64>::new();
                    red_lme_var_diag = Vec::<f64>::new();
                    red_lme_t_sq = Vec::<f64>::new();
                    red_lme_u_scratch = Vec::<f64>::new();
                    red_lme_joint_sigma_t_chol = faer::Mat::<f64>::zeros(0, 0);
                    red_lme_joint_rhs = Vec::<f64>::new();
                    red_lme_joint_k_inv = faer::Mat::<f64>::zeros(0, 0);
                } else {
                    // --- reduced model: copy kept columns into excl_x,
                    // rebuild suff-stats, build remapped targets ---
                    // (cold path — mirrors the GLM arm; comments cite GLM arm by name)
                    excl_col_remap.resize(n_predictors_total, -1);
                    p_red = build_exclusion_remap(
                        spec,
                        &ws.factor_excluded_flags,
                        &mut excl_kept_cols,
                        &mut excl_col_remap,
                    );
                    // Copy kept columns of x_full into excl_x (mirrors GLM arm).
                    excl_x = faer::Mat::<f32>::zeros(n_usize, p_red);
                    for (rj, &cj) in excl_kept_cols.iter().enumerate() {
                        for i in 0..n_usize {
                            excl_x[(i, rj)] = ws.x_full[(i, cj as usize)];
                        }
                    }
                    // Remap targets; dropped targets will carry NaN in the gather below.
                    for (slot, &ti) in spec.target_indices.iter().enumerate() {
                        let r = excl_col_remap[ti as usize];
                        if r >= 0 {
                            excl_targets.push(r as u32);
                            excl_target_slots.push(slot);
                        }
                    }

                    // Allocate reduced suff-stat + scratch buffers (cold path).
                    // Cluster-indexed buffers (sum_yc, cluster_sizes, v_diag_inv) are
                    // reused from the workspace — their dimension is max_n_clusters,
                    // independent of P. Only predictor-indexed buffers change size.
                    // Cold-path locals are fresh allocations rather than submatrix_mut/
                    // slice views into the workspace — no borrow obstacle prevents it,
                    // but uniform ownership avoids split-borrow lifetime gymnastics.
                    let max_k = ws.lme_sum_yc.len();
                    red_lme_xtx = faer::Mat::<f64>::zeros(p_red, p_red);
                    red_lme_xty = vec![0.0; p_red];
                    red_lme_yty = 0.0;
                    red_lme_sum_xc = faer::Mat::<f64>::zeros(p_red, max_k);
                    // Scratch buffers: xtvix/xtviy/xtvix_factor sized p_red×p_red.
                    red_lme_xtvix = faer::Mat::<f64>::zeros(p_red, p_red);
                    red_lme_xtviy = vec![0.0; p_red];
                    red_lme_xtvix_factor = faer::Mat::<f64>::zeros(p_red, p_red);
                    red_lme_betas = vec![0.0; p_red];
                    red_lme_var_diag = vec![0.0; p_red];
                    red_lme_t_sq = vec![0.0; p_red];
                    red_lme_u_scratch = vec![0.0; p_red];
                    // Joint Wald scratch: p_red × p_red.
                    red_lme_joint_sigma_t_chol = faer::Mat::<f64>::zeros(p_red, p_red);
                    red_lme_joint_rhs = vec![0.0; p_red];
                    // joint_k_inv starts as zeros; joint_wald_chi_sq self-initializes
                    // the identity before solving (see lme.rs step 2 comment).
                    red_lme_joint_k_inv = faer::Mat::<f64>::zeros(p_red, p_red);
                }

                // Per (sim, N) reset + suff-stats build.
                // Hot path: build from the full design (ws.x_full).
                // Cold path (needs_reduced_fit): build from excl_x (reduced columns).
                // Cluster structure is preserved in both cases — only predictor
                // COLUMNS shrink; rows and cluster_ids are untouched.
                // Brent-path suff-stats build — skipped on the general lmm
                // path, which builds its own multi-grouping suff via
                // ws.lmm.suff and never touches the lme_* buffers.
                let n_clusters = if ws.lmm.is_none() {
                    ws.reset_lme_suff_stats();
                    {
                        let (x_to_fit, suff_xtx, suff_xty, suff_yty, suff_sum_xc) =
                            if !needs_reduced_fit {
                                (
                                    ws.x_full.as_ref().subrows(0, n_usize),
                                    ws.lme_xtx.as_mut(),
                                    ws.lme_xty.as_mut_slice(),
                                    &mut ws.lme_yty,
                                    ws.lme_sum_xc.as_mut(),
                                )
                            } else {
                                (
                                    excl_x.as_ref(),
                                    red_lme_xtx.as_mut(),
                                    red_lme_xty.as_mut_slice(),
                                    &mut red_lme_yty,
                                    red_lme_sum_xc.as_mut(),
                                )
                            };
                        let mut suff = crate::lme::LmeSuffStats {
                            xtx: suff_xtx,
                            xty: suff_xty,
                            yty: suff_yty,
                            sum_xc: suff_sum_xc,
                            sum_yc: &mut ws.lme_sum_yc,
                            cluster_sizes: &mut ws.lme_cluster_sizes,
                            n_clusters_seen: &mut ws.lme_n_clusters_seen,
                            // The reduced/cold arm fits p_red ≤ P columns, so the
                            // workspace panel (PANEL_ROWS·P) is large enough for
                            // either arm.
                            panel_x: &mut ws.panel_x,
                            panel_y: &mut ws.panel_y,
                        };
                        let y_slice = &ws.y_full[..n_usize];
                        let cid_slice = &ws.cluster_ids[..n_usize];
                        suff.add_rows(x_to_fit, y_slice, cid_slice);
                    }
                    // Snapshot the cluster count seen by suff-stats
                    // before the scratch borrow lifts the field.
                    ws.lme_n_clusters_seen
                } else {
                    0 // unused on the general path
                };

                // Truth start (spec/design-derived; see the lme_fit doc
                // comment): the realised per-sim τ² — including the L5 ICC
                // jitter — is in ws.tau_squared_design, and residuals are
                // unit-variance by construction, so θ_true = √τ²_design
                // exactly. Read before the scratch borrow lifts the field.
                let theta_start = Some(ws.tau_squared_design.max(0.0).sqrt());

                // Build LmeScratch via inline reborrow — mirrors
                // build_lme_scratch in lme.rs::tests.
                // Hot path: borrow workspace P-sized buffers directly.
                // Cold path: borrow the reduced p_red-sized local buffers.
                //
                // `lmm_pc` captures per-component bitmask from the general lmm
                // path (LmmFit.pinned_components); Brent path encodes bit 0 from
                // boundary_hit; OLS/Glm leave the pre-zeroed 0. Written to
                // pc[n_idx] at the common write-site below.
                let mut lmm_pc = 0u32;
                let (targets_fit, fit) = if ws.lmm.is_some() {
                    // ---- general lmm path (non-degenerate ClusterSpec) ----
                    // Field-disjoint borrows: `lmm` lives behind ws.lmm; the
                    // id/data buffers are sibling fields.
                    let y_slice = &ws.y_full[..n_usize];
                    let cid_slice = &ws.cluster_ids[..n_usize];
                    if !needs_reduced_fit {
                        let lmm = ws.lmm.as_deref_mut().expect("checked is_some");
                        lmm.suff.reset();
                        lmm.suff.add_rows_multi(
                            ws.x_full.as_ref().subrows(0, n_usize),
                            y_slice,
                            cid_slice,
                            &ws.extra_grouping_ids,
                        );
                        // Truth-start (P1, activated post-blind-parity). MIRRORS
                        // introspect::fit_provided_data's general arm — hot loop
                        // and debug path must derive the same hint; change
                        // together. The stack buffer sidesteps the
                        // &lmm.theta_truth vs &mut lmm borrow conflict (zero-alloc).
                        // Sized MAX_THETA (mirrors MAX_THETA — change together);
                        // validate() (invariants 20/21) ensure nt ≤ MAX_THETA.
                        let mut tbuf = [0.0_f64; crate::lmm::MAX_THETA];
                        let nt = lmm.theta_truth.len();
                        debug_assert!(nt <= tbuf.len(), "nt={nt} exceeds MAX_THETA");
                        tbuf[..nt].copy_from_slice(&lmm.theta_truth);
                        let f = crate::lmm::fit_lmm(lmm, &spec.target_indices, Some(&tbuf[..nt]));
                        optim_diag::record_fit(f.n_eval, f.converged, f.boundary_hit == 1);
                        lmm_pc = f.pinned_components; // general lmm arm: full bitmask
                        let lmm = ws.lmm.as_deref().expect("checked is_some");
                        (
                            &spec.target_indices[..],
                            crate::lme::LmeFitView {
                                betas: &lmm.fit.betas,
                                var_diag: &lmm.fit.var_diag,
                                t_sq: &lmm.fit.t_sq,
                                factor: lmm.fit.factor.as_ref(),
                                sigma_sq: f.sigma_sq,
                                // Scalar τ̂² is not meaningful for a general
                                // multi-component fit and is unused on this hot
                                // path (only introspect's scalar Mle branch
                                // reads it).
                                tau_sq_hat: f64::NAN,
                                converged: f.converged,
                                boundary_hit: f.boundary_hit,
                                n_iter: 0, // not a Brent fit
                                n_evals: f.n_eval as u32,
                                joint_t_sq: f.joint_t_sq,
                            },
                        )
                    } else {
                        // Cold path: reduced design → fresh reduced-p lmm
                        // workspace (cold-path alloc, mirrors red_lme_*).
                        // Intercept-only layout: the reduced model excludes target columns;
                        // slope col indices from the full design don't map to p_red (Task 5+).
                        let cluster = spec.cluster.as_ref().expect("lmm ⇒ cluster");
                        excl_lmm = Some(Box::new(crate::lmm::LmmWorkspace::for_cluster_spec(
                            p_red,
                            cluster,
                            n_usize,
                            &[],
                        )));
                        let lmm = excl_lmm.as_deref_mut().expect("just set");
                        lmm.suff.add_rows_multi(
                            excl_x.as_ref(),
                            y_slice,
                            cid_slice,
                            &ws.extra_grouping_ids,
                        );
                        // Truth-start (P1) — the cold reduced-p workspace carries
                        // its own theta_truth (for_cluster_spec). Mirrors the hot
                        // branch + introspect; change together.
                        // Sized MAX_THETA (mirrors MAX_THETA — change together);
                        // validate() (invariants 20/21) ensure nt ≤ MAX_THETA.
                        let mut tbuf = [0.0_f64; crate::lmm::MAX_THETA];
                        let nt = lmm.theta_truth.len();
                        debug_assert!(nt <= tbuf.len(), "nt={nt} exceeds MAX_THETA");
                        tbuf[..nt].copy_from_slice(&lmm.theta_truth);
                        let f = crate::lmm::fit_lmm(lmm, &excl_targets, Some(&tbuf[..nt]));
                        optim_diag::record_fit(f.n_eval, f.converged, f.boundary_hit == 1);
                        lmm_pc = f.pinned_components; // general lmm arm: full bitmask
                        let lmm = excl_lmm.as_deref().expect("just set");
                        (
                            &excl_targets[..],
                            crate::lme::LmeFitView {
                                betas: &lmm.fit.betas,
                                var_diag: &lmm.fit.var_diag,
                                t_sq: &lmm.fit.t_sq,
                                factor: lmm.fit.factor.as_ref(),
                                sigma_sq: f.sigma_sq,
                                // See the hot-path note above — scalar τ̂² unused
                                // for a general multi-component fit.
                                tau_sq_hat: f64::NAN,
                                converged: f.converged,
                                boundary_hit: f.boundary_hit,
                                n_iter: 0,
                                n_evals: f.n_eval as u32,
                                joint_t_sq: f.joint_t_sq,
                            },
                        )
                    }
                } else if !needs_reduced_fit {
                    let scratch = crate::lme::LmeScratch {
                        xtx: ws.lme_xtx.as_ref(),
                        xty: &ws.lme_xty,
                        yty: ws.lme_yty,
                        ols_scratch: OlsScratch {
                            fit_betas: &mut ws.fit_betas,
                            fit_var_diag: &mut ws.fit_var_diag,
                            fit_t_sq: &mut ws.fit_t_sq,
                            fit_u_scratch: &mut ws.fit_u_scratch,
                            fit_factor: ws.fit_factor.as_mut(),
                            fit_rhs: ws.fit_rhs.as_mut(),
                        },
                        sum_xc: ws.lme_sum_xc.as_mut().into_const(),
                        sum_yc: &ws.lme_sum_yc,
                        cluster_sizes: &ws.lme_cluster_sizes,
                        n_clusters,
                        n_rows: n,
                        xtvix: ws.lme_xtvix.as_mut(),
                        xtviy: &mut ws.lme_xtviy,
                        xtvix_factor: ws.lme_xtvix_factor.as_mut(),
                        v_diag_inv: &mut ws.lme_v_diag_inv,
                        betas: &mut ws.lme_betas,
                        var_diag: &mut ws.lme_var_diag,
                        t_sq: &mut ws.lme_t_sq,
                        u_scratch: &mut ws.lme_u_scratch,
                        sigma_sq: 0.0,
                        brent_log_a: &mut ws.lme_brent_log_a,
                        brent_log_b: &mut ws.lme_brent_log_b,
                        brent_log_c: &mut ws.lme_brent_log_c,
                        brent_fa: &mut ws.lme_brent_fa,
                        brent_fb: &mut ws.lme_brent_fb,
                        brent_fc: &mut ws.lme_brent_fc,
                        joint_sigma_t_chol: ws.lme_joint_sigma_t_chol.as_mut(),
                        joint_rhs: &mut ws.lme_joint_rhs,
                        joint_k_inv: ws.lme_joint_k_inv.as_mut(),
                    };
                    let x_slice = ws.x_full.as_ref().subrows(0, n_usize);
                    let y_slice = &ws.y_full[..n_usize];
                    let cid_slice = &ws.cluster_ids[..n_usize];
                    let f = crate::lme::lme_fit(
                        x_slice,
                        y_slice,
                        cid_slice,
                        &spec.target_indices,
                        theta_start,
                        scratch,
                    );
                    (&spec.target_indices[..], f)
                } else {
                    // Cold path: reduced design. The ols_scratch fields are required
                    // by the struct but unused by lme_fit today (τ̂≈0 path uses
                    // profiled_deviance, not fit_suff_stats_t_sq). Views are sized
                    // p_red to satisfy the struct; no other invariant is load-bearing.
                    let scratch = crate::lme::LmeScratch {
                        xtx: red_lme_xtx.as_ref(),
                        xty: &red_lme_xty,
                        yty: red_lme_yty,
                        ols_scratch: OlsScratch {
                            fit_betas: &mut ws.fit_betas[..p_red],
                            fit_var_diag: &mut ws.fit_var_diag[..p_red],
                            fit_t_sq: &mut ws.fit_t_sq[..p_red],
                            fit_u_scratch: &mut ws.fit_u_scratch[..p_red],
                            fit_factor: ws.fit_factor.as_mut().submatrix_mut(0, 0, p_red, p_red),
                            fit_rhs: ws.fit_rhs.as_mut().subrows_mut(0, n_usize.max(p_red)),
                        },
                        sum_xc: red_lme_sum_xc.as_ref(),
                        sum_yc: &ws.lme_sum_yc,
                        cluster_sizes: &ws.lme_cluster_sizes,
                        n_clusters,
                        n_rows: n,
                        xtvix: red_lme_xtvix.as_mut(),
                        xtviy: &mut red_lme_xtviy,
                        xtvix_factor: red_lme_xtvix_factor.as_mut(),
                        v_diag_inv: &mut ws.lme_v_diag_inv,
                        betas: &mut red_lme_betas,
                        var_diag: &mut red_lme_var_diag,
                        t_sq: &mut red_lme_t_sq,
                        u_scratch: &mut red_lme_u_scratch,
                        sigma_sq: 0.0,
                        brent_log_a: &mut ws.lme_brent_log_a,
                        brent_log_b: &mut ws.lme_brent_log_b,
                        brent_log_c: &mut ws.lme_brent_log_c,
                        brent_fa: &mut ws.lme_brent_fa,
                        brent_fb: &mut ws.lme_brent_fb,
                        brent_fc: &mut ws.lme_brent_fc,
                        joint_sigma_t_chol: red_lme_joint_sigma_t_chol.as_mut(),
                        joint_rhs: &mut red_lme_joint_rhs,
                        joint_k_inv: red_lme_joint_k_inv.as_mut(),
                    };
                    let y_slice = &ws.y_full[..n_usize];
                    let cid_slice = &ws.cluster_ids[..n_usize];
                    let f = crate::lme::lme_fit(
                        excl_x.as_ref(),
                        y_slice,
                        cid_slice,
                        &excl_targets,
                        theta_start,
                        scratch,
                    );
                    (&excl_targets[..], f)
                };

                conv[n_idx] = u8::from(fit.converged);
                bh[n_idx] = fit.boundary_hit;
                // Brent path (ws.lmm.is_none()): derive bit 0 from boundary_hit.
                // General lmm path: lmm_pc was set from LmmFit.pinned_components above.
                pc[n_idx] = if ws.lmm.is_none() {
                    u32::from(fit.boundary_hit == 1)
                } else {
                    lmm_pc
                };
                // LME overall_crit is INFINITY (hardwired — critvals.rs §Mle),
                // so overall is always 0. No df adjustment needed.
                //
                // Parked: the OLS-F / GLM-LRT "overall" omnibus is not defined
                // for a mixed-effects fit, so every host suppresses it for LME
                // (and clustered GLMM) — `report_overall` is never set, hence
                // `overall_slice` is normally `None` and this write is dead. The
                // branch stays as the wiring point for a future joint-Wald
                // omnibus: replace the ∞ crit (critvals.rs §Mle) and this `0`.
                // The asymptotic joint Wald-χ² that LME *does* compute is the
                // `joint_unc`/`joint_cor` channel just below, reached via
                // per-marginal routing — not this omnibus slot.
                if let Some(ref mut overall) = overall_slice.as_deref_mut() {
                    overall[n_idx] = 0;
                }

                // Joint Wald-χ² significance. Per-N
                // crit value lives at `crit.joint_t_crit_sq[n_idx]`.
                // The LME joint Wald-χ² test lies outside
                // the family-wise correction set, so
                // `joint_cor == joint_unc` for LME designs.
                // Under exclusion the joint test runs over the reduced target set
                // (excluded targets are untestable by definition).
                let k_red = targets_fit.len(); // spec.target_indices.len() hot, excl_targets.len() cold
                let joint_crit = if !needs_reduced_fit {
                    crit.joint_t_crit_sq[n_idx]
                } else if k_red == 0 {
                    f64::INFINITY // no testable targets ⇒ joint never significant
                } else {
                    // Reduced target set: χ² df must match k_red. This is the one
                    // reduced-df crit NOT served by the (N, p_red) cache the OLS/GLM
                    // arms use: k_red is a TARGET count, not a column count, so the
                    // cache key cannot express it. Explicit recompute stays.
                    crate::critvals::chi2_ppf(1.0 - spec.crit_values.alpha, k_red as f64)
                };
                let joint_sig =
                    fit.converged && fit.joint_t_sq.is_finite() && fit.joint_t_sq > joint_crit;
                joint_unc_slice[n_idx] = u8::from(joint_sig);
                joint_cor_slice[n_idx] = u8::from(joint_sig);

                let row_start = n_idx * main_row;
                let row_end = row_start + main_row;
                let unc_slice = &mut unc[row_start..row_end];
                unc_slice.fill(0);
                let t_crit_unc = crit.t_crit_sq_uncorrected[n_idx];
                // PREDICTOR-indexed gather (module-header invariant).
                // Hot path: fit.t_sq is indexed by original predictor position (0..P);
                //   gather via spec.target_indices.
                // Cold path: fit.t_sq is indexed by reduced predictor position (0..p_red);
                //   gather via excl_col_remap[ti] (PREDICTOR-indexed, reduced).
                // Dropped slots → NaN (never significant).
                if fit.converged {
                    for (j, &ti) in spec.target_indices.iter().enumerate() {
                        let t_sq = if !needs_reduced_fit {
                            fit.t_sq[ti as usize]
                        } else {
                            match excl_col_remap[ti as usize] {
                                r if r >= 0 => fit.t_sq[r as usize],
                                _ => f64::NAN,
                            }
                        };
                        if !t_sq.is_nan() && t_sq > t_crit_unc {
                            unc_slice[j] = 1;
                        }
                    }
                }
                // Debug stat capture (sample-size index 0 only).
                // MLE has no contrasts: the per-target row is the
                // compact t_sq marginals, NaN when not converged or excluded.
                if n_idx == 0 {
                    if let Some(row) = stat_sink.as_deref_mut() {
                        row.fill(f64::NAN);
                        if fit.converged {
                            for (j, &ti) in spec.target_indices.iter().enumerate() {
                                row[j] = if !needs_reduced_fit {
                                    fit.t_sq[ti as usize]
                                } else {
                                    match excl_col_remap[ti as usize] {
                                        r if r >= 0 => fit.t_sq[r as usize],
                                        _ => f64::NAN,
                                    }
                                };
                            }
                        }
                    }
                }
                let cor_slice = &mut cor[row_start..row_end];
                if fit.converged {
                    // Build full-width (n_targets) t² buffer for apply_correction,
                    // NaN at dropped target slots. Mirrors the GLM arm's all_t_sq pattern.
                    let n_t = spec.target_indices.len();
                    let all_t_sq = &mut ws.compact_t_sq_scratch[..n_t];
                    all_t_sq.fill(f64::NAN);
                    if !needs_reduced_fit {
                        // Hot path: gather PREDICTOR-indexed t_sq into target-compact buffer.
                        for (k, &ti) in spec.target_indices.iter().enumerate() {
                            all_t_sq[k] = fit.t_sq[ti as usize];
                        }
                    } else {
                        // Cold path: scatter remapped values; dropped slots stay NaN.
                        for (k, &slot) in excl_target_slots.iter().enumerate() {
                            all_t_sq[slot] = fit.t_sq[targets_fit[k] as usize];
                        }
                    }
                    apply_correction(
                        spec.correction_method,
                        all_t_sq,
                        t_crit_unc,
                        &crit.correction_t_crit_sq[n_idx],
                        cor_slice,
                    );
                } else {
                    cor_slice.fill(0);
                }
                // Posthoc skipped — rejected at batch entry for LME.
                // Write exclusion flags after the fit (flags were set BEFORE the fit;
                // fx_row write is after so the codes captured are the final per-N codes).
                if n_factors > 0 {
                    let fx_row = &mut fx[n_idx * n_factors..(n_idx + 1) * n_factors];
                    fx_row.copy_from_slice(&ws.factor_excluded_flags);
                }
            }
        }
        EstimatorSpec::Glm => {
            // No cross-N reuse for IRLS: the working response
            // z = η + (y−p)/W changes per iteration, so each
            // (sim, N) pair is refit from the spec-derived truth start
            // β₀ = spec.effect_sizes (the DGP applies them directly to the
            // linear predictor on the logit scale).
            // No per-sim accumulator → no Logit-side reset needed.
            // Posthoc is rejected at batch entry above.
            for (n_idx, &n) in sample_sizes.iter().enumerate() {
                let n_usize = n as usize;
                let x_slice = ws.x_full.as_ref().subrows(0, n_usize);
                let y_slice = &ws.y_full[..n_usize];

                // GLMM branch (Glm + cluster): a clustered binary design routes
                // through the Laplace GLMM kernel instead of the plain IRLS path.
                // No factor exclusion is in scope for clustered specs, so this
                // mirrors the LME *general* arm's writeback exactly (z²-vs-crit +
                // multiplicity correction + joint Wald) and `continue`s past the
                // plain-logistic attempt loop below.
                if ws.glmm.is_some() {
                    let slope_cols: Vec<usize> = spec
                        .cluster_slope_design_cols
                        .iter()
                        .map(|&c| c as usize)
                        .collect();

                    // `test_formula` reduced fit (Phase 2): fit only the FIXED
                    // columns in `spec.fit_columns`. The GLMM workspace was sized
                    // to p_red in `build_glmm_workspace`, so the reduced X / β-start
                    // / targets below match its β-dimension. Z is unchanged — build_z
                    // still reads the full `x_full` (the random structure is
                    // generation-side). `fit_targets` maps full→reduced positions in
                    // `target_indices` order, so the output slots are untouched. When
                    // `test_reduces` is false the buffers are empty and the full spec
                    // slices feed the fit, byte-for-byte the prior behaviour.
                    let (excl_x, reduced_beta, reduced_targets): (
                        faer::Mat<f32>,
                        Vec<f64>,
                        Vec<u32>,
                    ) = if test_reduces {
                        let p_red = spec.fit_columns.len();
                        let mut xr = faer::Mat::<f32>::zeros(n_usize, p_red);
                        for (rc, &fc) in spec.fit_columns.iter().enumerate() {
                            for i in 0..n_usize {
                                xr[(i, rc)] = ws.x_full[(i, fc as usize)];
                            }
                        }
                        let beta = spec
                            .fit_columns
                            .iter()
                            .map(|&fc| spec.effect_sizes[fc as usize])
                            .collect();
                        let targets = spec
                            .target_indices
                            .iter()
                            .map(|&t| {
                                spec.fit_columns
                                    .iter()
                                    .position(|&c| c == t)
                                    .expect("a test target is always in the test design")
                                    as u32
                            })
                            .collect();
                        (xr, beta, targets)
                    } else {
                        (faer::Mat::<f32>::zeros(0, 0), Vec::new(), Vec::new())
                    };
                    let fit_x = if test_reduces {
                        excl_x.as_ref().subrows(0, n_usize)
                    } else {
                        ws.x_full.as_ref().subrows(0, n_usize)
                    };
                    let fit_targets: &[u32] = if test_reduces {
                        &reduced_targets
                    } else {
                        &spec.target_indices
                    };
                    let fit_beta: &[f64] = if test_reduces {
                        &reduced_beta
                    } else {
                        &spec.effect_sizes
                    };

                    // build_z borrows ws.glmm (&mut) disjoint from
                    // ws.x_full/cluster_ids/extra_grouping_ids (&) — direct field
                    // access lets NLL prove the storage is disjoint.
                    // build_z is only needed by the dense (crossed/nested) GLMM
                    // path; the no-extras blocked path reconstructs mᵢ per row from
                    // x + Λ_p, so building the dense n×k Z would be dead work.
                    {
                        let glmm = ws.glmm.as_deref_mut().expect("is_some");
                        if !glmm.groupings.extra_offsets.is_empty() {
                            crate::glmm::build_z(
                                glmm,
                                ws.x_full.as_ref().subrows(0, n_usize),
                                &ws.cluster_ids[..n_usize],
                                &ws.extra_grouping_ids,
                                &slope_cols,
                                n_usize,
                            );
                        }
                    }
                    // Truth start θ from the workspace's per-spec recipe (mirrors
                    // the LME arm's stack-buffer trick to sidestep the &theta_truth
                    // vs &mut glmm borrow conflict). Sized MAX_THETA — change with
                    // the LME arm.
                    let mut tbuf = [0.0_f64; crate::lmm::MAX_THETA];
                    let f = {
                        let glmm = ws.glmm.as_deref_mut().expect("is_some");
                        let nt = glmm.theta_truth.len();
                        debug_assert!(nt <= tbuf.len(), "nt={nt} exceeds MAX_THETA");
                        tbuf[..nt].copy_from_slice(&glmm.theta_truth);
                        crate::glmm::fit_glmm(
                            glmm,
                            fit_x,
                            &ws.y_full[..n_usize],
                            &ws.cluster_ids[..n_usize],
                            fit_targets,
                            Some(&tbuf[..nt]),
                            fit_beta,
                            n_usize,
                            spec.wald_se,
                        )
                    };

                    optim_diag::record_fit(f.n_eval, f.converged, f.boundary_hit == 1);
                    conv[n_idx] = u8::from(f.converged);
                    bh[n_idx] = f.boundary_hit;
                    pc[n_idx] = f.pinned_components;
                    tau_hat[n_idx] = f.tau_squared_hat;
                    // Overall: GLMM emits 0 (mirrors the LME arm — no per-sim LRT
                    // is computed by fit_glmm).
                    if let Some(ref mut overall) = overall_slice.as_deref_mut() {
                        overall[n_idx] = 0;
                    }

                    // Joint Wald-χ² significance — identical rule to the LME arm
                    // (`f.joint_t_sq > χ²(k, 1−α)`, outside the FW-correction set so
                    // joint_cor == joint_unc). The χ² crit is computed here via
                    // chi2_ppf because the shared table only populates
                    // joint_t_crit_sq for Mle (NaN for Glm) — same formula the LME
                    // cold path uses, k = #targets (no exclusion).
                    let k = spec.target_indices.len();
                    let joint_crit = if k == 0 {
                        f64::INFINITY
                    } else {
                        crate::critvals::chi2_ppf(1.0 - spec.crit_values.alpha, k as f64)
                    };
                    let joint_sig =
                        f.converged && f.joint_t_sq.is_finite() && f.joint_t_sq > joint_crit;
                    joint_unc_slice[n_idx] = u8::from(joint_sig);
                    joint_cor_slice[n_idx] = u8::from(joint_sig);

                    // Per-target uncorrected significance — PREDICTOR-indexed gather.
                    // `glmm.t_sq` has the FITTED width (p_red under test_formula,
                    // else full p); `fit_targets[j]` is target j's position in that
                    // fitted design (identity when not reducing). Output slot j stays
                    // in `spec.target_indices` order. No factor exclusion in clustered
                    // specs, so the only reduction is test_formula's.
                    let row_start = n_idx * main_row;
                    let row_end = row_start + main_row;
                    let unc_slice = &mut unc[row_start..row_end];
                    unc_slice.fill(0);
                    let t_crit_unc = crit.t_crit_sq_uncorrected[n_idx];
                    if f.converged {
                        let t_sq = &ws.glmm.as_deref().expect("is_some").t_sq;
                        for (j, &ti) in fit_targets.iter().enumerate() {
                            let v = t_sq[ti as usize];
                            if !v.is_nan() && v > t_crit_unc {
                                unc_slice[j] = 1;
                            }
                        }
                    }
                    // Debug stat capture (sample-size index 0 only) — compact t_sq
                    // marginals, NaN when not converged (mirrors the LME arm).
                    if n_idx == 0 {
                        if let Some(row) = stat_sink.as_deref_mut() {
                            row.fill(f64::NAN);
                            if f.converged {
                                let t_sq = &ws.glmm.as_deref().expect("is_some").t_sq;
                                for (j, &ti) in fit_targets.iter().enumerate() {
                                    row[j] = t_sq[ti as usize];
                                }
                            }
                        }
                    }
                    // Corrected significance — full-width t² buffer fed to
                    // apply_correction (mirrors the LME/GLM arms' all_t_sq pattern).
                    let cor_slice = &mut cor[row_start..row_end];
                    if f.converged {
                        let n_t = fit_targets.len();
                        {
                            let t_sq = &ws.glmm.as_deref().expect("is_some").t_sq;
                            let all_t_sq = &mut ws.compact_t_sq_scratch[..n_t];
                            for (k, &ti) in fit_targets.iter().enumerate() {
                                all_t_sq[k] = t_sq[ti as usize];
                            }
                        }
                        let all_t_sq = &mut ws.compact_t_sq_scratch[..n_t];
                        apply_correction(
                            spec.correction_method,
                            all_t_sq,
                            t_crit_unc,
                            &crit.correction_t_crit_sq[n_idx],
                            cor_slice,
                        );
                    } else {
                        cor_slice.fill(0);
                    }
                    // Posthoc skipped — rejected at batch entry for Glm+cluster.
                    // No factor exclusion in clustered specs, so no fx write.
                    continue;
                }

                // Update exclusion flags BEFORE the fit so the attempt loop
                // sees the correct per-(sim, N) sparse-exclusion codes (0/1).
                // min_count == 0 ⇒ feature disabled — flags stay zero from the
                // per-sim reset; the separation fallback shares the same
                // min_count > 0 gate, so no stale code-2 can arise.
                if n_factors > 0 && spec.factor_min_level_count > 0 {
                    // GLM rebuilds per (sim, N); recount from 0..n_usize.
                    // update_factor_exclusions always writes 0 or 1 to every
                    // flag, clearing any code-2 left by the separation fallback
                    // at the previous n_idx.
                    ws.factor_prefix_counts.fill(0);
                    update_factor_exclusions(
                        ws.x_full.as_ref(),
                        &spec.factor_n_levels,
                        1 + spec.n_non_factor as usize,
                        0,
                        n_usize,
                        spec.factor_min_level_count,
                        &mut ws.factor_prefix_counts,
                        &mut ws.factor_excluded_flags,
                    );
                }

                // Exclusion scratch — round-scope locals (cold path, mirrors
                // the OLS arm). Zero-alloc until an excluded round actually fires.
                // The remap is recomputed per attempt; the separation fallback
                // flips a flag between attempts.
                let mut excl_kept_cols: Vec<u32> = Vec::new();
                let mut excl_col_remap: Vec<i32> = Vec::new();
                let mut excl_targets: Vec<u32> = Vec::new();
                let mut excl_target_slots: Vec<usize> = Vec::new();
                let mut excl_x = faer::Mat::<f32>::zeros(0, 0); // sized on first use (f32 data plane)
                let mut excl_beta_start: Vec<f64> = Vec::new();

                // Attempt 0: sparse-reduced (or full) model.
                // Attempt 1: separation fallback — drop the included factor with
                // the smallest min level count, refit once. Still failing →
                // non-converged as today.
                // The fallback only fires when min_count > 0 (feature enabled);
                // min_count == 0 keeps exactly today's single-attempt behaviour.
                // All three break arms must set every field below — they are coupled sites.
                let mut fit_final_converged = false;
                let mut fit_final_t_sq_buf: Vec<f64> = Vec::new(); // compact over targets_fit
                let mut fit_final_target_slots: Vec<usize> = Vec::new(); // slots back to [0..n_targets)
                let mut fit_final_deviance = f64::NAN;
                let mut fit_final_deviance_null = f64::NAN;
                let mut fit_final_p_red: usize = n_predictors_total; // reduced count for overall crit

                for attempt in 0..2u8 {
                    let any_excluded = ws.factor_excluded_flags.iter().any(|&c| c != 0);
                    let needs_reduced_fit = any_excluded || test_reduces;
                    let p_red: usize;
                    let targets_fit: &[u32];

                    if !needs_reduced_fit {
                        p_red = n_predictors_total;
                        excl_target_slots.clear();
                        // Full-model path: target slots are identity (slot j ↔ index j).
                        for j in 0..spec.target_indices.len() {
                            excl_target_slots.push(j);
                        }
                        targets_fit = &spec.target_indices;
                    } else {
                        excl_col_remap.resize(n_predictors_total, -1);
                        p_red = build_exclusion_remap(
                            spec,
                            &ws.factor_excluded_flags,
                            &mut excl_kept_cols,
                            &mut excl_col_remap,
                        );
                        excl_x = faer::Mat::<f32>::zeros(n_usize, p_red);
                        for (rj, &cj) in excl_kept_cols.iter().enumerate() {
                            for i in 0..n_usize {
                                excl_x[(i, rj)] = ws.x_full[(i, cj as usize)];
                            }
                        }
                        // Truth start for the reduced model: gather the kept
                        // columns' true effects (mirrors the hot path's
                        // spec.effect_sizes). Recomputed per attempt, so the
                        // separation fallback gets it too.
                        excl_beta_start.clear();
                        excl_beta_start.extend(
                            excl_kept_cols
                                .iter()
                                .map(|&cj| spec.effect_sizes[cj as usize]),
                        );
                        // Remap targets; dropped targets carry NaN in the
                        // scatter-back (excl_target_slots drives the scatter).
                        excl_targets.clear();
                        excl_target_slots.clear();
                        for (slot, &ti) in spec.target_indices.iter().enumerate() {
                            let r = excl_col_remap[ti as usize];
                            if r >= 0 {
                                excl_targets.push(r as u32);
                                excl_target_slots.push(slot);
                            }
                        }
                        targets_fit = &excl_targets;
                    };

                    let x_fit: faer::MatRef<f32> = if !needs_reduced_fit {
                        x_slice
                    } else {
                        excl_x.as_ref()
                    };
                    // Truth start (spec-derived; see the glm.rs module header):
                    // full model takes effect_sizes as-is, the sparse-reduced
                    // model the per-attempt gather above. The separation-
                    // fallback refit (attempt 1) stays COLD: the fallback fires
                    // precisely when the dropped effect is extreme, so the
                    // gathered start sits in a saturated region far from the
                    // reduced-model fixpoint, where IRLS without step-halving
                    // (v1 parity) overshoots the first step and trips BETA_CAP
                    // — the cold start preserves the rescue behaviour.
                    let beta_start: Option<&[f64]> = if attempt > 0 {
                        None
                    } else if !needs_reduced_fit {
                        Some(&spec.effect_sizes)
                    } else {
                        Some(&excl_beta_start)
                    };
                    let t_red = targets_fit.len();

                    let scratch = crate::glm::GlmScratch {
                        irls_eta: &mut ws.irls_eta[..n_usize],
                        irls_p: &mut ws.irls_p[..n_usize],
                        irls_w: &mut ws.irls_w[..n_usize],
                        irls_z: &mut ws.irls_z[..n_usize],
                        irls_betas: &mut ws.irls_betas[..p_red],
                        irls_betas_new: &mut ws.irls_betas_new[..p_red],
                        irls_var_diag: &mut ws.irls_var_diag[..t_red],
                        irls_t_sq: &mut ws.irls_t_sq[..t_red],
                        irls_u_scratch: &mut ws.irls_u_scratch[..p_red],
                        irls_xtwx: ws.irls_xtwx.as_mut().submatrix_mut(0, 0, p_red, p_red),
                        irls_xtwz: &mut ws.irls_xtwz[..p_red],
                        irls_l: ws.irls_l.as_mut().submatrix_mut(0, 0, p_red, p_red),
                        irls_x_f64: &mut ws.irls_x_f64[..n_usize * p_red],
                        irls_wx: &mut ws.irls_wx[..n_usize * p_red],
                    };
                    let fit =
                        crate::glm::glm_irls_fit(x_fit, y_slice, targets_fit, beta_start, scratch);

                    if fit.converged || attempt == 1 {
                        // Final attempt: record results and break.
                        fit_final_converged = fit.converged;
                        fit_final_t_sq_buf = fit.t_sq.to_vec();
                        fit_final_target_slots = excl_target_slots.clone();
                        fit_final_deviance = fit.deviance;
                        fit_final_deviance_null = fit.deviance_null;
                        fit_final_p_red = p_red;
                        break;
                    }

                    // Separation fallback (attempt 0 non-converged, min_count > 0):
                    // drop the included factor with the smallest min level count, refit once.
                    // If min_count == 0 the loop runs only attempt 0 (we break above when
                    // attempt == 1 is the second go); but the fallback gate below also
                    // ensures we only enter the separation path when the feature is enabled.
                    if spec.factor_min_level_count == 0 {
                        // Feature disabled — treat as final (non-converged).
                        fit_final_converged = fit.converged;
                        fit_final_t_sq_buf = fit.t_sq.to_vec();
                        fit_final_target_slots = excl_target_slots.clone();
                        fit_final_deviance = fit.deviance;
                        fit_final_deviance_null = fit.deviance_null;
                        fit_final_p_red = p_red;
                        break;
                    }
                    match included_factor_with_smallest_min_count(
                        &ws.factor_prefix_counts,
                        &spec.factor_n_levels,
                        &ws.factor_excluded_flags,
                    ) {
                        Some(f) => {
                            ws.factor_excluded_flags[f] = 2;
                            // loop continues to attempt 1
                        }
                        None => {
                            // No factor to drop — record non-converged, done.
                            fit_final_converged = fit.converged;
                            fit_final_t_sq_buf = fit.t_sq.to_vec();
                            fit_final_target_slots = excl_target_slots.clone();
                            fit_final_deviance = fit.deviance;
                            fit_final_deviance_null = fit.deviance_null;
                            fit_final_p_red = p_red;
                            break;
                        }
                    }
                }

                // Post-fit writes — happen exactly once, using fit_final_*.
                // flags now carry post-fallback codes (0/1 = sparse, 2 = separation).
                conv[n_idx] = u8::from(fit_final_converged);
                bh[n_idx] = 0; // Logit: never a boundary condition

                let row_start = n_idx * main_row;
                let row_end = row_start + main_row;
                let unc_slice = &mut unc[row_start..row_end];
                unc_slice.fill(0);
                let t_crit_unc = crit.t_crit_sq_uncorrected[n_idx];
                let n_targets = spec.target_indices.len();

                // Build full-width t² buffer (length n_targets) with NaN at dropped
                // target slots, for both unc significance and apply_correction.
                // compact_t_sq_scratch (length ≥ n_predictors ≥ n_targets) is the
                // designated scratch for this pattern (mirrors the MLE gather path).
                let all_t_sq = &mut ws.compact_t_sq_scratch[..n_targets];
                all_t_sq.fill(f64::NAN);
                if fit_final_converged {
                    let any_excluded = ws.factor_excluded_flags.iter().any(|&c| c != 0);
                    let needs_reduced_fit = any_excluded || test_reduces;
                    if !needs_reduced_fit {
                        // Hot path: copy marginal t² values directly.
                        all_t_sq.copy_from_slice(&fit_final_t_sq_buf);
                    } else {
                        // fit_final_t_sq_buf is compact over kept targets;
                        // scatter back through excl_target_slots.
                        for (k, &slot) in fit_final_target_slots.iter().enumerate() {
                            all_t_sq[slot] = fit_final_t_sq_buf[k];
                        }
                    }
                    for (j, &t_sq) in all_t_sq.iter().enumerate() {
                        if !t_sq.is_nan() && t_sq > t_crit_unc {
                            unc_slice[j] = 1;
                        }
                    }
                }
                // Debug stat capture (sample-size index 0 only).
                // GLM has no contrasts: `all_t_sq` is the per-target row,
                // NaN where not converged or where the target's factor was excluded.
                if n_idx == 0 {
                    if let Some(row) = stat_sink.as_deref_mut() {
                        row.copy_from_slice(all_t_sq);
                    }
                }
                let cor_slice = &mut cor[row_start..row_end];
                if fit_final_converged {
                    apply_correction(
                        spec.correction_method,
                        all_t_sq,
                        t_crit_unc,
                        &crit.correction_t_crit_sq[n_idx],
                        cor_slice,
                    );
                } else {
                    cor_slice.fill(0);
                }
                // This deviance LRT is the legitimate omnibus for an *unclustered*
                // GLM. A clustered logistic (GLMM) reaches this same code path
                // (estimator == Glm + cluster), but its deviance is cluster-naive,
                // so the LRT would be anti-conservative — hosts suppress
                // `report_overall` for clustered fits, leaving `overall_slice`
                // `None` here. Parked for a future cluster-aware joint Wald.
                if let Some(ref mut overall) = overall_slice.as_deref_mut() {
                    let any_excluded = ws.factor_excluded_flags.iter().any(|&c| c != 0);
                    let needs_reduced_fit = any_excluded || test_reduces;
                    // Under exclusion the LRT has fit_final_p_red−1 df, not the
                    // full-P precomputed df — read the reduced-model crit from
                    // the shared (N, p_red) cache (same build_crit_tables path
                    // as the full table; intercept-only maps to INFINITY there).
                    // GLM marginal/correction crits stay on the full table:
                    // Wald-z is df-free, so they are exact under exclusion.
                    let crit_overall = if !needs_reduced_fit {
                        crit.overall_crit[n_idx]
                    } else {
                        reduced_crit_entry(
                            &mut ws.reduced_crit_cache,
                            n,
                            fit_final_p_red as u32,
                            spec,
                        )?
                        .crit
                        .overall_crit[0]
                    };
                    let mut sig: u8 = 0;
                    if fit_final_converged
                        && fit_final_deviance.is_finite()
                        && fit_final_deviance_null.is_finite()
                    {
                        let lrt = fit_final_deviance_null - fit_final_deviance;
                        if lrt.is_finite() && lrt > crit_overall {
                            sig = 1;
                        }
                    }
                    overall[n_idx] = sig;
                }
                // Posthoc skipped — rejected at batch entry for Logit.
                // Write exclusion flags after the fit loop, capturing
                // post-fallback codes (code 2 from separation set above).
                if n_factors > 0 {
                    let fx_row = &mut fx[n_idx * n_factors..(n_idx + 1) * n_factors];
                    fx_row.copy_from_slice(&ws.factor_excluded_flags);
                }
            }
        }
    }

    // Silence unused-variable lints for the two `Option<&mut [u8]>` params
    // that only the OLS family branch reads via `as_deref_mut()`. The Logit
    // and LME branches deliberately ignore them.
    let _ = post_unc_slice;
    let _ = post_cor_slice;
    Ok(())
}

/// Progress + cancellation reporter shared by the rayon and sequential
/// dispatch paths. Returns `Err(EngineError::Cancelled)` when the sink
/// returns `false` so the caller can stop iterating.
fn report_progress(
    counter_ref: &AtomicU64,
    cancelled_ref: &std::sync::atomic::AtomicBool,
    progress_step: u64,
    n_sims: u64,
    sink: Option<&dyn ProgressSink>,
) -> Result<(), EngineError> {
    // Per-sim cancellation poll (cheap atomic load). Separate from the ~50×/run
    // `report` checkpoint below so cancel latency is bounded to ~one sim even
    // when checkpoints are many slow fits apart — see `ProgressSink::is_cancelled`.
    if let Some(s) = sink {
        if s.is_cancelled() {
            cancelled_ref.store(true, Ordering::Relaxed);
            return Err(EngineError::Cancelled);
        }
    }
    let prev = counter_ref.fetch_add(1, Ordering::Relaxed);
    let now = prev + 1;
    let should_report = now % progress_step == 0 || now == n_sims;
    if should_report {
        if let Some(s) = sink {
            if !s.report(now, n_sims) {
                cancelled_ref.store(true, Ordering::Relaxed);
                return Err(EngineError::Cancelled);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build all C(k,2) pairwise contrasts for each posthoc block in `spec.posthoc`,
/// concatenating block results in block order. Returns empty Vec when `spec.posthoc`
/// is empty.
///
/// Canonical contrast order per block: `for a in 0..k { for b in (a+1)..k { L[b] − L[a] } }`,
/// where L[0] = zero vector (reference), L[i>0] = unit vector at `target_indices[i-1]`.
fn build_posthoc_contrasts(spec: &SimulationSpec) -> Result<Vec<Vec<f64>>, EngineError> {
    if spec.posthoc.is_empty() {
        return Ok(Vec::new());
    }
    let p =
        1 + spec.n_non_factor as usize + spec.n_factor_dummies as usize + spec.interactions.len();
    let mut out = Vec::new();
    for block in &spec.posthoc {
        let dummies = &block.target_indices;
        let k = dummies.len() + 1; // number of levels (1 reference + dummies)
                                   // Range-check all dummy indices.
        for &ti in dummies {
            if (ti as usize) >= p {
                return Err(EngineError::InvalidSpec(format!(
                    "posthoc target index {ti} out of range (n_predictors = {p})"
                )));
            }
        }
        // Emit C(k,2) contrasts in canonical order.
        for a in 0..k {
            for b in (a + 1)..k {
                // vec(L[b]) − vec(L[a]), where L[0]=zero, L[i>0]=unit at dummies[i-1].
                let mut c = vec![0.0_f64; p];
                if b > 0 {
                    c[dummies[b - 1] as usize] = 1.0;
                }
                if a > 0 {
                    c[dummies[a - 1] as usize] -= 1.0;
                }
                out.push(c);
            }
        }
    }
    Ok(out)
}

/// Studentized-range `k` for a Tukey-HSD target sitting at kernel column `col`.
///
/// Kernel column layout is `[intercept=0, continuous=1..=n_non_factor, factor
/// dummies…]`. Factor `f` (with `L_f` levels) owns `L_f − 1` consecutive dummy
/// columns; `factor_n_levels` lists `L_f` in that same layout order. Returns the
/// owning factor's `L` as `f64` when `col` lands in a factor's dummy block, else
/// `f64::NAN` (intercept or continuous predictor — not a Tukey-eligible target).
/// `q_tukey_ppf` maps `k < 2` / NaN to NaN, so a misdirected target fails.
fn tukey_k_for_kernel_col(col: u32, n_non_factor: u32, factor_n_levels: &[i32]) -> f64 {
    let dummy_base = 1 + n_non_factor;
    if col < dummy_base {
        // intercept or a continuous predictor — no factor.
        return f64::NAN;
    }
    let mut cursor = dummy_base;
    for &nl in factor_n_levels {
        let l = nl.max(0) as u32;
        let width = l.saturating_sub(1); // dummy columns for this factor
        if col < cursor + width {
            return l as f64;
        }
        cursor += width;
    }
    // Past the last factor's dummies — shouldn't happen for a validated spec.
    f64::NAN
}

/// Build per-sample-size correction thresholds for one posthoc block.
///
/// `factor_n_levels_for_posthoc`: one entry = this block's k (the factor's level
/// count). Under `TukeyHsd`, each contrast in the block gets `tukey_k = k`; for
/// all other correction methods the `factor_n_levels_for_posthoc` slice is unused
/// (the regular `CritValueTable::build` path handles it).
///
/// `estimator` is plumbed through for consistency — posthoc is only supported
/// for Ols, so the Glm/Mle arms are effectively dead, but the symmetric signature
/// preserves layout uniformity.
fn build_posthoc_correction_crit_with_levels(
    crit: &crate::spec::CritValues,
    sample_sizes: &[u32],
    n_predictors: u32,
    n_contrasts: u32,
    correction_method: CorrectionMethod,
    estimator: EstimatorSpec,
    factor_n_levels_for_posthoc: &[i32],
) -> Result<Vec<Vec<f64>>, EngineError> {
    if n_contrasts == 0 {
        return Ok(sample_sizes.iter().map(|_| Vec::new()).collect());
    }
    let post_alpha = crit.posthoc_alpha.unwrap_or(crit.alpha);
    let post_crit = crate::spec::CritValues {
        alpha: post_alpha,
        posthoc_alpha: Some(post_alpha),
    };
    let table = if correction_method == CorrectionMethod::TukeyHsd {
        // Build one Tukey-k entry per contrast: all contrasts in this block
        // belong to the same factor with k = factor_n_levels_for_posthoc[0].
        let k = factor_n_levels_for_posthoc.first().copied().unwrap_or(0) as f64;
        let tukey_k_per_contrast: Vec<f64> = vec![k; n_contrasts as usize];
        CritValueTable::build_with_tukey_k(
            &post_crit,
            sample_sizes,
            n_predictors,
            n_contrasts,
            correction_method,
            estimator,
            &tukey_k_per_contrast,
        )?
    } else {
        CritValueTable::build(
            &post_crit,
            sample_sizes,
            n_predictors,
            n_contrasts,
            correction_method,
            estimator,
        )?
    };
    Ok(table.correction_t_crit_sq)
}

/// Posthoc block shapes — one entry per `spec.posthoc` block, in request order.
/// kᵢ = the block's level count (dummies + 1); contrasts = C(kᵢ, 2).
fn posthoc_block_shapes(spec: &SimulationSpec) -> Vec<PosthocBlockShape> {
    spec.posthoc
        .iter()
        .map(|p| {
            let k = (p.target_indices.len() + 1) as u32;
            PosthocBlockShape {
                n_levels: k,
                n_contrasts: k * (k - 1) / 2,
            }
        })
        .collect()
}

/// Build every df-dependent critical-value table for a model with
/// `n_predictors` fitted columns: the main `CritValueTable` plus the per-N
/// post-hoc correction rows (flat across post-hoc blocks; empty inner vecs
/// when the spec has no post-hoc).
///
/// Single authority for the df rules. Called once per batch with the full
/// predictor count (the hot-path tables) and per sparse-exclusion round with
/// the reduced count `p_red` via `reduced_crit_entry` — both paths share this
/// code, so reduced-model thresholds cannot drift from full-model ones.
/// `n_targets` is always the planned target count (marginals + contrasts):
/// the correction-rank alphas follow the *planned* family size even when some
/// targets are untestable in a given round; only the df transform follows
/// `n_predictors`.
fn build_crit_tables(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_predictors: u32,
) -> Result<(CritValueTable, Vec<Vec<f64>>), EngineError> {
    let n_targets = spec.target_indices.len() + spec.contrast_pairs.len();
    // Per-target Tukey `k` (the factor level count `L` of the factor each target
    // belongs to). Only consumed by the TukeyHsd arm; cheap to always compute.
    // Target order matches the per-N correction row: marginals first (in
    // `target_indices` order), then contrasts (in `contrast_pairs` order).
    let tukey_k_per_target: Vec<f64> = if spec.correction_method == CorrectionMethod::TukeyHsd {
        let mut k = Vec::with_capacity(n_targets);
        for &col in &spec.target_indices {
            k.push(tukey_k_for_kernel_col(
                col,
                spec.n_non_factor,
                &spec.factor_n_levels,
            ));
        }
        for &(pos, _neg) in &spec.contrast_pairs {
            // A contrast tests β_pos − β_neg. Tukey HSD compares means within a
            // single factor; the positive side identifies that factor. If pos
            // and neg live in different factors (or one is non-factor), `k` is
            // taken from pos's factor — a same-factor pairwise contrast (the only
            // sensible Tukey target) has both sides in the same block, so `k` is
            // that factor's `L` either way.
            k.push(tukey_k_for_kernel_col(
                pos,
                spec.n_non_factor,
                &spec.factor_n_levels,
            ));
        }
        k
    } else {
        Vec::new()
    };
    let crit = CritValueTable::build_with_tukey_k(
        &spec.crit_values,
        sample_sizes,
        n_predictors,
        n_targets as u32,
        spec.correction_method,
        spec.estimator,
        &tukey_k_per_target,
    )?;

    // Build correction crit per block, concatenating into one flat-per-n_idx Vec.
    let posthoc_blocks = posthoc_block_shapes(spec);
    let posthoc_n_contrasts: usize = posthoc_blocks.iter().map(|b| b.n_contrasts as usize).sum();
    let posthoc_correction_crit_sq: Vec<Vec<f64>> = {
        if posthoc_n_contrasts == 0 {
            sample_sizes.iter().map(|_| Vec::new()).collect()
        } else {
            // Start with empty vecs per sample-size slot.
            let mut result: Vec<Vec<f64>> = (0..sample_sizes.len()).map(|_| Vec::new()).collect();
            for block in &posthoc_blocks {
                let block_crit = build_posthoc_correction_crit_with_levels(
                    &spec.crit_values,
                    sample_sizes,
                    n_predictors,
                    block.n_contrasts,
                    spec.correction_method,
                    spec.estimator,
                    &[block.n_levels as i32],
                )?;
                for (n_idx, slot) in result.iter_mut().enumerate() {
                    slot.extend_from_slice(&block_crit[n_idx]);
                }
            }
            result
        }
    };
    Ok((crit, posthoc_correction_crit_sq))
}

// ---------------------------------------------------------------------------
// Exclusion helpers
// ---------------------------------------------------------------------------

/// Look up (or build and memoize) the reduced-model crit entry for a
/// sparse-exclusion round. Key is `(n, p_red)`: every df-dependent threshold
/// depends only on the kept column count, never on which factors were dropped
/// (see `SimWorkspace::reduced_crit_cache`). Built by `build_crit_tables` —
/// the same constructor as the hot-path tables.
fn reduced_crit_entry<'a>(
    cache: &'a mut Vec<ReducedCritEntry>,
    n: u32,
    p_red: u32,
    spec: &SimulationSpec,
) -> Result<&'a ReducedCritEntry, EngineError> {
    if let Some(i) = cache.iter().position(|e| e.n == n && e.p_red == p_red) {
        return Ok(&cache[i]);
    }
    let (crit, mut posthoc_corr) = build_crit_tables(spec, &[n], p_red)?;
    // Single-N build → exactly one per-N row (empty when no posthoc blocks).
    let posthoc_corr_row = posthoc_corr.pop().unwrap_or_default();
    cache.push(ReducedCritEntry {
        n,
        p_red,
        crit,
        posthoc_corr_row,
    });
    Ok(cache.last().expect("entry just pushed"))
}

/// Grow the per-level prefix counts with rows [from..to), reading each
/// factor's level back from its dummy block in `x` (every generation arm
/// writes dummies as exact 0.0/1.0 and transforms touch only continuous
/// columns, so `== 1.0` recovery is exact). Then write sparse-exclusion
/// codes: flags[f] = 1 where any level of factor f sits below min_count.
/// Code 2 (separation) is assigned later by the GLM fallback; this function
/// only writes 0/1.
///
/// Calling convention: OLS passes `from = previous prefix end` (incremental,
/// cumulative counts); GLM/MLE reset `prefix_counts` to zero and call with
/// `from = 0` each grid point because they rebuild their fits from scratch per N.
#[allow(clippy::too_many_arguments)]
fn update_factor_exclusions(
    x: faer::MatRef<f32>,
    factor_n_levels: &[i32],
    dummy_base: usize, // first dummy column: 1 + n_non_factor
    from: usize,
    to: usize,
    min_count: u32,
    prefix_counts: &mut [u32],
    flags: &mut [u8],
) {
    let n_factors = factor_n_levels.len();
    // 1. Count the new rows: level = 1 + position of the 1.0 in the factor's
    //    dummy block, or 0 (reference) when the block is all zeros.
    let mut prop_off = 0usize;
    let mut col = dummy_base;
    #[allow(clippy::needless_range_loop)]
    for f in 0..n_factors {
        let l = factor_n_levels[f].max(0) as usize;
        let width = l.saturating_sub(1);
        for i in from..to {
            let mut lvl = 0usize;
            for d in 0..width {
                // Dummies are written exact 0.0f32/1.0f32 in data_gen — exact in f32.
                if x[(i, col + d)] == 1.0f32 {
                    lvl = d + 1;
                    break;
                }
            }
            prefix_counts[prop_off + lvl] += 1;
        }
        prop_off += l;
        col += width;
    }
    // 2. Refresh flags (reset; min_count == 0 disables the feature).
    let mut prop_off = 0usize;
    for f in 0..n_factors {
        let l = factor_n_levels[f].max(0) as usize;
        let min_c = prefix_counts[prop_off..prop_off + l]
            .iter()
            .copied()
            .min()
            .unwrap_or(0);
        flags[f] = u8::from(min_count > 0 && min_c < min_count);
        prop_off += l;
    }
}

/// Build kept-column list + full→reduced remap from the exclusion flags.
/// Kernel layout: [0=intercept | 1..=n_nf continuous | dummies | interactions].
/// An interaction column is dropped iff any of its component columns is a
/// dropped dummy. Returns p_red (number of kept columns).
pub(crate) fn build_exclusion_remap(
    spec: &SimulationSpec,
    flags: &[u8],
    kept_cols: &mut Vec<u32>,
    col_remap: &mut [i32],
) -> usize {
    let n_nf = spec.n_non_factor as usize;
    let dummy_base = 1 + n_nf;
    let n_fd = spec.n_factor_dummies as usize;
    col_remap.fill(-1);
    kept_cols.clear();
    // Per-dummy-column keep mask, derived from the owning factor's flag.
    let mut col = dummy_base;
    let mut keep_dummy = vec![true; n_fd]; // cold path; allocation acceptable
    for (f, &nl) in spec.factor_n_levels.iter().enumerate() {
        let width = (nl.max(0) as usize).saturating_sub(1);
        if flags[f] != 0 {
            for d in 0..width {
                keep_dummy[col + d - dummy_base] = false;
            }
        }
        col += width;
    }
    // Test-design (`test_formula`) keep-set: a column survives only if the
    // fitted design retains it. Empty `fit_columns` ⇒ no reduction (every
    // column passes), so the function is byte-identical to its factor-only
    // self. Composes with factor exclusion by intersection:
    // keep(col) = in_test_design(col) ∧ not_factor_excluded(col).
    let in_test =
        |col: usize| spec.fit_columns.is_empty() || spec.fit_columns.contains(&(col as u32));
    // Intercept + continuous: kept unless the test design drops them.
    #[allow(clippy::needless_range_loop)]
    for c in 0..dummy_base {
        if in_test(c) {
            col_remap[c] = kept_cols.len() as i32;
            kept_cols.push(c as u32);
        }
    }
    #[allow(clippy::needless_range_loop)]
    for d in 0..n_fd {
        let col = dummy_base + d;
        if keep_dummy[d] && in_test(col) {
            col_remap[col] = kept_cols.len() as i32;
            kept_cols.push(col as u32);
        }
    }
    // Interactions: appended after the dummies, one column per inner vec.
    let inter_base = dummy_base + n_fd;
    for (j, comp) in spec.interactions.iter().enumerate() {
        let col = inter_base + j;
        let dropped = comp.iter().any(|&c| {
            let cu = c as usize;
            cu >= dummy_base && cu < inter_base && !keep_dummy[cu - dummy_base]
        });
        if !dropped && in_test(col) {
            col_remap[col] = kept_cols.len() as i32;
            kept_cols.push(col as u32);
        }
    }
    kept_cols.len()
}

/// Pick the included factor (flag == 0) whose minimum per-level count is
/// smallest, as the candidate to drop when GLM IRLS fails to converge (the
/// separation fallback). Returns `Some(factor_index)` or `None` when every
/// factor is already flagged (no candidate available) or there are no factors.
///
/// `prefix_counts` is in `factor_proportions` layout (Σ `factor_n_levels[f]`
/// entries per factor, packed contiguously), matching the layout written by
/// `update_factor_exclusions`. `flags` is one entry per factor.
fn included_factor_with_smallest_min_count(
    prefix_counts: &[u32],
    factor_n_levels: &[i32],
    flags: &[u8],
) -> Option<usize> {
    let mut best: Option<(usize, u32)> = None; // (factor_index, min_count)
    let mut prop_off = 0usize;
    for (f, &nl) in factor_n_levels.iter().enumerate() {
        let l = nl.max(0) as usize;
        if flags[f] == 0 {
            // Factor is currently included — find its minimum per-level count.
            let min_c = prefix_counts[prop_off..prop_off + l]
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            match best {
                None => best = Some((f, min_c)),
                Some((_, prev_min)) if min_c < prev_min => best = Some((f, min_c)),
                _ => {}
            }
        }
        prop_off += l;
    }
    best.map(|(f, _)| f)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
        OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
    };

    fn minimal_spec() -> SimulationSpec {
        // 1 intercept + 1 continuous predictor; 1 target.
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.5],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    #[test]
    fn run_batch_shapes_match() {
        let spec = minimal_spec();
        let result = run_batch(&spec, &[100, 200], 50, 42, None).unwrap();
        assert_eq!(result.shape.n_sims, 50);
        assert_eq!(result.shape.n_sample_sizes, 2);
        assert_eq!(result.shape.n_targets, 1);
        assert_eq!(result.uncorrected.len(), 50 * 2);
        assert_eq!(result.corrected.len(), 50 * 2);
        assert_eq!(result.converged.len(), 50 * 2);
        assert!(result.posthoc_unc.is_empty());
        assert!(result.posthoc_cor.is_empty());
    }

    /// EP-1 invariant: the uncorrected/corrected significance buffers are sized
    /// `n_sims × n_sample_sizes × (target_indices.len() + contrast_pairs.len())`,
    /// NOT by `target_indices.len()` alone. Consumers that assume the shorter size
    /// have caused a hot-loop panic, a plot OOB, and a silent WASM merge drop.
    /// This test is the first in the suite to exercise a run with BOTH non-empty
    /// `target_indices` (len 2) AND non-empty `contrast_pairs` (len 1), so that
    /// any regression to the wrong stride fires immediately.
    #[test]
    fn run_batch_buffer_sized_by_marginals_plus_contrasts() {
        // two_factor_spec: n_non_factor=1, n_factor_dummies=3 (factor0 → col 2;
        // factor1 → cols 3 & 4), one interaction at col 5.
        // target_indices=[1, 2]: 2 marginal targets (x1 and factor0 dummy).
        // contrast_pairs=[(3, 4)]: pairwise contrast between the two factor1
        // dummies — distinct from both marginals, so n_targets = 2 + 1 = 3.
        let mut spec = two_factor_spec();
        spec.contrast_pairs = vec![(3, 4)];

        let n_sims: u32 = 50;
        let sample_sizes = [100u32, 200];
        let n_sample_sizes = sample_sizes.len(); // 2

        let n_marginals = spec.target_indices.len(); // 2
        let n_contrasts = spec.contrast_pairs.len(); // 1
        let expected = n_sims as usize * n_sample_sizes * (n_marginals + n_contrasts); // 50*2*3 = 300

        let result = run_batch(&spec, &sample_sizes, n_sims, 42, None).unwrap();

        assert_eq!(
            result.uncorrected.len(),
            expected,
            "uncorrected buffer must be n_sims*n_sample_sizes*(marginals+contrasts)"
        );
        assert_eq!(
            result.corrected.len(),
            expected,
            "corrected buffer must be n_sims*n_sample_sizes*(marginals+contrasts)"
        );
        assert_eq!(result.shape.n_targets, (n_marginals + n_contrasts) as u32);
    }

    #[test]
    fn run_batch_st_matches_run_batch_on_same_seed() {
        // Per-sim seeding via `pcg_mix64(base_seed, sim_id)` is identical in
        // both variants; only the dispatch differs. So the output must be
        // bit-equal across MT and ST given the same (spec, seed, n_sims).
        let spec = minimal_spec();
        let n_sims = 64u32;
        let seed = 42u64;
        let mt = run_batch(&spec, &[100], n_sims, seed, None).unwrap();
        let st = run_batch_st(&spec, &[100], n_sims, seed, None).unwrap();
        assert_eq!(mt.uncorrected, st.uncorrected);
        assert_eq!(mt.corrected, st.corrected);
        assert_eq!(mt.converged, st.converged);
        assert_eq!(mt.boundary_hit, st.boundary_hit);
        assert_eq!(mt.shape.n_sims, st.shape.n_sims);
    }

    /// Native determinism (Property D): `run_batch` (multi-core, rayon) and
    /// `run_batch_st` (single-core) return BIT-IDENTICAL `BatchResult`s for the
    /// same `(spec, base_seed, n_sims)`, independent of thread count. Holds
    /// because every per-sim stream keys purely on the global `sim_id` via
    /// `pcg_mix64(base_seed, sim_id)` (never on a per-thread offset), and every
    /// `BatchResult` field is an integer significance/convergence tensor whose
    /// chunk merge is a concatenation in canonical `sim_id` order — there is no
    /// floating-point reduction whose associativity could depend on the rayon
    /// split, so MT and ST cannot diverge.
    ///
    /// `two_factor_spec` is the fixture on purpose: two marginal targets + a
    /// factor block + an interaction populate `uncorrected`/`corrected`/
    /// `converged`/`boundary_hit`/`factor_excluded` together, and `n_sims = 2000`
    /// over a two-point n-grid forces several rayon chunks so a
    /// thread-split-dependent merge bug would actually surface here.
    ///
    /// Forward guard: if a future change lands a non-integer (f64) field in
    /// `BatchResult`, this gate stays valid only if that field sums in canonical
    /// `sim_id` order. A rayon-order f64 reduction would make MT/ST diverge in
    /// the low bits and regress this test — that is the signal, not noise to
    /// loosen.
    #[test]
    fn run_batch_bit_identical_1_vs_n_threads() {
        let spec = two_factor_spec();
        let base_seed = 2137u64;
        let n_sims = 2000u32;
        let n_grid = [200u32, 500];
        let mt = run_batch(&spec, &n_grid, n_sims, base_seed, None).unwrap();
        let st = run_batch_st(&spec, &n_grid, n_sims, base_seed, None).unwrap();

        // Every BatchResult tensor is integer-valued, so equality IS bit-identity.
        assert_eq!(
            mt.uncorrected, st.uncorrected,
            "uncorrected significance diverged"
        );
        assert_eq!(
            mt.corrected, st.corrected,
            "corrected significance diverged"
        );
        assert_eq!(mt.posthoc_unc, st.posthoc_unc, "posthoc_unc diverged");
        assert_eq!(mt.posthoc_cor, st.posthoc_cor, "posthoc_cor diverged");
        assert_eq!(mt.converged, st.converged, "convergence diverged");
        assert_eq!(mt.boundary_hit, st.boundary_hit, "boundary_hit diverged");
        assert_eq!(mt.joint_unc, st.joint_unc, "joint_unc diverged");
        assert_eq!(mt.joint_cor, st.joint_cor, "joint_cor diverged");
        assert_eq!(mt.overall, st.overall, "overall diverged");
        assert_eq!(
            mt.factor_excluded, st.factor_excluded,
            "factor_excluded diverged"
        );
        assert_eq!(mt.shape.n_sims, st.shape.n_sims);
        assert_eq!(mt.shape.n_sample_sizes, st.shape.n_sample_sizes);
        assert_eq!(mt.shape.n_targets, st.shape.n_targets);
        assert_eq!(mt.shape.n_factors, st.shape.n_factors);
    }

    #[test]
    fn run_batch_power_increases_with_n() {
        let spec = minimal_spec();
        let result = run_batch(&spec, &[30, 200], 500, 42, None).unwrap();
        let stride = 1usize;
        let mut p30 = 0usize;
        let mut p200 = 0usize;
        for s in 0..500 {
            p30 += result.uncorrected[s * 2 * stride] as usize;
            p200 += result.uncorrected[s * 2 * stride + stride] as usize;
        }
        assert!(p200 > p30, "power(200) {p200} must exceed power(30) {p30}");
    }

    #[test]
    fn run_batch_validates_sample_sizes() {
        let spec = minimal_spec();
        // Non-increasing.
        let err = run_batch(&spec, &[100, 100], 10, 42, None).unwrap_err();
        assert!(matches!(err, EngineError::InvalidSpec(_)));
        // Empty.
        let err = run_batch(&spec, &[], 10, 42, None).unwrap_err();
        assert!(matches!(err, EngineError::InvalidSpec(_)));
    }

    #[test]
    fn run_batch_rejects_bad_target_index() {
        let mut spec = minimal_spec();
        spec.target_indices = vec![5];
        let err = run_batch(&spec, &[100], 10, 42, None).unwrap_err();
        assert!(matches!(err, EngineError::InvalidSpec(_)));
    }

    #[test]
    fn run_batch_rejects_correlation_length_mismatch() {
        // correlation.len() must equal n_non_factor². A 1-predictor spec
        // with a length-2 correlation vector is rejected as InvalidSpec.
        let mut spec = minimal_spec();
        spec.correlation = vec![1.0, 0.0];
        let err = run_batch(&spec, &[100], 10, 42, None).unwrap_err();
        assert!(
            matches!(err, EngineError::InvalidSpec(_)),
            "expected InvalidSpec for correlation length mismatch, got {err:?}"
        );
    }

    #[test]
    fn run_batch_rejects_non_psd_correlation() {
        // A symmetric-but-indefinite correlation matrix fails the trial Cholesky
        // and is rejected as CorrelationNotPSD. [[1,2],[2,1]] has an eigenvalue of −1.
        let mut spec = minimal_spec();
        spec.n_non_factor = 2;
        spec.correlation = vec![1.0, 2.0, 2.0, 1.0];
        spec.var_types = vec![Distribution::Normal, Distribution::Normal];
        spec.var_params = vec![0.0, 0.0];
        spec.effect_sizes = vec![0.0, 0.5, 0.5];
        spec.target_indices = vec![1, 2];
        let err = run_batch(&spec, &[100], 10, 42, None).unwrap_err();
        assert!(
            matches!(err, EngineError::CorrelationNotPSD),
            "expected CorrelationNotPSD for indefinite correlation, got {err:?}"
        );
    }

    #[test]
    fn run_batch_rejects_non_psd_correlation_in_non_optimistic_scenario() {
        // The up-front PSD check sits before scenario dispatch, so a non-PSD
        // INPUT matrix is rejected even for a non-optimistic scenario — whose
        // per-sim path would otherwise silently PSD-repair its own perturbed
        // copy. Uses the classic in-range-but-indefinite 3×3 matrix
        // [[1,.9,.9],[.9,1,-.9],[.9,-.9,1]] (det < 0): passes range/symmetry/
        // diagonal but fails the trial Cholesky.
        let mut spec = minimal_spec();
        spec.n_non_factor = 3;
        spec.correlation = vec![
            1.0, 0.9, 0.9, //
            0.9, 1.0, -0.9, //
            0.9, -0.9, 1.0,
        ];
        spec.var_types = vec![
            Distribution::Normal,
            Distribution::Normal,
            Distribution::Normal,
        ];
        spec.var_params = vec![0.0, 0.0, 0.0];
        spec.effect_sizes = vec![0.0, 0.5, 0.5, 0.5];
        spec.target_indices = vec![1, 2, 3];
        // A non-zero correlation_noise_sd flips is_optimistic() to false,
        // routing data_gen to perturb_correlation → psd_repair_and_factor.
        spec.scenario = ScenarioPerturbations {
            name: "doomer".into(),
            correlation_noise_sd: 0.1,
            ..ScenarioPerturbations::optimistic()
        };
        assert!(!spec.scenario.is_optimistic());
        let err = run_batch(&spec, &[100], 10, 42, None).unwrap_err();
        assert!(
            matches!(err, EngineError::CorrelationNotPSD),
            "expected CorrelationNotPSD before scenario dispatch, got {err:?}"
        );
    }

    struct CancelAtSecondCall {
        seen: std::sync::atomic::AtomicU32,
    }
    impl ProgressSink for CancelAtSecondCall {
        fn report(&self, _current: u64, _total: u64) -> bool {
            let prev = self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            prev < 1
        }
    }

    #[test]
    fn run_batch_cancel_propagates() {
        let spec = minimal_spec();
        let sink = CancelAtSecondCall {
            seen: std::sync::atomic::AtomicU32::new(0),
        };
        // progress_step = n_sims / 50 = 20 → reports fire every 20 sims
        // (~50 reports per run). Cancellation on the 2nd report should abort.
        let err = run_batch(&spec, &[100], 1000, 42, Some(&sink)).unwrap_err();
        assert!(matches!(err, EngineError::Cancelled));
    }

    /// `report` never cancels; only the per-sim `is_cancelled` poll does. Proves
    /// cancellation is decoupled from the ~50×/run report checkpoints, so a slow
    /// run does not wait `progress_step` fits before observing cancel.
    struct CancelViaPoll {
        polls: std::sync::atomic::AtomicU32,
    }
    impl ProgressSink for CancelViaPoll {
        fn report(&self, _current: u64, _total: u64) -> bool {
            true // never cancels via the checkpoint path
        }
        fn is_cancelled(&self) -> bool {
            self.polls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                >= 3
        }
    }

    #[test]
    fn run_batch_cancel_via_per_sim_poll() {
        let spec = minimal_spec();
        let sink = CancelViaPoll {
            polls: std::sync::atomic::AtomicU32::new(0),
        };
        let err = run_batch(&spec, &[100], 1000, 42, Some(&sink)).unwrap_err();
        assert!(matches!(err, EngineError::Cancelled));
    }

    /// Build a minimal Logit spec — intercept at column 0 = logit(0.3),
    /// β₁ = 0.5. Effects layout follows the OLS convention.
    fn minimal_logit_spec() -> SimulationSpec {
        let p = 0.3_f64;
        let intercept = (p / (1.0 - p)).ln();
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![intercept, 0.5],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Binary,
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            intercept,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    #[test]
    fn run_batch_accepts_logit_optimistic() {
        let spec = minimal_logit_spec();
        let result = run_batch(&spec, &[200], 10, 42, None).expect("Logit optimistic must run");
        assert_eq!(result.shape.n_sims, 10);
        assert_eq!(result.shape.n_sample_sizes, 1);
        assert_eq!(result.shape.n_targets, 1);
    }

    #[test]
    fn run_batch_accepts_logit_with_scenarios() {
        // Gate-removal regression: binary outcomes accept non-optimistic
        // scenarios (β-jitter applies as log-odds heterogeneity).
        let mut spec = minimal_logit_spec();
        spec.scenario = ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.1,
            ..Default::default()
        };
        let result = run_batch(&spec, &[200], 10, 42, None).expect("logit + scenarios must run");
        assert_eq!(result.shape.n_sims, 10);
        assert_eq!(result.shape.n_targets, 1);
    }

    #[test]
    fn run_batch_rejects_logit_with_posthoc() {
        let mut spec = minimal_logit_spec();
        spec.posthoc = vec![crate::spec::PosthocSpec {
            factor_index: 0,
            target_indices: vec![1],
        }];
        let err = run_batch(&spec, &[200], 10, 42, None).unwrap_err();
        assert!(matches!(err, EngineError::InvalidSpec(_)));
    }

    #[test]
    fn run_batch_logit_seed_reproducibility() {
        let p = 0.3_f64;
        let intercept = (p / (1.0 - p)).ln();
        let mut spec = minimal_logit_spec();
        spec.effect_sizes = vec![intercept, 0.5];
        spec.target_indices = vec![1];

        let r1 = run_batch(&spec, &[200], 50, 123, None).unwrap();
        let r2 = run_batch(&spec, &[200], 50, 123, None).unwrap();
        assert_eq!(r1.uncorrected, r2.uncorrected);
        assert_eq!(r1.corrected, r2.corrected);
        assert_eq!(r1.converged, r2.converged);
    }

    /// GLMM dispatch: a clustered logit (random intercept) routes Glm + cluster
    /// to `glmm::fit_glmm`, surfaces `n_variance_components == 1`, and writes a
    /// finite τ̂² for every converged fit.
    #[test]
    fn glm_cluster_batch_runs_and_surfaces_variance_components() {
        // y ~ x1 + (1 | g), logit, 12 clusters × 20 = 240, single random intercept.
        let mut spec = minimal_logit_spec();
        spec.cluster = Some(crate::spec::ClusterSpec::intercept_only(
            crate::spec::ClusterSizing::FixedClusters { n_clusters: 12 },
            0.20,
        ));
        let batch = run_batch(&spec, &[240], 64, 2137, None).unwrap();
        assert_eq!(batch.shape.n_variance_components, 1);
        let any_conv = batch.converged.iter().any(|&c| c != 0);
        assert!(any_conv, "some GLMM fits must converge");
        for i in 0..batch.converged.len() {
            if batch.converged[i] != 0 {
                assert!(
                    batch.tau_squared_hat[i].is_finite(),
                    "converged ⇒ finite τ̂²"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // LME tests
    // -----------------------------------------------------------------------

    /// Build a minimal LME spec: 1 continuous predictor (β₁ = 0.5),
    /// 5 clusters, τ² = 0.25.
    fn minimal_lme_spec() -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.5],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: Some(crate::spec::ClusterSpec {
                sizing: crate::spec::ClusterSizing::FixedClusters { n_clusters: 5 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            }),
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// Power is monotone non-decreasing in N.
    #[test]
    fn run_batch_lme_power_monotone_in_n() {
        let spec = minimal_lme_spec();
        // sample_sizes=[50, 100, 200]; n_sims=500; 1 target.
        let n_sims: u32 = 500;
        let sample_sizes = &[50u32, 100, 200];
        let result = run_batch(&spec, sample_sizes, n_sims, 42, None).unwrap();

        // layout: uncorrected[sim * n_sample_sizes * n_targets + n_idx * n_targets + t_idx]
        // n_targets=1, n_sample_sizes=3.
        let ns = n_sims as usize;
        let mut power = [0usize; 3];
        for s in 0..ns {
            for (n_idx, p) in power.iter_mut().enumerate() {
                *p += result.uncorrected[s * 3 + n_idx] as usize;
            }
        }

        assert!(
            power[1] >= power[0],
            "LME power not monotone: p50={}, p100={} (should have p100 >= p50)",
            power[0],
            power[1]
        );
        assert!(
            power[2] >= power[1],
            "LME power not monotone: p100={}, p200={} (should have p200 >= p100)",
            power[1],
            power[2]
        );
    }

    /// Seed reproducibility — same spec + same seed → identical buffers.
    #[test]
    fn run_batch_lme_seed_reproducibility() {
        let spec = minimal_lme_spec();
        let r1 = run_batch(&spec, &[50], 100, 77, None).unwrap();
        let r2 = run_batch(&spec, &[50], 100, 77, None).unwrap();
        assert_eq!(r1.uncorrected, r2.uncorrected, "uncorrected mismatch");
        assert_eq!(r1.corrected, r2.corrected, "corrected mismatch");
        assert_eq!(r1.converged, r2.converged, "converged mismatch");
        assert_eq!(r1.boundary_hit, r2.boundary_hit, "boundary_hit mismatch");
        assert_eq!(r1.joint_unc, r2.joint_unc, "joint_unc mismatch");
        assert_eq!(r1.joint_cor, r2.joint_cor, "joint_cor mismatch");
    }

    /// Joint Wald-χ² buffers (LME shape rules):
    ///   - joint_unc / joint_cor have length n_sims × n_sample_sizes.
    ///   - LME `joint_cor == joint_unc` (joint test is outside the correction
    ///     family), and every entry is a 0/1 flag.
    #[test]
    fn run_batch_lme_joint_buffers_populated() {
        let spec = minimal_lme_spec();
        let n_sims: u32 = 200;
        let result = run_batch(&spec, &[200], n_sims, 42, None).unwrap();
        let n = n_sims as usize;

        assert_eq!(result.joint_unc.len(), n, "joint_unc wrong length");
        assert_eq!(result.joint_cor.len(), n, "joint_cor wrong length");

        // LME: joint_cor must equal joint_unc bit-for-bit.
        assert_eq!(
            result.joint_unc, result.joint_cor,
            "joint_cor must equal joint_unc for LME"
        );
        // Each entry is a 0/1 significance flag.
        for &b in &result.joint_unc {
            assert!(b <= 1, "joint_unc entries must be 0/1 flags, got {b}");
        }
    }

    #[test]
    fn run_batch_ols_joint_buffers_all_zero() {
        // OLS spec — joint_unc / joint_cor must stay at the zero default.
        let spec = minimal_spec();
        let result = run_batch(&spec, &[100, 200], 50, 42, None).unwrap();
        for &b in &result.joint_unc {
            assert_eq!(b, 0, "OLS joint_unc must be all zero");
        }
        for &b in &result.joint_cor {
            assert_eq!(b, 0, "OLS joint_cor must be all zero");
        }
    }

    #[test]
    fn run_batch_overall_buffer_gated_on_flag() {
        let mut spec = minimal_spec();
        // Default flag is false → buffer should be empty.
        let r_off = run_batch(&spec, &[200], 20, 42, None).unwrap();
        assert!(
            r_off.overall.is_empty(),
            "overall must be empty when flag is off"
        );

        spec.report_overall = true;
        let r_on = run_batch(&spec, &[200], 20, 42, None).unwrap();
        assert_eq!(
            r_on.overall.len(),
            20,
            "overall buffer must be n_sims × n_sample_sizes = 20"
        );
        // Every entry is a 0/1 flag.
        for &b in &r_on.overall {
            assert!(b <= 1, "overall entries must be 0/1 flags, got {b}");
        }
    }

    #[test]
    fn run_batch_overall_lme_always_zero() {
        let mut spec = minimal_lme_spec();
        spec.report_overall = true;
        let r = run_batch(&spec, &[50], 30, 42, None).unwrap();
        assert_eq!(r.overall.len(), 30);
        for &b in &r.overall {
            assert_eq!(
                b, 0,
                "LME must emit 0 in overall regardless of report_overall"
            );
        }
    }

    #[test]
    fn run_batch_overall_zero_on_non_converged() {
        let mut spec = minimal_logit_spec();
        spec.report_overall = true;
        spec.max_failed_fraction = 1.0;
        // Drive separation so most fits fail.
        spec.effect_sizes = vec![0.0, 20.0]; // huge effect → separation at any N
        let r = run_batch(&spec, &[40], 30, 42, None).unwrap();
        for (i, &b) in r.overall.iter().enumerate() {
            if r.converged[i] == 0 {
                assert_eq!(b, 0, "non-converged sim {i} must have overall=0");
            }
        }
    }

    /// LME without ClusterSpec → InvalidSpec.
    #[test]
    fn run_batch_lme_rejects_no_cluster_spec() {
        let mut spec = minimal_lme_spec();
        spec.cluster = None;
        let err = run_batch(&spec, &[50], 10, 42, None).unwrap_err();
        assert!(
            matches!(err, EngineError::InvalidSpec(_)),
            "Expected InvalidSpec, got {err:?}"
        );
    }

    /// LME + posthoc → InvalidSpec (posthoc stays Ols-only). Scenarios on the
    /// Mle path are no longer rejected — see `run_batch_lme_accepts_scenarios`.
    #[test]
    fn run_batch_lme_rejects_posthoc() {
        let mut spec = minimal_lme_spec();
        spec.posthoc = vec![crate::spec::PosthocSpec {
            factor_index: 0,
            target_indices: vec![1],
        }];
        let err = run_batch(&spec, &[50], 10, 42, None).unwrap_err();
        assert!(
            matches!(err, EngineError::InvalidSpec(_)),
            "Expected InvalidSpec for LME + posthoc, got {err:?}"
        );
    }

    /// A non-optimistic scenario runs on the Mle path (gate removed): the
    /// generation-side RE/residual/correlation perturbations apply to the
    /// Gaussian LMM exactly as they do for OLS/GLMM. Just assert it completes —
    /// monotonicity/byte-identity are covered by the focused tests below.
    #[test]
    fn run_batch_lme_accepts_scenarios() {
        let mut spec = minimal_lme_spec();
        spec.scenario = ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.1,
            ..Default::default()
        };
        assert!(!spec.scenario.is_optimistic());
        let res = run_batch(&spec, &[50], 10, 42, None);
        assert!(
            res.is_ok(),
            "Mle + non-optimistic scenario must run, got {res:?}"
        );
    }

    /// Removing the gate must not perturb the optimistic Mle path: the default
    /// (optimistic) scenario only ever skipped the now-deleted early return, so
    /// its output is structurally untouched. Pin a representative power count so
    /// a future change that leaks scenario state into the optimistic path trips.
    #[test]
    fn run_batch_lme_optimistic_unchanged() {
        let spec = minimal_lme_spec();
        let res = run_batch(&spec, &[100], 200, 2137, None).unwrap();
        let power: usize = res.uncorrected.iter().map(|&b| b as usize).sum();
        // Captured pre-gate-removal (optimistic path is identical before/after).
        assert_eq!(power, 197, "optimistic Mle power count drifted: {power}");
    }

    /// `icc_noise_sd > 0` measurably perturbs Mle results vs the zero-knob run:
    /// the per-sim τ² jitter (data_gen D6) changes the random-intercept variance,
    /// so the converged power count moves. The base run takes the optimistic fast
    /// path (no scenario RNG); the jittered run enters the scenario path, but with
    /// every other knob at zero its correlation/var-type/residual draws are
    /// identity no-ops and the data stream (tag 0) matches the base run — so any
    /// difference can come only from the τ² jitter.
    #[test]
    fn run_batch_lme_icc_noise_perturbs() {
        let base = minimal_lme_spec();
        let base_res = run_batch(&base, &[80], 400, 99, None).unwrap();
        let base_power: usize = base_res.uncorrected.iter().map(|&b| b as usize).sum();

        let mut jittered = minimal_lme_spec();
        jittered.scenario = ScenarioPerturbations {
            name: "doomer".into(),
            lme: Some(crate::spec::LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 0.0,
                icc_noise_sd: 0.5, // large relative to base τ²=0.25 ⇒ visible shift
            }),
            ..Default::default()
        };
        assert!(!jittered.scenario.is_optimistic());
        let jit_res = run_batch(&jittered, &[80], 400, 99, None).unwrap();
        let jit_power: usize = jit_res.uncorrected.iter().map(|&b| b as usize).sum();

        assert_ne!(
            base_power, jit_power,
            "icc_noise_sd>0 must move Mle power vs zero-knob: base={base_power} jit={jit_power}"
        );
    }

    // -----------------------------------------------------------------------
    // Posthoc pairwise contrast tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_posthoc_contrasts_is_all_pairwise() {
        // 1 continuous (col 1) + a 3-level factor (dummies at cols 2,3); reference = level 0.
        // p = 1 (intercept) + 1 (n_non_factor) + 2 (n_factor_dummies) = 4.
        let mut spec = minimal_spec();
        spec.n_non_factor = 1;
        spec.n_factor_dummies = 2;
        spec.factor_n_levels = vec![3];
        spec.factor_proportions = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        spec.effect_sizes = vec![0.0, 0.0, 0.3, 0.5];
        spec.target_indices = vec![2, 3];
        spec.posthoc = vec![crate::spec::PosthocSpec {
            factor_index: 0,
            target_indices: vec![2, 3], // dummy cols in level order
        }];
        let cs = build_posthoc_contrasts(&spec).unwrap();
        assert_eq!(cs.len(), 3, "C(3,2) pairwise contrasts");
        assert_eq!(cs[0], vec![0.0, 0.0, 1.0, 0.0], "contrast (0,1): L1 vs ref");
        assert_eq!(cs[1], vec![0.0, 0.0, 0.0, 1.0], "contrast (0,2): L2 vs ref");
        assert_eq!(cs[2], vec![0.0, 0.0, -1.0, 1.0], "contrast (1,2): L2 vs L1");
    }

    #[test]
    fn build_posthoc_contrasts_empty_when_no_posthoc() {
        let spec = minimal_spec();
        let cs = build_posthoc_contrasts(&spec).unwrap();
        assert!(cs.is_empty());
    }

    // -----------------------------------------------------------------------
    // Posthoc Tukey-k crit test
    // -----------------------------------------------------------------------

    /// Build a minimal OLS spec with a 3-level factor, zero main targets
    /// (target_indices=[], contrast_pairs=[]), report_overall=true, and one
    /// 3-level posthoc block (→ 3 pairwise contrasts). This is the canonical
    /// ANOVA "overall + all-contrasts" request where n_targets == 0.
    ///
    /// Before the fix: panics at `chunks_exact_mut(0)`.
    /// After the fix: returns a valid BatchResult with empty main buffers and a
    /// populated 3-contrast posthoc block.
    fn zero_main_targets_posthoc_spec() -> SimulationSpec {
        // intercept + 1 continuous + 2 factor dummies = 4 predictors
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 2,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![3],
            factor_proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.0, 0.3, 0.5],
            target_indices: vec![], // ZERO main targets
            contrast_pairs: vec![], // ZERO contrasts
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![crate::spec::PosthocSpec {
                factor_index: 0,
                target_indices: vec![2, 3], // dummy cols for the 3-level factor
            }],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: true, // omnibus F still runs
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    #[test]
    fn run_batch_zero_main_targets_with_posthoc() {
        // Regression test: n_targets == 0 must not panic at chunks_exact_mut(0).
        // The result must have empty main buffers and a populated 3-contrast
        // posthoc block.
        let spec = zero_main_targets_posthoc_spec();
        let n_sims = 40u32;
        let sample_sizes = &[120u32];
        let result = run_batch(&spec, sample_sizes, n_sims, 42, None)
            .expect("n_targets==0 with posthoc must not panic");

        // Shape checks.
        assert_eq!(result.shape.n_sims, n_sims);
        assert_eq!(result.shape.n_sample_sizes, 1);
        assert_eq!(result.shape.n_targets, 0);
        assert_eq!(result.shape.posthoc_blocks.len(), 1);
        assert_eq!(result.shape.posthoc_blocks[0].n_contrasts, 3);

        // Main buffers must be empty.
        assert!(result.uncorrected.is_empty(), "uncorrected must be empty");
        assert!(result.corrected.is_empty(), "corrected must be empty");

        // Converged/boundary_hit still have n_sims entries.
        assert_eq!(result.converged.len(), n_sims as usize);
        assert_eq!(result.boundary_hit.len(), n_sims as usize);

        // Posthoc buffers: n_sims * 1_ss * 3_contrasts = 120 bytes each.
        let expected_posthoc_len = (n_sims as usize) * 3;
        assert_eq!(result.posthoc_unc.len(), expected_posthoc_len);
        assert_eq!(result.posthoc_cor.len(), expected_posthoc_len);
        // At n=120, a medium effect (0.3/0.5 dummies) should fire at least sometimes.
        let posthoc_fire: usize = result.posthoc_unc.iter().map(|&b| b as usize).sum();
        assert!(
            posthoc_fire > 0,
            "at least some posthoc contrasts should be significant (effect=0.3/0.5, n=120)"
        );

        // overall buffer: n_sims entries (report_overall=true).
        assert_eq!(result.overall.len(), n_sims as usize);
        // At n=120 with medium effects, F should fire.
        let overall_fire: usize = result.overall.iter().map(|&b| b as usize).sum();
        assert!(
            overall_fire > 0,
            "at least some overall F tests should be significant"
        );
    }

    #[test]
    fn run_batch_st_zero_main_targets_with_posthoc() {
        // Same test on the sequential path.
        let spec = zero_main_targets_posthoc_spec();
        let result = run_batch_st(&spec, &[120], 40, 42, None)
            .expect("run_batch_st n_targets==0 must not panic");
        assert_eq!(result.shape.n_targets, 0);
        assert!(result.uncorrected.is_empty());
        assert_eq!(result.posthoc_unc.len(), 40 * 3);
    }

    #[test]
    fn posthoc_tukey_crit_is_finite_and_between_unc_and_bonferroni() {
        let crit = CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        };
        let sample_sizes = [120u32];
        let n_pred = 4u32; // intercept + 1 cont + 2 dummies
        let factor_n_levels = [3i32];
        let tukey = build_posthoc_correction_crit_with_levels(
            &crit,
            &sample_sizes,
            n_pred,
            3,
            CorrectionMethod::TukeyHsd,
            EstimatorSpec::Ols,
            &factor_n_levels,
        )
        .unwrap();
        assert_eq!(tukey.len(), 1);
        assert_eq!(tukey[0].len(), 3);
        for v in &tukey[0] {
            assert!(
                v.is_finite() && *v > 0.0,
                "tukey posthoc crit must be finite & positive, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Exclusion helper tests
    // -----------------------------------------------------------------------

    /// Build a 2-factor spec: factor 0 is 2-level (1 dummy col at dummy_base),
    /// factor 1 is 3-level (2 dummy cols at dummy_base+1..+2), plus one
    /// interaction column `x1:factor0_dummy` at inter_base.
    ///
    /// Full column layout (p = 5):
    ///   col 0: intercept
    ///   col 1: continuous x1 (n_non_factor = 1)
    ///   col 2: factor0 dummy (dummy_base = 2, width = 1)
    ///   col 3: factor1 dummy[0] (level 1 vs ref)
    ///   col 4: factor1 dummy[1] (level 2 vs ref)
    ///   col 5: interaction x1 * col2 (inter_base = 5)
    fn two_factor_spec() -> SimulationSpec {
        // n_non_factor=1, n_factor_dummies=3, interactions=[vec![1,2]].
        // effect_sizes length = 1 + 1 + 3 + 1 = 6.
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 3, // factor0: 1 dummy + factor1: 2 dummies
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![2, 3], // factor0: 2 levels, factor1: 3 levels
            factor_proportions: vec![0.5, 0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            factor_sampled: Vec::new(),
            // effect_sizes: intercept, x1, f0_dummy, f1_dummy0, f1_dummy1, interaction
            effect_sizes: vec![0.0, 0.3, 0.2, 0.2, 0.3, 0.1],
            target_indices: vec![1, 2],
            contrast_pairs: vec![],
            // interaction: x1 (col 1) * factor0 dummy (col 2)
            interactions: vec![vec![1, 2]],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// `build_exclusion_remap` with factor 0 flagged (flag[0] = 1):
    ///   - dropped columns: col 2 (factor0 dummy) and col 5 (interaction x1*col2)
    ///   - kept columns: 0, 1, 3, 4 → p_red = 4
    ///   - col_remap[2] = -1, col_remap[5] = -1
    ///   - col_remap[1] = 1 (continuous target survives)
    #[test]
    fn build_exclusion_remap_drops_factor0_and_interaction() {
        let spec = two_factor_spec();
        // p = 1 + 1 + 3 + 1 = 6 columns total
        let p = 1
            + spec.n_non_factor as usize
            + spec.n_factor_dummies as usize
            + spec.interactions.len();
        let flags = [1u8, 0]; // factor0 excluded, factor1 kept
        let mut kept_cols: Vec<u32> = Vec::new();
        let mut col_remap = vec![-1i32; p];
        let p_red = build_exclusion_remap(&spec, &flags, &mut kept_cols, &mut col_remap);

        // intercept + x1 + 2 factor1 dummies = 4 kept columns
        assert_eq!(p_red, 4, "p_red must be 4 with factor0 dropped");
        assert_eq!(
            kept_cols,
            vec![0u32, 1, 3, 4],
            "kept cols: intercept, x1, f1 dummies"
        );

        // col_remap for kept columns: 0→0, 1→1, 3→2, 4→3
        assert_eq!(col_remap[0], 0, "intercept remap");
        assert_eq!(col_remap[1], 1, "x1 remap survives");
        // factor0 dummy and interaction are dropped
        assert_eq!(col_remap[2], -1, "factor0 dummy must be dropped");
        assert_eq!(col_remap[5], -1, "interaction x1:f0 must be dropped");
        // factor1 dummies get new positions
        assert_eq!(col_remap[3], 2, "f1 dummy0 remap");
        assert_eq!(col_remap[4], 3, "f1 dummy1 remap");
    }

    /// `fit_columns` empty ⇒ `build_exclusion_remap` is byte-identical to its
    /// factor-only behaviour (backward-compat invariant). Same fixture and
    /// expectations as `build_exclusion_remap_drops_factor0_and_interaction`.
    #[test]
    fn build_exclusion_remap_empty_fit_columns_is_factor_only() {
        let spec = two_factor_spec(); // fit_columns == []
        assert!(
            spec.fit_columns.is_empty(),
            "fixture must carry no test-design reduction"
        );
        let p = 1
            + spec.n_non_factor as usize
            + spec.n_factor_dummies as usize
            + spec.interactions.len();
        let flags = [1u8, 0]; // factor0 excluded
        let mut kept_cols: Vec<u32> = Vec::new();
        let mut col_remap = vec![-1i32; p];
        let p_red = build_exclusion_remap(&spec, &flags, &mut kept_cols, &mut col_remap);
        assert_eq!(p_red, 4);
        assert_eq!(kept_cols, vec![0u32, 1, 3, 4]);
        assert_eq!(col_remap, vec![0, 1, -1, 2, 3, -1]);
    }

    /// `fit_columns` can drop a *continuous* column — something factor flags
    /// cannot express. Keeping {intercept, x1} and dropping x2 (col 2) yields a
    /// 2-column reduced design. This is the `test_formula` keep-set entry point.
    #[test]
    fn build_exclusion_remap_honors_fit_columns_drops_continuous() {
        // build_exclusion_remap only reads n_non_factor / n_factor_dummies /
        // factor_n_levels / interactions / fit_columns; the rest of minimal_spec
        // is irrelevant to the column-keep logic exercised here.
        let mut spec = minimal_spec();
        spec.n_non_factor = 2; // cols: 0 intercept, 1 x1, 2 x2
        spec.fit_columns = vec![0, 1]; // keep intercept + x1, drop x2
        let p = 3;
        let flags: [u8; 0] = []; // no factors
        let mut kept_cols: Vec<u32> = Vec::new();
        let mut col_remap = vec![-1i32; p];
        let p_red = build_exclusion_remap(&spec, &flags, &mut kept_cols, &mut col_remap);
        assert_eq!(p_red, 2, "dropping x2 leaves intercept + x1");
        assert_eq!(kept_cols, vec![0u32, 1]);
        assert_eq!(col_remap, vec![0, 1, -1], "x2 (col 2) maps to -1");
    }

    /// Factor exclusion and `fit_columns` reduction COMPOSE by intersection:
    /// keep(col) = in_fit_columns(col) ∧ not_factor_excluded(col). Here factor0
    /// is flagged (drops its dummy col 2 + interaction col 5) AND fit_columns
    /// drops the continuous x1 (col 1), keeping only intercept + factor1's two
    /// dummies.
    #[test]
    fn build_exclusion_remap_composes_factor_and_fit_columns() {
        let mut spec = two_factor_spec();
        spec.fit_columns = vec![0, 3, 4]; // intercept + f1 dummies; drops x1
        let p = 1
            + spec.n_non_factor as usize
            + spec.n_factor_dummies as usize
            + spec.interactions.len();
        let flags = [1u8, 0]; // factor0 excluded
        let mut kept_cols: Vec<u32> = Vec::new();
        let mut col_remap = vec![-1i32; p];
        let p_red = build_exclusion_remap(&spec, &flags, &mut kept_cols, &mut col_remap);
        assert_eq!(
            p_red, 3,
            "intercept + 2 factor1 dummies survive the intersection"
        );
        assert_eq!(kept_cols, vec![0u32, 3, 4]);
        // col1 (x1) dropped by fit_columns; col2/col5 dropped by factor0 flag.
        assert_eq!(col_remap, vec![0, -1, -1, 1, 2, -1]);
    }

    // --- test_formula reduced-fit end-to-end regressions (one per estimator arm) ---
    // Shared DGP: y ~ 0.1·x1 + 0.8·x2 with corr(x1,x2)=0.7. The FULL fit recovers
    // x1's *partial* coefficient (0.1 → weak power); dropping x2 via fit_columns
    // recovers x1's *marginal* coefficient (0.1 + 0.7·0.8 = 0.66 → strong power).
    // With the trigger off, reduced == full (the bug); the assertions encode the
    // post-fix power jump. One test per arm because each refits differently.

    fn corr_two_continuous(estimator: EstimatorSpec, outcome: OutcomeKind) -> SimulationSpec {
        let mut s = minimal_spec();
        s.n_non_factor = 2;
        s.correlation = vec![1.0, 0.7, 0.7, 1.0]; // flat col-major 2×2, ρ=0.7
        s.var_types = vec![Distribution::Normal, Distribution::Normal];
        s.var_params = vec![0.0, 0.0];
        s.effect_sizes = vec![0.0, 0.1, 0.8]; // intercept, x1 (weak), x2 (strong)
        s.target_indices = vec![1]; // report x1 only
        s.estimator = estimator;
        s.outcome_kind = outcome;
        s
    }

    /// Power of the single reported target: 1 target × 1 sample size ⇒
    /// `uncorrected.len() == n_sims`, so power is the mean of the hit flags.
    fn power_single_target(spec: &SimulationSpec, n: u32, n_sims: u32, seed: u64) -> f64 {
        let r = run_batch(spec, &[n], n_sims, seed, None).unwrap();
        r.uncorrected.iter().map(|&v| v as usize).sum::<usize>() as f64 / n_sims as f64
    }

    #[test]
    fn test_formula_reduced_fit_ols_recovers_marginal_coefficient() {
        let full = corr_two_continuous(EstimatorSpec::Ols, OutcomeKind::Continuous);
        let mut reduced = corr_two_continuous(EstimatorSpec::Ols, OutcomeKind::Continuous);
        reduced.fit_columns = vec![0, 1]; // keep intercept + x1, drop x2 (col 2)

        let p_full = power_single_target(&full, 120, 1000, 2137);
        let p_red = power_single_target(&reduced, 120, 1000, 2137);

        assert!(
            p_full < 0.30,
            "full fit reports x1's weak partial power, got {p_full}"
        );
        assert!(
            p_red > 0.90,
            "reduced fit recovers x1's strong marginal power, got {p_red}"
        );
        assert!(
            p_red - p_full > 0.50,
            "reduced power must jump vs full: {p_red} vs {p_full}"
        );
    }

    #[test]
    fn test_formula_reduced_fit_glm_recovers_marginal_coefficient() {
        let full = corr_two_continuous(EstimatorSpec::Glm, OutcomeKind::Binary);
        let mut reduced = corr_two_continuous(EstimatorSpec::Glm, OutcomeKind::Binary);
        reduced.fit_columns = vec![0, 1]; // drop x2 (col 2)

        // Binary outcomes carry less information than Gaussian → larger N for a
        // clean partial-vs-marginal gap.
        let p_full = power_single_target(&full, 400, 1000, 2137);
        let p_red = power_single_target(&reduced, 400, 1000, 2137);

        assert!(
            p_full < 0.40,
            "full logit fit reports x1's weak partial power, got {p_full}"
        );
        assert!(
            p_red > 0.60,
            "reduced logit fit recovers x1's marginal power, got {p_red}"
        );
        assert!(
            p_red - p_full > 0.25,
            "reduced power must jump vs full: {p_red} vs {p_full}"
        );
    }

    #[test]
    fn test_formula_reduced_fit_lme_recovers_marginal_coefficient() {
        // Intercept-only cluster random effect on top of the shared DGP.
        let lme_cluster = || {
            Some(crate::spec::ClusterSpec {
                sizing: crate::spec::ClusterSizing::FixedClusters { n_clusters: 5 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            })
        };
        let mut full = corr_two_continuous(EstimatorSpec::Mle, OutcomeKind::Continuous);
        full.cluster = lme_cluster();
        let mut reduced = corr_two_continuous(EstimatorSpec::Mle, OutcomeKind::Continuous);
        reduced.cluster = lme_cluster();
        reduced.fit_columns = vec![0, 1]; // drop x2 (col 2)

        let p_full = power_single_target(&full, 200, 800, 2137);
        let p_red = power_single_target(&reduced, 200, 800, 2137);

        assert!(
            p_full < 0.35,
            "full LME fit reports x1's weak partial power, got {p_full}"
        );
        assert!(
            p_red > 0.80,
            "reduced LME fit recovers x1's marginal power, got {p_red}"
        );
        assert!(
            p_red - p_full > 0.40,
            "reduced power must jump vs full: {p_red} vs {p_full}"
        );
    }

    #[test]
    fn test_formula_reduced_fit_glmm_recovers_marginal_coefficient() {
        // Clustered logit (random intercept) on the shared DGP. The reduced fit
        // refits only x1 + (1 | g); x1 absorbs x2's marginal log-odds while the
        // random intercept is unchanged (Z/θ-truth stay full). Binary + GLMM
        // carry less information than the OLS twin → loose thresholds, like the
        // plain-GLM case.
        let glmm_cluster = || {
            Some(crate::spec::ClusterSpec {
                sizing: crate::spec::ClusterSizing::FixedClusters { n_clusters: 8 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            })
        };
        let mut full = corr_two_continuous(EstimatorSpec::Glm, OutcomeKind::Binary);
        full.cluster = glmm_cluster();
        let mut reduced = corr_two_continuous(EstimatorSpec::Glm, OutcomeKind::Binary);
        reduced.cluster = glmm_cluster();
        reduced.fit_columns = vec![0, 1]; // drop x2 (col 2)

        // GLMM fits are costly (joint [θ|β] BOBYQA + PIRLS); keep N·clusters
        // small. The partial/marginal gap (0.1 vs 0.66 log-odds) is wide enough
        // that 150 sims at N=120 separates them well outside MC noise.
        let p_full = power_single_target(&full, 120, 150, 2137);
        let p_red = power_single_target(&reduced, 120, 150, 2137);

        assert!(
            p_full < 0.40,
            "full GLMM fit reports x1's weak partial power, got {p_full}"
        );
        assert!(
            p_red > 0.60,
            "reduced GLMM fit recovers x1's marginal power, got {p_red}"
        );
        assert!(
            p_red - p_full > 0.25,
            "reduced power must jump vs full: {p_red} vs {p_full}"
        );
    }

    /// Hand-built x_full dummy block: 6 rows, 3 dummy columns (cols 2,3,4).
    /// factor0 has 1 dummy (col 2); factor1 has 2 dummies (cols 3,4).
    ///
    /// Row encoding:
    ///   row 0: factor0=level0 (ref), factor1=level0 (ref)   → col2=0,col3=0,col4=0
    ///   row 1: factor0=level1,       factor1=level0 (ref)   → col2=1,col3=0,col4=0
    ///   row 2: factor0=level0 (ref), factor1=level1         → col2=0,col3=1,col4=0
    ///   row 3: factor0=level1,       factor1=level2         → col2=1,col3=0,col4=1
    ///   row 4: factor0=level0 (ref), factor1=level0 (ref)   → col2=0,col3=0,col4=0
    ///   row 5: factor0=level1,       factor1=level1         → col2=1,col3=1,col4=0
    ///
    /// After 6 rows (from=0, to=6):
    ///   factor0: level0(ref)=3, level1=3 → min=3
    ///   factor1: level0(ref)=3, level1=2, level2=1 → min=1
    ///
    /// With min_count=2: factor0 not flagged (min=3≥2), factor1 not flagged (min=1<2 → flagged).
    /// With min_count=4: both flagged (min=3<4 and min=1<4).
    /// With min_count=0: never flagged.
    #[test]
    fn update_factor_exclusions_counts_and_boundary() {
        let n_nf = 1usize; // one non-factor continuous col
        let dummy_base = 1 + n_nf; // = 2
        let ncols = dummy_base + 3; // intercept + x1 + col2 + col3 + col4 = 5
        let nrows = 6usize;
        let factor_n_levels: Vec<i32> = vec![2, 3];

        // Build x: intercept=1, x1=anything, then dummy cols.
        let mut x = faer::Mat::<f32>::zeros(nrows, ncols);
        for i in 0..nrows {
            x[(i, 0)] = 1.0; // intercept
        }
        // factor0 dummies (col 2): rows 1,3,5 are level1.
        x[(1, 2)] = 1.0;
        x[(3, 2)] = 1.0;
        x[(5, 2)] = 1.0;
        // factor1 dummies (cols 3,4): rows 2,5 → level1 (col3=1); row 3 → level2 (col4=1).
        x[(2, 3)] = 1.0;
        x[(5, 3)] = 1.0;
        x[(3, 4)] = 1.0;

        // factor_proportions layout: [f0_l0, f0_l1, f1_l0, f1_l1, f1_l2]
        let mut prefix_counts = vec![0u32; 2 + 3];
        let mut flags = vec![0u8; 2];

        // --- min_count = 0 → feature disabled, never flagged ---
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            nrows,
            0,
            &mut prefix_counts,
            &mut flags,
        );
        assert_eq!(flags, [0, 0], "min_count=0 must never flag any factor");
        // Counts: f0: [3, 3], f1: [3, 2, 1]
        assert_eq!(prefix_counts[0], 3, "f0 level0 count");
        assert_eq!(prefix_counts[1], 3, "f0 level1 count");
        assert_eq!(prefix_counts[2], 3, "f1 level0 count");
        assert_eq!(prefix_counts[3], 2, "f1 level1 count");
        assert_eq!(prefix_counts[4], 1, "f1 level2 count");

        // --- min_count = 2 → factor1 flagged (min=1 < 2), factor0 not (min=3 ≥ 2) ---
        prefix_counts.fill(0);
        flags.fill(0);
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            nrows,
            2,
            &mut prefix_counts,
            &mut flags,
        );
        assert_eq!(
            flags[0], 0,
            "factor0 must not be flagged at min_count=2 (min=3)"
        );
        assert_eq!(
            flags[1], 1,
            "factor1 must be flagged at min_count=2 (min=1)"
        );

        // --- min_count = 4 over first 5 rows → both factors flagged (f0 min=2, f1 min=1) ---
        prefix_counts.fill(0);
        flags.fill(0);
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            5, // first 5 rows only
            4,
            &mut prefix_counts,
            &mut flags,
        );
        // f0: level0 rows {0,2,4}=3, level1 rows {1,3}=2 → min=2 < 4 → flagged
        assert_eq!(
            flags[0], 1,
            "factor0 must be flagged at min_count=4 (min=2 in 5 rows)"
        );
        // f1: level0 rows {0,1,4}=3, level1 rows {2}=1, level2 rows {3}=1 → min=1 < 4 → flagged
        assert_eq!(
            flags[1], 1,
            "factor1 must be flagged at min_count=4 (min=1 in 5 rows)"
        );

        // Boundary: exactly 5 obs per level, min_count=5 → not flagged.
        // Build a 10-row matrix: 5 rows factor0=ref, 5 rows factor0=level1.
        let ncols2 = dummy_base + 1; // only factor0 (1 dummy)
        let factor_n_levels2: Vec<i32> = vec![2];
        let mut x2 = faer::Mat::<f32>::zeros(10, ncols2);
        for i in 0..10 {
            x2[(i, 0)] = 1.0;
            if i >= 5 {
                x2[(i, 2)] = 1.0; // level1
            }
        }
        let mut pc2 = vec![0u32; 2];
        let mut fl2 = vec![0u8; 1];
        update_factor_exclusions(
            x2.as_ref(),
            &factor_n_levels2,
            dummy_base,
            0,
            10,
            5, // min=5, counts=[5,5] → not flagged
            &mut pc2,
            &mut fl2,
        );
        assert_eq!(fl2[0], 0, "exactly min_count observations → not flagged");

        // One fewer: 4 on one side, 6 on the other, min_count=5 → flagged.
        let mut pc3 = vec![0u32; 2];
        let mut fl3 = vec![0u8; 1];
        // Adjust x2 so rows 0..4 are ref, rows 4..10 are level1 (4 ref, 6 level1).
        let mut x3 = faer::Mat::<f32>::zeros(10, ncols2);
        for i in 0..10 {
            x3[(i, 0)] = 1.0;
            if i >= 4 {
                x3[(i, 2)] = 1.0;
            }
        }
        update_factor_exclusions(
            x3.as_ref(),
            &factor_n_levels2,
            dummy_base,
            0,
            10,
            5,
            &mut pc3,
            &mut fl3,
        );
        assert_eq!(fl3[0], 1, "min=4 < min_count=5 must be flagged");
    }

    /// OLS calls update_factor_exclusions incrementally (from=last..n_usize).
    /// This test pins that two incremental calls produce identical counts and
    /// flags as one full scan over the same rows.
    #[test]
    fn update_factor_exclusions_incremental_matches_full_scan() {
        // Reuse the 6-row dummy fixture from update_factor_exclusions_counts_and_boundary.
        let n_nf = 1usize;
        let dummy_base = 1 + n_nf; // = 2
        let ncols = dummy_base + 3;
        let nrows = 6usize;
        let factor_n_levels: Vec<i32> = vec![2, 3];

        let mut x = faer::Mat::<f32>::zeros(nrows, ncols);
        for i in 0..nrows {
            x[(i, 0)] = 1.0;
        }
        x[(1, 2)] = 1.0;
        x[(3, 2)] = 1.0;
        x[(5, 2)] = 1.0;
        x[(2, 3)] = 1.0;
        x[(5, 3)] = 1.0;
        x[(3, 4)] = 1.0;

        let min_count = 2u32;

        // Full scan: from=0, to=6.
        let mut full_counts = vec![0u32; 2 + 3];
        let mut full_flags = vec![0u8; 2];
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            6,
            min_count,
            &mut full_counts,
            &mut full_flags,
        );

        // Incremental: two calls covering [0..3) then [3..6).
        let mut inc_counts = vec![0u32; 2 + 3];
        let mut inc_flags = vec![0u8; 2];
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            3,
            min_count,
            &mut inc_counts,
            &mut inc_flags,
        );
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            3,
            6,
            min_count,
            &mut inc_counts,
            &mut inc_flags,
        );

        assert_eq!(
            inc_counts, full_counts,
            "incremental counts must match full scan"
        );
        assert_eq!(
            inc_flags, full_flags,
            "incremental flags must match full scan"
        );
    }

    // -----------------------------------------------------------------------
    // Step 6.7 — OLS reduced-model tests
    // -----------------------------------------------------------------------

    /// Build an OLS spec: y ~ x1 + g, g 2-level, with the given continuous-
    /// predictor beta, factor-dummy beta, and factor proportions.
    /// Column layout: [0=intercept, 1=x1, 2=g_dummy].
    /// effect_sizes = [intercept=0, x1_beta, g_beta].
    /// target_indices = [1, 2] (test both x1 and g[2]).
    fn factor_spec(
        x1_beta: f64,
        g_beta: f64,
        proportions: Vec<f64>,
        factor_min_level_count: u32,
    ) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 1, // 2-level factor: 1 dummy
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![2],
            factor_proportions: proportions, // [p_ref, p_level1]
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, x1_beta, g_beta],
            target_indices: vec![1, 2], // x1 and g[2]
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            // Overall-F on, so the healthy-path parity test also pins the
            // overall buffer (dfn/dfd switch to p_red under exclusion).
            report_overall: true,
            factor_min_level_count,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// Sparse-excluded factor: proportions [0.95, 0.05] at N=40 → minority level
    /// gets ~2 rows (< 5) → excluded in every sim. Target index 1 (g[2]) is the
    /// factor dummy target; it must never be significant and factor_excluded must
    /// be 1. Target index 0 (x1) is the continuous predictor; it can still pass.
    #[test]
    fn sparse_factor_is_excluded_and_targets_report_zero() {
        let spec = factor_spec(0.5, 0.8, vec![0.95, 0.05], 5);
        let n_sims = 200u32;
        let result = run_batch(&spec, &[40], n_sims, 42, None).unwrap();

        // Factor excluded in all sims (code 1 = sparse exclusion).
        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "all sims must have factor_excluded=1 at N=40 with proportions [0.95,0.05]"
        );
        // Every sim converged (reduced model drops the dummy and fits fine).
        assert!(
            result.converged.iter().all(|&c| c == 1),
            "all sims must converge on the reduced model"
        );
        // g[2] target (slot 1) never significant; x1 (slot 0) often significant.
        let mut x1_hits = 0usize;
        for sim in 0..n_sims as usize {
            assert_eq!(
                result.uncorrected[sim * 2 + 1],
                0,
                "sim {sim}: excluded g[2] target must never be significant"
            );
            x1_hits += result.uncorrected[sim * 2] as usize;
        }
        assert!(
            x1_hits > 0,
            "x1 (beta=0.5) at N=40 must achieve some power across {n_sims} sims"
        );
    }

    /// Healthy factor: proportions [0.5, 0.5] at N=40 → counts [20, 20] — no
    /// exclusion. The BatchResult must be bit-identical to the same spec with
    /// factor_min_level_count=0 (hot path untouched — verifies the healthy path
    /// is bit-for-bit identical after the exclusion branch was added).
    #[test]
    fn healthy_factor_is_never_excluded_and_results_are_unchanged() {
        let n_sims = 100u32;
        let seed = 99u64;
        let sample_sizes = &[40u32];
        // Run with min_level_count active (but proportions guarantee no exclusion).
        let spec_with_guard = factor_spec(0.5, 0.8, vec![0.5, 0.5], 5);
        // Run with min_level_count disabled (classic hot path, no exclusion logic).
        let spec_no_guard = factor_spec(0.5, 0.8, vec![0.5, 0.5], 0);

        let r_guard = run_batch(&spec_with_guard, sample_sizes, n_sims, seed, None).unwrap();
        let r_no = run_batch(&spec_no_guard, sample_sizes, n_sims, seed, None).unwrap();

        // factor_excluded must be all 0 — neither run excluded the factor.
        assert!(
            r_guard.factor_excluded.iter().all(|&c| c == 0),
            "healthy factor must never produce exclusion code 1"
        );
        // Both runs must be bit-identical on the significance buffers.
        assert_eq!(
            r_guard.uncorrected, r_no.uncorrected,
            "healthy path: uncorrected must be bit-identical with/without guard"
        );
        assert_eq!(
            r_guard.corrected, r_no.corrected,
            "healthy path: corrected must be bit-identical with/without guard"
        );
        assert_eq!(
            r_guard.converged, r_no.converged,
            "healthy path: converged must be bit-identical with/without guard"
        );
        assert_eq!(
            r_guard.overall, r_no.overall,
            "healthy path: overall-F must be bit-identical with/without guard"
        );
    }

    /// Grid test: proportions [0.9, 0.1] at N=20 (count 2 < 5 → excluded) and
    /// N=100 (count 10 ≥ 5 → included). Exclusion codes must reflect per-N
    /// behaviour; at N=100 the g[2] target achieves some power.
    #[test]
    fn exclusion_is_per_n_on_a_grid() {
        let spec = factor_spec(0.3, 0.8, vec![0.9, 0.1], 5);
        let n_sims = 200u32;
        // sample_sizes stride = 2: [n_idx=0 → N=20, n_idx=1 → N=100].
        let result = run_batch(&spec, &[20, 100], n_sims, 42, None).unwrap();

        // n_factors=1, n_sample_sizes=2 → factor_excluded layout:
        // (n_sims × n_sample_sizes × n_factors) sim-major.
        // factor_excl_stride = n_sample_sizes * n_factors = 2.
        let mut all_excluded_at_n20 = true;
        let mut all_included_at_n100 = true;
        for sim in 0..n_sims as usize {
            // sim-major: factor_excluded[sim*2 + n_idx].
            if result.factor_excluded[sim * 2] != 1 {
                all_excluded_at_n20 = false;
            }
            if result.factor_excluded[sim * 2 + 1] != 0 {
                all_included_at_n100 = false;
            }
        }
        assert!(
            all_excluded_at_n20,
            "N=20 with proportions [0.9,0.1]: factor must be excluded in all sims"
        );
        assert!(
            all_included_at_n100,
            "N=100 with proportions [0.9,0.1]: factor must be included in all sims"
        );

        // At N=100 the factor is included: g[2] with beta=0.8 must have some hits.
        // main_row = 2 (2 targets), layout: uncorrected[sim * 2*n_samples + n_idx*2 + target].
        // n_sample_sizes=2, n_targets=2 → sim_main_stride=4.
        // slot for (n_idx=1, target=1) = n_idx*2+1 = 3 within the sim row.
        let mut g_hits_n100 = 0usize;
        for sim in 0..n_sims as usize {
            g_hits_n100 += result.uncorrected[sim * 4 + 3] as usize;
        }
        assert!(
            g_hits_n100 > 0,
            "g[2] target at N=100 with beta=0.8 must achieve nonzero power ({g_hits_n100} hits)"
        );
    }

    /// A factor where every row is at the reference level (all dummies zero)
    /// has a non-reference count of 0, which is below any min_count > 0.
    /// The reference-level count lands in slot 0; flag == 1.
    #[test]
    fn update_factor_exclusions_flags_factor_with_absent_level() {
        let n_nf = 0usize;
        let dummy_base = 1 + n_nf; // = 1
        let ncols = dummy_base + 1; // intercept + 1 dummy
        let nrows = 5usize;
        let factor_n_levels: Vec<i32> = vec![2];

        // All rows at reference (dummy stays 0).
        let mut x = faer::Mat::<f32>::zeros(nrows, ncols);
        for i in 0..nrows {
            x[(i, 0)] = 1.0;
        }

        let mut counts = vec![0u32; 2];
        let mut flags = vec![0u8; 1];
        update_factor_exclusions(
            x.as_ref(),
            &factor_n_levels,
            dummy_base,
            0,
            nrows,
            1,
            &mut counts,
            &mut flags,
        );

        // All 5 rows counted in the reference slot; non-reference slot = 0.
        assert_eq!(counts[0], 5, "reference slot must hold all rows");
        assert_eq!(counts[1], 0, "non-reference level is absent");
        assert_eq!(
            flags[0], 1,
            "absent non-reference level must flag the factor"
        );
    }

    /// 3-level factor with proportions [0.90, 0.05, 0.05] at N=40: both minority
    /// levels get floor(0.05·40)=2 rows, which is below factor_min_level_count=5,
    /// so the factor is excluded in every sim. All 3 pairwise posthoc contrasts
    /// must be zero in every sim (se_sq guard at posthoc.rs:97 zeros dropped
    /// columns); the continuous x1 predictor must still achieve hits (sanity that
    /// the reduced fit ran).
    ///
    /// Column layout (p=4): [0=intercept, 1=x1, 2=f_dummy0, 3=f_dummy1].
    /// posthoc target_indices=[2,3] are the dummy column positions; the C(3,2)=3
    /// contrasts land at posthoc_unc[sim*3..sim*3+3] (1 sample size, 3 contrasts).
    #[test]
    fn excluded_factor_posthoc_contrasts_report_zero() {
        // proportions: [ref=0.90, level1=0.05, level2=0.05]
        // At N=40: level1 count = floor(0.05·40) = 2 < 5 → excluded every sim.
        let spec = SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 2, // 3-level factor: 2 dummies
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![3],
            factor_proportions: vec![0.90, 0.05, 0.05],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.5, 0.3, 0.3], // intercept, x1, f_dummy0, f_dummy1
            target_indices: vec![1],                // x1 only — sanity check
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![crate::spec::PosthocSpec {
                factor_index: 0,
                target_indices: vec![2, 3], // dummy column indices for the 3-level factor
            }],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 5,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        };

        let n_sims = 150u32;
        let result = run_batch(&spec, &[40], n_sims, 42, None)
            .expect("excluded-factor posthoc spec must not error");

        // Every sim must converge (reduced model drops both dummies and fits fine).
        assert!(
            result.converged.iter().all(|&c| c == 1),
            "all sims must converge on the reduced model"
        );

        // Factor excluded in all sims (code 1 = sparse exclusion).
        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "all sims must have factor_excluded=1 at N=40 with proportions [0.90,0.05,0.05]"
        );

        // All posthoc contrasts must be zero across every sim.
        // Layout: posthoc_unc[sim * 3 + contrast_idx] (1 sample size, 3 contrasts).
        assert_eq!(
            result.shape.posthoc_blocks.len(),
            1,
            "must have exactly one posthoc block"
        );
        assert_eq!(
            result.shape.posthoc_blocks[0].n_contrasts, 3,
            "3-level factor produces C(3,2)=3 pairwise contrasts"
        );
        assert_eq!(
            result.posthoc_unc.len(),
            n_sims as usize * 3,
            "posthoc_unc must be n_sims * 3 bytes"
        );
        assert!(
            result.posthoc_unc.iter().all(|&b| b == 0),
            "all posthoc contrasts must be zero when factor is excluded in every sim"
        );
        assert!(
            result.posthoc_cor.iter().all(|&b| b == 0),
            "corrected posthoc contrasts must also be zero"
        );

        // x1 (target slot 0) must achieve some hits — sanity that reduced fit ran.
        let x1_hits: usize = result.uncorrected.iter().map(|&b| b as usize).sum();
        assert!(
            x1_hits > 0,
            "x1 (beta=0.5) at N=40 must achieve some power across {n_sims} sims"
        );
    }

    // -----------------------------------------------------------------------
    // Step 7.3 — GLM reduced-model + separation-fallback tests
    // -----------------------------------------------------------------------

    /// Build a logit spec: y ~ x1 + g (binary outcome), g 2-level, with the
    /// given continuous-predictor beta, factor-dummy beta, factor proportions,
    /// and min_level_count. Column layout: [0=intercept, 1=x1, 2=g_dummy].
    fn logit_factor_spec(
        x1_beta: f64,
        g_beta: f64,
        intercept: f64,
        proportions: Vec<f64>,
        factor_min_level_count: u32,
    ) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 1, // 2-level factor: 1 dummy
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![2],
            factor_proportions: proportions, // [p_ref, p_level1]
            factor_sampled: Vec::new(),
            // intercept is effect_sizes[0] (column 0 = 1.0 for all rows);
            // also stored in spec.intercept for display / LRT reference.
            effect_sizes: vec![intercept, x1_beta, g_beta],
            target_indices: vec![1, 2], // x1 and g[2]
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Binary,
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            intercept,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// GLM sparse-factor exclusion: logit y ~ x1 + g, g 2-level with
    /// proportions [0.95, 0.05] at N=60 → minority count ≈ 3 < 5 → excluded
    /// in every sim. The GLM arm must fit the reduced model (drop g_dummy),
    /// so: all sims converge, g[2] target power = 0, x1 has nonzero power.
    #[test]
    fn glm_sparse_factor_excluded_like_ols() {
        // intercept = logit(0.3) gives a reasonable baseline probability.
        let p = 0.3_f64;
        let intercept = (p / (1.0 - p)).ln();
        // proportions [0.95, 0.05]: at N=60, minority count ≈ 3 < 5 → excluded every sim.
        let spec = logit_factor_spec(0.5, 2.0, intercept, vec![0.95, 0.05], 5);
        let n_sims = 200u32;
        let result = run_batch(&spec, &[60], n_sims, 42, None).unwrap();

        // Factor excluded in all sims (code 1 = sparse exclusion).
        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "all sims must have factor_excluded=1 at N=60 with GLM proportions [0.95,0.05]"
        );
        // Every sim must converge (reduced model drops the dummy and fits fine).
        assert!(
            result.converged.iter().all(|&c| c == 1),
            "all GLM sims must converge on the sparse-reduced model"
        );
        // g[2] target (slot 1) never significant; x1 (slot 0) often significant.
        // Layout: uncorrected[sim * n_sample_sizes * n_targets + n_idx * n_targets + target_slot]
        // = uncorrected[sim * 2 + target_slot] (1 sample size, 2 targets).
        let mut x1_hits = 0usize;
        for sim in 0..n_sims as usize {
            assert_eq!(
                result.uncorrected[sim * 2 + 1],
                0,
                "sim {sim}: excluded g[2] target must never be significant in GLM"
            );
            x1_hits += result.uncorrected[sim * 2] as usize;
        }
        assert!(
            x1_hits > 0,
            "x1 (beta=0.5) at N=60 must achieve some GLM power across {n_sims} sims"
        );
    }

    /// GLM separation fallback: factor [0.3, 0.7] at N=100 → majority count=30,
    /// minority count=70 (neither is sparse-excluded with min_count=5). Extreme
    /// baseline p≈0.02 + huge factor effect (β_g=100) → minority cell is all-1
    /// → near-perfect separation (majority is nearly all-0, so ~55% of sims have
    /// exact separation) → IRLS hits BETA_CAP (|β|>30) in many sims.
    /// Fallback sets code 2 for those sims, rescuing convergence.
    ///
    /// Parameters chosen to guarantee ≥5% code-2 rate: P(exact separation per sim)
    /// ≈ P(all 30 majority rows are y=0) = 0.98^30 ≈ 0.55; even imperfect-separation
    /// sims often hit BETA_CAP as β_g grows past 30 before deviance stabilises.
    ///
    /// Asserts: (a) some sims carry code 2; (b) convergence count with
    /// min_count=5 (fallback active) is HIGHER than min_count=0 (no fallback);
    /// (c) sims with code 2 still report x1 significance hits (reduced model ran).
    #[test]
    fn glm_separation_fallback_rescues_sims() {
        // logit(0.02) ≈ -3.892 — extreme baseline, majority cell nearly all-0.
        // β_g = 100: minority P(y=1) ≈ 1.0 → all-1 minority cell in almost every sim.
        // proportions [0.3, 0.7]: majority count=30 ≥ 5, minority count=70 ≥ 5 → neither excluded.
        let p_base = 0.02_f64;
        let intercept = (p_base / (1.0 - p_base)).ln();
        let n_sims = 400u32;
        let seed = 2137u64;
        let sample_sizes = &[100u32];

        // min_count=5: separation fallback active (feature enabled).
        let spec_fallback = logit_factor_spec(0.3, 100.0, intercept, vec![0.3, 0.7], 5);
        // min_count=0: no fallback (today's behaviour — feature disabled).
        let spec_no_fallback = logit_factor_spec(0.3, 100.0, intercept, vec![0.3, 0.7], 0);

        let r_fallback = run_batch(&spec_fallback, sample_sizes, n_sims, seed, None).unwrap();
        let r_no_fallback = run_batch(&spec_no_fallback, sample_sizes, n_sims, seed, None).unwrap();

        // (a) Some sims must have code 2 (separation fallback fired).
        let code2_count = r_fallback
            .factor_excluded
            .iter()
            .filter(|&&c| c == 2)
            .count();
        assert!(
            code2_count > (n_sims as usize / 20), // ≥ 5 % of sims
            "separation fallback must fire in at least 5% of sims; got {code2_count}/{n_sims}"
        );

        // (b) Fallback rescues convergence: more converged sims with fallback than without.
        let conv_fallback: usize = r_fallback.converged.iter().map(|&c| c as usize).sum();
        let conv_no_fallback: usize = r_no_fallback.converged.iter().map(|&c| c as usize).sum();
        assert!(
            conv_fallback > conv_no_fallback,
            "fallback must rescue more sims: fallback={conv_fallback} no_fallback={conv_no_fallback}"
        );

        // (c) Among code-2 sims, x1 should still achieve significance hits
        // (the reduced model — without the factor — ran and detected x1 effect).
        // Layout: uncorrected[sim * n_targets + target_slot], n_targets=2 (x1 at slot 0).
        let mut x1_hits_in_code2 = 0usize;
        for sim in 0..n_sims as usize {
            if r_fallback.factor_excluded[sim] == 2 {
                x1_hits_in_code2 += r_fallback.uncorrected[sim * 2] as usize;
            }
        }
        assert!(
            x1_hits_in_code2 > 0,
            "x1 must achieve some hits among {code2_count} code-2 sims (reduced model ran)"
        );
    }

    // -----------------------------------------------------------------------
    // df-correctness tests for the overall crit under sparse-factor exclusion
    // -----------------------------------------------------------------------

    /// Build a 5-level OLS factor spec: y ~ x1 + g(5 levels), x1 β=0 (null),
    /// factor always excluded. Column layout (p=6):
    /// [0=intercept, 1=x1, 2=g_dummy0, 3=g_dummy1, 4=g_dummy2, 5=g_dummy3].
    /// `factor_min_level_count` typically = 5.
    fn factor_5level_spec(
        x1_beta: f64,
        proportions: Vec<f64>,
        factor_min_level_count: u32,
    ) -> SimulationSpec {
        // 5-level factor → 4 dummies; effect_sizes len = 1 + 1 + 4 = 6.
        let n_dummies = 4u32;
        let mut effect_sizes = vec![0.0; 1 + 1 + n_dummies as usize];
        effect_sizes[1] = x1_beta; // x1
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: n_dummies,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![5],
            factor_proportions: proportions,
            factor_sampled: Vec::new(),
            effect_sizes,
            target_indices: vec![1], // x1 only (null β=0 for FPR test)
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: true,
            factor_min_level_count,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// Build a 5-level GLM (logit) factor spec: y ~ x1 + g(5 levels), x1 β=0
    /// (null), factor always excluded. intercept fixes baseline probability.
    fn logit_5level_spec(
        x1_beta: f64,
        intercept: f64,
        proportions: Vec<f64>,
        factor_min_level_count: u32,
    ) -> SimulationSpec {
        let n_dummies = 4u32;
        let mut effect_sizes = vec![0.0; 1 + 1 + n_dummies as usize];
        effect_sizes[0] = intercept;
        effect_sizes[1] = x1_beta;
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: n_dummies,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![5],
            factor_proportions: proportions,
            factor_sampled: Vec::new(),
            effect_sizes,
            target_indices: vec![1], // x1 only (null for FPR test)
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Binary,
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            intercept,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: true,
            factor_min_level_count,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// OLS overall F under exclusion uses reduced dfs, not full-P dfs.
    ///
    /// 5-level factor (4 dummies) → df gap: full-P crit = F(5, n-6) ≈ 2.49,
    /// correct crit = F(1, n-2) ≈ 4.10 at N=40. With the bug the null FPR would
    /// be ≈ 0.12 (anti-conservative); with the fix it should sit near α=0.05.
    /// Factor always excluded (proportions [0.92, 0.02, 0.02, 0.02, 0.02], N=40,
    /// min_count=5) → reduced model = intercept + x1, x1 β=0 → null FPR ≈ 0.05.
    #[test]
    fn ols_overall_under_exclusion_uses_reduced_df() {
        // Proportions: reference 0.92, four rare levels each at 0.02.
        // At N=40 each rare level gets floor(0.02*40)=0 or 1 row < 5 → excluded every sim.
        let spec = factor_5level_spec(0.0, vec![0.92, 0.02, 0.02, 0.02, 0.02], 5);
        let n_sims = 800u32;
        let result = run_batch(&spec, &[40], n_sims, 1337, None).unwrap();

        // Sanity: factor excluded in every sim.
        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "factor must be excluded in all sims for the df-correctness test to be valid"
        );

        // overall layout: n_sims × n_sample_sizes = n_sims × 1 (sim-major).
        let hits: usize = result.overall.iter().map(|&b| b as usize).sum();
        let fpr = hits as f64 / n_sims as f64;
        assert!(
            (0.02..=0.08).contains(&fpr),
            "OLS overall null FPR under exclusion must be near α=0.05, got {fpr:.4} ({hits}/{n_sims})"
        );
    }

    /// GLM (logit) overall LRT under exclusion uses reduced dfs, not full-P dfs.
    ///
    /// 5-level factor (4 dummies) → df gap: bug crit = χ²(5)=11.07 → FPR ≈ 0.001;
    /// correct crit = χ²(1)=3.84 → FPR ≈ 0.05.
    /// Factor always excluded (proportions [0.92, 0.02, 0.02, 0.02, 0.02], N=60,
    /// min_count=5) → reduced model = intercept + x1, x1 β=0 → null FPR ≈ 0.05.
    #[test]
    fn glm_overall_under_exclusion_uses_reduced_df() {
        // Baseline probability 0.5 (intercept=0) — well-behaved logit, no separation risk.
        let spec = logit_5level_spec(0.0, 0.0, vec![0.92, 0.02, 0.02, 0.02, 0.02], 5);
        let n_sims = 400u32;
        let result = run_batch(&spec, &[60], n_sims, 2137, None).unwrap();

        // Sanity: factor excluded in every sim.
        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "factor must be excluded in all sims for the df-correctness test to be valid"
        );

        // overall layout: n_sims × n_sample_sizes = n_sims × 1 (sim-major).
        let hits: usize = result.overall.iter().map(|&b| b as usize).sum();
        let fpr = hits as f64 / n_sims as f64;
        assert!(
            (0.02..=0.10).contains(&fpr),
            "GLM overall null FPR under exclusion must be near α=0.05, got {fpr:.4} ({hits}/{n_sims})"
        );
    }

    // -----------------------------------------------------------------------
    // Step 8.2 — LME (Mle) reduced-model exclusion test
    // -----------------------------------------------------------------------

    /// Build a clustered LME spec with one continuous predictor (x1, col 1) and
    /// a 2-level within-cluster factor (g, col 2). Column layout (p=3):
    ///   col 0: intercept
    ///   col 1: x1 (continuous, β=x1_beta)
    ///   col 2: g dummy (2-level factor, β=g_beta)
    /// Cluster: FixedClusters { n_clusters: 5 }, τ²=0.25.
    /// effect_sizes = [0.0, x1_beta, g_beta].
    /// target_indices = [1, 2] (both x1 and g[2]).
    fn lme_factor_spec(
        x1_beta: f64,
        g_beta: f64,
        proportions: Vec<f64>,
        factor_min_level_count: u32,
    ) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 1, // 2-level factor: 1 dummy
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![2],
            factor_proportions: proportions, // [p_ref, p_level1]
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, x1_beta, g_beta],
            target_indices: vec![1, 2], // x1 and g[2]
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: Some(crate::spec::ClusterSpec {
                sizing: crate::spec::ClusterSizing::FixedClusters { n_clusters: 5 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            }),
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// LME sparse-factor exclusion: clustered LME y ~ x1 + g, g 2-level with
    /// proportions [0.95, 0.05] at N=60 → minority count ≈ 3 < 5 → excluded in
    /// every sim. Asserts:
    ///   (a) all sims have factor_excluded=1 (sparse-exclusion code)
    ///   (b) g[2] target (slot 1) unc power == 0 in every sim
    ///   (c) x1 (slot 0) achieves nonzero power (reduced model ran and converged)
    ///   (d) convergence count with min_count=5 (reduced fit) is ≥ convergence
    ///       count with min_count=0 (full fit with sparse factor included) — the
    ///       reduced model is at least as stable as the full model
    #[test]
    fn lme_sparse_factor_excluded_and_x1_converges() {
        // proportions [0.95, 0.05]: at N=60, minority count ≈ 3 < 5 → excluded every sim.
        let n_sims = 200u32;
        let seed = 42u64;
        let sample_sizes = &[60u32];

        let spec_min5 = lme_factor_spec(0.5, 1.5, vec![0.95, 0.05], 5);
        // min_count=0: full design (g included even when sparse) — today's baseline.
        let spec_min0 = lme_factor_spec(0.5, 1.5, vec![0.95, 0.05], 0);

        let r_min5 = run_batch(&spec_min5, sample_sizes, n_sims, seed, None).unwrap();
        let r_min0 = run_batch(&spec_min0, sample_sizes, n_sims, seed, None).unwrap();

        // (a) Factor excluded in all sims (code 1 = sparse exclusion).
        assert!(
            r_min5.factor_excluded.iter().all(|&c| c == 1),
            "all sims must have factor_excluded=1 at N=60 with LME proportions [0.95,0.05]"
        );

        // (b) g[2] target (slot 1) never significant when excluded.
        // Layout: uncorrected[sim * n_sample_sizes * n_targets + n_idx * n_targets + slot]
        // = uncorrected[sim * 2 + slot] (1 sample size, 2 targets).
        for sim in 0..n_sims as usize {
            assert_eq!(
                r_min5.uncorrected[sim * 2 + 1],
                0,
                "sim {sim}: excluded g[2] target must never be significant in LME"
            );
        }

        // (c) x1 achieves some power (reduced model ran and was fit correctly).
        let x1_hits: usize = (0..n_sims as usize)
            .map(|sim| r_min5.uncorrected[sim * 2] as usize)
            .sum();
        assert!(
            x1_hits > 0,
            "x1 (beta=0.5) at N=60 must achieve some LME power across {n_sims} sims ({x1_hits} hits)"
        );

        // (d) Reduced model converges at least as often as the full model (sparse
        // factor harms LME convergence; removing it can only help or be neutral).
        let conv_min5: usize = r_min5.converged.iter().map(|&c| c as usize).sum();
        let conv_min0: usize = r_min0.converged.iter().map(|&c| c as usize).sum();
        assert!(
            conv_min5 >= conv_min0,
            "reduced LME model must converge at least as often as full model: \
             min5={conv_min5} min0={conv_min0}"
        );
    }

    /// Regression test: LME joint Wald crit uses k_red df under sparse-factor
    /// exclusion, not the full-k df from the precomputed table.
    ///
    /// **Bug being pinned:** before the fix, `joint_crit` was always
    /// `crit.joint_t_crit_sq[n_idx]` = χ²(0.95, 2) = 5.991, even when the factor
    /// was excluded and the joint test ran over k_red=1 targets.
    ///
    /// **Test design — exact per-sim identity:** with k_red=1 the joint Wald
    /// collapses to x1's marginal Wald z² (see
    /// `joint_wald_collapses_to_wald_z_sq_when_k_eq_1`), and χ²(0.95, 1) equals
    /// the marginal z² crit (3.841). So under the fix, `joint_unc[sim]` must
    /// EQUAL x1's `uncorrected[sim]` bit-for-bit. Under the bug the joint gate
    /// was 5.991 — equality breaks for every sim whose z² lands in
    /// (3.841, 5.991). x1_beta=0.26 puts the z² distribution astride that
    /// window (~50% marginal power), so many sims fall in the gap.
    /// Verified to FAIL against the pre-fix code (full-k crit) on 2026-06-05.
    #[test]
    fn lme_joint_wald_crit_reduced_df_under_exclusion() {
        let n_sims = 400u32;
        let seed = 137u64;

        // y ~ x1 + g, g excluded every sim (sparse [0.95,0.05], min=5) → k_red=1.
        let spec = lme_factor_spec(0.26, 0.0, vec![0.95, 0.05], 5);
        let result = run_batch(&spec, &[60], n_sims, seed, None).unwrap();

        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "all sims must have factor_excluded=1"
        );

        // main_row = 2 targets (x1 slot 0, g[2] slot 1); one sample size.
        let mut joint_hits = 0usize;
        for sim in 0..n_sims as usize {
            let x1_sig = result.uncorrected[sim * 2];
            let joint_sig = result.joint_unc[sim];
            assert_eq!(
                joint_sig, x1_sig,
                "sim {sim}: k_red=1 joint Wald must collapse to x1's marginal \
                 verdict (same z², same 3.841 crit) — mismatch means the joint \
                 gate is still the full-k crit"
            );
            joint_hits += joint_sig as usize;
        }
        // Non-degenerate: the z² distribution must straddle the crit, otherwise
        // the equality assertion could never catch a wrong (higher) gate.
        assert!(
            joint_hits > 0 && joint_hits < n_sims as usize,
            "x1_beta=0.26 must give intermediate joint power, got {joint_hits}/{n_sims}"
        );
    }

    /// Regression test: the OLS marginal t² crit uses the reduced-model df
    /// under sparse-factor exclusion, not the full-model df from the
    /// precomputed table.
    ///
    /// **Test design — exact per-sim identity:** with the factor always
    /// excluded, the reduced model is intercept + x1 (p_red = 2), so the
    /// overall F tests the single slope: F = t², and its crit F(0.95; 1, N−2)
    /// equals the marginal t² crit at df = N−2. Both gates read the same
    /// reduced (N, p_red) table, so `uncorrected[sim]` must equal
    /// `overall[sim]` bit-for-bit. With a full-df marginal read the gates
    /// differ — t²crit(df = N−P = 6) ≈ 5.987 vs F(1, 10) ≈ 4.965 at N=12 —
    /// and every sim whose t² lands in that gap breaks the equality.
    /// x1_beta=0.65 puts the t² distribution astride the window (~50% power).
    /// Verified to FAIL against the full-df marginal read on 2026-06-05.
    #[test]
    fn ols_marginal_crit_reduced_df_under_exclusion() {
        let n_sims = 400u32;
        // y ~ x1 + g(5 levels, [0.92, 4×0.02]) at N=12 → every rare level gets
        // < 5 rows → excluded every sim; reduced model = intercept + x1.
        let spec = factor_5level_spec(0.65, vec![0.92, 0.02, 0.02, 0.02, 0.02], 5);
        let result = run_batch(&spec, &[12], n_sims, 4242, None).unwrap();

        assert!(
            result.factor_excluded.iter().all(|&c| c == 1),
            "factor must be excluded in all sims"
        );

        // main_row = 1 target (x1), one sample size → unc and overall both
        // index by sim directly.
        let mut hits = 0usize;
        for sim in 0..n_sims as usize {
            assert_eq!(
                result.uncorrected[sim], result.overall[sim],
                "sim {sim}: with p_red=2 the marginal t² IS the overall F and \
                 t²crit(N−2) == F(1, N−2) — a mismatch means the marginal gate \
                 still reads the full-model df"
            );
            hits += result.uncorrected[sim] as usize;
        }
        // Non-degenerate: t² must straddle the crit, otherwise the equality
        // could never catch a wrong (higher) marginal gate.
        assert!(
            hits > 0 && hits < n_sims as usize,
            "x1_beta=0.65 at N=12 must give intermediate power, got {hits}/{n_sims}"
        );
    }

    /// The reduced-model table comes from the same constructor as the full
    /// one; pin its values at (N=12, p_red=2). The marginal t² crit is
    /// t²(0.975; 10 df) ≈ 4.9646027, which equals the overall crit
    /// F(0.95; 1, 10) by the t²(ν) = F(1, ν) identity — the t and F inverse
    /// routines are independent, so agreement is a real cross-check, not an
    /// echo. The GLM overall crit at p_red=2 is χ²(0.95; 1) ≈ 3.8414588.
    #[test]
    fn reduced_crit_table_matches_known_quantiles() {
        let spec = factor_5level_spec(0.0, vec![0.92, 0.02, 0.02, 0.02, 0.02], 5);
        let (red, posthoc_rows) = build_crit_tables(&spec, &[12], 2).unwrap();
        assert!(
            (red.t_crit_sq_uncorrected[0] - 4.964_602_7).abs() < 1e-5,
            "t² crit at df=10: got {}",
            red.t_crit_sq_uncorrected[0]
        );
        assert!(
            (red.overall_crit[0] - red.t_crit_sq_uncorrected[0]).abs() < 1e-7,
            "t²(ν) = F(1, ν) identity: t²={} F={}",
            red.t_crit_sq_uncorrected[0],
            red.overall_crit[0]
        );
        // No posthoc blocks → one (empty) per-N row for the single-N build.
        assert_eq!(posthoc_rows.len(), 1);
        assert!(posthoc_rows[0].is_empty());

        let glm = logit_5level_spec(0.0, 0.0, vec![0.92, 0.02, 0.02, 0.02, 0.02], 5);
        let (red_glm, _) = build_crit_tables(&glm, &[60], 2).unwrap();
        assert!(
            (red_glm.overall_crit[0] - 3.841_458_8).abs() < 1e-6,
            "χ² crit at df=1: got {}",
            red_glm.overall_crit[0]
        );
    }

    // -----------------------------------------------------------------------
    // M2: the general lmm dispatch path (crossed + nested random intercepts).
    // -----------------------------------------------------------------------

    fn crossed_nested_mle_spec() -> SimulationSpec {
        use crate::spec::{ClusterSizing, ClusterSpec};
        use engine_contract::{GroupingRelation, GroupingSpec};
        // 6 subjects × 4 items + 2 children/subject; atom 48.
        let mut spec = minimal_lme_spec();
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, 0.20);
        cluster.extra_groupings = vec![
            GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                tau_squared: 0.15,
                slopes: vec![],
            },
            GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                tau_squared: 0.08,
                slopes: vec![],
            },
        ];
        spec.cluster = Some(cluster);
        spec
    }

    /// Crossed+nested specs run end-to-end through the Mle arm: majority
    /// convergence, finite power counts, joint significance populated.
    #[test]
    fn general_path_runs_end_to_end() {
        let spec = crossed_nested_mle_spec();
        let r = run_batch_st(&spec, &[96, 192], 200, 2137, None).unwrap();
        let conv: u32 = r.converged.iter().map(|&c| u32::from(c)).sum();
        assert!(conv > 180, "convergence {conv}/400");
        assert_eq!(r.boundary_hit.len(), 400);
    }

    /// Same seed ⇒ same result on the general path (the determinism twin the
    /// Brent arm carries in `run_batch_lme_seed_reproducibility`).
    #[test]
    fn general_path_seed_reproducible() {
        let spec = crossed_nested_mle_spec();
        let r1 = run_batch_st(&spec, &[96], 100, 2137, None).unwrap();
        let r2 = run_batch_st(&spec, &[96], 100, 2137, None).unwrap();
        assert_eq!(r1.uncorrected, r2.uncorrected);
        assert_eq!(r1.converged, r2.converged);
        assert_eq!(r1.boundary_hit, r2.boundary_hit);
    }

    /// st ≡ mt on the general path (mirror of the
    /// `run_batch_st_matches_run_batch_on_same_seed` twin).
    #[test]
    fn general_path_st_mt_equal() {
        let spec = crossed_nested_mle_spec();
        let st = run_batch_st(&spec, &[96], 100, 2137, None).unwrap();
        let mt = run_batch(&spec, &[96], 100, 2137, None).unwrap();
        assert_eq!(st.uncorrected, mt.uncorrected);
        assert_eq!(st.converged, mt.converged);
        assert_eq!(st.boundary_hit, mt.boundary_hit);
    }

    // -----------------------------------------------------------------------
    // M3: random-slopes dispatch path — (1 + x1 | g).
    // -----------------------------------------------------------------------

    /// y ~ x1 + (1+x1|g): one continuous predictor, random intercept + slope
    /// on x1, 20 clusters. `cluster_slope_design_cols=[1]` maps slope 0 → x_full
    /// column 1 (x1). Effect sizes chosen for non-trivial but sub-1 power at N=240.
    fn slope_power_spec() -> SimulationSpec {
        use crate::spec::{ClusterSizing, ClusterSpec};
        use engine_contract::SlopeTerm;

        let mut spec = minimal_lme_spec();
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![SlopeTerm {
                column: engine_contract::ColumnId(0), // column 0 = the continuous predictor
                variance: 0.10,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        });
        // x_full column 1 = x1 (col 0 = intercept, col 1 = first continuous).
        spec.cluster_slope_design_cols = vec![1];
        spec
    }

    /// A `(1 + x1 | g)` power run produces a well-formed BatchResult:
    ///   - `n_variance_components == 2` (intercept + slope)
    ///   - every pinned_components mask uses only the 2 low bits (bitmask for
    ///     2 components, indexed 0 and 1).
    #[test]
    fn slope_power_run_end_to_end() {
        let spec = slope_power_spec();
        let batch = run_batch(&spec, &[240], 200, 2137, None).unwrap();
        assert_eq!(batch.shape.n_variance_components, 2);
        // Every converged fit's pinned mask uses only the 2 low bits.
        assert!(batch.pinned_components.iter().all(|&m| m & !0b11 == 0));
    }
}
