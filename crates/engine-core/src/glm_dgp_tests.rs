//! GLM fit tests that depend on engine-core's data-generation layer.
//!
//! These four tests drive `glmm::mcpower::glm_irls_fit` on data produced by
//! `data_gen::generate_sim_data` (via a logit `SimulationSpec`). The fit kernel
//! moved to the `glmm` crate in the carve, but `SimulationSpec` /
//! `generate_sim_data` stay here — so these DGP-coupled tests live in
//! engine-core (the other, inline-data GLM tests stay in `glmm`). Bodies are
//! verbatim from the former `glmm/src/glm.rs` test block; only the physical home
//! and the `glm_irls_fit`/`GlmScratch` import path (now `glmm::mcpower`) changed.

use crate::data_gen::generate_sim_data;
use crate::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use crate::workspace::SimWorkspace;
use faer::Mat;
use glmm::mcpower::{glm_irls_fit, GlmScratch, MAX_IRLS_ITERS};

/// Build a `GlmScratch` borrowing every IRLS field of `ws`. Used by tests
/// to avoid duplicating the inline struct literal at each call site.
fn glm_scratch(ws: &mut SimWorkspace) -> GlmScratch<'_> {
    GlmScratch {
        irls_eta: &mut ws.irls_eta,
        irls_p: &mut ws.irls_p,
        irls_w: &mut ws.irls_w,
        irls_z: &mut ws.irls_z,
        irls_betas: &mut ws.irls_betas,
        irls_betas_new: &mut ws.irls_betas_new,
        irls_var_diag: &mut ws.irls_var_diag,
        irls_t_sq: &mut ws.irls_t_sq,
        irls_u_scratch: &mut ws.irls_u_scratch,
        irls_xtwx: ws.irls_xtwx.as_mut(),
        irls_xtwz: &mut ws.irls_xtwz,
        irls_l: ws.irls_l.as_mut(),
        irls_x_f64: &mut ws.irls_x_f64,
        irls_wx: &mut ws.irls_wx,
    }
}

/// Helper: copy `ws.x_full[..n]` and `ws.y_full[..n]` out into owned
/// `(Mat<f32>, Vec<f32>)` so the caller can re-borrow `ws` mutably for
/// the glm scratch without violating aliasing.
fn copy_xy(ws: &SimWorkspace, n: usize, p: usize) -> (Mat<f32>, Vec<f32>) {
    let mut x = Mat::<f32>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = ws.x_full[(i, j)];
        }
    }
    let y: Vec<f32> = ws.y_full[..n].to_vec();
    (x, y)
}

/// Build a Logit spec with `n_non_factor` continuous predictors,
/// `effect_sizes` of length `1 + n_non_factor` (intercept + per-predictor).
fn logit_spec(n_non_factor: u32, effect_sizes: Vec<f64>) -> SimulationSpec {
    let p = (n_non_factor as usize) * (n_non_factor as usize);
    let mut corr = vec![0.0; p];
    for j in 0..(n_non_factor as usize) {
        corr[j * n_non_factor as usize + j] = 1.0;
    }
    let intercept = effect_sizes[0];
    SimulationSpec {
        n_non_factor,
        n_factor_dummies: 0,
        correlation: corr,
        var_types: vec![Distribution::Normal; n_non_factor as usize],
        var_pinned: vec![],
        var_params: vec![0.0; n_non_factor as usize],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes,
        target_indices: (1..=n_non_factor).collect(),
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

/// EST-15: IRLS converges within MAX_IRLS_ITERS on well-behaved binary
/// data with non-trivial signal, returning finite β̂ / z². No recovery
/// value is pinned — only that the iteration terminates as *converged*
/// with a finite iteration count and finite inference.
#[test]
fn glm_converges_on_separable_signal() {
    let intercept = (0.3_f64 / 0.7).ln();
    let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
    let n = 5000;
    let p = 4;
    let mut ws = SimWorkspace::new(n, p, 3, 0, None);
    generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
    let (x, y) = copy_xy(&ws, n, p);

    let targets: Vec<u32> = vec![0, 1, 2, 3];
    let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
    assert!(fit.converged, "IRLS should converge on well-behaved data");
    assert!(
        fit.n_iter > 0 && fit.n_iter <= MAX_IRLS_ITERS,
        "n_iter {} must be in (0, MAX_IRLS_ITERS={}]",
        fit.n_iter,
        MAX_IRLS_ITERS
    );
    for &b in fit.betas.iter() {
        assert!(b.is_finite(), "β̂ must be finite on converged fit");
    }
    for &t in fit.t_sq.iter() {
        // Finiteness only: direction + magnitude are already pinned by the
        // sibling `glm_wald_z_sq_direction_and_ballpark` (same spec/seed,
        // asserts z²[β₁] > 50 at n=5000). A tautological `>= 0.0` on a
        // squared quantity adds nothing beyond the finite check here.
        assert!(t.is_finite(), "z² must be finite on converged fit");
    }
}

/// W1 truth-start: warm (β₀ = spec.effect_sizes) and cold (β = 0) starts
/// on the same bytes converge to the same |Δdev| < DEVIANCE_TOL fixpoint —
/// β̂ within 1e-5 abs, z = √z² within 1e-4 abs (the campaign parity
/// floors) — and the warm path spends no more iterations than the cold
/// one. A warm start that shifted the fixpoint (rather than the path to
/// it) or slowed convergence would fail here.
#[test]
fn glm_truth_start_matches_cold_fixpoint() {
    let intercept = (0.3_f64 / 0.7).ln();
    let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
    let n = 1000;
    let p = 4;
    let mut ws = SimWorkspace::new(n, p, 3, 0, None);
    generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
    let (x, y) = copy_xy(&ws, n, p);
    let targets: Vec<u32> = vec![0, 1, 2, 3];

    let cold = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
    assert!(cold.converged, "cold fit must converge");
    let cold_betas = cold.betas.to_vec();
    let cold_t_sq = cold.t_sq.to_vec();
    let cold_iters = cold.n_iter;

    let warm = glm_irls_fit(
        x.as_ref(),
        &y,
        &targets,
        Some(&spec.effect_sizes),
        glm_scratch(&mut ws),
    );
    assert!(warm.converged, "warm fit must converge");
    for (j, (&bw, &bc)) in warm.betas.iter().zip(&cold_betas).enumerate() {
        assert!(
            (bw - bc).abs() < 1e-5,
            "β̂[{j}]: warm {bw} vs cold {bc} exceeds 1e-5"
        );
    }
    for (t, (&tw, &tc)) in warm.t_sq.iter().zip(&cold_t_sq).enumerate() {
        assert!(
            (tw.sqrt() - tc.sqrt()).abs() < 1e-4,
            "z[{t}]: warm {} vs cold {} exceeds 1e-4",
            tw.sqrt(),
            tc.sqrt()
        );
    }
    assert!(
        warm.n_iter <= cold_iters,
        "warm n_iter {} must not exceed cold {cold_iters}",
        warm.n_iter
    );
}

/// Warm-path allocation guard for the IRLS fit (mirrors `ols.rs`'s
/// `fit_suff_stats_warm_path_bounded_alloc`). Per fit the allocations are
/// faer's `Llt` internals for the in-place β solve (one per IRLS
/// iteration) plus a single post-loop `.L()` that exposes the cached factor
/// `L`. On faer 0.24 `.L()` returns a borrow (`MatRef`) instead of an owned
/// `Mat`, so that post-loop materialisation no longer allocates: this fixture
/// converges in 3 iterations and now allocates 8 blocks/fit (800 over
/// `N_CALLS`), down from 9/fit on faer 0.20 (itself down from 12/fit before
/// the in-loop deferral).
///
/// `BOUND` locks the measured warm-path block count for this fixture; it is
/// iteration-count-dependent, so a change here flags either a new
/// per-iteration allocation or a shift in convergence behaviour. If a future
/// faer version changes its Cholesky internals, update the bound — do not
/// relax it to whatever passes today.
///
/// `#[ignore]` because `dhat::Profiler` measures process-wide allocations
/// and other concurrent tests in the same binary contaminate the count.
/// Run explicitly with:
///   `cargo test -p engine-core glm_fit_warm_path_bounded_alloc -- --ignored --test-threads=1`
#[test]
#[ignore]
fn glm_fit_warm_path_bounded_alloc() {
    const N: usize = 1000;
    const P: usize = 4;
    const N_CALLS: usize = 100;
    // 8 blocks/fit × 100 calls on faer 0.24 (was 9/fit on 0.20; the `.L()`
    // borrow dropped the post-loop factor materialisation). Re-measured
    // after the truth start landed: this well-signalled fixture converges
    // in the same iteration count warm as cold, so the count is unchanged
    // — the warm start pays off on slow-converging (e.g. rare-events)
    // fits, not here.
    const BOUND: u64 = 800;

    let intercept = (0.3_f64 / 0.7).ln();
    let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
    let mut ws = SimWorkspace::new(N, P, 3, 0, None);
    generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
    let (x, y) = copy_xy(&ws, N, P);
    let targets: Vec<u32> = vec![1, 2, 3];

    // Warmup outside the profiler window; assert convergence so the bench
    // measures the converged path (the only one that caches L). Truth-
    // started like the shipped hot loop — the bound is iteration-count-
    // dependent, so the fixture must walk the same warm path.
    let warm = glm_irls_fit(
        x.as_ref(),
        &y,
        &targets,
        Some(&spec.effect_sizes),
        glm_scratch(&mut ws),
    );
    assert!(warm.converged, "bench fixture must converge");

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        let _ = glm_irls_fit(
            x.as_ref(),
            &y,
            &targets,
            Some(&spec.effect_sizes),
            glm_scratch(&mut ws),
        );
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks <= BOUND,
        "glm_irls_fit allocated {} blocks across {} warm-path calls (BOUND = {})",
        stats.total_blocks,
        N_CALLS,
        BOUND
    );
}

/// C11 — GLM Wald z² direction and ballpark (n=5000, β₁=0.5)
#[test]
fn glm_wald_z_sq_direction_and_ballpark() {
    // Extends glm_converges_on_separable_signal with direction + z² band assertions.
    // At n=5000 with β₁=0.5, β₂=-0.3, β₃=0.0, seed=42 (deterministic), the
    // fitted β̂ should be in the correct direction and the Wald z² for β₁ should
    // be in a realistic range (analytic SE ≈ 0.028 → z² ≈ 300 at n=5000).
    let intercept = (0.3_f64 / 0.7).ln();
    let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
    let n = 5000;
    let p = 4;
    let mut ws = SimWorkspace::new(n, p, 3, 0, None);
    generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
    let (x, y) = copy_xy(&ws, n, p);
    let targets: Vec<u32> = vec![0, 1, 2, 3];
    let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
    assert!(fit.converged, "IRLS should converge on well-behaved data");

    // β̂₁ should be near +0.5 (positive direction).
    assert!(
        fit.betas[1] > 0.35 && fit.betas[1] < 0.65,
        "β̂₁ = {} expected near 0.5",
        fit.betas[1]
    );
    // β̂₂ should be near -0.3 (negative direction).
    assert!(
        fit.betas[2] > -0.45 && fit.betas[2] < -0.15,
        "β̂₂ = {} expected near -0.3",
        fit.betas[2]
    );
    // Wald z² for target 1 (β₁): at n=5000 with β₁=0.5, z² should be substantial.
    // A biased IRLS returning z²≈0 or z²>1e6 would fail this bound.
    assert!(
        fit.t_sq[1] > 50.0 && fit.t_sq[1] < 2000.0,
        "z²[β₁] = {} expected in (50, 2000) at n=5000",
        fit.t_sq[1]
    );
}
