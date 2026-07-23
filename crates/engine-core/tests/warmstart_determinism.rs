//! Determinism guard for the grid-sequential θ̂ warm-start (Phase 2).
//!
//! The warm-start carries the previous (smaller-N) grid point's converged θ̂ into
//! the next point's mixed-model fit. It is safe ONLY because the carry is walked
//! in fixed ascending-N order by the single worker that owns each sim — it never
//! crosses sims, so work-stealing order cannot perturb it. This test pins that
//! guarantee: the same mixed spec + seed must produce a byte-identical
//! `BatchResult` whether run on several rayon workers (`run_batch`) or on a single
//! thread with no rayon (`run_batch_st`). A regression that let the carry leak
//! across sims (the rejected cross-sim accumulator) would diverge here.
//!
//! Covers both mixed arms: a Gaussian LMM (`Mle`) and a clustered logistic GLMM
//! (`Glm` + cluster) — the two paths that feed the carry.

use engine_core::batch::{run_batch, run_batch_st, set_n_threads};
use engine_core::spec::{
    BatchResult, ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution,
    EstimatorSpec, HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations,
    SimulationSpec,
};

const BASE_SEED: u64 = 2137;
const N_SIMS: u32 = 64;
// Ascending multi-point grid so the carry actually fires (each point but the
// first warm-starts from its predecessor).
const GRID: [u32; 4] = [60, 120, 180, 240];

/// Random-intercept mixed spec over `k` continuous predictors, τ² = 0.25, 20
/// clusters. `estimator`/`outcome_kind` select the LMM vs clustered-GLMM path.
fn mixed_spec(estimator: EstimatorSpec, outcome_kind: OutcomeKind) -> SimulationSpec {
    let k = 3usize;
    let mut correlation = vec![0.0; k * k];
    for i in 0..k {
        correlation[i * k + i] = 1.0;
    }
    let mut effect_sizes = vec![0.0; 1 + k];
    effect_sizes[1] = 0.4;
    effect_sizes[2] = 0.4;
    SimulationSpec {
        n_non_factor: k as u32,
        n_factor_dummies: 0,
        correlation,
        var_types: vec![Distribution::Normal; k],
        var_pinned: vec![],
        var_params: vec![0.0; k],
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
        target_indices: vec![1, 2],
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
        outcome_kind,
        link: None,
        estimator,
        wald_se: Default::default(),
        nagq: 1,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.5,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
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
        extra_slope_cols: Vec::new(),
        fit_columns: Vec::new(),
    }
}

/// Byte-identity of two `BatchResult`s. f64 vectors compare on `to_bits` so NaN
/// (the unconverged/unclustered sentinel) matches NaN bit-for-bit.
fn assert_identical(a: &BatchResult, b: &BatchResult, label: &str) {
    assert_eq!(a.uncorrected, b.uncorrected, "{label}: uncorrected");
    assert_eq!(a.corrected, b.corrected, "{label}: corrected");
    assert_eq!(a.posthoc_unc, b.posthoc_unc, "{label}: posthoc_unc");
    assert_eq!(a.posthoc_cor, b.posthoc_cor, "{label}: posthoc_cor");
    assert_eq!(a.converged, b.converged, "{label}: converged");
    assert_eq!(a.boundary_hit, b.boundary_hit, "{label}: boundary_hit");
    assert_eq!(
        a.pinned_components, b.pinned_components,
        "{label}: pinned_components"
    );
    assert_eq!(a.joint_unc, b.joint_unc, "{label}: joint_unc");
    assert_eq!(a.joint_cor, b.joint_cor, "{label}: joint_cor");
    assert_eq!(a.overall, b.overall, "{label}: overall");
    assert_eq!(
        a.factor_excluded, b.factor_excluded,
        "{label}: factor_excluded"
    );
    let a_tau: Vec<u64> = a.tau_squared_hat.iter().map(|x| x.to_bits()).collect();
    let b_tau: Vec<u64> = b.tau_squared_hat.iter().map(|x| x.to_bits()).collect();
    assert_eq!(a_tau, b_tau, "{label}: tau_squared_hat (bit-identical)");
}

/// Multi-thread (`run_batch`, several rayon workers) vs single-thread
/// (`run_batch_st`, no rayon) must be byte-identical for both mixed arms.
#[test]
fn warm_start_is_thread_count_invariant() {
    // Force a multi-worker pool so `run_batch` genuinely interleaves sims across
    // threads (a 1-core CI box would otherwise make this a no-op). Must run before
    // any `run_batch` in this (dedicated) test binary — the pool is set once.
    set_n_threads(4).expect("set thread pool for the determinism test");

    for (estimator, outcome, label) in [
        (EstimatorSpec::Mle, OutcomeKind::Continuous, "LMM"),
        (EstimatorSpec::Glm, OutcomeKind::Binary, "GLMM"),
    ] {
        let spec = mixed_spec(estimator, outcome);
        let parallel =
            run_batch(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch (parallel)");
        let serial =
            run_batch_st(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch_st (serial)");
        assert_identical(&parallel, &serial, label);

        // A second parallel run must match the first (re-entrancy / carry-reset).
        let parallel2 =
            run_batch(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch (parallel #2)");
        assert_identical(&parallel, &parallel2, label);
    }
}
