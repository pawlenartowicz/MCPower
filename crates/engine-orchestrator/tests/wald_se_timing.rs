//! Wall-clock cost of the two GLMM Wald-SE modes — pure Rust, no FFI.
//!
//! `#[ignore]`d (it is a manual benchmark, not a gate). Run ONLY on a stabilized
//! + clock-locked machine, single-threaded, pinned to one P-core:
//!
//! ```text
//! RAYON_NUM_THREADS=1 taskset -c 0 \
//!   cargo test --release -p engine-orchestrator --test wald_se_timing -- --ignored --nocapture
//! ```
//!
//! Times `run_batch_st` (single-threaded, the stable signal) under each mode on a
//! fixed clustered-logit spec; reports MIN-of-3 elapsed (timing noise is one-sided)
//! after a discarded warm-up. The two modes differ ONLY in the per-fit covariance
//! step layered on the shared Laplace fit:
//!   * rx       — Schur block solve            (opt-in speed knob)
//!   * hessian  — FD-Hessian of the deviance   (per-fit, default)

use engine_contract::{ColumnId, SlopeTerm};
use engine_core::batch::run_batch_st;
use engine_core::spec::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
    WaldSe,
};
use std::time::Instant;

const SEED: u64 = 2137;
const N_SIMS: u32 = 2000; // timing loop length; fits/sec is per-fit so this only sets resolution

/// All-defaults OLS spec with `k` standard-normal predictors (copied from the
/// frozen throughput-bench constructor so the DGP matches the rest of the suite).
fn base_spec(k: u32) -> SimulationSpec {
    let k_us = k as usize;
    let mut correlation = vec![0.0; k_us * k_us];
    for i in 0..k_us {
        correlation[i * k_us + i] = 1.0;
    }
    SimulationSpec {
        n_non_factor: k,
        n_factor_dummies: 0,
        correlation,
        var_types: vec![Distribution::Normal; k_us],
        var_pinned: vec![],
        var_params: vec![0.0; k_us],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes: vec![0.0; 1 + k_us],
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
        wald_se: WaldSe::default(),
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.25,
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

/// Clustered-logit GLMM: random intercept (+ optional slope on x1). τ²≈0.822 is
/// the latent-scale ICC-0.2 translation; x1=0.5, baseline_p 0.3.
fn glmm_spec(n_clusters: u32, with_slope: bool) -> SimulationSpec {
    let mut s = base_spec(1);
    s.outcome_kind = OutcomeKind::Binary;
    s.estimator = EstimatorSpec::Glm;
    s.intercept = (0.3f64 / 0.7).ln();
    s.effect_sizes[0] = s.intercept;
    s.effect_sizes[1] = 0.5;
    s.target_indices = vec![1];
    let mut cluster = ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters },
        tau_squared: 0.822,
        slopes: vec![],
        extra_groupings: vec![],
    };
    if with_slope {
        cluster.slopes.push(SlopeTerm {
            column: ColumnId(0),
            variance: 0.10,
            corr_with_intercept: 0.3,
            corr_with: vec![],
        });
        s.cluster_slope_design_cols = vec![1];
    }
    s.cluster = Some(cluster);
    s
}

/// MIN elapsed seconds of `reps` calls to `f`, after one discarded warm-up.
fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warm-up (cold cache + freq ramp) — discarded
    let mut t = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        f();
        t = t.min(start.elapsed().as_secs_f64());
    }
    t
}

#[test]
#[ignore = "manual wall-clock benchmark; run on a locked machine with --ignored --nocapture"]
fn wald_se_mode_timing() {
    let nt = std::fs::read_to_string("/sys/devices/system/cpu/intel_pstate/no_turbo")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".into());
    println!(
        "\nno_turbo = {nt} (want 1 = LOCKED)   RAYON_NUM_THREADS = {}\n",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into())
    );

    // (label, n, n_clusters, slope) — n is an n_clusters multiple (even cluster sizes).
    let cells: &[(&str, u32, u32, bool)] = &[
        ("intercept_8clust", 160, 8, false),
        ("slope_8clust", 160, 8, true),
    ];

    for &(label, n, n_clusters, slope) in cells {
        println!(
            "=== {label}: n={n}, {n_clusters} clusters{} ===",
            if slope { ", random slope" } else { "" }
        );

        let spec0 = glmm_spec(n_clusters, slope);

        let time_mode = |mode: WaldSe| -> f64 {
            let mut spec = spec0.clone();
            spec.wald_se = mode;
            best(3, || {
                run_batch_st(&spec, &[n], N_SIMS, SEED, None).expect("batch");
            })
        };

        let t_rx = time_mode(WaldSe::Rx);
        let t_hess = time_mode(WaldSe::Hessian);

        let fps = |t: f64| N_SIMS as f64 / t;
        let usf = |t: f64| 1e6 * t / N_SIMS as f64;
        println!(
            "  per-fit (warm loop):\n\
                 rx      {:>8.1} us/fit  {:>8.0} fits/s  (1.00x)\n\
                 hessian {:>8.1} us/fit  {:>8.0} fits/s  ({:.2}x rx)",
            usf(t_rx),
            fps(t_rx),
            usf(t_hess),
            fps(t_hess),
            t_hess / t_rx,
        );
        println!();
    }
}
