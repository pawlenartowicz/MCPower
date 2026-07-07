//! Parity-bug fix: v1 applies `heterogeneity` (β-jitter) to LME — v2 dropped it on the floor
//! by gating on `family == Family::Ols`. After the fused y-loop, the gate is
//! `Ols | LmeIntercept`, so LME with `heterogeneity > 0` must actually perturb
//! the linear predictor.
//!
//! Test: for fixed seed, the per-row residual contribution
//! `y[i] - Σ βⱼ·xᵢⱼ - ε[i] - u_c[i]` is exactly zero (modulo FP noise) when
//! `heterogeneity = 0` and has materially non-zero spread when `heterogeneity > 0`.
//! Under the bug this was zero in both cases — that is precisely what this test
//! pins against regression.

use engine_core::data_gen::generate_sim_data;
use engine_core::spec::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use engine_core::workspace::SimWorkspace;

const N_CLUSTERS: usize = 5;
const TAU_SQUARED: f64 = 0.25;
const BETAS: [f64; 3] = [0.0, 0.5, -0.4];
const ROW_N: usize = 80;
const N_SIMS: u64 = 200;
const BASE_SEED: u64 = 2137;

fn make_lme_spec(het: f64) -> SimulationSpec {
    let mut s = SimulationSpec {
        n_non_factor: 2,
        n_factor_dummies: 0,
        correlation: vec![1.0, 0.0, 0.0, 1.0],
        var_types: vec![Distribution::Normal, Distribution::Normal],
        var_pinned: vec![],
        var_params: vec![0.0, 0.0],
        upload_normal: vec![],
        upload_normal_shape: (0, 0),
        upload_data: vec![],
        upload_data_shape: (0, 0),
        bootstrap_frame_map: vec![],
        between_var_indices: vec![],
        factor_n_levels: vec![],
        factor_proportions: vec![],
        factor_sampled: Vec::new(),
        effect_sizes: BETAS.to_vec(),
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
        outcome_kind: OutcomeKind::Continuous,
        estimator: EstimatorSpec::Mle,
        wald_se: Default::default(),
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.1,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters {
                n_clusters: N_CLUSTERS as u32,
            },
            tau_squared: TAU_SQUARED,
            slopes: vec![],
            extra_groupings: vec![],
        }),
        scenario: ScenarioPerturbations {
            heterogeneity: het,
            ..ScenarioPerturbations::optimistic()
        },
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        extra_slope_cols: Vec::new(),
        fit_columns: Vec::new(),
    };
    s.het_coeffs = s.compute_het_coeffs();
    s
}

/// Per-sim residual contribution at row 0:
///   delta = y[0] - (Σ βⱼ·x₀ⱼ) - ε[0] - u_{cluster(0)}
/// Under correct heterogeneity wiring this equals the β-jitter contribution
///   Σⱼ N(0, het·|βⱼ|) · x₀ⱼ
/// (∼ 0 when het = 0 modulo FP; nonzero with variance ≳ het²·Σ βⱼ² otherwise).
fn collect_jitter_at_row0(het: f64) -> Vec<f64> {
    let spec = make_lme_spec(het);
    let n_pred = 1 + spec.n_non_factor as usize;
    let mut ws = SimWorkspace::new(
        ROW_N,
        n_pred,
        spec.n_non_factor as usize,
        0,
        Some(&ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters {
                n_clusters: N_CLUSTERS as u32,
            },
            0.25,
        )),
    );
    let mut out = Vec::with_capacity(N_SIMS as usize);
    for sim_id in 0..N_SIMS {
        generate_sim_data(&spec, sim_id, BASE_SEED, &mut ws).unwrap();
        let mut clean_lp = 0.0;
        for (j, &beta) in BETAS.iter().enumerate().take(n_pred) {
            // x_full is f32 (data plane); widen to f64 for stats.
            clean_lp += beta * ws.x_full[(0, j)] as f64;
        }
        let cluster_id = 0 % N_CLUSTERS; // interleaved layout
        let u_c = ws.cluster_u_draws[cluster_id] as f64;
        out.push(ws.y_full[0] as f64 - clean_lp - ws.residuals[0] as f64 - u_c);
    }
    out
}

fn sample_var(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

#[test]
fn lme_heterogeneity_is_applied_to_linear_predictor() {
    let off = collect_jitter_at_row0(0.0);
    let on = collect_jitter_at_row0(0.3);

    let var_off = sample_var(&off);
    let var_on = sample_var(&on);

    // Sanity: with het=0 the jitter contribution is zero, so δ collapses to the
    // f32 narrowing of `y_full` (the data plane is f32; `δ = f32(S) − S` for the
    // f64 sum S = lp + residual + u_c). That floor is ~1e-7, not 0 — pin to the
    // f32-plane budget, still tight enough to catch a gate flip back to OLS-only.
    let max_off = off.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        max_off < 1e-6,
        "heterogeneity=0 should produce zero β-jitter contribution; max |δ| = {max_off:.3e}"
    );
    assert!(
        var_off < 1e-12,
        "heterogeneity=0 jitter variance should be ~0; got {var_off:.3e}"
    );

    // With het=0.3 the jitter contribution has variance ~ E_X[Σ (h·βⱼ)²·xⱼ²]
    // For β=(0, 0.5, -0.4), Var(X_j) ≈ 1 ⇒ expected ≈ 0.09·(0.25 + 0.16) ≈ 0.037.
    // Loose lower bound to stay robust under sample noise.
    assert!(
        var_on > 0.01,
        "heterogeneity=0.3 jitter variance should be ≳ 0.01; got {var_on:.3e}"
    );
    // And it must be materially larger than the het=0 baseline by orders of magnitude.
    assert!(
        var_on > 1e12 * var_off.max(1e-30),
        "heterogeneity=0.3 var ({var_on:.3e}) should dwarf heterogeneity=0 var ({var_off:.3e})"
    );
}

/// X / residuals / cluster draws must be bit-identical between het=0 and het>0
/// runs at the same seed — the per-study β-jitter δ is drawn from a
/// domain-separated stream (sim_id ^ STREAM_TAG_HET), never the main data
/// stream. If this fails, heterogeneity is consuming main-stream RNG and has
/// broken the contract that it is a y-only knob.
#[test]
fn heterogeneity_does_not_perturb_x_or_residual_streams() {
    let spec_off = make_lme_spec(0.0);
    let spec_on = make_lme_spec(0.3);
    let n_pred = 1 + spec_off.n_non_factor as usize;
    let mut ws_off = SimWorkspace::new(
        ROW_N,
        n_pred,
        spec_off.n_non_factor as usize,
        0,
        Some(&ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters {
                n_clusters: N_CLUSTERS as u32,
            },
            0.25,
        )),
    );
    let mut ws_on = SimWorkspace::new(
        ROW_N,
        n_pred,
        spec_on.n_non_factor as usize,
        0,
        Some(&ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters {
                n_clusters: N_CLUSTERS as u32,
            },
            0.25,
        )),
    );

    generate_sim_data(&spec_off, 7, BASE_SEED, &mut ws_off).unwrap();
    generate_sim_data(&spec_on, 7, BASE_SEED, &mut ws_on).unwrap();

    for i in 0..ROW_N {
        for j in 0..n_pred {
            assert_eq!(
                ws_off.x_full[(i, j)],
                ws_on.x_full[(i, j)],
                "X stream diverges at ({i},{j}) — heterogeneity must not consume X-draw RNG"
            );
        }
        assert_eq!(
            ws_off.residuals[i], ws_on.residuals[i],
            "residual stream diverges at {i} — heterogeneity must not consume residual-draw RNG"
        );
    }
    for c in 0..N_CLUSTERS {
        assert_eq!(
            ws_off.cluster_u_draws[c], ws_on.cluster_u_draws[c],
            "cluster random-effect stream diverges at {c}"
        );
    }
}
