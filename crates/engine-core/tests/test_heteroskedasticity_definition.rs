//! Heteroskedasticity redesign (2026-05-29): residual variance follows the
//! renormalized-multiplicative monotone model
//!   `Var(εᵢ) = σ²·exp(γ·zᵢ)/exp(γ²/2)`,  `γ = ln(λ)/4`,
//! where `z` is the standardized driver (linear predictor Xβ when `driver=None`,
//! else a chosen `x_full` column) and λ is the variance ratio — scenario-only
//! (`scenario.heteroskedasticity_ratio`; 1.0 = homoskedastic). The model
//! contributes only the driver.
//! The `/exp(γ²/2)` factor preserves mean residual variance for a normal driver.

use engine_core::data_gen::generate_sim_data;
use engine_core::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use engine_core::workspace::SimWorkspace;

/// Two zero-mean unit-variance uncorrelated normal predictors; β = (0, 0.5, 0.3).
/// `driver` populates `heteroskedasticity_driver` directly (the x_full column
/// index: 0 = intercept, 1 = x1, 2 = x2); `lambda` is the scenario λ (1.0 =
/// homoskedastic, keeps the scenario optimistic).
fn make_spec(driver: Option<u32>, lambda: f64) -> SimulationSpec {
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
        effect_sizes: vec![0.0, 0.5, 0.3],
        target_indices: vec![1, 2],
        contrast_pairs: vec![],
        interactions: vec![],
        correction_method: CorrectionMethod::None,
        crit_values: CritValues {
            alpha: 0.05,
            posthoc_alpha: None,
        },
        heteroskedasticity_driver: driver,
        residual_dist: ResidualDist::Normal,
        residual_pinned: false,
        outcome_kind: OutcomeKind::Continuous,
        estimator: EstimatorSpec::Ols,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.1,
        cluster: None,
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: lambda,
            ..ScenarioPerturbations::optimistic()
        },
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    };
    s.het_coeffs = s.compute_het_coeffs();
    s
}

#[test]
fn coeffs_populate_lp_and_per_column_moments() {
    // β = (0, 0.5, 0.3) over zero-mean uncorrelated unit-variance columns:
    // lp_pop_mean == 0 (symmetry), lp_pop_std == sqrt(0.25 + 0.09) > 0, and the
    // per-column vecs are full-design [intercept, x1, x2] = ([1,0,0],[0,1,1]).
    let s = make_spec(None, 2.0);
    assert!(
        s.het_coeffs.lp_pop_mean.abs() < 1e-12,
        "lp_pop_mean must be 0, got {}",
        s.het_coeffs.lp_pop_mean
    );
    let expected_std = (0.25_f64 + 0.09).sqrt();
    assert!(
        (s.het_coeffs.lp_pop_std - expected_std).abs() < 1e-12,
        "lp_pop_std = {}, expected {expected_std}",
        s.het_coeffs.lp_pop_std
    );
    assert_eq!(s.het_coeffs.col_mean, vec![1.0, 0.0, 0.0]);
    assert_eq!(s.het_coeffs.col_std, vec![0.0, 1.0, 1.0]);
}

#[test]
fn residual_scale_matches_exp_form_for_lp_driver() {
    let lambda = 4.0_f64;
    let spec = make_spec(None, lambda);
    let n_pred = 3;
    let n = 400;
    let mut ws = SimWorkspace::new(n, n_pred, 2, 0, None);
    generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

    let mu = spec.het_coeffs.lp_pop_mean;
    let sd = spec.het_coeffs.lp_pop_std;
    let gamma = lambda.ln() / 4.0;
    let inv_norm = (-gamma * gamma / 2.0).exp();
    let mut unscaled_sq = 0.0;
    for i in 0..n {
        // x_full and residuals are f32 (data plane); widen to f64 for stats.
        let lp = 0.5 * ws.x_full[(i, 1)] as f64 + 0.3 * ws.x_full[(i, 2)] as f64;
        let z = (lp - mu) / sd;
        let mult = (gamma * z).exp() * inv_norm;
        let raw = ws.residuals[i] as f64 / mult.sqrt();
        unscaled_sq += raw * raw;
    }
    let var = unscaled_sq / n as f64;
    // After dividing out the exact per-row multiplier the residuals are standard
    // normal again; var ≈ 1 within sampling noise at n=400.
    assert!((var - 1.0).abs() < 0.3, "unscaled residual var = {var}");
}

#[test]
fn mean_residual_variance_preserved_across_lambda() {
    // The /exp(γ²/2) renormalization keeps E[Var(ε)] = σ² = 1 as λ moves, so
    // dialing heteroskedasticity does not change total noise. Exact in expectation
    // for a normal driver; checked as mean(residualᵢ²) ≈ 1.
    let n_pred = 3;
    let n = 4000;
    for &lambda in &[1.0_f64, 2.0, 4.0] {
        let spec = make_spec(None, lambda);
        let mut ws = SimWorkspace::new(n, n_pred, 2, 0, None);
        generate_sim_data(&spec, 7, 2024, &mut ws).unwrap();
        let mean_sq: f64 = (0..n)
            .map(|i| (ws.residuals[i] * ws.residuals[i]) as f64)
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean_sq - 1.0).abs() < 0.1,
            "λ={lambda}: mean residual² = {mean_sq}, expected ≈ 1.0"
        );
    }
}

#[test]
fn lambda_only_scenario_scales_residuals_but_not_x() {
    // λ is the only non-neutral knob: the run leaves the optimistic fast path,
    // but X must stay bit-identical to the off run (the data stream is
    // untouched; only the residual multiplier changes), and at least one
    // residual must differ (λ is live).
    let n_pred = 3;
    let n = 300;
    let spec_off = make_spec(None, 1.0);
    let spec_l4 = make_spec(None, 4.0);
    let mut ws_off = SimWorkspace::new(n, n_pred, 2, 0, None);
    let mut ws_l4 = SimWorkspace::new(n, n_pred, 2, 0, None);
    generate_sim_data(&spec_off, 1, 42, &mut ws_off).unwrap();
    generate_sim_data(&spec_l4, 1, 42, &mut ws_l4).unwrap();
    for i in 0..n {
        for j in 0..n_pred {
            assert_eq!(
                ws_off.x_full[(i, j)].to_bits(),
                ws_l4.x_full[(i, j)].to_bits(),
                "X diverges at ({i},{j}) — λ must not touch the data stream"
            );
        }
    }
    assert!(
        (0..n).any(|i| ws_off.residuals[i] != ws_l4.residuals[i]),
        "scenario λ=4 must produce heteroskedastic residuals, not the off pattern"
    );
}

#[test]
fn different_drivers_produce_different_residuals() {
    // Driving variance off x1 vs x2 yields a different per-row scaling (hence
    // different power downstream), proving the driver is actually honored
    // (the v1 bug ignored it).
    let n_pred = 3;
    let n = 300;
    let spec_x1 = make_spec(Some(1), 4.0);
    let spec_x2 = make_spec(Some(2), 4.0);
    let mut ws1 = SimWorkspace::new(n, n_pred, 2, 0, None);
    let mut ws2 = SimWorkspace::new(n, n_pred, 2, 0, None);
    generate_sim_data(&spec_x1, 4, 42, &mut ws1).unwrap();
    generate_sim_data(&spec_x2, 4, 42, &mut ws2).unwrap();
    // Same RNG draws, but different driver columns ⇒ different multipliers.
    assert!(
        (0..n).any(|i| ws1.residuals[i] != ws2.residuals[i]),
        "x1-driven and x2-driven het must differ"
    );
}

#[test]
fn fused_path_y_equals_lp_plus_scaled_residual() {
    // y[i] == (Σ βⱼ·xᵢⱼ) + scaled_residual[i] (heterogeneity=0, no cluster).
    // Catches a regression where het scaling is reordered relative to y assembly.
    let spec = make_spec(None, 4.0);
    let n_pred = 3;
    let n = 200;
    let mut ws = SimWorkspace::new(n, n_pred, 2, 0, None);
    generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
    for i in 0..n {
        // Mirror the engine exactly: accumulate lp + residual in f64 (the fit-
        // precision plane), narrow to f32 once at the store. Recomputing in f32
        // would round per-op and diverge by 1 ULP from the engine's single narrow.
        let lp = 0.5 * ws.x_full[(i, 1)] as f64 + 0.3 * ws.x_full[(i, 2)] as f64;
        let expected_y = (lp + ws.residuals[i] as f64) as f32;
        assert_eq!(
            ws.y_full[i], expected_y,
            "fused y-assembly mismatch at i={i}: y={}, lp={lp}, residual={}",
            ws.y_full[i], ws.residuals[i]
        );
    }
}

#[test]
fn lme_het_scales_level1_only_cluster_draw_homoskedastic() {
    // Under heteroskedasticity an LME (clustered) run scales the LEVEL-1 residual
    // only; the cluster draw u_c stays homoskedastic, and mean residual variance is
    // preserved (so the mean ICC τ²/(τ²+σ²) is unchanged as λ moves). u_c is drawn
    // before the row loop and does not depend on λ, so cluster draws are
    // bit-identical across λ.
    use engine_core::spec::{ClusterSizing, ClusterSpec};
    let n = 2000;
    let n_pred = 3;
    let n_clusters = 20;
    let sizing = ClusterSizing::FixedClusters { n_clusters };

    let mut spec_off = make_spec(None, 1.0);
    spec_off.cluster = Some(ClusterSpec {
        sizing: sizing.clone(),
        tau_squared: 0.5,
        slopes: vec![],
        extra_groupings: vec![],
    });
    let mut spec_doomer = make_spec(None, 4.0);
    spec_doomer.cluster = Some(ClusterSpec {
        sizing: sizing.clone(),
        tau_squared: 0.5,
        slopes: vec![],
        extra_groupings: vec![],
    });

    let mut ws_off = SimWorkspace::new(n, n_pred, 2, 0, Some(&ClusterSpec::intercept_only(sizing.clone(), 0.25)));
    let mut ws_doomer = SimWorkspace::new(n, n_pred, 2, 0, Some(&ClusterSpec::intercept_only(sizing.clone(), 0.25)));
    generate_sim_data(&spec_off, 11, 2024, &mut ws_off).unwrap();
    generate_sim_data(&spec_doomer, 11, 2024, &mut ws_doomer).unwrap();

    // (1) Cluster draws u_c are homoskedastic: identical regardless of λ.
    for c in 0..n_clusters as usize {
        assert_eq!(
            ws_off.cluster_u_draws[c], ws_doomer.cluster_u_draws[c],
            "cluster draw u_c[{c}] must be unaffected by heteroskedasticity"
        );
    }
    // (2) Level-1 residuals DO change under doomer (het is applied to ε).
    assert!(
        (0..n).any(|i| ws_off.residuals[i] != ws_doomer.residuals[i]),
        "doomer must apply level-1 heteroskedasticity"
    );
    // (3) Mean level-1 residual variance is preserved ⇒ mean ICC preserved.
    let mean_sq: f64 = (0..n)
        .map(|i| (ws_doomer.residuals[i] * ws_doomer.residuals[i]) as f64)
        .sum::<f64>()
        / n as f64;
    assert!(
        (mean_sq - 1.0).abs() < 0.15,
        "mean level-1 residual² under doomer = {mean_sq}, expected ≈ 1.0 (variance preserved)"
    );
}

#[test]
fn row_stable_under_het() {
    // Residual scale at row i depends only on row i's X plus spec scalars, so
    // x_full and y prefixes stay bit-identical across max_n.
    let spec = make_spec(None, 4.0);
    let n_pred = 3;
    let mut ws_big = SimWorkspace::new(1000, n_pred, 2, 0, None);
    let mut ws_small = SimWorkspace::new(200, n_pred, 2, 0, None);
    generate_sim_data(&spec, 5, 42, &mut ws_big).unwrap();
    generate_sim_data(&spec, 5, 42, &mut ws_small).unwrap();
    for i in 0..200 {
        for j in 0..n_pred {
            assert_eq!(
                ws_big.x_full[(i, j)],
                ws_small.x_full[(i, j)],
                "x diverges at ({i},{j})"
            );
        }
        assert_eq!(ws_big.y_full[i], ws_small.y_full[i], "y diverges at {i}");
    }
}
