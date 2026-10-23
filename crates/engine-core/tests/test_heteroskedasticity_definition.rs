//! Heteroskedasticity redesign (2026-05-29): residual variance follows the
//! renormalized-multiplicative monotone model
//!   `Var(εᵢ) = σ²·exp(γ·zᵢ)/exp(γ²/2)`,  `γ = ln(λ)/4`,
//! where `z` is the standardized driver (linear predictor Xβ when `driver=None`,
//! else a chosen `x_full` column) and λ is the variance ratio — scenario-only
//! (`scenario.heteroskedasticity_ratio`; 1.0 = homoskedastic). The model
//! contributes only the driver.
//! The `/exp(γ²/2)` factor preserves mean residual variance for a normal driver.

mod common;

use engine_core::data_gen::generate_sim_data;
use engine_core::spec::{Distribution, ScenarioPerturbations, SimulationSpec};
use engine_core::workspace::SimWorkspace;

/// Two zero-mean unit-variance uncorrelated normal predictors; β = (0, 0.5, 0.3).
/// `driver` populates `heteroskedasticity_driver` directly (the x_full column
/// index: 0 = intercept, 1 = x1, 2 = x2); `lambda` is the scenario λ (1.0 =
/// homoskedastic, keeps the scenario optimistic).
fn make_spec(driver: Option<u32>, lambda: f64) -> SimulationSpec {
    let mut s = SimulationSpec {
        n_non_factor: 2,
        correlation: vec![1.0, 0.0, 0.0, 1.0],
        var_types: vec![Distribution::Normal, Distribution::Normal],
        var_params: vec![0.0, 0.0],
        effect_sizes: vec![0.0, 0.5, 0.3],
        target_indices: vec![1, 2],
        heteroskedasticity_driver: driver,
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: lambda,
            ..ScenarioPerturbations::optimistic()
        },
        ..common::base_spec()
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

/// Exact per-row residual multiplier for the linear-predictor driver: with the
/// same raw CLASS_RESID draws (λ = 1 leaves them unscaled), the λ = 4 residual
/// is `ε₁ᵢ·sqrt(exp(γ·zᵢ)/exp(γ²/2))`. The `/exp(γ²/2)` renormalisation keeps
/// E[Var(ε)] = σ² as λ moves (exact in expectation for a normal driver), and
/// dropping it shifts every row by exp(γ²/4) ≈ 3% at λ = 4, far outside the
/// f32 rounding tolerance used here.
#[test]
fn residual_scale_matches_exp_form_for_lp_driver() {
    let lambda = 4.0_f64;
    let spec_off = make_spec(None, 1.0);
    let spec = make_spec(None, lambda);
    let n_pred = 3;
    let n = 400;
    let mut ws_off = SimWorkspace::new(n, n_pred, 2, 0, None);
    let mut ws = SimWorkspace::new(n, n_pred, 2, 0, None);
    generate_sim_data(&spec_off, 0, 42, &mut ws_off).unwrap();
    generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

    let mu = spec.het_coeffs.lp_pop_mean;
    let sd = spec.het_coeffs.lp_pop_std;
    let gamma = lambda.ln() / 4.0;
    let inv_norm = (-gamma * gamma / 2.0).exp();
    for i in 0..n {
        // x_full and residuals are f32 (data plane); widen to f64.
        let lp = 0.5 * ws.x_full[(i, 1)] as f64 + 0.3 * ws.x_full[(i, 2)] as f64;
        let z = (lp - mu) / sd;
        let mult = (gamma * z).exp() * inv_norm;
        let expected = ws_off.residuals[i] as f64 * mult.sqrt();
        let got = ws.residuals[i] as f64;
        assert!(
            (got - expected).abs() <= 1e-6 * expected.abs().max(1e-3),
            "row {i}: residual {got}, expected {expected}"
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

    let mut ws_off = SimWorkspace::new(
        n,
        n_pred,
        2,
        0,
        Some(&ClusterSpec::intercept_only(sizing.clone(), 0.25)),
    );
    let mut ws_doomer = SimWorkspace::new(
        n,
        n_pred,
        2,
        0,
        Some(&ClusterSpec::intercept_only(sizing.clone(), 0.25)),
    );
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

/// Generates one sim of `n` rows and checks the analytic driver moments in
/// `spec.het_coeffs` against the sample moments of what the generator wrote:
/// `lp_clean = Σⱼ βⱼ·xᵢⱼ` over every `x_full` column (interactions included)
/// for `lp_pop_mean`/`lp_pop_std`, and each column for `col_mean`/`col_std`.
/// Tolerance is 5 Monte-Carlo SEs: SE(mean) = s/√n and
/// SE(sd) = s·√((m₄/s⁴ − 1)/(4n)), both from the sample, plus a 1e-9 floor
/// for the constant intercept column.
fn assert_driver_moments_match_generated(spec: &SimulationSpec, n: usize, label: &str) {
    let n_nf = spec.n_non_factor as usize;
    let p = 1 + n_nf + spec.n_factor_dummies as usize + spec.interactions.len();
    let mut ws = SimWorkspace::new(n, p, n_nf, spec.factor_n_levels.len(), None);
    generate_sim_data(spec, 0, 2137, &mut ws).unwrap();

    let check = |values: &[f64], mean: f64, sd: f64, what: &str| {
        let nf = values.len() as f64;
        let m = values.iter().sum::<f64>() / nf;
        let m2 = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / nf;
        let m4 = values.iter().map(|v| (v - m).powi(4)).sum::<f64>() / nf;
        let s = m2.sqrt();
        let se_mean = s / nf.sqrt();
        let se_sd = if m2 > 0.0 {
            s * ((m4 / (m2 * m2) - 1.0).max(0.0) / (4.0 * nf)).sqrt()
        } else {
            0.0
        };
        assert!(
            (m - mean).abs() <= 5.0 * se_mean + 1e-9,
            "{label} {what}: analytic mean {mean}, sample mean {m} (SE {se_mean})"
        );
        assert!(
            (s - sd).abs() <= 5.0 * se_sd + 1e-9,
            "{label} {what}: analytic sd {sd}, sample sd {s} (SE {se_sd})"
        );
    };

    let lp: Vec<f64> = (0..n)
        .map(|i| {
            (0..p)
                .map(|j| ws.x_full[(i, j)] as f64 * spec.effect_sizes[j])
                .sum()
        })
        .collect();
    check(
        &lp,
        spec.het_coeffs.lp_pop_mean,
        spec.het_coeffs.lp_pop_std,
        "lp",
    );
    assert_eq!(spec.het_coeffs.col_mean.len(), p, "{label}: col_mean width");
    assert_eq!(spec.het_coeffs.col_std.len(), p, "{label}: col_std width");
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| ws.x_full[(i, j)] as f64).collect();
        check(
            &col,
            spec.het_coeffs.col_mean[j],
            spec.het_coeffs.col_std[j],
            &format!("column {j}"),
        );
    }
}

#[test]
fn driver_moments_cover_correlated_gaussian_interactions() {
    // x1, x2, x3 standard normal with r12 = 0.5, r13 = 0.3, r23 = −0.2 and the
    // terms x1:x2, x2:x3, x1:x2:x3. E[x1·x2] = r12 ≠ 0 moves lp_pop_mean, and
    // the 3-way column covaries with x1 (E[x1²x2x3] = r23 + 2·r12·r13).
    let mut s = SimulationSpec {
        n_non_factor: 3,
        correlation: vec![1.0, 0.5, 0.3, 0.5, 1.0, -0.2, 0.3, -0.2, 1.0],
        var_types: vec![Distribution::Normal; 3],
        var_params: vec![0.0; 3],
        interactions: vec![vec![1, 2], vec![2, 3], vec![1, 2, 3]],
        effect_sizes: vec![0.1, 0.25, 0.25, -0.2, 0.22, 0.15, 0.1],
        target_indices: vec![1, 2, 3, 4, 5, 6],
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: 2.0,
            ..ScenarioPerturbations::optimistic()
        },
        ..common::base_spec()
    };
    s.het_coeffs = s.compute_het_coeffs();
    assert_driver_moments_match_generated(&s, 400_000, "correlated gaussian");
}

#[test]
fn driver_moments_cover_binary_by_continuous_interactions() {
    // `y ~ treatment + stress * sleep` plus treatment:stress, with treatment
    // binary (p = 0.3) and every column uncorrelated.
    let mut s = SimulationSpec {
        n_non_factor: 3,
        correlation: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        var_types: vec![
            Distribution::Binary,
            Distribution::Normal,
            Distribution::Normal,
        ],
        var_params: vec![0.3, 0.0, 0.0],
        interactions: vec![vec![2, 3], vec![1, 2]],
        effect_sizes: vec![0.0, 0.5, 0.25, 0.25, 0.22, 0.3],
        target_indices: vec![1, 2, 3, 4, 5],
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: 2.0,
            ..ScenarioPerturbations::optimistic()
        },
        ..common::base_spec()
    };
    s.het_coeffs = s.compute_het_coeffs();
    assert_driver_moments_match_generated(&s, 400_000, "binary x continuous");
}

#[test]
fn driver_moments_cover_factor_by_continuous_interactions() {
    // One standard-normal x1 and a 3-level factor (0.5/0.3/0.2) with both
    // dummies interacted with x1: [1, x1, d1, d2, d1:x1, d2:x1]. The two
    // dummies of one factor are never 1 together, so Cov(d1, d2) = −p₁p₂.
    let mut s = SimulationSpec {
        n_non_factor: 1,
        n_factor_dummies: 2,
        factor_n_levels: vec![3],
        factor_proportions: vec![0.5, 0.3, 0.2],
        interactions: vec![vec![2, 1], vec![3, 1]],
        effect_sizes: vec![0.0, 0.25, 0.5, 0.8, 0.3, -0.4],
        target_indices: vec![1, 2, 3, 4, 5],
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: 2.0,
            ..ScenarioPerturbations::optimistic()
        },
        ..common::base_spec()
    };
    s.het_coeffs = s.compute_het_coeffs();
    assert_driver_moments_match_generated(&s, 400_000, "factor x continuous");
}

#[test]
fn driver_moments_cover_uploaded_binary_interaction() {
    // NORTA-mode uploaded binary (p = 0.4) is the same 0/1 threshold marginal
    // as `Binary`, so its moments are mean p, SD sqrt(p(1 − p)), not 0/1.
    // Columns: [1, u, x, u:x], uncorrelated.
    let mut s = SimulationSpec {
        n_non_factor: 2,
        correlation: vec![1.0, 0.0, 0.0, 1.0],
        var_types: vec![Distribution::UploadedBinary, Distribution::Normal],
        var_params: vec![0.4, 0.0],
        interactions: vec![vec![1, 2]],
        effect_sizes: vec![0.0, 0.5, 0.25, 0.3],
        target_indices: vec![1, 2, 3],
        scenario: ScenarioPerturbations {
            heteroskedasticity_ratio: 2.0,
            ..ScenarioPerturbations::optimistic()
        },
        ..common::base_spec()
    };
    s.het_coeffs = s.compute_het_coeffs();
    assert_driver_moments_match_generated(&s, 400_000, "uploaded binary x continuous");
}
