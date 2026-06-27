//! Per-study heterogeneity imposes a hard power ceiling at ≈ Φ(1/h).
//!
//! Heterogeneity h draws one effect shift δ ~ N(0, (h·|β|)²) PER SIMULATION (one
//! realized study per sim), clipped so a realized effect is pulled toward 0 but
//! never across it. Because the draw is per study, not per observation, it does
//! NOT average out: at any N the test can only detect studies whose realized
//! effect stayed non-zero, so power plateaus at
//!   Φ(1/h) + (1 − Φ(1/h))·α     (clipped studies are true nulls, reject at α)
//! independent of N — not climbing to 1.0. This is the behaviour the
//! `2026-06-08-heterogeneity-per-study-spec` change introduced; it mirrors the
//! Python probes in that plan's `notes/heterogeneity_ceiling_test*.py`.
//!
//! The two statistical tests are `#[ignore]`d — they are L3 validation (large
//! N × many sims, ~100 s in debug) and run on request, not per commit:
//!   cargo test -p engine-core --release --test test_heterogeneity_ceiling -- --ignored
//! The discriminator is the plateau: power ≈ Φ(1/h) < 1 and FLAT between N = 5000
//! and N = 20000, where a per-observation model would instead sit at 1.0.
//! `het_zero_is_clean_signal` is cheap (pure generation) and stays in the suite.

use engine_core::batch::run_batch;
use engine_core::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use engine_core::workspace::SimWorkspace;

const ALPHA: f64 = 0.05;
const SEED: u64 = 2137;
const N_SIMS: u32 = 3000;

/// One intercept (β₀ = 0) + one continuous predictor with slope `beta_x1`,
/// identity correlation, OLS — the single-effect design the ceiling is derived
/// for. `het` is the scenario heterogeneity knob (τ = het·|β|).
fn ols_spec(beta_x1: f64, het: f64) -> SimulationSpec {
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
        effect_sizes: vec![0.0, beta_x1],
        target_indices: vec![1],
        contrast_pairs: vec![],
        interactions: vec![],
        correction_method: CorrectionMethod::None,
        crit_values: CritValues {
            alpha: ALPHA,
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
        scenario: ScenarioPerturbations {
            heterogeneity: het,
            ..ScenarioPerturbations::optimistic()
        },
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    }
}

/// Uncorrected power of the single target at sample size `n`. `uncorrected` is
/// shape (n_sims, n_sample_sizes, n_targets) row-major; with one N and one
/// target its length is `n_sims`, so the mean over bytes is the power.
fn power_at(spec: &SimulationSpec, n: u32) -> f64 {
    let r = run_batch(spec, &[n], N_SIMS, SEED, None).expect("run_batch must succeed");
    let hits: u32 = r.uncorrected.iter().map(|&b| b as u32).sum();
    hits as f64 / N_SIMS as f64
}

// ---------------------------------------------------------------------------
// Ceiling: power ≈ Φ(1/h), flat in N. β = 0.5 (huge effect: optimistic would be
// saturated at 1.0 at both N, so any value below 1 is the heterogeneity ceiling,
// not estimation noise). Φ(1/h) hardcoded from the spec table.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "L3 statistical validation — large N × many sims; run on request in --release"]
fn ceiling_at_phi_inv_h() {
    // (h, Φ(1/h), abs tolerance on the ceiling). Tolerance covers the MC SE
    // (~4σ at N_SIMS) plus the finite-N residual; it widens as the ceiling
    // drops because Bernoulli variance peaks away from 1.
    let cases = [
        (0.4_f64, 0.993_790_f64, 0.012_f64), // 1/h = 2.5
        (0.5, 0.977_250, 0.016),             // 1/h = 2.0
        (1.0, 0.841_345, 0.030),             // 1/h = 1.0
    ];

    for (h, phi, tol) in cases {
        let spec = ols_spec(0.5, h);
        let p_lo = power_at(&spec, 5_000);
        let p_hi = power_at(&spec, 20_000);

        // Clipped studies (would-be sign-flips) become true nulls → reject at α,
        // nudging the ceiling above Φ(1/h) by (1−Φ)·α. Compare to that refined value.
        let ceiling = phi + (1.0 - phi) * ALPHA;
        assert!(
            (p_hi - ceiling).abs() < tol,
            "h={h}: power at N=20000 ({p_hi:.4}) should sit at the Φ(1/h) ceiling {ceiling:.4} (±{tol})"
        );

        // The hard-ceiling discriminator: power does NOT climb toward 1.0 with N.
        // A per-observation model would average the jitter away and reach ~1.0 by
        // N≈1000; here both N sit on the same plateau (CRN-paired studies, the δ
        // stream is keyed on sim_id only, so study s has one realized effect at
        // both N). |Δ| is bounded by the two-sample MC noise, not a real climb.
        assert!(
            (p_hi - p_lo).abs() < 0.02,
            "h={h}: power must be flat in N (no averaging-out): N=5000 {p_lo:.4} vs N=20000 {p_hi:.4}"
        );

        // For the lower ceilings the plateau is unmistakably below 1.0 — a real
        // ceiling, not near-saturation. (h=0.4's 0.994 is too close to 1 to assert
        // this cleanly at this n_sims; its ceiling is pinned by the band above.)
        if h >= 0.5 {
            assert!(
                p_hi < 0.99,
                "h={h}: power {p_hi:.4} must stay strictly below 1.0 — the ceiling is structural"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Null calibration is untouched: a declared null (β = 0 ⇒ sⱼ = 0 ⇒ δ = 0,
// never clipped) stays a true null and rejects at ≈ α even with heterogeneity
// on. The clip only ever fires on a non-zero effect, so it cannot inflate the
// null rejection rate — this guards that.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "L3 statistical validation — large N × many sims; run on request in --release"]
fn null_target_rejects_at_alpha() {
    let spec = ols_spec(0.0, 0.5); // null effect, heterogeneity active
    let p = power_at(&spec, 20_000);
    // SE ≈ sqrt(α(1−α)/n_sims) ≈ 0.004; ±0.015 is ~4 SE.
    assert!(
        (p - ALPHA).abs() < 0.015,
        "null target with heterogeneity on must reject at ≈ α = {ALPHA}; got {p:.4}"
    );
}

// ---------------------------------------------------------------------------
// h = 0 is the untouched clean-signal path. With heterogeneity off the per-study
// stream is never created and lp = lp_clean, so y is exactly Xβ + ε (+ u_re = 0,
// no cluster) — no jitter leaks in. The cross-version byte-identity guarantee
// (h = 0 == pre-change) rests on this plus the unchanged main RNG stream and is
// covered by `golden_rng.rs` and the data_gen h = 0 goldens; this pins the local
// invariant that the het machinery does not perturb the h = 0 signal.
// ---------------------------------------------------------------------------
#[test]
fn het_zero_is_clean_signal() {
    const ROW_N: usize = 256;
    let spec = ols_spec(0.5, 0.0);
    let n_pred = 1 + spec.n_non_factor as usize;
    let mut ws = SimWorkspace::new(ROW_N, n_pred, spec.n_non_factor as usize, 0, None);

    for sim_id in 0..16u64 {
        engine_core::data_gen::generate_sim_data(&spec, sim_id, SEED, &mut ws).unwrap();
        for i in 0..ROW_N {
            // Replicate data_gen's f64 accumulation order exactly: Σⱼ xᵢⱼ·βⱼ,
            // then + residual (+ 0 for the absent random effect), cast once to f32.
            let mut lp = 0.0_f64;
            for j in 0..n_pred {
                lp += ws.x_full[(i, j)] as f64 * spec.effect_sizes[j];
            }
            let y_clean = (lp + ws.residuals[i] as f64) as f32;
            assert_eq!(
                ws.y_full[i], y_clean,
                "h=0 must be the clean signal Xβ+ε (no jitter) at sim {sim_id}, row {i}"
            );
        }
    }
}
