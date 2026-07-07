//! Scenario `heterogeneity` (β-jitter) resolution and the binary extension.
//!
//! Pins the additive resolution `het_total = (baseline + scenario).max(0)`:
//! the scenario knob alone (baseline 0, what every host sends) must perturb
//! y — under the old multiplier formula `baseline × max(1 + scenario, 1)` it
//! was a silent no-op (the v1-parity bug). Also pins the v2 binary extension:
//! β-jitter fires for Bernoulli outcomes as log-odds jitter on the effects
//! plus odds-scale jitter on the intercept (SD = het, NOT het·|β₀|), while
//! continuous outcomes keep the intercept excluded (v1 parity).
//!
//! RNG-stream discipline (jitter must not consume X/residual draws) is pinned
//! by `test_heterogeneity_lme.rs`; the stream comparisons here lean on it.

use engine_core::batch::run_batch;
use engine_core::data_gen::generate_sim_data;
use engine_core::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use engine_core::workspace::SimWorkspace;

const ROW_N: usize = 400;
const N_SIMS: u64 = 200;
const BASE_SEED: u64 = 2137;

/// 1 intercept + 1 continuous predictor, identity correlation. With a single
/// predictor the scenario block path adds zero correlation noise, so X and
/// residual streams stay bit-identical to the optimistic fast path — any y
/// difference is the jitter.
fn base_spec(outcome_kind: OutcomeKind, effect_sizes: Vec<f64>) -> SimulationSpec {
    let intercept = effect_sizes[0];
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
        effect_sizes,
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
        outcome_kind,
        estimator: match outcome_kind {
            OutcomeKind::Continuous => EstimatorSpec::Ols,
            OutcomeKind::Binary => EstimatorSpec::Glm,
        },
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
        extra_slope_cols: Vec::new(),
        fit_columns: Vec::new(),
    }
}

/// A scenario whose ONLY live knob is `heterogeneity` — non-optimistic when
/// het != 0, with zero X-side and residual perturbations.
fn het_only_scenario(het: f64) -> ScenarioPerturbations {
    ScenarioPerturbations {
        name: "custom".into(),
        heterogeneity: het,
        ..Default::default()
    }
}

/// Generate y for all sims under `spec`; returns (x_col1, residuals, y) per sim
/// flattened row-major.
fn collect_streams(spec: &SimulationSpec) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_pred = 1 + spec.n_non_factor as usize;
    let mut ws = SimWorkspace::new(ROW_N, n_pred, spec.n_non_factor as usize, 0, None);
    let mut xs = Vec::with_capacity(ROW_N * N_SIMS as usize);
    let mut res = Vec::with_capacity(ROW_N * N_SIMS as usize);
    let mut ys = Vec::with_capacity(ROW_N * N_SIMS as usize);
    for sim_id in 0..N_SIMS {
        generate_sim_data(spec, sim_id, BASE_SEED, &mut ws).unwrap();
        for i in 0..ROW_N {
            xs.push(ws.x_full[(i, 1)] as f64);
            res.push(ws.residuals[i] as f64);
            ys.push(ws.y_full[i] as f64);
        }
    }
    (xs, res, ys)
}

// ---------------------------------------------------------------------------
// Dead-path regression (the v1-parity bug): scenario heterogeneity alone —
// baseline 0.0, exactly what every host sends — must change y. Under the old
// `spec.heterogeneity × max(1 + scenario.heterogeneity, 1)` formula this was
// 0 × 1.2 = 0: a silent no-op.
// ---------------------------------------------------------------------------
#[test]
fn scenario_heterogeneity_alone_perturbs_y() {
    let clean = base_spec(OutcomeKind::Continuous, vec![0.0, 0.5]);
    let mut jittered = clean.clone();
    jittered.scenario = het_only_scenario(0.2);

    let (x_a, r_a, y_a) = collect_streams(&clean);
    let (x_b, r_b, y_b) = collect_streams(&jittered);

    // Same X and residual streams (jitter is y-only; single predictor means
    // the scenario block path reproduces the optimistic design bit-for-bit).
    assert_eq!(
        x_a, x_b,
        "X stream must not depend on scenario heterogeneity"
    );
    assert_eq!(
        r_a, r_b,
        "residual stream must not depend on scenario heterogeneity"
    );

    // ... but y must move: count differing rows across all sims.
    let n_diff = y_a.iter().zip(&y_b).filter(|(a, b)| a != b).count();
    assert!(
        n_diff > (y_a.len() * 9) / 10,
        "scenario heterogeneity alone must perturb y (dead-knob regression); \
         only {n_diff}/{} rows differ",
        y_a.len()
    );
}

// ---------------------------------------------------------------------------
// Clamp at zero: a negative scenario value with baseline 0 resolves to 0 —
// bit-identical y to the optimistic no-jitter run.
// ---------------------------------------------------------------------------
#[test]
fn negative_scenario_heterogeneity_clamps_to_zero() {
    let clean = base_spec(OutcomeKind::Continuous, vec![0.0, 0.5]);
    let mut negative = clean.clone();
    negative.scenario = het_only_scenario(-0.5);

    let (_, _, y_clean) = collect_streams(&clean);
    let (_, _, y_neg) = collect_streams(&negative);
    assert_eq!(
        y_clean, y_neg,
        "(0 + -0.5).max(0) = 0: negative scenario heterogeneity must be a no-jitter run"
    );
}

// ---------------------------------------------------------------------------
// Binary extension: β-jitter fires for Bernoulli outcomes. Same X and uniform
// streams, but the jittered lp flips Bernoulli outcomes near the threshold.
// ---------------------------------------------------------------------------
#[test]
fn binary_jitter_fires_for_scenario_heterogeneity() {
    let p = 0.3_f64;
    let intercept = (p / (1.0 - p)).ln();
    let clean = base_spec(OutcomeKind::Binary, vec![intercept, 0.5]);
    let mut jittered = clean.clone();
    jittered.scenario = het_only_scenario(0.4);

    let (x_a, u_a, y_a) = collect_streams(&clean);
    let (x_b, u_b, y_b) = collect_streams(&jittered);

    assert_eq!(
        x_a, x_b,
        "X stream must not depend on scenario heterogeneity"
    );
    assert_eq!(
        u_a, u_b,
        "uniform stream must not depend on scenario heterogeneity"
    );

    let n_diff = y_a.iter().zip(&y_b).filter(|(a, b)| a != b).count();
    assert!(
        n_diff > 0,
        "binary β-jitter must flip some Bernoulli outcomes (got 0 flips in {} draws)",
        y_a.len()
    );
    // Sanity: flips are a minority — jitter perturbs, it doesn't scramble.
    assert!(
        n_diff < y_a.len() / 2,
        "binary β-jitter flipped {n_diff}/{} rows — implausibly many",
        y_a.len()
    );
}

// ---------------------------------------------------------------------------
// Binary intercept jitter is odds-scale: SD = het (NOT het·|β₀|). With all
// effect coefficients at 0, the per-study intercept draw gives every row in a
// sim the same p = sigmoid(β₀ + het·Z) (Z drawn once per sim), so the event
// rate averages sigmoid(β₀ + het·Z) over the N_SIMS study draws and must match
// the Gauss-quadrature value of E[sigmoid(β₀ + het·Z)] — which differs from
// sigmoid(β₀) by the Jensen shift, and from the |β₀|-scaled variant by far more
// than the sampling error.
// ---------------------------------------------------------------------------
#[test]
fn binary_intercept_jitter_is_odds_scale() {
    let p0 = 0.1_f64;
    let b0 = (p0 / (1.0 - p0)).ln(); // ≈ -2.197
    let het = 1.0_f64; // engine-level capability; hosts cap via configs

    let mut spec = base_spec(OutcomeKind::Binary, vec![b0, 0.0]);
    spec.scenario = het_only_scenario(het);
    let (_, _, ys) = collect_streams(&spec);
    let rate = ys.iter().sum::<f64>() / ys.len() as f64;

    // E[sigmoid(b0 + het·Z)] by simple quadrature over Z ∈ [-8, 8].
    let mut expected = 0.0;
    let mut mass = 0.0;
    let step = 1e-3;
    let mut z = -8.0_f64;
    while z <= 8.0 {
        let w = (-z * z / 2.0).exp();
        expected += w / (1.0 + (-(b0 + het * z)).exp());
        mass += w;
        z += step;
    }
    expected /= mass;

    // Per-study draw: the intercept shift is one draw per sim, so the rate's
    // sampling error is set by the N_SIMS=200 *study* draws, not the 80k rows —
    // SE ≈ SD_Z[sigmoid(β₀+het·Z)]/√200 ≈ 0.008 (the per-row Bernoulli SE ≈ 0.001
    // is negligible beside it). ±0.03 is ~3.7 of those SEs; the wrong |β₀|-scaled
    // variant sits 0.083 away, so this band still cleanly rejects it.
    assert!(
        (rate - expected).abs() < 0.03,
        "empirical rate {rate:.4} should match odds-scale E[sigmoid(β₀+het·Z)] = {expected:.4}"
    );
    // And the Jensen shift must be visible: rate well above sigmoid(β₀) = 0.1.
    assert!(
        rate > p0 + 0.01,
        "intercept jitter must shift the event rate above {p0} (got {rate:.4})"
    );
}

// ---------------------------------------------------------------------------
// Continuous intercept stays clean (v1 parity): with a nonzero intercept and
// zero effect coefficients, jitter scales are all zero — y is bit-identical
// to the no-jitter run even though the jitter draws are consumed.
// ---------------------------------------------------------------------------
#[test]
fn continuous_intercept_jitter_is_excluded() {
    let clean = base_spec(OutcomeKind::Continuous, vec![0.7, 0.0]);
    let mut jittered = clean.clone();
    jittered.scenario = het_only_scenario(0.4);

    let (_, _, y_clean) = collect_streams(&clean);
    let (_, _, y_jit) = collect_streams(&jittered);
    assert_eq!(
        y_clean, y_jit,
        "continuous intercept must not be jittered (effects-only, v1 parity)"
    );
}

// ---------------------------------------------------------------------------
// End-to-end: logit + doomer runs and lands below optimistic power for a
// medium-effect case (behavior-constraining, not "runs without error").
// ---------------------------------------------------------------------------
#[test]
fn binary_doomer_power_below_optimistic() {
    let p = 0.3_f64;
    let intercept = (p / (1.0 - p)).ln();
    let optimistic = base_spec(OutcomeKind::Binary, vec![intercept, 0.5]);

    let mut doomer = optimistic.clone();
    // configs/scenarios.json doomer values (residual knobs inert for binary).
    doomer.scenario = ScenarioPerturbations {
        name: "doomer".into(),
        heterogeneity: 0.4,
        heteroskedasticity_ratio: 4.0,
        correlation_noise_sd: 0.3,
        distribution_change_prob: 0.8,
        new_distributions: vec![
            Distribution::RightSkewed,
            Distribution::LeftSkewed,
            Distribution::Uniform,
        ],
        residual_change_prob: 0.8,
        residual_dists: vec![ResidualDist::HighKurtosis, ResidualDist::RightSkewed],
        residual_df: 5.0,
        sampled_factor_proportions: true,
        lme: None,
    };

    let n_sims = 1000u32;
    let r_opt = run_batch(&optimistic, &[150], n_sims, 42, None).expect("optimistic must run");
    let r_doom = run_batch(&doomer, &[150], n_sims, 42, None).expect("doomer must run");

    let count = |r: &engine_core::spec::BatchResult| -> u32 {
        r.uncorrected.iter().map(|&b| b as u32).sum()
    };
    let (c_opt, c_doom) = (count(&r_opt), count(&r_doom));
    assert!(
        c_doom < c_opt,
        "doomer power ({c_doom}/{n_sims}) must be below optimistic ({c_opt}/{n_sims})"
    );
}
