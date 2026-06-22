//! Interaction power + intercept-absorption guards.
//!
//! Tests run through the full kernel path via `run_batch`, building
//! `SimulationSpec` directly (not via the spec-builder).
//!
//! Layout for 2-predictor interaction spec:
//!   col 0 = intercept, col 1 = x1, col 2 = x2, col 3 = x1:x2
//!   effect_sizes = [intercept(0.0), x1, x2, x1:x2]
//!   target_indices = [3]  (the interaction column)
//!
//! Layout for 3-predictor interaction spec:
//!   col 0 = intercept, col 1 = x1, col 2 = x2, col 3 = x3, col 4 = x1:x2:x3
//!   effect_sizes = [intercept(0.0), x1, x2, x3, x1:x2:x3]
//!   target_indices = [4]

use engine_core::run_batch;
use engine_core::spec::HeteroskedasticityCoeffs;
use engine_core::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};

/// Base spec: 2 continuous predictors + their 2-way interaction (4 kernel cols).
/// `effect_sizes` must have length 4: [intercept, x1, x2, x1:x2].
fn ols_interaction_spec(effect_sizes: Vec<f64>) -> SimulationSpec {
    SimulationSpec {
        n_non_factor: 2,
        n_factor_dummies: 0,
        // 2×2 identity correlation matrix (flat column-major)
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
        effect_sizes,
        // kernel col 3 = x1:x2 = 1 + n_non_factor(2) + n_factor_dummies(0) + 0
        target_indices: vec![3],
        contrast_pairs: vec![],
        interactions: vec![vec![1, 2]], // col 3 = col1 * col2
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
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 1.0,
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

/// Test 1: OLS interaction power is HIGH for a large effect.
///
/// With β_interaction=0.5 at N=400 (2 predictors + their interaction, σ≈1),
/// power should be well above 0.7.
#[test]
fn ols_interaction_power_is_high_for_large_effect() {
    let spec = ols_interaction_spec(vec![0.0, 0.3, 0.3, 0.5]);
    let n_sims: u32 = 600;
    let result = run_batch(&spec, &[400], n_sims, 42, None).unwrap();

    // shape: (n_sims=600, n_sample_sizes=1, n_targets=1) row-major → index=sim
    let n_targets = spec.target_indices.len(); // 1
    let n_sample_sizes = 1usize;
    assert_eq!(
        result.uncorrected.len(),
        n_sims as usize * n_sample_sizes * n_targets
    );

    let mut hits = 0u32;
    for sim in 0..n_sims as usize {
        if result.uncorrected[sim * n_sample_sizes * n_targets] == 1 {
            hits += 1;
        }
    }
    let power = hits as f64 / n_sims as f64;
    assert!(
        power > 0.7,
        "interaction power should be high for β=0.5 at N=400, got {power:.3}"
    );
}

/// Test 2: OLS NULL interaction (β=0) false-positive rate ≈ α=0.05.
///
/// For two standard-normal zero-mean predictors, E[x1·x2]=0 when uncorrelated,
/// so the intercept absorbs any non-zero product mean and the false-positive rate
/// is calibrated at α. A gross deviation (≫ 0.05) would indicate a bias bug in
/// the interaction column materialisation.
#[test]
fn ols_interaction_false_positive_rate_near_alpha() {
    let spec = ols_interaction_spec(vec![0.0, 0.3, 0.3, 0.0]);
    let n_sims: u32 = 2000;
    let result = run_batch(&spec, &[400], n_sims, 42, None).unwrap();

    let n_targets = spec.target_indices.len(); // 1
    let n_sample_sizes = 1usize;

    let mut hits = 0u32;
    for sim in 0..n_sims as usize {
        if result.uncorrected[sim * n_sample_sizes * n_targets] == 1 {
            hits += 1;
        }
    }
    let rate = hits as f64 / n_sims as f64;
    // Tolerance ±0.025 (≈ 5 SE at n=2000). Gross failure (e.g. rate ≫ 0.05)
    // would indicate the intercept is NOT absorbing the product mean — a real bug.
    // Do NOT widen this band without first verifying no systematic bias.
    assert!(
        (rate - 0.05).abs() < 0.025,
        "null interaction false-positive rate should be ≈0.05 (±0.025), got {rate:.4}"
    );
}

/// Test 3: Logit (GLM) interaction runs end-to-end and most fits converge.
///
/// Confirms the materialised interaction column flows through the Glm estimator
/// path without blowing up.
#[test]
fn logit_interaction_runs_and_converges() {
    let mut spec = ols_interaction_spec(vec![0.0, 0.5, 0.5, 0.5]);
    spec.outcome_kind = OutcomeKind::Binary;
    spec.estimator = EstimatorSpec::Glm;

    let n_sims: u32 = 300;
    let result = run_batch(&spec, &[500], n_sims, 42, None).unwrap();

    // converged shape: (n_sims × n_sample_sizes) = 300 × 1 = 300
    assert_eq!(result.converged.len(), n_sims as usize);

    let n_converged = result.converged.iter().filter(|&&c| c == 1).count();
    assert!(
        n_converged > 270,
        "most logit fits should converge (>270/300), got {n_converged}"
    );
}

/// Test 4: Three-way interaction x1:x2:x3 materialises and can be targeted.
///
/// Layout with n_non_factor=3, interactions=[[1,2,3]]:
///   col 0=intercept, col1=x1, col2=x2, col3=x3, col4=x1:x2:x3
///   target_indices=[4]
///
/// Asserts the result buffer has the right shape, confirming the column is
/// created and indexed correctly.
#[test]
fn three_way_interaction_materialises_and_targets() {
    // Start from the 2-predictor base and extend to 3 predictors.
    let mut spec = ols_interaction_spec(vec![0.0, 0.2, 0.2, 0.0]);
    spec.n_non_factor = 3;
    // 3×3 identity correlation matrix (flat column-major)
    spec.correlation = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    spec.var_types = vec![Distribution::Normal; 3];
    spec.var_params = vec![0.0; 3];
    // effect_sizes: [intercept(0), x1(0.2), x2(0.2), x3(0.2), x1:x2:x3(0.5)]
    spec.effect_sizes = vec![0.0, 0.2, 0.2, 0.2, 0.5];
    // col4 = product of kernel cols 1, 2, 3
    spec.interactions = vec![vec![1, 2, 3]];
    // target the interaction column: 1 + n_non_factor(3) + n_factor_dummies(0) + 0 = 4
    spec.target_indices = vec![4];

    let n_sims: u32 = 200;
    let result = run_batch(&spec, &[400], n_sims, 42, None).unwrap();

    // Buffer shape: n_sims=200, n_sample_sizes=1, n_targets=1 → 200 entries
    assert_eq!(
        result.uncorrected.len(),
        200,
        "3-way interaction buffer should have 200 entries (200 sims × 1 sample-size × 1 target)"
    );
    // Sanity: all entries are 0 or 1
    for &v in &result.uncorrected {
        assert!(v == 0 || v == 1, "unexpected value {v} in uncorrected");
    }

    // Discriminating power floor on the 3-way target (index 0 in the 1-target, 1-sample-size
    // buffer: sim * n_sample_sizes(1) * n_targets(1) + 0 + 0 = sim).
    // β=0.5 at N=400 with orthogonal normal predictors gives power near 1.0;
    // > 0.1 is deliberately conservative so a mutation that zero-fills the
    // interaction target (the ONLY 3-way path) is caught without pinning a brittle value.
    let hits = result.uncorrected.iter().filter(|&&v| v == 1).count();
    let power = hits as f64 / n_sims as f64;
    assert!(
        power > 0.1,
        "3-way interaction power should be well above zero for β=0.5 at N=400, got {power:.3}"
    );
}

/// Test 5 (LME smoke test): Interaction runs through the MLE estimator end-to-end.
///
/// Uses the 2-predictor interaction spec with a cluster structure (5 clusters,
/// tau²=0.25). Asserts the run completes, most fits converge, and the interaction
/// power estimate is in a sane (non-degenerate) range.
#[test]
fn lme_interaction_smoke_test() {
    let mut spec = ols_interaction_spec(vec![0.0, 0.5, 0.5, 0.5]);
    spec.estimator = EstimatorSpec::Mle;
    spec.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 5 },
        tau_squared: 0.25,
        slopes: vec![],
        extra_groupings: vec![],
    });

    let n_sims: u32 = 200;
    let result = run_batch(&spec, &[200], n_sims, 42, None).unwrap();

    // converged shape: (n_sims × n_sample_sizes) = 200 × 1 = 200
    assert_eq!(
        result.converged.len(),
        n_sims as usize,
        "LME converged buffer should have n_sims entries"
    );

    let n_converged = result.converged.iter().filter(|&&c| c == 1).count();
    assert!(
        n_converged > 150,
        "most LME fits should converge (>150/200), got {n_converged}"
    );

    // Discriminating floor: β=0.5 at n=200 with 5 clusters should produce
    // well above chance power; > 0.3 rules out a no-op estimator while
    // tolerating small-cluster variance without pinning a brittle golden.
    let n_targets = spec.target_indices.len(); // 1
    let n_sample_sizes = 1usize;
    let mut hits = 0u32;
    for sim in 0..n_sims as usize {
        if result.uncorrected[sim * n_sample_sizes * n_targets] == 1 {
            hits += 1;
        }
    }
    let power = hits as f64 / n_sims as f64;
    assert!(
        power > 0.3,
        "LME interaction power should be well above chance for β=0.5 at n=200, got {power:.3}"
    );
}
