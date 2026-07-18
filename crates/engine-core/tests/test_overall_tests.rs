//! Integration tests for the v1-parity overall (joint) significance flag.

use engine_core::run_batch;
use engine_core::spec::HeteroskedasticityCoeffs;
use engine_core::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};

fn ols_spec(n_non_factor: u32, effect_sizes: Vec<f64>) -> SimulationSpec {
    let p = n_non_factor as usize;
    let mut corr = vec![0.0; p * p];
    for j in 0..p {
        corr[j * p + j] = 1.0;
    }
    SimulationSpec {
        n_non_factor,
        n_factor_dummies: 0,
        correlation: corr,
        var_types: vec![Distribution::Normal; p],
        var_pinned: vec![],
        var_params: vec![0.0; p],
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
        outcome_kind: OutcomeKind::Continuous,
        link: None,
        estimator: EstimatorSpec::Ols,
        wald_se: Default::default(),
        nagq: 1,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 1.0,
        cluster: None,
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: true,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        extra_slope_cols: Vec::new(),
        fit_columns: Vec::new(),
    }
}

fn logit_spec(n_non_factor: u32, effect_sizes: Vec<f64>) -> SimulationSpec {
    let mut s = ols_spec(n_non_factor, effect_sizes);
    s.outcome_kind = OutcomeKind::Binary;
    s.estimator = EstimatorSpec::Glm;
    s
}

fn lme_spec() -> SimulationSpec {
    let mut s = ols_spec(1, vec![0.0, 0.5]);
    s.estimator = EstimatorSpec::Mle;
    s.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 5 },
        tau_squared: 0.25,
        slopes: vec![],
        extra_groupings: vec![],
    });
    s
}

#[test]
fn run_batch_ols_overall_buffer_shape_when_flag_on() {
    // report_overall=true must populate one 0/1 reject decision per sim. This is
    // the deterministic buffer-shape mechanic — the magnitude of the rejection
    // rate is a statistical property (recorded as L3 seed, not asserted at L1/L2).
    let spec = ols_spec(3, vec![0.0, 0.3, 0.3, 0.3]);
    let n_sims: u32 = 400;
    let result = run_batch(&spec, &[200], n_sims, 42, None).unwrap();
    assert_eq!(result.overall.len(), n_sims as usize);
    for (i, &b) in result.overall.iter().enumerate() {
        assert!(
            b == 0 || b == 1,
            "overall[{i}] must be a 0/1 decision, got {b}"
        );
    }
}

#[test]
fn run_batch_overall_empty_when_flag_off() {
    let mut spec = ols_spec(3, vec![0.0, 0.5, 0.5, 0.5]);
    spec.report_overall = false;
    let result = run_batch(&spec, &[200], 100, 42, None).unwrap();
    assert!(
        result.overall.is_empty(),
        "report_overall=false must produce empty overall Vec; got len={}",
        result.overall.len()
    );
}

#[test]
fn run_batch_logit_overall_buffer_shape_and_nonconverged_zero() {
    // Logit overall buffer shape (one 0/1 decision per sim) plus the deterministic
    // invariant that a non-converged sim always emits 0. The rejection-rate
    // magnitude is statistical (L3 seed), not asserted here.
    let intercept = (0.3_f64 / 0.7).ln();
    let spec = logit_spec(2, vec![intercept, 1.0, 0.0]);
    let n_sims: u32 = 300;
    let result = run_batch(&spec, &[400], n_sims, 42, None).unwrap();
    assert_eq!(result.overall.len(), n_sims as usize);
    for (i, &b) in result.overall.iter().enumerate() {
        assert!(
            b == 0 || b == 1,
            "overall[{i}] must be a 0/1 decision, got {b}"
        );
        if result.converged[i] == 0 {
            assert_eq!(b, 0, "non-converged Logit sim {i} must emit 0");
        }
    }
}

#[test]
fn run_batch_lme_overall_always_zero() {
    let spec = lme_spec();
    let result = run_batch(&spec, &[50], 30, 42, None).unwrap();
    assert_eq!(result.overall.len(), 30);
    for &b in &result.overall {
        assert_eq!(b, 0, "LME must emit 0 in overall regardless of effect size");
    }
}
