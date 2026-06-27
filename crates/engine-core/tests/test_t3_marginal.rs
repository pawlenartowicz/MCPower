//! When a non-factor column uses DIST_HIGH_KURTOSIS (var_type=4),
//! the resulting column is drawn from the censored-t(3) marginal via the
//! lookup table, standardized at build time to the censored distribution's
//! own SD — so the empirical variance is ~1.0 and the excess kurtosis ~6.4
//! (the censored table's law; full t(3) kurtosis is infinite).
//! This test verifies that the DIST_HIGH_KURTOSIS path uses the table (not a
//! passthrough z): variance alone no longer discriminates (both give ~1.0),
//! so the passthrough detector is the excess kurtosis — a normal z would
//! give ~0, the table ~6.4.

use engine_core::data_gen::generate_sim_data;
use engine_core::marginals::T3PpfTable;
use engine_core::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};
use engine_core::workspace::SimWorkspace;

#[test]
fn high_kurtosis_marginal_has_unit_variance() {
    let spec = SimulationSpec {
        n_non_factor: 1,
        n_factor_dummies: 0,
        correlation: vec![1.0],
        var_types: vec![Distribution::HighKurtosis],
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
        effect_sizes: vec![0.0, 0.0],
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
        wald_se: Default::default(),
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.1,
        cluster: None,
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: Some(T3PpfTable::build_default()),
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        fit_columns: Vec::new(),
    };

    let n = 100_000;
    let mut ws = SimWorkspace::new(n, 2, 1, 0, None);
    generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

    let mut sum = 0.0;
    let mut sq = 0.0;
    let mut quart = 0.0;
    for i in 0..n {
        let xi = ws.x_full[(i, 1)];
        sum += xi;
        sq += xi * xi;
        quart += xi * xi * xi * xi;
    }
    let mean = sum as f64 / n as f64;
    let var = sq as f64 / n as f64 - mean * mean;
    // Table is standardized at build to the censored distribution's own SD.
    // MC band: SE(var̂) = √((exkurt+2)/n) ≈ 0.0095 at n=1e5 → ±0.05 is >5σ.
    assert!(
        (0.95..=1.05).contains(&var),
        "var={var}, expected ~1.0 (standardized censored t(3))."
    );
    // Passthrough detector: a normal z gives excess kurtosis ≈ 0; the
    // censored t(3) table gives ≈ 6.4. Variance no longer discriminates.
    let exkurt = quart as f64 / n as f64 / (var * var) - 3.0;
    assert!(
        exkurt > 3.0,
        "exkurt={exkurt}, expected ~6.4 (censored t(3)). \
         If exkurt≈0, the DIST_HIGH_KURTOSIS arm is passing z through."
    );
}
