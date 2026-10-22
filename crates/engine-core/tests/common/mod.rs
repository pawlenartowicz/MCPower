//! Shared fixtures for the engine-core integration tests.

use engine_core::spec::{
    CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
    OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};

/// Baseline kernel spec, identical to the unit tests' `test_support::base_spec`:
/// intercept + one standard-normal predictor (β₁ = 0.5, the single target),
/// OLS on a continuous outcome, optimistic scenario, no factors, clusters,
/// uploads, interactions or posthoc. Tests state only what they change:
/// `SimulationSpec { n_non_factor: 2, ..common::base_spec() }`.
pub fn base_spec() -> SimulationSpec {
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
        factor_sampled: vec![],
        effect_sizes: vec![0.0, 0.5],
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
        link: None,
        estimator: EstimatorSpec::Ols,
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.1,
        cluster: None,
        scenario: ScenarioPerturbations::optimistic(),
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        extra_slope_cols: vec![],
        fit_columns: vec![],
        curvature_hints: None,
    }
}
