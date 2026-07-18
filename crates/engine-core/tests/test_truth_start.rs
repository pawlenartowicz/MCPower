//! `truth_start` scenario gate at the fit sites: a mixed-model fit takes the
//! blind cold start when `scenario.truth_start` is false and the DGP-truth warm
//! start when it is true. Both must converge (the warm start is a valid start,
//! not a shortcut that corrupts the answer). Power-equality is NOT asserted —
//! BOBYQA is start-sensitive, so warm and cold may settle in different optima;
//! that start-sensitivity is the whole point of the assumption.
//!
//! The captured data is identical across both settings: `truth_start` gates only
//! the optimizer's starting θ, never data generation. So the debug refit runs
//! the same bytes twice, differing only in the θ hint.

use engine_core::introspect::{fit_provided_data, run_introspect, IntrospectMask};
use engine_core::spec::{
    ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, Distribution, EstimatorSpec,
    HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
};

const BASE_SEED: u64 = 2137;

/// A single random-intercept LMM case (Mle + FixedClusters), τ² = 0.25.
fn mixed_spec(truth_start: bool) -> SimulationSpec {
    let k = 3usize;
    let mut correlation = vec![0.0; k * k];
    for i in 0..k {
        correlation[i * k + i] = 1.0;
    }
    let mut effect_sizes = vec![0.0; 1 + k];
    effect_sizes[1] = 0.25;
    effect_sizes[2] = 0.25;
    let mut scenario = ScenarioPerturbations::optimistic();
    scenario.truth_start = truth_start;
    SimulationSpec {
        n_non_factor: k as u32,
        n_factor_dummies: 0,
        correlation,
        var_types: vec![Distribution::Normal; k],
        var_pinned: vec![],
        var_params: vec![0.0; k],
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
        link: None,
        estimator: EstimatorSpec::Mle,
        wald_se: Default::default(),
        nagq: 1,
        intercept: 0.0,
        posthoc: vec![],
        max_failed_fraction: 0.25,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
        }),
        scenario,
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall: false,
        factor_min_level_count: 0,
        cluster_slope_design_cols: vec![],
        extra_slope_cols: Vec::new(),
        fit_columns: Vec::new(),
    }
}

/// Capture one dataset (data is truth_start-independent) and refit it through the
/// introspect (debug) path under both settings; both converge, betas finite.
#[test]
fn truth_start_gate_converges_both_ways() {
    let cold_spec = mixed_spec(false);
    // Data generation ignores truth_start, so capture once from the cold spec.
    let out = run_introspect(
        &cold_spec,
        400,
        1,
        BASE_SEED,
        IntrospectMask {
            stats: false,
            data: true,
            crit: true,
            power: false,
        },
    )
    .expect("introspect");
    let d = out.data.expect("data capture");

    let cold = fit_provided_data(
        &cold_spec,
        &d.design,
        d.nrow,
        d.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .expect("cold fit");
    assert!(cold.converged, "cold (truth_start=false) fit must converge");
    assert!(
        cold.betas.iter().all(|b| b.is_finite()),
        "cold betas finite"
    );

    let warm_spec = mixed_spec(true);
    let warm = fit_provided_data(
        &warm_spec,
        &d.design,
        d.nrow,
        d.ncol,
        &d.outcome,
        d.cluster_ids.as_deref(),
    )
    .expect("warm fit");
    assert!(warm.converged, "warm (truth_start=true) fit must converge");
    assert!(
        warm.betas.iter().all(|b| b.is_finite()),
        "warm betas finite"
    );
}
