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

mod common;

use engine_core::introspect::{fit_provided_data, run_introspect, IntrospectMask};
use engine_core::spec::{
    ClusterSizing, ClusterSpec, Distribution, EstimatorSpec, ScenarioPerturbations, SimulationSpec,
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
        correlation,
        var_types: vec![Distribution::Normal; k],
        var_params: vec![0.0; k],
        effect_sizes,
        target_indices: vec![1, 2],
        estimator: EstimatorSpec::Mle,
        max_failed_fraction: 0.25,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
        }),
        scenario,
        ..common::base_spec()
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

    // The gate must reach the optimizer: the θ search is start-sensitive, so
    // refitting the same bytes from the truth start and from the blind start
    // lands on different low bits somewhere in β̂ or SE. Identical bits mean
    // the hint was dropped before the fit.
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(
        bits(&cold.betas) != bits(&warm.betas) || bits(&cold.se) != bits(&warm.se),
        "truth_start had no effect on the refit: cold and warm results are bit-identical"
    );
}
