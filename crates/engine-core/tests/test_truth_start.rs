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

use engine_core::batch::{optim_diag, run_batch_st};
use engine_core::introspect::{IntrospectMask, fit_provided_data, run_introspect};
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
        BASE_SEED,
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
        BASE_SEED,
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

/// The same case with a random slope on x1, which routes the hot loop through
/// the general LMM fit (the `lme` Brent path never reads a start).
fn slope_spec(truth_start: bool) -> SimulationSpec {
    let mut spec = mixed_spec(truth_start);
    spec.cluster.as_mut().unwrap().slopes = vec![engine_contract::SlopeTerm {
        column: engine_contract::ColumnId(0),
        variance: 0.10,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    // x_full col 1 = x1 (col 0 = intercept).
    spec.cluster_slope_design_cols = vec![1];
    spec
}

/// BOBYQA objective evaluations and fit count a single-N, single-threaded
/// `run_batch_st` spends. Process-wide counters, diffed around the run; this is
/// the only `run_batch_st` call in this test binary, so no other test feeds them.
fn evals_and_fits(spec: &SimulationSpec) -> (u64, u64) {
    let before = optim_diag::snapshot();
    run_batch_st(spec, &[400], 20, BASE_SEED, None).expect("run_batch_st");
    let after = optim_diag::snapshot();
    (after[0] - before[0], after[1] - before[1])
}

/// Hot-loop gate: at a single grid point (no carried θ̂) the fit starts from the
/// DGP truth only when `scenario.truth_start` is set, so the two settings must
/// cost a different number of objective evaluations over the same sims. Equal
/// counts mean the start was dropped before the fit.
#[test]
fn truth_start_gate_reaches_the_hot_loop_fit() {
    let (cold_evals, cold_fits) = evals_and_fits(&slope_spec(false));
    let (warm_evals, warm_fits) = evals_and_fits(&slope_spec(true));
    assert!(
        cold_fits > 0 && warm_fits > 0,
        "the slope LMM must be fitted"
    );
    assert_ne!(
        cold_evals, warm_evals,
        "truth_start changed nothing in the hot loop ({cold_evals} evals both ways)"
    );
}
