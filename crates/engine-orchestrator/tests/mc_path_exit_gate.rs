//! For each of two contracts (Glm, Mle), runs `find_power` twice with
//! identical inputs but different rayon thread-pool sizes (1 vs auto).
//! Asserts the results are bit-equal — proving the worker-count invariance
//! contract that every port relies on.

mod common;

use common::{lme_spec, logit_spec};
use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec, OutcomeKind, SimulationContract};
use engine_orchestrator::{
    find_power, CancellationToken, EstimatorExtras, PowerResult, ScenarioResult,
};
use engine_spec_builder::build_contract;

const N: usize = 200;
const N_SIMS: usize = 800;
const BASE_SEED: u64 = 2137;

fn run_with(
    contracts: &[SimulationContract],
    n_threads: Option<usize>,
) -> ScenarioResult<PowerResult> {
    let cancel = CancellationToken::new();
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = n_threads {
        builder = builder.num_threads(n);
    }
    let pool = builder.build().expect("rayon pool");
    pool.install(|| {
        find_power(contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok")
    })
}

/// Bit-equal comparison that doesn't trip on NaN self-inequality.
///
/// `f64::PartialEq` propagates IEEE-754's `NaN != NaN`, which makes a direct
/// `assert_eq!` on `ScenarioResult` spuriously fail whenever a result field
/// (e.g. `Glm::baseline_prob_realized` or `Mle::tau_estimate`) is
/// `NaN` in both runs. The orchestrator's *contract* surface is the msgpack
/// byte encoding, and msgpack encodes NaN to a
/// fixed bit pattern — so encoding both runs and comparing the byte vectors
/// is the true "bit-equal across worker counts" check.
fn assert_bit_equal_across_workers(contracts: Vec<SimulationContract>) {
    let r_auto = run_with(&contracts, None);
    let r_one = run_with(&contracts, Some(1));
    let bytes_auto = rmp_serde::to_vec(&r_auto).expect("encode r_auto");
    let bytes_one = rmp_serde::to_vec(&r_one).expect("encode r_one");
    assert_eq!(
        bytes_auto, bytes_one,
        "find_power must be bit-equal across worker counts (num_threads=auto vs 1) — \
         msgpack-encoded ScenarioResult bytes differ"
    );
    assert_eq!(r_auto.scenarios.len(), 1);
}

#[test]
fn mc_path_exit_gate_glm() {
    let contracts = build_contract(&logit_spec(), OutcomeKind::Binary, None, -0.5, vec![])
        .expect("build_contract glm");
    // Sanity-check the estimator_extras variant before paying for the full
    // two-pool run.
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let (_, pr) = &r.scenarios[0];
    assert!(
        matches!(pr.estimator_extras, EstimatorExtras::Glm { .. }),
        "expected Glm estimator_extras, got {:?}",
        pr.estimator_extras
    );

    assert_bit_equal_across_workers(contracts);
}

/// GLMM end-to-end: a clustered-logit contract (Glm + random intercept) routes
/// through `fit_glmm` (Task 6) and surfaces `EstimatorExtras::Glm` with a valid
/// `singular_fit_rate` and a finite `tau_squared_hat_mean` (the engine half of
/// the §5 Laplace-bias warning). Guards the batch.rs dispatch + the
/// `from_batch` Glm aggregation wiring.
#[test]
fn find_power_clustered_logit_surfaces_glm_extras() {
    let contracts = build_contract(
        &logit_spec(),
        OutcomeKind::Binary,
        None, // Binary ⇒ estimator defaults to Glm; cluster present ⇒ GLMM path
        -0.5,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract clustered glm");

    // Smaller workload than the bit-equality gates above: GLMM runs a joint
    // BOBYQA per (sim, N), so 64 sims at N=120 keeps this fast in debug while
    // still converging enough fits to populate the extras.
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, 120, 64, BASE_SEED, None, &cancel).expect("find_power ok");
    let (_, pr) = &r.scenarios[0];
    let EstimatorExtras::Glm {
        singular_fit_rate,
        singular_n,
        tau_squared_hat_mean,
        tau_squared_hat_n,
        ..
    } = &pr.estimator_extras
    else {
        panic!("expected Glm estimator_extras, got {:?}", pr.estimator_extras);
    };
    assert!(*singular_n > 0, "some GLMM fits must converge");
    assert!(
        (0.0..=1.0).contains(singular_fit_rate),
        "singular_fit_rate out of range: {singular_fit_rate}"
    );
    assert!(*tau_squared_hat_n > 0, "converged fits must contribute τ̂²");
    assert!(
        tau_squared_hat_mean.is_finite(),
        "converged ⇒ finite mean τ̂², got {tau_squared_hat_mean}"
    );
}

#[test]
fn mc_path_exit_gate_mle() {
    let contracts = build_contract(
        &lme_spec(),
        OutcomeKind::Continuous,
        Some(EstimatorSpec::Mle),
        0.0,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract mle");

    let cancel = CancellationToken::new();
    let r = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let (_, pr) = &r.scenarios[0];
    assert!(
        matches!(pr.estimator_extras, EstimatorExtras::Mle { .. }),
        "expected Mle estimator_extras, got {:?}",
        pr.estimator_extras
    );

    assert_bit_equal_across_workers(contracts);
}
