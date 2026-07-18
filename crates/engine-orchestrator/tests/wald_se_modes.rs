//! End-to-end gate for the `WaldSe` mode switch on the clustered-binary GLMM.
//!
//! `WaldSe::Hessian` (the default) sources the per-fit Wald SE from the
//! FD-Hessian covariance (glmer `use.hessian = TRUE`), which is ≥ the
//! `WaldSe::Rx` Schur SE — so the Wald test is at least as conservative and
//! Hessian power must NOT exceed Rx power. Both runs share data/seed, so the
//! comparison is paired (only the SE denominator differs).

mod common;
use common::logit_spec;

use engine_contract::{ClusterSizing, ClusterSpec, OutcomeKind, SimulationContract, WaldSe};
use engine_orchestrator::{find_power, CancellationToken};
use engine_spec_builder::build_contract;

/// Minimal clustered-binary GLMM contract (random intercept, 20 clusters,
/// ICC≈0.3) under the requested Wald-SE mode.
fn small_glmm_contract(wald_se: WaldSe) -> SimulationContract {
    let mut contracts = build_contract(
        &logit_spec(),
        OutcomeKind::Binary, // Binary ⇒ Glm; cluster present ⇒ GLMM path
        None,
        None,
        -0.5,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract clustered glm");
    let mut c = contracts.remove(0);
    c.wald_se = wald_se;
    c
}

#[test]
fn glmm_find_power_hessian_more_conservative_than_rx() {
    let cancel = CancellationToken::new();

    let p_rx = find_power(
        &[small_glmm_contract(WaldSe::Rx)],
        120,
        200,
        2137,
        None,
        &cancel,
    )
    .unwrap()
    .scenarios[0]
        .1
        .power_uncorrected[0];

    let p_h = find_power(
        &[small_glmm_contract(WaldSe::Hessian)],
        120,
        200,
        2137,
        None,
        &cancel,
    )
    .unwrap()
    .scenarios[0]
        .1
        .power_uncorrected[0];

    assert!(
        p_h <= p_rx + 0.02,
        "hessian power {p_h} should not exceed rx power {p_rx}"
    );
}
