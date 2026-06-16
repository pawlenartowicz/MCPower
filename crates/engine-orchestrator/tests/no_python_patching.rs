//! Confirms a non-Python caller can run Glm + Mle power analyses through
//! `engine_spec_builder::build_contract` + `engine_orchestrator::find_power`
//! with zero Python touchpoints.

mod common;

use common::{lme_spec, logit_spec};
use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec, OutcomeKind};
use engine_orchestrator::{find_power, CancellationToken};
use engine_spec_builder::build_contract;

#[test]
fn no_python_glm() {
    let contracts = build_contract(&logit_spec(), OutcomeKind::Binary, None, -0.5, vec![]).unwrap();
    let cancel = CancellationToken::new();
    let result = find_power(&contracts, 200, 64, 2137, None, &cancel).expect("find_power");
    assert_eq!(result.scenarios.len(), 1);
    let pr = &result.scenarios[0].1;
    assert_eq!(pr.power_uncorrected.len(), 1, "one marginal target");
    // Logit x1 (effect 0.3) at N=200/64 sims, seed 2137 ⇒ power ≈ 0.47, all fits
    // converge. Pin a band that detects the real effect — a 0/NaN/degenerate
    // statistic (the failure a shape-only check missed) lands outside it.
    assert!(
        (0.25..0.75).contains(&pr.power_uncorrected[0]),
        "glm power {}",
        pr.power_uncorrected[0]
    );
    assert!(pr.convergence_rate > 0.9, "glm convergence {}", pr.convergence_rate);
}

#[test]
fn no_python_mle() {
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
    .unwrap();
    let cancel = CancellationToken::new();
    let result = find_power(&contracts, 200, 64, 2137, None, &cancel).expect("find_power");
    assert_eq!(result.scenarios.len(), 1);
    let pr = &result.scenarios[0].1;
    assert_eq!(pr.power_uncorrected.len(), 1, "one marginal target");
    // LME (MLE) x1 (effect 0.3), 20 clusters τ²=0.3 at N=200/64 sims, seed 2137 ⇒
    // power ≈ 0.98, all fits converge.
    assert!(pr.power_uncorrected[0] > 0.8, "mle power {}", pr.power_uncorrected[0]);
    assert!(pr.convergence_rate > 0.9, "mle convergence {}", pr.convergence_rate);
}
