//! Smoke test for the `Continuous + cluster=Some + Ols` pairing.
//!
//! Generates clustered data and fits OLS ignoring the clustering structure.
//! Asserts the run completes without error and returns a finite power in [0,1].

mod common;

use common::minimal_ols_contract;
use engine_contract::{ClusterSizing, ClusterSpec};
use engine_orchestrator::{find_power, CancellationToken};

#[test]
fn clustered_generation_solved_with_ols_runs() {
    // Continuous + cluster=Some + Ols: generate clustered data, fit OLS ignoring clustering.
    let mut c = minimal_ols_contract();
    c.generation.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
        tau_squared: 0.3,
        slopes: vec![],
        extra_groupings: vec![],
    });
    // estimator is already Ols (set explicitly here for clarity)
    assert_eq!(c.estimator, engine_contract::EstimatorSpec::Ols);

    let cancel = CancellationToken::new();
    let result = find_power(&[c], 200, 200, 2137, None, &cancel)
        .expect("find_power ok for clustered-OLS pairing");

    assert_eq!(result.scenarios.len(), 1);
    let (_, pr) = &result.scenarios[0];
    let power = pr.power_uncorrected[0];
    // Discriminating floor: β=0.5 at N=200 yields high power; anything below
    // 0.2 indicates a no-op engine (broken generation or estimation path).
    // The call is seeded/deterministic, so this can never be flaky.
    assert!(
        power > 0.2,
        "β=0.5 at N=200 must produce power > 0.2, got {power}"
    );
    assert!(
        power.is_finite() && (0.0..=1.0).contains(&power),
        "expected finite power in [0,1], got {power}"
    );
}
