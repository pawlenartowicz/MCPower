//! End-to-end: a crossed-slope contract (`y ~ x + (1+x | primary) + (1+x | item)`)
//! flows through `find_power` with no API change — the additive `GroupingSpec.slopes`
//! lowers into the kernel, data-gen draws covariate-weighted REs, and the blocked
//! LMM solver fits them. Pins that the extra-grouping slope reaches the kernel
//! (four variance components surface) and that the fits mostly converge.

mod common;
use common::minimal_ols_contract;

use engine_contract::{
    ClusterSizing, ClusterSpec, ColumnId, EstimatorSpec, GroupingRelation, GroupingSpec,
    SimulationContract, SlopeTerm,
};
use engine_orchestrator::{find_power, CancellationToken, EstimatorExtras};

fn crossed_slope_contract() -> SimulationContract {
    let slope = |v: f64, rho: f64| SlopeTerm {
        column: ColumnId(0), // the continuous Direct fixed effect
        variance: v,
        corr_with_intercept: rho,
        corr_with: vec![],
    };
    let mut c = minimal_ols_contract();
    c.estimator = EstimatorSpec::Mle;
    c.generation.cluster = Some(ClusterSpec {
        sizing: ClusterSizing::FixedClusters { n_clusters: 5 },
        tau_squared: 0.25,
        slopes: vec![slope(0.10, 0.2)], // (1 + x | primary)
        extra_groupings: vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.16,
            slopes: vec![slope(0.09, 0.1)], // (1 + x | item)
        }],
    });
    c
}

#[test]
fn crossed_slope_power_runs_end_to_end() {
    let c = crossed_slope_contract();
    assert!(c.validate().is_ok(), "crossed-slope contract must validate");
    let cancel = CancellationToken::new();
    // atom = 5 (primary) · 4 (crossed) = 20; 80 = 4 balanced blocks.
    let res = find_power(&[c], 80, 60, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    assert_eq!(pr.n, 80, "N is an atom multiple, no floor");
    // Primary intercept + slope and crossed intercept + slope: four variance
    // components. A dropped extra-grouping slope would surface three.
    let EstimatorExtras::Mle {
        boundary_rate_per_component,
        ..
    } = &pr.estimator_extras
    else {
        panic!("expected Mle extras, got {:?}", pr.estimator_extras);
    };
    assert_eq!(
        boundary_rate_per_component.len(),
        4,
        "{boundary_rate_per_component:?}"
    );
    assert!(
        pr.convergence_rate > 0.5,
        "crossed-slope convergence_rate {}",
        pr.convergence_rate
    );
}
