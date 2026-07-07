//! End-to-end: a crossed-slope contract (`y ~ x + (1+x | primary) + (1+x | item)`)
//! flows through `find_power` with no API change — the additive `GroupingSpec.slopes`
//! lowers into the kernel, data-gen draws covariate-weighted REs, and the blocked
//! LMM solver fits them. Pins that the whole pipeline produces a valid power.

mod common;
use common::minimal_ols_contract;

use engine_contract::{
    ClusterSizing, ClusterSpec, ColumnId, EstimatorSpec, GroupingRelation, GroupingSpec,
    SimulationContract, SlopeTerm,
};
use engine_orchestrator::{find_power, CancellationToken};

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
    let power = pr.power_uncorrected[0];
    assert!(
        power > 0.0 && power <= 1.0,
        "crossed-slope power must be a valid probability, got {power}"
    );
}
