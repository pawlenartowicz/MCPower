//! Small-cluster GLMM advisory: a clustered binary/count model that tests a
//! cluster-level effect with fewer than `limits.glmm_reliable_clusters`
//! clusters gets a `grid_warnings` entry, on all four entry points. The check
//! runs before any simulation, so the tests use few sims and assert only on
//! the warning text.

mod common;
use common::{lme_spec, logit_spec};

use engine_contract::{
    ClusterSizing, ClusterSpec, ColumnId, DfMethod, GroupingRelation, GroupingSpec, LinkKind,
    OutcomeKind, SimulationContract,
};
use engine_orchestrator::{
    ByValue, CancellationToken, GridMode, SampleSizeMethod, find_power, find_sample_size,
    single_core_find_power, single_core_find_sample_size,
};
use engine_spec_builder::build_contract;

const N_SIMS: usize = 10;
const SEED: u64 = 2137;
const MARKER: &str = "cluster-level effect";

fn fixed(n_clusters: u32) -> ClusterSizing {
    ClusterSizing::FixedClusters { n_clusters }
}

fn cluster(sizing: ClusterSizing) -> ClusterSpec {
    ClusterSpec {
        sizing,
        tau_squared: 0.3,
        slopes: vec![],
        extra_groupings: vec![],
    }
}

/// GLMM `y ~ x1 + x2 + x3 + b + (1|g)` of the given family, target `x1`
/// (column 0). With `cluster_level` the target is drawn once per cluster.
fn glmm_family(
    kind: OutcomeKind,
    link: Option<LinkKind>,
    sizing: ClusterSizing,
    cluster_level: bool,
) -> SimulationContract {
    let intercept = match kind {
        OutcomeKind::Count => 0.5,
        _ => -0.5,
    };
    let mut c = build_contract(
        &logit_spec(),
        kind,
        link,
        None,
        intercept,
        vec![cluster(sizing)],
    )
    .expect("build_contract glmm")
    .remove(0);
    c.df_method = DfMethod::BetweenWithin;
    if cluster_level {
        c.generation.cluster_level_columns = vec![ColumnId(0)];
    }
    c
}

/// Logit GLMM, see [`glmm_family`].
fn glmm(sizing: ClusterSizing, cluster_level: bool) -> SimulationContract {
    glmm_family(OutcomeKind::Binary, None, sizing, cluster_level)
}

fn power_warnings(c: SimulationContract, n: usize) -> [Vec<String>; 2] {
    let cancel = CancellationToken::new();
    let multi = find_power(std::slice::from_ref(&c), n, N_SIMS, SEED, None, &cancel).unwrap();
    let single = single_core_find_power(&[c], n, N_SIMS, SEED, None, &cancel).unwrap();
    [
        multi.scenarios[0].1.grid_warnings.clone(),
        single.scenarios[0].1.grid_warnings.clone(),
    ]
}

/// Grid sample-size search (step 50, linear, unit divisor 1) over `bounds`,
/// multi-core then single-core.
fn sample_size_warnings(c: SimulationContract, bounds: (usize, usize)) -> [Vec<String>; 2] {
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let cancel = CancellationToken::new();
    let multi = find_sample_size(
        std::slice::from_ref(&c),
        0.8,
        bounds,
        N_SIMS,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .unwrap();
    let single =
        single_core_find_sample_size(&[c], 0.8, bounds, N_SIMS, method, 1, SEED, None, &cancel)
            .unwrap();
    [
        multi.scenarios[0].1.grid_warnings.clone(),
        single.scenarios[0].1.grid_warnings.clone(),
    ]
}

fn has_warning(w: &[String]) -> bool {
    w.iter().any(|m| m.contains(MARKER))
}

#[test]
fn power_warns_for_cluster_level_target_below_threshold() {
    const FULL_120: &str = "6 clusters at N=120 (fewer than 20): with few clusters the \
        between-within t test of a cluster-level effect in a clustered binary/count model \
        rejects more often than alpha, so power for cluster-level effects is overstated; \
        add clusters";
    for w in power_warnings(glmm(fixed(6), true), 120) {
        let msg = w.iter().find(|m| m.contains(MARKER));
        assert_eq!(msg.map(String::as_str), Some(FULL_120), "got: {w:?}");
    }
}

#[test]
fn power_is_silent_for_within_target_or_enough_clusters() {
    for (label, c) in [
        ("within-cluster target", glmm(fixed(6), false)),
        ("20 clusters", glmm(fixed(20), true)),
    ] {
        for w in power_warnings(c, 120) {
            assert!(!has_warning(&w), "{label}: got {w:?}");
        }
    }
}

#[test]
fn power_is_silent_for_lmm_asymptotic_and_crossed_glmm() {
    let mut lmm = build_contract(
        &lme_spec(),
        OutcomeKind::Continuous,
        None,
        None,
        0.0,
        vec![cluster(fixed(6))],
    )
    .expect("build_contract lmm")
    .remove(0);
    lmm.df_method = DfMethod::BetweenWithin;
    lmm.generation.cluster_level_columns = vec![ColumnId(0)];

    let mut asymptotic = glmm(fixed(6), true);
    asymptotic.df_method = DfMethod::Asymptotic;

    let mut crossed = glmm(fixed(6), true);
    crossed
        .generation
        .cluster
        .as_mut()
        .unwrap()
        .extra_groupings
        .push(GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.2,
            slopes: vec![],
        });

    for (label, c, n) in [
        ("lmm", lmm, 120),
        ("asymptotic", asymptotic, 120),
        ("crossed", crossed, 240),
    ] {
        for w in power_warnings(c, n) {
            assert!(!has_warning(&w), "{label}: got {w:?}");
        }
    }
}

/// `cluster_size = 10` over N ∈ {100, 150, …, 300}: 10–30 clusters, first
/// reaching 20 at N = 200.
#[test]
fn sample_size_names_the_first_grid_point_with_enough_clusters() {
    let c = glmm(ClusterSizing::FixedSize { cluster_size: 10 }, true);
    for w in sample_size_warnings(c, (100, 300)) {
        let msg = w.iter().find(|m| m.contains(MARKER));
        assert!(
            msg.is_some_and(|m| m.starts_with("fewer than 20 clusters at grid points below N=200")),
            "got: {w:?}"
        );
    }
}

#[test]
fn power_warns_for_probit_and_poisson_glmm() {
    for (label, kind, link) in [
        ("probit", OutcomeKind::Binary, Some(LinkKind::Probit)),
        ("poisson", OutcomeKind::Count, None),
    ] {
        for w in power_warnings(glmm_family(kind, link, fixed(6), true), 120) {
            assert!(
                w.iter()
                    .any(|m| m.contains(MARKER)
                        && m.starts_with("6 clusters at N=120 (fewer than 20)")),
                "{label}: got {w:?}"
            );
        }
    }
}

/// FixedClusters 6 over bounds (100, 300): atom 6, step 54 (50 rounded up to
/// the atom), so N ∈ {102, 156, 210, 264, 300}, 6 clusters at each.
#[test]
fn sample_size_warns_at_every_grid_point_with_fixed_clusters() {
    for w in sample_size_warnings(glmm(fixed(6), true), (100, 300)) {
        let msg = w.iter().find(|m| m.contains(MARKER));
        assert!(
            msg.is_some_and(|m| m.starts_with("fewer than 20 clusters at every grid point")),
            "got: {w:?}"
        );
    }
}

/// cluster_size 10 over N ∈ {200, 250, 300}: 20–30 clusters, never below.
#[test]
fn sample_size_is_silent_when_the_grid_starts_at_the_threshold() {
    let c = glmm(ClusterSizing::FixedSize { cluster_size: 10 }, true);
    for w in sample_size_warnings(c, (200, 300)) {
        assert!(!has_warning(&w), "got: {w:?}");
    }
}
