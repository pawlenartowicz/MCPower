//! Integration tests for cluster-aware grid construction wired through
//! `find_sample_size` and `single_core_find_sample_size`.
//! Also covers the `find_power` single-N cluster snap.

mod common;
use common::minimal_ols_contract;

use engine_contract::{ClusterSizing, ClusterSpec, SimulationContract};
use engine_orchestrator::{
    find_power, find_sample_size, single_core_find_sample_size, ByValue, CancellationToken,
    GridMode, SampleSizeMethod,
};
use engine_orchestrator::grid::build_grid;

fn clustered_contract(sizing: ClusterSizing, tau: f64) -> SimulationContract {
    let mut c = minimal_ols_contract();
    c.generation.cluster = Some(ClusterSpec {
        sizing,
        tau_squared: tau,
        slopes: vec![],
        extra_groupings: vec![],
    });
    c
}

#[test]
fn fixed_clusters_grid_points_are_atom_multiples_and_warns() {
    let c = clustered_contract(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.3);
    let cancel = CancellationToken::new();
    let res = find_sample_size(
        &[c],
        0.8,
        (30, 205),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 12 },
            mode: GridMode::Linear,
        },
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let (_, ss) = &res.scenarios[0];
    let ns: Vec<usize> = ss.grid_or_trace.iter().map(|p| p.n).collect();
    assert!(ns.iter().all(|&n| n % 20 == 0), "{ns:?}");
    assert!(
        ss.grid_warnings.iter().any(|w| w.contains("lowered")),
        "expected 'lowered' warning, got: {:?}",
        ss.grid_warnings
    );
}

#[test]
fn fixed_size_cluster_below_floor_errors() {
    // cluster_size=1 < min_rows_per_cluster=2 → ClusterSizeTooSmall error
    let c = clustered_contract(ClusterSizing::FixedSize { cluster_size: 1 }, 0.3);
    let cancel = CancellationToken::new();
    let r = find_sample_size(
        &[c],
        0.8,
        (30, 200),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 12 },
            mode: GridMode::Linear,
        },
        2137,
        None,
        &cancel,
    );
    assert!(
        r.is_err(),
        "expected error for cluster_size < min_rows_per_cluster"
    );
}

#[test]
fn find_power_floors_n_to_atom_and_warns() {
    // FixedClusters { n_clusters: 20 } → atom = 20.
    // 95 / 20 = 4 complete clusters → floored to 80.
    let c = clustered_contract(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.3);
    let cancel = CancellationToken::new();
    let res = find_power(&[c], 95, 50, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    assert_eq!(pr.n, 80, "95 floored to a multiple of 20");
    assert!(
        pr.grid_warnings.iter().any(|w| w.contains("80")),
        "expected warning mentioning 80, got: {:?}",
        pr.grid_warnings
    );
}

// ── Grid purity / mergeability invariants ───────────────────────────────────

#[test]
fn grid_is_pure_function_of_inputs() {
    // The grid is a function only of (from, to, by, mode, atom, hard_min) —
    // never of observed power — so the search stays mergeable across WASM
    // workers and safe for multi-effect targets. Same inputs ⇒ byte-identical
    // grid + warnings.
    let a = build_grid(
        30,
        200,
        ByValue::Auto { count: 12 },
        GridMode::Linear,
        20,
        40,
    )
    .unwrap();
    let b = build_grid(
        30,
        200,
        ByValue::Auto { count: 12 },
        GridMode::Linear,
        20,
        40,
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn single_core_and_multicore_request_identical_grid() {
    // Multi-core `find_sample_size` and single-core (WASM-worker)
    // `single_core_find_sample_size` must request the SAME sample sizes for a
    // clustered FixedSize spec — the grid is built from inputs before any
    // simulation, so it is identical even though the RNG-driven power values are
    // only statistically (not byte-) equal across worker counts.
    let sizing = ClusterSizing::FixedSize { cluster_size: 10 };
    let method = SampleSizeMethod::Grid {
        by: ByValue::Auto { count: 12 },
        mode: GridMode::Linear,
    };
    let cancel = CancellationToken::new();

    let multi = find_sample_size(
        &[clustered_contract(sizing.clone(), 0.3)],
        0.8,
        (50, 200),
        30,
        method,
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let single = single_core_find_sample_size(
        &[clustered_contract(sizing, 0.3)],
        0.8,
        (50, 200),
        30,
        method,
        2137,
        None,
        &cancel,
    )
    .unwrap();

    let ns = |r: &engine_orchestrator::ScenarioResult<engine_orchestrator::SampleSizeResult>| {
        r.scenarios[0]
            .1
            .grid_or_trace
            .iter()
            .map(|p| p.n)
            .collect::<Vec<usize>>()
    };
    let multi_ns = ns(&multi);
    let single_ns = ns(&single);
    assert!(multi_ns.iter().all(|&n| n % 10 == 0), "{multi_ns:?}");
    assert_eq!(
        multi_ns, single_ns,
        "multi-core and single-core must request the same grid N's"
    );
}
