//! Integration tests for cluster-aware grid construction wired through
//! `find_sample_size` and `single_core_find_sample_size`.
//! Also covers the `find_power` single-N cluster snap.

mod common;
use common::minimal_ols_contract;

use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec, SimulationContract};
use engine_orchestrator::{
    bounds_to_rows, find_power, find_sample_size, resolve_unit_divisor, sample_size_result_to_host,
    single_core_find_power, single_core_find_sample_size, ByValue, CancellationToken, GridMode,
    HostValue, OrchestratorError, SampleSizeMethod, SampleSizeUnit,
};

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
        1,
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
        1,
        2137,
        None,
        &cancel,
    );
    assert!(
        matches!(
            r,
            Err(OrchestratorError::ClusterSizeTooSmall { got: 1, min: 2 })
        ),
        "expected ClusterSizeTooSmall for cluster_size < min_rows_per_cluster, got {r:?}"
    );
}

/// Rejection twin: the single-core (WASM-worker) `single_core_find_sample_size`
/// must reject a below-floor FixedSize cluster the same way the multi-core
/// `find_sample_size` does in `fixed_size_cluster_below_floor_errors`.
#[test]
fn single_core_find_sample_size_rejects_cluster_size_too_small() {
    // cluster_size=1 < min_rows_per_cluster=2 → ClusterSizeTooSmall error
    let c = clustered_contract(ClusterSizing::FixedSize { cluster_size: 1 }, 0.3);
    let cancel = CancellationToken::new();
    let r = single_core_find_sample_size(
        &[c],
        0.8,
        (30, 200),
        50,
        SampleSizeMethod::Grid {
            by: ByValue::Auto { count: 12 },
            mode: GridMode::Linear,
        },
        1,
        2137,
        None,
        &cancel,
    );
    assert!(
        matches!(
            r,
            Err(OrchestratorError::ClusterSizeTooSmall { got: 1, min: 2 })
        ),
        "single_core: expected ClusterSizeTooSmall, got {r:?}"
    );
}

/// Also the end-to-end run of the `Continuous + cluster + Ols` pairing:
/// clustered data generated, OLS fitted ignoring the clustering.
#[test]
fn find_power_floors_n_to_atom_and_warns() {
    // FixedClusters { n_clusters: 20 } → atom = 20.
    // 95 / 20 = 4 complete clusters → floored to 80.
    let c = clustered_contract(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.3);
    assert_eq!(c.estimator, EstimatorSpec::Ols);
    let cancel = CancellationToken::new();
    let res = find_power(&[c], 95, 50, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    assert_eq!(pr.n, 80, "95 floored to a multiple of 20");
    assert!(
        pr.grid_warnings.iter().any(|w| w.contains("80")),
        "expected warning mentioning 80, got: {:?}",
        pr.grid_warnings
    );
    // Discriminating floor: β=0.5 at N=80 yields high power; anything below
    // 0.2 indicates a no-op generation or estimation path.
    assert!(
        pr.power_uncorrected[0] > 0.2,
        "β=0.5 at N=80 must produce power > 0.2, got {}",
        pr.power_uncorrected[0]
    );
}

/// EP-3 dispatch-twin — cluster-atom floor and `grid_warnings` through the
/// single-core path.
///
/// `find_power_floors_n_to_atom_and_warns` (above) verifies the single-N
/// cluster snap only via the multi-core `find_power`. The WASM worker path
/// runs `single_core_find_power`; this test pins the same atom-floor + warning
/// behaviour for that twin so a preflight divergence (floor applied only in one
/// path) surfaces in tests rather than silently on the WASM site.
#[test]
fn ep3_single_core_floors_n_to_atom_and_warns() {
    // Same fixture as find_power_floors_n_to_atom_and_warns: atom = 20,
    // N = 95 → 4 complete clusters → floored to 80.
    let c = clustered_contract(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.3);
    let cancel = CancellationToken::new();
    let res = single_core_find_power(&[c], 95, 50, 2137, None, &cancel).unwrap();
    let (_, pr) = &res.scenarios[0];
    assert_eq!(
        pr.n, 80,
        "single_core: 95 must be floored to a multiple of 20"
    );
    assert!(
        pr.grid_warnings.iter().any(|w| w.contains("80")),
        "single_core: expected warning mentioning 80, got: {:?}",
        pr.grid_warnings
    );
}

/// End-to-end unit path: FixedSize 8, bounds 10..50 clusters step 5 →
/// grid N 80..400 step 40; the host map's `sample_sizes_unit` reads
/// 10, 15, …, 50 while `sample_sizes` stays in rows. Both dispatch twins.
#[test]
fn fixed_size_clusters_unit_scales_bounds_and_relabels_host_columns() {
    let c = clustered_contract(ClusterSizing::FixedSize { cluster_size: 8 }, 0.3);
    let divisor = resolve_unit_divisor(std::slice::from_ref(&c), SampleSizeUnit::Clusters).unwrap();
    assert_eq!(divisor, 8);
    let (bounds, method) = bounds_to_rows(
        (10, 50),
        SampleSizeMethod::Grid {
            by: ByValue::Fixed(5),
            mode: GridMode::Linear,
        },
        divisor,
    );
    assert_eq!(bounds, (80, 400));
    let cancel = CancellationToken::new();

    let multi = find_sample_size(
        std::slice::from_ref(&c),
        0.8,
        bounds,
        30,
        method,
        divisor,
        2137,
        None,
        &cancel,
    )
    .unwrap();
    let single =
        single_core_find_sample_size(&[c], 0.8, bounds, 30, method, divisor, 2137, None, &cancel)
            .unwrap();

    for res in [&multi, &single] {
        let (label, ss) = &res.scenarios[0];
        assert_eq!(ss.unit_divisor, 8);
        let ns: Vec<usize> = ss.grid_or_trace.iter().map(|p| p.n).collect();
        assert_eq!(ns, vec![80, 120, 160, 200, 240, 280, 320, 360, 400]);
        let hv = sample_size_result_to_host(ss, label);
        let HostValue::Map(pairs) = &hv else {
            panic!("not a map")
        };
        let get = |k: &str| &pairs.iter().find(|(kk, _)| *kk == k).unwrap().1;
        match get("sample_sizes_unit") {
            HostValue::VecU64(v) => assert_eq!(v, &vec![10, 15, 20, 25, 30, 35, 40, 45, 50]),
            other => panic!("sample_sizes_unit: {other:?}"),
        }
        match get("sample_sizes") {
            HostValue::VecU64(v) => assert_eq!(v[0], 80),
            other => panic!("sample_sizes: {other:?}"),
        }
        assert!(matches!(get("unit_divisor"), HostValue::Usize(8)));
    }
}
