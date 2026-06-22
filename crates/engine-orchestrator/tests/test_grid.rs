use engine_orchestrator::grid::{build_grid, first_n_at_target};
use engine_orchestrator::{ByValue, GridMode, OrchestratorError};

#[test]
fn linear_grid_matches_python_range() {
    // Python: list(range(30, 51, 5)) -> [30, 35, 40, 45, 50]
    let (g, _) = build_grid(30, 50, ByValue::Fixed(5), GridMode::Linear, 1, 1).expect("ok");
    assert_eq!(g, vec![30, 35, 40, 45, 50]);
}

#[test]
fn linear_grid_inclusive_endpoint_partial_step() {
    // Real builder pins the `to` endpoint even when the regular step skips it.
    // from=30, to=49, step=5: walk gives [30,35,40,45], then 49 is pinned.
    let (g, _) = build_grid(30, 49, ByValue::Fixed(5), GridMode::Linear, 1, 1).expect("ok");
    assert_eq!(g, vec![30, 35, 40, 45, 49]);
}

#[test]
fn log_grid_n_points_dedup_sorted() {
    let (g, _) = build_grid(10, 1000, ByValue::Fixed(4), GridMode::Log, 1, 1).expect("ok");
    assert!(g.first() == Some(&10));
    assert!(g.last() == Some(&1000));
    let mut sorted = g.clone();
    sorted.sort();
    assert_eq!(g, sorted);
    sorted.dedup();
    assert_eq!(g, sorted);
}

#[test]
fn log_grid_rejects_zero_from() {
    let r = build_grid(0, 100, ByValue::Fixed(4), GridMode::Log, 1, 1);
    assert!(matches!(
        r,
        Err(OrchestratorError::InvalidGridBounds { .. })
    ));
}

/// build_grid rejects inverted bounds (to < from).
/// Fixed(0) no longer errors — the real builder maps it to atom (>=1), so it
/// yields a valid atom-stepped grid. Inverted-bounds guard still fires.
#[test]
fn grid_rejects_inverted_bounds_and_zero_step() {
    for mode in [GridMode::Linear, GridMode::Log] {
        // to < from: always an error.
        assert!(matches!(
            build_grid(100, 50, ByValue::Fixed(5), mode, 1, 1),
            Err(OrchestratorError::InvalidGridBounds { .. })
        ));
    }
    // Fixed(0) -> ceil_to(0, atom=1).max(1) = 1 -> valid step=1 grid; not an error.
    let (g, _) = build_grid(10, 15, ByValue::Fixed(0), GridMode::Linear, 1, 1)
        .expect("Fixed(0) yields a valid step-1 grid");
    assert!(!g.is_empty(), "grid must be non-empty");
    assert!(
        g.windows(2).all(|p| p[0] < p[1]),
        "grid must be strictly ascending"
    );
    assert_eq!(g[0], 10);
    assert_eq!(*g.last().unwrap(), 15);
}

#[test]
fn first_n_at_target_finds_smallest_n() {
    let powers = vec![
        vec![0.5, 0.3],
        vec![0.7, 0.6],
        vec![0.85, 0.82],
        vec![0.9, 0.9],
    ];
    let sizes = vec![30, 50, 100, 200];
    assert_eq!(first_n_at_target(&powers, &sizes, 0.8, 0), Some(100));
    assert_eq!(first_n_at_target(&powers, &sizes, 0.8, 1), Some(100));
    assert_eq!(first_n_at_target(&powers, &sizes, 0.95, 0), None);
}

#[test]
fn first_n_at_target_accepts_percent_form() {
    // Python contract: target_power > 1.0 is treated as a percentage (80 → 0.8).
    let powers = vec![vec![0.85]];
    let sizes = vec![100];
    assert_eq!(first_n_at_target(&powers, &sizes, 80.0, 0), Some(100));
    assert_eq!(first_n_at_target(&powers, &sizes, 0.8, 0), Some(100));
}
