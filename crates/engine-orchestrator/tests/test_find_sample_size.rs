mod common;
use common::{minimal_ols_contract, minimal_ols_contract_labelled};

use engine_orchestrator::{
    ByValue, CancellationToken, CrossingFit, GridMode, OrchestratorError, ProgressEvent,
    ProgressSink, SampleSizeMethod, SampleSizeResult, ScenarioResult, find_sample_size,
    merge_sample_size_results, single_core_find_sample_size,
};

/// Grid whose interior brackets the 0.8 crossing of `minimal_ols_contract`
/// (β = 0.5 crosses near N ≈ 30), so the crossing fit is `Fitted`, not
/// `AtOrBelowMin`.
const INTERIOR_CROSSING_BOUNDS: (usize, usize) = (10, 60);

fn interior_crossing_method() -> SampleSizeMethod {
    SampleSizeMethod::Grid {
        by: ByValue::Fixed(10),
        mode: GridMode::Linear,
    }
}

fn is_empty_contracts_error(
    r: &Result<ScenarioResult<SampleSizeResult>, OrchestratorError>,
) -> bool {
    matches!(r, Err(OrchestratorError::InvalidScenarios(m)) if m.contains("empty"))
}

// Dispatch-twin: empty contracts must be rejected on both sample-size entry
// points, mirroring find_power_rejects_empty_scenarios.
#[test]
fn find_sample_size_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r = find_sample_size(&[], 0.8, (50, 200), 25, method, 1, 0, None, &cancel);
    assert!(is_empty_contracts_error(&r), "{r:?}");
}

#[test]
fn single_core_find_sample_size_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r = single_core_find_sample_size(&[], 0.8, (50, 200), 25, method, 1, 0, None, &cancel);
    assert!(is_empty_contracts_error(&r), "{r:?}");
}

// Dispatch-twin: a pre-set cancel flag must short-circuit the grid loop on both
// sample-size entry points, mirroring find_power_cancellation_short_circuits —
// the grid loop is a distinct cancel checkpoint from find_power's single-N path.
#[test]
fn find_sample_size_cancellation_short_circuits() {
    let contracts = vec![minimal_ols_contract_labelled("a")];
    let cancel = CancellationToken::new();
    cancel.cancel();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r = find_sample_size(&contracts, 0.8, (50, 200), 25, method, 1, 0, None, &cancel);
    assert!(matches!(
        r,
        Err(engine_orchestrator::OrchestratorError::Cancelled { .. })
    ));
}

#[test]
fn single_core_find_sample_size_cancellation_short_circuits() {
    let contracts = vec![minimal_ols_contract_labelled("a")];
    let cancel = CancellationToken::new();
    cancel.cancel();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 1, 0, None, &cancel);
    assert!(matches!(
        r,
        Err(engine_orchestrator::OrchestratorError::Cancelled { .. })
    ));
}

#[test]
fn grid_method_returns_one_pr_per_grid_point() {
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r =
        find_sample_size(&contracts, 0.8, (50, 200), 25, method, 1, 13, None, &cancel).expect("ok");
    assert_eq!(r.scenarios.len(), 1);
    let ssr = &r.scenarios[0].1;
    assert_eq!(ssr.grid_or_trace.len(), 4); // 50, 100, 150, 200
    assert_eq!(ssr.grid_or_trace[0].n, 50);
    assert_eq!(ssr.grid_or_trace[3].n, 200);
    assert_eq!(ssr.first_achieved.len(), 1);
    approx::assert_relative_eq!(ssr.target_power, 0.8);
}

#[test]
fn grid_first_achieved_none_when_target_unreachable() {
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    // Target 0.999 sits above this fixture's achievable power ceiling (~0.97 for
    // β=0.5 at N≤60), so it is genuinely unreachable. n_sims must stay large: with
    // few sims the measured rate can reach 100% whenever every sim happens to be
    // significant, spuriously "achieving" the target — that knife-edge made the
    // old n_sims=10 version flip None→Some on any RNG reshuffle.
    let r = find_sample_size(
        &contracts,
        0.999,
        (30, 60),
        400,
        method,
        1,
        0,
        None,
        &cancel,
    )
    .expect("ok");
    let ssr = &r.scenarios[0].1;
    assert_eq!(ssr.first_achieved, vec![None]);
}

/// Dispatch twins: at the same seed, `find_sample_size` (`run_batch`) and
/// `single_core_find_sample_size` (`run_batch_st`) walk the same per-sim seeds at
/// every grid point, so the whole result (grid, counts, first-N, crossing fits)
/// must be bit-identical. Thread-count invariance itself is owned by engine-core
/// (`run_batch_bit_identical_1_vs_n_threads`).
#[test]
fn single_core_find_sample_size_is_bit_identical_to_find_sample_size() {
    let contracts = vec![
        minimal_ols_contract_labelled("first"),
        minimal_ols_contract_labelled("second"),
    ];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };

    let mc = find_sample_size(
        &contracts,
        0.8,
        (50, 200),
        64,
        method,
        1,
        2137,
        None,
        &cancel,
    )
    .expect("ok");
    let sc = single_core_find_sample_size(
        &contracts,
        0.8,
        (50, 200),
        64,
        method,
        1,
        2137,
        None,
        &cancel,
    )
    .expect("ok");

    assert_eq!(mc.scenarios.len(), 2);
    for (_, ssr) in &mc.scenarios {
        let grid_ns: Vec<usize> = ssr.grid_or_trace.iter().map(|pr| pr.n).collect();
        assert_eq!(grid_ns, vec![50, 100, 150, 200]);
        let n_targets = ssr.grid_or_trace[0].target_indices.len();
        // Catches the vec![] stub: one joint entry per target, each a grid point.
        assert_eq!(ssr.first_joint_achieved.len(), n_targets);
        for n in ssr.first_joint_achieved.iter().flatten() {
            assert!(grid_ns.contains(n), "first_joint_achieved {n} off-grid");
        }
        // Crossing-fit shapes mirror the empirical vectors.
        assert_eq!(ssr.fitted.len(), ssr.first_achieved.len());
        assert_eq!(ssr.fitted_joint.len(), ssr.first_joint_achieved.len());
    }
    assert_eq!(
        mc, sc,
        "dispatch twins must be bit-identical at the same seed"
    );
}

#[test]
fn fitted_shapes_and_in_grid_consistency() {
    // fitted parallels first_achieved, fitted_joint parallels
    // first_joint_achieved, and unclustered runs carry cluster_atom == 1.
    // Any Fitted entry must cross inside the searched grid with an atom-ceiled
    // headline and CI bounds bracketing n_star.
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let r = find_sample_size(
        &contracts,
        0.8,
        INTERIOR_CROSSING_BOUNDS,
        200,
        interior_crossing_method(),
        1,
        13,
        None,
        &cancel,
    )
    .expect("ok");
    let ssr = &r.scenarios[0].1;
    assert_eq!(ssr.fitted.len(), ssr.first_achieved.len());
    assert_eq!(ssr.fitted_joint.len(), ssr.first_joint_achieved.len());
    assert!(
        matches!(ssr.fitted[0], CrossingFit::Fitted { .. }),
        "anti-vacuity: the grid must bracket the crossing, got {:?}",
        ssr.fitted[0]
    );
    assert_eq!(ssr.cluster_atom, 1, "unclustered atom");
    for cf in ssr.fitted.iter().chain(ssr.fitted_joint.iter()) {
        if let CrossingFit::Fitted {
            n_star,
            n_achievable,
            ci_lo,
            ci_hi,
        } = cf
        {
            assert!(
                (10.0..=60.0).contains(n_star),
                "n_star within the grid, got {n_star}"
            );
            let na = *n_achievable as f64;
            assert!(
                na >= *n_star - 1e-6 && na < *n_star + 1.0 + 1e-6,
                "n_achievable {n_achievable} must be the ceil of n_star {n_star} (atom 1)"
            );
            if let Some(lo) = ci_lo {
                assert!(*lo <= *n_star + 1e-9, "ci_lo {lo} <= n_star {n_star}");
            }
            if let Some(hi) = ci_hi {
                assert!(*hi >= *n_star - 1e-9, "ci_hi {hi} >= n_star {n_star}");
            }
        }
    }
}

#[test]
fn merge_of_single_part_reproduces_fit_exactly() {
    // The merge path discards per-worker fitted values and recomputes all four
    // derivations from pooled counts via the same shared helper the run path
    // used. With one part the pooled counts ARE the part's counts, so the
    // recompute must be value-identical — fit included.
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let part = single_core_find_sample_size(
        &contracts,
        0.8,
        INTERIOR_CROSSING_BOUNDS,
        200,
        interior_crossing_method(),
        1,
        13,
        None,
        &cancel,
    )
    .expect("ok");
    let merged = merge_sample_size_results(std::slice::from_ref(&part)).expect("merge ok");
    let (p, m) = (&part.scenarios[0].1, &merged.scenarios[0].1);
    // Anti-vacuity: a real interior crossing, so the interpolation and the
    // Wilson-band CI are recomputed (not just the AtOrBelowMin short-cut).
    assert!(
        matches!(p.fitted[0], CrossingFit::Fitted { .. }),
        "the grid must bracket the crossing, got {:?}",
        p.fitted[0]
    );
    assert_eq!(p.first_achieved, m.first_achieved);
    assert_eq!(p.first_joint_achieved, m.first_joint_achieved);
    assert_eq!(
        p.fitted, m.fitted,
        "merge recompute must match the run path"
    );
    assert_eq!(p.fitted_joint, m.fitted_joint);
    assert_eq!(p.cluster_atom, m.cluster_atom);
}

#[test]
fn grid_find_sample_size_progress_npoint_per_grid_point_and_zero_sentinel() {
    // NPointCompleted fires once per evaluated grid point.
    // SimsCompleted in Grid find_sample_size carries n == 0 as a per-grid-point
    // sentinel, distinct from find_power which carries n > 0.
    #[derive(Default)]
    struct Sink {
        npoint: usize,
        sims_n: Vec<usize>,
    }
    impl ProgressSink for Sink {
        fn on_event(&mut self, e: ProgressEvent) {
            match e {
                ProgressEvent::NPointCompleted { .. } => self.npoint += 1,
                ProgressEvent::SimsCompleted { n, .. } => self.sims_n.push(n),
                _ => {}
            }
        }
    }
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let mut sink = Sink::default();
    // Grid (50,200) by 50 → 4 points: 50, 100, 150, 200.
    find_sample_size(
        &contracts,
        0.8,
        (50, 200),
        25,
        method,
        1,
        13,
        Some(&mut sink),
        &cancel,
    )
    .expect("ok");
    assert_eq!(sink.npoint, 4, "one NPointCompleted per grid point (4)");
    assert!(
        !sink.sims_n.is_empty(),
        "grid search must emit SimsCompleted"
    );
    assert!(
        sink.sims_n.iter().all(|&n| n == 0),
        "Grid find_sample_size SimsCompleted must carry the n=0 sentinel, got {:?}",
        sink.sims_n
    );
}

/// Grid progress counts model fits: `Started.total_sims` =
/// n_sims × n_scenarios × grid_points (one draw is fitted at every grid N),
/// ticks are cumulative across scenarios with that same denominator, and the
/// final boundary tick reaches the call total.
#[test]
fn grid_progress_total_counts_fits_across_scenarios_and_grid() {
    #[derive(Default)]
    struct CumSink {
        started_total: u64,
        ticks: Vec<(u64, u64)>,
    }
    impl ProgressSink for CumSink {
        fn on_event(&mut self, e: ProgressEvent) {
            match e {
                ProgressEvent::Started { total_sims, .. } => self.started_total = total_sims,
                ProgressEvent::SimsCompleted {
                    completed, total, ..
                } => self.ticks.push((completed, total)),
                _ => {}
            }
        }
    }
    let contracts = vec![
        minimal_ols_contract_labelled("a"),
        minimal_ols_contract_labelled("b"),
    ];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let mut sink = CumSink::default();
    // Grid (50,200) by 50 → 4 points; 25 sims × 2 scenarios × 4 = 200 fits.
    find_sample_size(
        &contracts,
        0.8,
        (50, 200),
        25,
        method,
        1,
        13,
        Some(&mut sink),
        &cancel,
    )
    .expect("ok");
    assert_eq!(
        sink.started_total, 200,
        "total fits = n_sims × n_scenarios × grid_points"
    );
    assert!(
        sink.ticks.iter().all(|&(_, t)| t == 200),
        "every tick's total must equal Started.total_sims: {:?}",
        sink.ticks
    );
    assert_eq!(
        sink.ticks.iter().map(|&(c, _)| c).max(),
        Some(200),
        "the final boundary tick must reach the call total"
    );
    assert!(
        sink.ticks.iter().any(|&(c, _)| c > 100 && c < 200),
        "scenario 2 ticks must be offset above scenario 1's range: {:?}",
        sink.ticks
    );

    // Single-core twin: same totals, and (sequential) strictly ordered.
    let mut st_sink = CumSink::default();
    single_core_find_sample_size(
        &contracts,
        0.8,
        (50, 200),
        25,
        method,
        1,
        13,
        Some(&mut st_sink),
        &cancel,
    )
    .expect("ok");
    assert_eq!(st_sink.started_total, 200);
    assert!(
        !st_sink.ticks.is_empty(),
        "single-core grid must emit per-sim ticks"
    );
    assert!(st_sink.ticks.iter().all(|&(_, t)| t == 200));
    assert!(
        st_sink.ticks.windows(2).all(|w| w[0].0 <= w[1].0),
        "sequential path must be monotone: {:?}",
        st_sink.ticks
    );
    assert_eq!(st_sink.ticks.last().map(|&(c, _)| c), Some(200));
}

#[test]
fn first_joint_achieved_shape_after_merge() {
    // merge_sample_size_results also populates first_joint_achieved
    // with the correct shape (not the vec![] stub).
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };

    let part1 =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 1, 13, None, &cancel)
            .expect("ok");
    let part2 =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 1, 99, None, &cancel)
            .expect("ok");

    let merged = merge_sample_size_results(&[part1, part2]).expect("merge ok");
    let ssr = &merged.scenarios[0].1;
    let n_targets = ssr.grid_or_trace[0].target_indices.len();
    assert_eq!(
        ssr.first_joint_achieved.len(),
        n_targets,
        "merged first_joint_achieved must have one entry per target"
    );
    let grid_ns: Vec<usize> = ssr.grid_or_trace.iter().map(|pr| pr.n).collect();
    for n in ssr.first_joint_achieved.iter().flatten() {
        assert!(
            grid_ns.contains(n),
            "merged first_joint_achieved N must be a grid point, got {n}"
        );
    }
}
