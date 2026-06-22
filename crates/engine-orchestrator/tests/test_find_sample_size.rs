mod common;
use common::{minimal_ols_contract, minimal_ols_contract_labelled};

use engine_orchestrator::{
    find_sample_size, merge_sample_size_results, single_core_find_sample_size, ByValue,
    CancellationToken, CrossingFit, GridMode, ProgressEvent, ProgressSink, SampleSizeMethod,
};

// Dispatch-twin: empty contracts must be rejected on both sample-size entry
// points, mirroring find_power_rejects_empty_scenarios.
#[test]
fn find_sample_size_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    assert!(find_sample_size(&[], 0.8, (50, 200), 25, method, 0, None, &cancel).is_err());
}

#[test]
fn single_core_find_sample_size_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    assert!(
        single_core_find_sample_size(&[], 0.8, (50, 200), 25, method, 0, None, &cancel).is_err()
    );
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
    let r = find_sample_size(&contracts, 0.8, (50, 200), 25, method, 0, None, &cancel);
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
    let r = single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 0, None, &cancel);
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
        find_sample_size(&contracts, 0.8, (50, 200), 25, method, 13, None, &cancel).expect("ok");
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
    let r =
        find_sample_size(&contracts, 0.999, (30, 60), 400, method, 0, None, &cancel).expect("ok");
    let ssr = &r.scenarios[0].1;
    assert_eq!(ssr.first_achieved, vec![None]);
}

#[test]
fn find_sample_size_is_deterministic_across_thread_counts() {
    let c1 = minimal_ols_contract_labelled("first");
    let c2 = minimal_ols_contract_labelled("second");
    let contracts = vec![c1, c2];

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_auto = rayon::ThreadPoolBuilder::new().build().unwrap();
    let cancel = CancellationToken::new();

    // Grid mode for predictable iteration count.
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };

    let r1 = pool1
        .install(|| find_sample_size(&contracts, 0.8, (50, 200), 64, method, 2137, None, &cancel))
        .unwrap();
    let r_auto = pool_auto
        .install(|| find_sample_size(&contracts, 0.8, (50, 200), 64, method, 2137, None, &cancel))
        .unwrap();

    // Vacuous-green guards: verify output is well-formed and non-degenerate.
    assert_eq!(r_auto.scenarios.len(), 2);
    for (_, ssr) in &r_auto.scenarios {
        assert!(!ssr.grid_or_trace.is_empty());
        assert!(ssr
            .grid_or_trace
            .iter()
            .all(|pr| pr.power_uncorrected.iter().all(|p| p.is_finite())));
    }

    // Determinism gate: results must be bit-equal regardless of thread count.
    assert_eq!(
        r1, r_auto,
        "find_sample_size must be bit-equal across thread counts"
    );
}

#[test]
fn single_core_find_sample_size_matches_find_sample_size_shape() {
    // single_core_find_sample_size returns the same ScenarioResult<SampleSizeResult>
    // shape as the multi-core find_sample_size (same scenarios, label, grid length
    // + n's, first_achieved length, target_power). RNG paths are not bit-equal
    // across worker counts, so only the shape is asserted.
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };

    let mc =
        find_sample_size(&contracts, 0.8, (50, 200), 25, method, 13, None, &cancel).expect("ok");
    let sc =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 13, None, &cancel)
            .expect("ok");

    assert_eq!(mc.scenarios.len(), sc.scenarios.len());
    assert_eq!(
        mc.scenarios[0].0, sc.scenarios[0].0,
        "scenario label parity"
    );
    let (mc_ssr, sc_ssr) = (&mc.scenarios[0].1, &sc.scenarios[0].1);
    let mc_ns: Vec<usize> = mc_ssr.grid_or_trace.iter().map(|pr| pr.n).collect();
    let sc_ns: Vec<usize> = sc_ssr.grid_or_trace.iter().map(|pr| pr.n).collect();
    assert_eq!(mc_ns, sc_ns, "grid n's parity");
    assert_eq!(mc_ssr.first_achieved.len(), sc_ssr.first_achieved.len());
    approx::assert_relative_eq!(mc_ssr.target_power, sc_ssr.target_power);
    // Catches the vec![] stub: length must match target count, not be zero.
    assert_eq!(
        sc_ssr.first_joint_achieved.len(),
        sc_ssr.grid_or_trace[0].target_indices.len(),
        "first_joint_achieved must match target count (not the vec![] stub)"
    );
    // Crossing-fit shapes mirror the empirical vectors on BOTH twin paths.
    assert_eq!(mc_ssr.fitted.len(), mc_ssr.first_achieved.len());
    assert_eq!(sc_ssr.fitted.len(), sc_ssr.first_achieved.len());
    assert_eq!(mc_ssr.fitted_joint.len(), mc_ssr.first_joint_achieved.len());
    assert_eq!(sc_ssr.fitted_joint.len(), sc_ssr.first_joint_achieved.len());
    assert_eq!(mc_ssr.cluster_atom, sc_ssr.cluster_atom, "atom parity");
}

#[test]
fn fitted_shapes_and_in_grid_consistency() {
    // fitted parallels first_achieved, fitted_joint parallels
    // first_joint_achieved, and unclustered runs carry cluster_atom == 1.
    // Any Fitted entry must cross inside the searched grid with an atom-ceiled
    // headline and CI bounds bracketing n_star.
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let r =
        find_sample_size(&contracts, 0.8, (50, 200), 200, method, 13, None, &cancel).expect("ok");
    let ssr = &r.scenarios[0].1;
    assert_eq!(ssr.fitted.len(), ssr.first_achieved.len());
    assert_eq!(ssr.fitted_joint.len(), ssr.first_joint_achieved.len());
    assert!(!ssr.fitted.is_empty(), "anti-vacuity: targets exist");
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
                (50.0..=200.0).contains(n_star),
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
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let part =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 200, method, 13, None, &cancel)
            .expect("ok");
    let merged = merge_sample_size_results(std::slice::from_ref(&part)).expect("merge ok");
    let (p, m) = (&part.scenarios[0].1, &merged.scenarios[0].1);
    assert!(!p.fitted.is_empty(), "anti-vacuity: fit was computed");
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
        13,
        Some(&mut sink),
        &cancel,
    )
    .expect("ok");
    assert_eq!(sink.npoint, 4, "one NPointCompleted per grid point (4)");
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
fn first_joint_achieved_shape_and_grid_point_membership() {
    // first_joint_achieved has one entry per target (k = 1..=n_targets),
    // and every Some(n) must be a grid point. No strict monotonicity asserted
    // (MC noise can cause ties/inversions on a coarse grid).
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let result =
        find_sample_size(&contracts, 0.8, (50, 200), 25, method, 13, None, &cancel).expect("ok");

    let ssr = &result.scenarios[0].1;
    let n_targets = ssr.grid_or_trace[0].target_indices.len();
    assert_eq!(
        ssr.first_joint_achieved.len(),
        n_targets,
        "first_joint_achieved must have one entry per target"
    );
    let grid_ns: Vec<usize> = ssr.grid_or_trace.iter().map(|pr| pr.n).collect();
    for n in ssr.first_joint_achieved.iter().flatten() {
        assert!(
            grid_ns.contains(n),
            "first_joint_achieved N must be a grid point, got {n}"
        );
    }
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
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 13, None, &cancel)
            .expect("ok");
    let part2 =
        single_core_find_sample_size(&contracts, 0.8, (50, 200), 25, method, 99, None, &cancel)
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
