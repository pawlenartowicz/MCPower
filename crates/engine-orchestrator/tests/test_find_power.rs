//! End-to-end orchestrator test against a real engine_core::run_batch call.

mod common;
use common::{
    marginal_plus_contrast_contract, minimal_ols_contract, minimal_ols_contract_labelled,
};

use engine_orchestrator::{
    find_power, single_core_find_power, CancellationToken, NoOpSink, ProgressEvent, ProgressSink,
};

#[derive(Default)]
struct EventLog {
    tags: Vec<&'static str>,
}
impl ProgressSink for EventLog {
    fn on_event(&mut self, e: ProgressEvent) {
        self.tags.push(match e {
            ProgressEvent::Started { .. } => "Started",
            ProgressEvent::ScenarioStarted { .. } => "ScenarioStarted",
            ProgressEvent::SimsCompleted { .. } => "SimsCompleted",
            ProgressEvent::NPointCompleted { .. } => "NPointCompleted",
            ProgressEvent::ScenarioCompleted { .. } => "ScenarioCompleted",
            ProgressEvent::Cancelled => "Cancelled",
            ProgressEvent::Completed => "Completed",
        });
    }
}

/// EP-1 regression: power/ci/count vectors are sized
/// `target_indices.len() + contrast_pairs.len()`, never by `target_indices`
/// alone. The escape (hot-loop panic, plot OOB, silent merge drop) survived
/// because no end-to-end run carried a contrast — this is that run. Mirrored to
/// the `single_core_find_power` twin so the WASM path is guarded too (EP-3).
#[test]
fn find_power_vectors_sized_by_marginals_plus_contrasts() {
    let contracts = vec![marginal_plus_contrast_contract()];
    let cancel = CancellationToken::new();
    let result = find_power(&contracts, 100, 50, 42, None, &cancel).expect("ok");
    let pr = &result.scenarios[0].1;
    assert_eq!(pr.target_indices.len(), 1, "one marginal target");
    assert_eq!(pr.contrast_pairs.len(), 1, "one pairwise contrast");
    let expected = pr.target_indices.len() + pr.contrast_pairs.len();
    assert_eq!(pr.power_uncorrected.len(), expected);
    assert_eq!(pr.power_corrected.len(), expected);
    assert_eq!(pr.ci_uncorrected.len(), expected);
    assert_eq!(pr.ci_corrected.len(), expected);
    assert_eq!(pr.success_counts_uncorrected.len(), expected);

    // Single-core twin must agree on the contrast tail and the same length.
    let sc = single_core_find_power(&contracts, 100, 50, 42, None, &cancel).expect("ok");
    let sp = &sc.scenarios[0].1;
    assert_eq!(sp.power_uncorrected.len(), expected);
    assert_eq!(sp.contrast_pairs, pr.contrast_pairs);
}

#[test]
fn find_power_single_scenario_returns_one_entry() {
    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let mut sink = EventLog::default();
    let result = find_power(&contracts, 100, 50, 42, Some(&mut sink), &cancel).expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    assert_eq!(result.scenarios[0].0, "optimistic");
    let pr = &result.scenarios[0].1;
    assert_eq!(pr.n, 100);
    assert_eq!(pr.n_sims, 50);
    assert_eq!(pr.target_indices, vec![1]);
    assert!(
        pr.power_uncorrected[0] > 0.0,
        "real effect (beta=0.5) at N=100 must produce non-zero power, got {}",
        pr.power_uncorrected[0]
    );
    assert_eq!(sink.tags.first(), Some(&"Started"));
    assert_eq!(sink.tags.last(), Some(&"Completed"));
    // find_power evaluates a single N-point, so NPointCompleted fires
    // exactly once (count, not mere presence — 0 or 2 emissions must fail).
    let n_npoint = sink
        .tags
        .iter()
        .filter(|&&t| t == "NPointCompleted")
        .count();
    assert_eq!(
        n_npoint, 1,
        "exactly one NPointCompleted for single-N find_power"
    );
    // ScenarioStarted precedes the scenario's work (NPointCompleted) and
    // ScenarioCompleted follows it — assert the ordering, not just presence.
    let pos = |tag: &str| sink.tags.iter().position(|&t| t == tag);
    let started = pos("ScenarioStarted").expect("ScenarioStarted emitted");
    let npoint = pos("NPointCompleted").expect("NPointCompleted emitted");
    let completed = pos("ScenarioCompleted").expect("ScenarioCompleted emitted");
    assert!(
        started < npoint && npoint < completed,
        "expected ScenarioStarted < NPointCompleted < ScenarioCompleted, got {started} {npoint} {completed}"
    );
}

/// `ScenarioStarted.total` equals `contracts.len()` and does not
/// change across iterations.
#[test]
fn scenario_started_total_equals_contract_count() {
    #[derive(Default)]
    struct TotalSink {
        totals: Vec<usize>,
    }
    impl ProgressSink for TotalSink {
        fn on_event(&mut self, e: ProgressEvent) {
            if let ProgressEvent::ScenarioStarted { total, .. } = e {
                self.totals.push(total);
            }
        }
    }

    let contracts = vec![
        minimal_ols_contract_labelled("a"),
        minimal_ols_contract_labelled("b"),
        minimal_ols_contract_labelled("c"),
    ];
    let cancel = CancellationToken::new();
    let mut sink = TotalSink::default();
    find_power(&contracts, 50, 25, 1, Some(&mut sink), &cancel).expect("ok");
    // One ScenarioStarted per scenario, each carrying total == contracts.len().
    assert_eq!(sink.totals.len(), contracts.len());
    assert!(
        sink.totals.iter().all(|&t| t == contracts.len()),
        "ScenarioStarted.total must equal contracts.len() on every iteration: {:?}",
        sink.totals
    );
}

/// `SimsCompleted` events in `find_power` carry the actual sample
/// size `n > 0` (the sentinel `n = 0` is only used in Grid-mode
/// find_sample_size).
#[test]
fn sims_completed_carries_actual_n() {
    #[derive(Default)]
    struct SimsSink {
        ns: Vec<usize>,
    }
    impl ProgressSink for SimsSink {
        fn on_event(&mut self, e: ProgressEvent) {
            if let ProgressEvent::SimsCompleted { n, .. } = e {
                self.ns.push(n);
            }
        }
    }

    let contracts = vec![minimal_ols_contract()];
    let cancel = CancellationToken::new();
    let mut sink = SimsSink::default();
    find_power(&contracts, 100, 200, 7, Some(&mut sink), &cancel).expect("ok");
    assert!(
        !sink.ns.is_empty(),
        "find_power must emit SimsCompleted events"
    );
    assert!(
        sink.ns.iter().all(|&n| n == 100),
        "SimsCompleted in find_power must carry the actual n (100), not the sentinel 0: {:?}",
        sink.ns
    );
}

/// `SimsCompleted` is cumulative across the whole call: every tick's `total`
/// equals `Started.total_sims` (= n_sims × n_scenarios), the second scenario's
/// ticks land above the first scenario's range (the per-scenario offset is
/// applied), and the final boundary tick reaches the call total. Monotonicity
/// is NOT asserted on the multi-core path — rayon workers may race tick
/// emission by a step.
#[test]
fn sims_completed_is_cumulative_across_scenarios() {
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
    let mut sink = CumSink::default();
    find_power(&contracts, 50, 100, 3, Some(&mut sink), &cancel).expect("ok");

    assert_eq!(sink.started_total, 200, "total fits = n_sims × n_scenarios");
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
}

/// The single-core twin emits the same cumulative stream — and, being
/// sequential, it is strictly ordered: monotone non-decreasing, ending
/// exactly at the call total.
#[test]
fn single_core_sims_completed_is_cumulative_and_monotone() {
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
    let mut sink = CumSink::default();
    engine_orchestrator::single_core_find_power(&contracts, 50, 100, 3, Some(&mut sink), &cancel)
        .expect("ok");

    assert_eq!(sink.started_total, 200);
    assert!(
        !sink.ticks.is_empty(),
        "single-core must emit per-sim ticks (worker-pool hosts sum them)"
    );
    assert!(sink.ticks.iter().all(|&(_, t)| t == 200));
    assert!(
        sink.ticks.windows(2).all(|w| w[0].0 <= w[1].0),
        "sequential path must be monotone: {:?}",
        sink.ticks
    );
    assert_eq!(sink.ticks.last().map(|&(c, _)| c), Some(200));
}

#[test]
fn find_power_multi_scenario_preserves_label_order() {
    let contracts = vec![
        minimal_ols_contract_labelled("optimistic"),
        minimal_ols_contract_labelled("pessimistic"),
    ];
    let cancel = CancellationToken::new();
    let mut sink = NoOpSink;
    let result = find_power(&contracts, 50, 25, 7, Some(&mut sink), &cancel).expect("ok");
    let labels: Vec<&str> = result.scenarios.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(labels, vec!["optimistic", "pessimistic"]);
}

#[test]
fn find_power_identical_scenarios_are_bit_identical() {
    // Seed-pairing regression: every scenario in one call runs on the same
    // call-level seed, so two scenarios with identical knobs are paired runs
    // and must return bit-identical results. (Per-index re-seeding made the
    // two arms differ by pure Monte Carlo noise.)
    let contracts = vec![
        minimal_ols_contract_labelled("a"),
        minimal_ols_contract_labelled("b"),
    ];
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, 20, 200, 2137, None, &cancel).expect("ok");
    assert_eq!(r.scenarios.len(), 2);
    let (pr_a, pr_b) = (&r.scenarios[0].1, &r.scenarios[1].1);
    // Anti-vacuity: power must sit strictly inside (0, 1) so an unpaired RNG
    // stream would almost surely move the success count.
    assert!(
        pr_a.power_uncorrected[0] > 0.0 && pr_a.power_uncorrected[0] < 1.0,
        "fixture must keep power off the boundaries, got {}",
        pr_a.power_uncorrected[0]
    );
    assert_eq!(pr_a, pr_b, "identical-knob scenarios must be bit-identical");
}

#[test]
fn find_power_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let r = find_power(&[], 100, 50, 0, None, &cancel);
    assert!(r.is_err());
}

// Dispatch-twin: the single-core entry must reject empty contracts exactly
// like its multi-core twin (the shared preflight guard must cover both paths).
#[test]
fn single_core_find_power_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    assert!(single_core_find_power(&[], 100, 50, 0, None, &cancel).is_err());
}

#[test]
fn find_power_with_two_scenarios_is_deterministic_across_thread_counts() {
    // Two minimal OLS contracts with distinct labels. Both are zero-perturbation
    // so there are no stochastic scenario-construction paths to diverge.
    let c1 = minimal_ols_contract_labelled("first");
    let c2 = minimal_ols_contract_labelled("second");
    let contracts = vec![c1, c2];

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_auto = rayon::ThreadPoolBuilder::new().build().unwrap();
    let cancel = CancellationToken::new();

    let r1 = pool1
        .install(|| find_power(&contracts, 100, 64, 2137, None, &cancel))
        .unwrap();
    let r_auto = pool_auto
        .install(|| find_power(&contracts, 100, 64, 2137, None, &cancel))
        .unwrap();

    // Vacuous-green guards: verify output is well-formed and non-degenerate.
    assert_eq!(r_auto.scenarios.len(), 2);
    for (_, pr) in &r_auto.scenarios {
        assert_eq!(pr.n_sims, 64);
        assert!(!pr.power_uncorrected.is_empty());
        assert!(pr.power_uncorrected.iter().all(|p| p.is_finite()));
    }
    let total: f64 = r_auto
        .scenarios
        .iter()
        .flat_map(|(_, pr)| pr.power_uncorrected.iter())
        .sum();
    assert!(
        total > 0.0,
        "no scenario produced non-zero power — degenerate fixture"
    );

    // Determinism gate: results must be bit-equal regardless of thread count.
    assert_eq!(r1, r_auto, "results must be bit-equal across thread counts");
}

#[test]
fn find_power_cancellation_short_circuits() {
    let contracts = vec![
        minimal_ols_contract_labelled("a"),
        minimal_ols_contract_labelled("b"),
    ];
    let cancel = CancellationToken::new();
    cancel.cancel();
    let r = find_power(&contracts, 50, 25, 0, None, &cancel);
    assert!(matches!(
        r,
        Err(engine_orchestrator::OrchestratorError::Cancelled { .. })
    ));
}

// Dispatch-twin: the single-core entry (the WASM path) runs a separate sim loop,
// so it needs its own cancel checkpoint — a pre-set flag must short-circuit it
// exactly like its multi-core twin above.
#[test]
fn single_core_find_power_cancellation_short_circuits() {
    let contracts = vec![
        minimal_ols_contract_labelled("a"),
        minimal_ols_contract_labelled("b"),
    ];
    let cancel = CancellationToken::new();
    cancel.cancel();
    let r = single_core_find_power(&contracts, 50, 25, 0, None, &cancel);
    assert!(matches!(
        r,
        Err(engine_orchestrator::OrchestratorError::Cancelled { .. })
    ));
}
