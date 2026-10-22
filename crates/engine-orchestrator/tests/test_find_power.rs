//! End-to-end orchestrator test against a real engine_core::run_batch call.

mod common;
use common::{
    marginal_plus_contrast_contract, minimal_ols_contract, minimal_ols_contract_labelled,
};

use engine_orchestrator::{
    CancellationToken, NoOpSink, ProgressEvent, ProgressSink, find_power, single_core_find_power,
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
    assert_eq!(sp.success_counts_uncorrected.len(), expected);
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
/// applied), and the final boundary tick reaches the call total. Ordering is
/// pinned by `multi_core_sims_completed_is_strictly_increasing`.
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

/// The multi-core entry delivers `SimsCompleted` strictly increasing, like the
/// single-core twin. At `n_sims <= 50` the engine reports after every sim, so
/// rayon workers contend for the sink on every tick; 40 calls × 4 scenarios
/// give thousands of chances for a stale tick to slip through.
#[test]
fn multi_core_sims_completed_is_strictly_increasing() {
    #[derive(Default)]
    struct Ticks(Vec<u64>);
    impl ProgressSink for Ticks {
        fn on_event(&mut self, e: ProgressEvent) {
            if let ProgressEvent::SimsCompleted { completed, .. } = e {
                self.0.push(completed);
            }
        }
    }

    let contracts: Vec<_> = ["a", "b", "c", "d"]
        .into_iter()
        .map(minimal_ols_contract_labelled)
        .collect();
    let cancel = CancellationToken::new();
    for seed in 0..40 {
        let mut sink = Ticks::default();
        find_power(&contracts, 30, 50, seed, Some(&mut sink), &cancel).expect("ok");
        assert!(
            sink.0.windows(2).all(|w| w[0] < w[1]),
            "seed {seed}: SimsCompleted must be strictly increasing: {:?}",
            sink.0
        );
        assert_eq!(
            sink.0.last(),
            Some(&200),
            "seed {seed}: final tick reaches the total"
        );
    }
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

fn is_empty_contracts_error(
    r: &Result<
        engine_orchestrator::ScenarioResult<engine_orchestrator::PowerResult>,
        engine_orchestrator::OrchestratorError,
    >,
) -> bool {
    matches!(r, Err(engine_orchestrator::OrchestratorError::InvalidScenarios(m)) if m.contains("empty"))
}

#[test]
fn find_power_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let r = find_power(&[], 100, 50, 0, None, &cancel);
    assert!(is_empty_contracts_error(&r), "{r:?}");
}

// Dispatch-twin: the single-core entry must reject empty contracts exactly
// like its multi-core twin (the shared preflight guard must cover both paths).
#[test]
fn single_core_find_power_rejects_empty_scenarios() {
    let cancel = CancellationToken::new();
    let r = single_core_find_power(&[], 100, 50, 0, None, &cancel);
    assert!(is_empty_contracts_error(&r), "{r:?}");
}

/// Dispatch twins: at the same seed, `find_power` (`run_batch`) and
/// `single_core_find_power` (`run_batch_st`) walk the same per-sim seeds, so the
/// whole result must be bit-identical — any divergence in lowering, aggregation
/// or result assembly between the twins shows up here. Thread-count invariance
/// itself is owned by engine-core (`run_batch_bit_identical_1_vs_n_threads`).
#[test]
fn find_power_and_single_core_twin_are_bit_identical() {
    let contracts = vec![
        minimal_ols_contract_labelled("first"),
        minimal_ols_contract_labelled("second"),
    ];
    let cancel = CancellationToken::new();
    let mc = find_power(&contracts, 20, 200, 2137, None, &cancel).unwrap();
    let sc = single_core_find_power(&contracts, 20, 200, 2137, None, &cancel).unwrap();

    // Anti-vacuity: both scenarios ran and produced interior power.
    assert_eq!(mc.scenarios.len(), 2);
    for (_, pr) in &mc.scenarios {
        assert_eq!(pr.n_sims, 200);
        assert!(
            pr.power_uncorrected[0] > 0.0 && pr.power_uncorrected[0] < 1.0,
            "fixture must keep power off the boundaries, got {}",
            pr.power_uncorrected[0]
        );
    }
    assert_eq!(
        mc, sc,
        "dispatch twins must be bit-identical at the same seed"
    );
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

/// A one-term `Joint` on a one-predictor OLS model is the overall F test, and
/// F(1, n − 2) = t² on the same fit, so every sim decides the overall test
/// and the marginal the same way: the two powers are equal exactly.
#[test]
fn one_term_joint_overall_matches_marginal_power_exactly() {
    use engine_contract::TestTarget;
    let mut c = minimal_ols_contract();
    c.test.targets = vec![
        TestTarget::Marginal { term: 1 },
        TestTarget::Joint { terms: vec![1] },
    ];
    let cancel = CancellationToken::new();
    for (n, seed) in [(20, 1), (40, 2), (80, 3)] {
        let r = find_power(std::slice::from_ref(&c), n, 1600, seed, None, &cancel).expect("ok");
        let pr = &r.scenarios[0].1;
        let marginal = pr.power_uncorrected[0];
        // Power strictly inside (0, 1), so equal decisions are not trivially equal.
        assert!(
            marginal > 0.0 && marginal < 1.0,
            "n = {n}: power {marginal}"
        );
        assert_eq!(
            pr.overall_significant_rate,
            Some(marginal),
            "n = {n}: overall F must reject exactly when the marginal t does"
        );
        assert_eq!(
            pr.overall_significant_count,
            pr.success_counts_uncorrected[0]
        );
    }
}
