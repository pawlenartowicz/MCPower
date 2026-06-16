//! Pure-math tests for merge_power_results. No engine calls.

use engine_orchestrator::{
    merge_power_results, merge_sample_size_results, ByValue, Ci, CrossingFit, EstimatorExtras,
    GridMode, OrchestratorError, PosthocPower, PowerResult, SampleSizeMethod, SampleSizeResult,
    ScenarioResult,
};
use engine_orchestrator::aggregation::wilson_ci;

fn ols_pr(n_sims: u64, succ_unc: Vec<u64>, succ_cor: Vec<u64>, conv: u64) -> PowerResult {
    PowerResult {
        n: 100,
        n_sims,
        target_indices: vec![0, 1],
        contrast_pairs: vec![],
        power_uncorrected: succ_unc.iter().map(|&k| k as f64 / n_sims as f64).collect(),
        power_corrected: succ_cor.iter().map(|&k| k as f64 / n_sims as f64).collect(),
        ci_uncorrected: vec![Ci { lo: 0.0, hi: 1.0 }; 2],
        ci_corrected: vec![Ci { lo: 0.0, hi: 1.0 }; 2],
        convergence_rate: conv as f64 / n_sims as f64,
        boundary_hit: vec![0; n_sims as usize],
        estimator_extras: EstimatorExtras::Ols {},
        overall_significant_rate: None,
        success_counts_uncorrected: succ_unc,
        success_counts_corrected: succ_cor,
        convergence_count: conv,
        overall_significant_count: 0,
        overall_significant_ci: None,
        success_count_histogram_uncorrected: vec![],
        success_count_histogram_corrected: vec![],
        grid_warnings: vec![],
        posthoc: vec![],
        factor_exclusion_counts: vec![],
        factor_separation_counts: vec![],
    }
}

fn wrap(parts: Vec<PowerResult>) -> Vec<ScenarioResult<PowerResult>> {
    parts
        .into_iter()
        .map(|pr| ScenarioResult {
            scenarios: vec![("s".into(), pr)],
        })
        .collect()
}

#[test]
fn merge_pools_success_counts() {
    let a = ols_pr(100, vec![80, 60], vec![75, 55], 100);
    let b = ols_pr(200, vec![160, 110], vec![150, 100], 200);
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;
    assert_eq!(pr.n_sims, 300);
    assert_eq!(pr.success_counts_uncorrected, vec![240, 170]);
    assert_eq!(pr.success_counts_corrected, vec![225, 155]);
    assert_eq!(pr.convergence_count, 300);
    // Rate recomputed from pooled count / pooled n_sims, not averaged per-worker.
    assert_eq!(pr.convergence_rate, 1.0);
    assert!((pr.power_uncorrected[0] - 240.0 / 300.0).abs() < 1e-12);
    // Target 1 rate (currently unasserted).
    assert!(
        (pr.power_uncorrected[1] - 170.0 / 300.0).abs() < 1e-12,
        "power_unc[1]"
    );
    // CIs recomputed from pooled integer counts, not averaged per-worker floats.
    assert_eq!(pr.ci_uncorrected[0], wilson_ci(240, 300), "ci_unc[0]");
    assert_eq!(pr.ci_uncorrected[1], wilson_ci(170, 300), "ci_unc[1]");
}

#[test]
fn merge_concatenates_boundary_hit() {
    let mut a = ols_pr(2, vec![1, 0], vec![1, 0], 2);
    a.boundary_hit = vec![0, 1];
    let mut b = ols_pr(3, vec![2, 1], vec![1, 0], 3);
    b.boundary_hit = vec![0, 0, 2];
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    assert_eq!(merged.scenarios[0].1.boundary_hit, vec![0, 1, 0, 0, 2]);
}

#[test]
fn merge_rejects_mismatched_scenarios() {
    let a = ScenarioResult {
        scenarios: vec![("s1".into(), ols_pr(10, vec![5, 5], vec![5, 5], 10))],
    };
    let b = ScenarioResult {
        scenarios: vec![("s2".into(), ols_pr(10, vec![5, 5], vec![5, 5], 10))],
    };
    let r = merge_power_results(&[a, b]);
    assert!(matches!(r, Err(OrchestratorError::IncompatibleMerge(_))));
}

/// Contrast entries ride after the marginals in the power/count vectors
/// (one per contrast_pairs entry). The merge must pool ALL slots — before the
/// fix it sized the pool by target_indices.len() alone and silently truncated
/// the contrast tail in multi-worker (WASM) merges.
#[test]
fn merge_pools_contrast_entries() {
    // 1 marginal + 1 contrast → vectors of length 2.
    let mk = |n_sims: u64, succ: Vec<u64>| {
        let mut pr = ols_pr(n_sims, succ.clone(), succ, n_sims);
        pr.target_indices = vec![1];
        pr.contrast_pairs = vec![(2, 1)];
        pr.ci_uncorrected = vec![Ci { lo: 0.0, hi: 1.0 }; 2];
        pr.ci_corrected = vec![Ci { lo: 0.0, hi: 1.0 }; 2];
        pr
    };
    let merged =
        merge_power_results(&wrap(vec![mk(100, vec![80, 20]), mk(100, vec![70, 30])])).unwrap();
    let pr = &merged.scenarios[0].1;
    assert_eq!(pr.contrast_pairs, vec![(2, 1)]);
    assert_eq!(pr.success_counts_uncorrected, vec![150, 50]);
    assert_eq!(pr.power_uncorrected.len(), 2, "contrast slot must survive the merge");
    assert!((pr.power_uncorrected[1] - 50.0 / 200.0).abs() < 1e-12);
}

#[test]
fn merge_rejects_mismatched_contrast_pairs() {
    let mut a = ols_pr(10, vec![5, 5], vec![5, 5], 10);
    a.contrast_pairs = vec![(2, 1)];
    let b = ols_pr(10, vec![5, 5], vec![5, 5], 10);
    let r = merge_power_results(&wrap(vec![a, b]));
    assert!(matches!(r, Err(OrchestratorError::IncompatibleMerge(_))));
}

/// merge_power_results is order-independent — pooling `[A, B]` and
/// `[B, A]` yields identical pooled rates (counter sums commute).
#[test]
fn merge_is_order_independent() {
    let a = ols_pr(100, vec![80, 60], vec![75, 55], 100);
    let b = ols_pr(200, vec![160, 110], vec![150, 100], 200);
    let ab = merge_power_results(&wrap(vec![a.clone(), b.clone()])).unwrap();
    let ba = merge_power_results(&wrap(vec![b, a])).unwrap();
    let pr_ab = &ab.scenarios[0].1;
    let pr_ba = &ba.scenarios[0].1;
    assert_eq!(
        pr_ab.success_counts_uncorrected,
        pr_ba.success_counts_uncorrected
    );
    assert_eq!(
        pr_ab.success_counts_corrected,
        pr_ba.success_counts_corrected
    );
    assert_eq!(pr_ab.power_uncorrected, pr_ba.power_uncorrected);
    assert_eq!(pr_ab.power_corrected, pr_ba.power_corrected);
    assert_eq!(pr_ab.n_sims, pr_ba.n_sims);
}

fn ssr_grid(parts_per_n: Vec<PowerResult>) -> SampleSizeResult {
    SampleSizeResult {
        grid_or_trace: parts_per_n,
        first_achieved: vec![None],
        first_joint_achieved: vec![],
        fitted: vec![],
        fitted_joint: vec![],
        first_overall_achieved: None,
        fitted_overall: None,
        cluster_atom: 1,
        target_power: 0.8,
        method: SampleSizeMethod::Grid {
            by: ByValue::Fixed(100),
            mode: GridMode::Linear,
        },
        grid_warnings: vec![],
    }
}

fn wrap_ssr(parts: Vec<SampleSizeResult>) -> Vec<ScenarioResult<SampleSizeResult>> {
    parts
        .into_iter()
        .map(|r| ScenarioResult {
            scenarios: vec![("s".into(), r)],
        })
        .collect()
}

#[test]
fn merge_sample_size_results_grid_pools_per_n() {
    // Two parts, grid [100, 200, 300]. At each N, the pooled counts should
    // sum, and `first_achieved` should be recomputed from the pooled rates.
    let mk = |n: usize, n_sims: u64, k_unc: u64| {
        let mut pr = ols_pr(
            n_sims,
            vec![k_unc, k_unc / 2],
            vec![k_unc, k_unc / 2],
            n_sims,
        );
        pr.n = n;
        pr
    };

    // Part A: 100 sims at each N. Per-target k = 50/80/95 (rates 0.5/0.8/0.95).
    let part_a = ssr_grid(vec![mk(100, 100, 50), mk(200, 100, 80), mk(300, 100, 95)]);
    // Part B: 100 sims at each N. Per-target k = 60/82/98.
    let part_b = ssr_grid(vec![mk(100, 100, 60), mk(200, 100, 82), mk(300, 100, 98)]);
    let merged = merge_sample_size_results(&wrap_ssr(vec![part_a, part_b])).unwrap();
    let r = &merged.scenarios[0].1;
    assert_eq!(r.grid_or_trace.len(), 3);
    assert_eq!(r.grid_or_trace[0].n_sims, 200);
    assert_eq!(r.grid_or_trace[0].success_counts_uncorrected, vec![110, 55]);
    assert_eq!(r.grid_or_trace[1].success_counts_uncorrected, vec![162, 81]);
    assert_eq!(r.grid_or_trace[2].success_counts_uncorrected, vec![193, 96]);
    // first_achieved recomputed from pooled corrected rates: target 0 first
    // crosses 0.8 at N=200 (162/200 = 0.81), target 1 never reaches 0.8.
    assert_eq!(r.first_achieved, vec![Some(200), None]);
}

#[test]
fn merge_recomputes_crossing_fit_from_pooled_counts_golden() {
    // Pooled target-0 counts land exactly on the fit.rs golden fixture
    // (p̂ = [0.60, 0.75, 0.85, 0.95] of n_sims = 100 over grid [50,100,150,200],
    // target 0.8): n_star = 125, CI by Wilson band inversion. Golden values
    // computed externally with the same Wilson (z = 1.96) formula. Proves the
    // merge path runs the fit on POOLED counts, not on any per-part value.
    let mk = |n: usize, k0: u64, k1: u64| {
        let mut pr = ols_pr(50, vec![k0, k1], vec![k0, k1], 50);
        pr.n = n;
        pr
    };
    let part_a = ssr_grid(vec![mk(50, 30, 5), mk(100, 40, 10), mk(150, 45, 15), mk(200, 50, 20)]);
    let part_b = ssr_grid(vec![mk(50, 30, 5), mk(100, 35, 10), mk(150, 40, 15), mk(200, 45, 20)]);
    let merged = merge_sample_size_results(&wrap_ssr(vec![part_a, part_b])).unwrap();
    let r = &merged.scenarios[0].1;
    // Pooled corrected counts: [60, 75, 85, 95] of 100 — the golden series.
    let pooled: Vec<u64> = r
        .grid_or_trace
        .iter()
        .map(|pr| pr.success_counts_corrected[0])
        .collect();
    assert_eq!(pooled, vec![60, 75, 85, 95]);
    match &r.fitted[0] {
        CrossingFit::Fitted {
            n_star,
            n_achievable,
            ci_lo,
            ci_hi,
        } => {
            assert!((n_star - 125.0).abs() < 1e-9, "n_star {n_star}");
            assert_eq!(*n_achievable, 125);
            assert!((ci_lo.unwrap() - 90.83640209948472).abs() < 1e-9);
            assert!((ci_hi.unwrap() - 163.5595794316606).abs() < 1e-9);
        }
        other => panic!("expected Fitted for target 0, got {other:?}"),
    }
    // Target 1 pools to [10, 20, 30, 40] of 100: never reaches 0.8 and the
    // fitted endpoint 0.4 sits under the low-power hint gate ⇒ no hint.
    assert_eq!(r.fitted[1], CrossingFit::NotReached { n_approx: None });
    assert_eq!(r.cluster_atom, 1, "atom carried through merge");
}

#[test]
fn merge_redrives_overall_crossing_from_pooled_counts() {
    // Per-part overall counts pool to [60, 80, 95] of n_sims = 200 over grid
    // [50,100,150]. The merge path must re-derive fitted_overall and
    // first_overall_achieved from the POOLED overall counts, not copy a part.
    let mk = |n: usize, k_overall: u64| {
        let mut pr = ols_pr(100, vec![0, 0], vec![0, 0], 100);
        pr.n = n;
        pr.overall_significant_rate = Some(k_overall as f64 / 100.0);
        pr.overall_significant_count = k_overall;
        pr
    };
    // Part A overall counts 30/40/47, Part B 30/40/48 ⇒ pooled 60/80/95.
    let part_a = ssr_grid(vec![mk(50, 30), mk(100, 40), mk(150, 47)]);
    let part_b = ssr_grid(vec![mk(50, 30), mk(100, 40), mk(150, 48)]);
    let merged = merge_sample_size_results(&wrap_ssr(vec![part_a, part_b])).unwrap();
    let r = &merged.scenarios[0].1;
    let pooled: Vec<u64> = r
        .grid_or_trace
        .iter()
        .map(|pr| pr.overall_significant_count)
        .collect();
    assert_eq!(pooled, vec![60, 80, 95]);
    // Pooled overall rates 0.30/0.40/0.475 — never reach 0.8, and the fitted
    // endpoint 0.475 sits under the low-power hint gate ⇒ NotReached, no hint.
    // That this is derived (not a copied per-part value) is the point: each
    // part's own endpoint rate was 0.47/0.48, neither equal to the pooled fit.
    assert_eq!(
        r.fitted_overall,
        Some(CrossingFit::NotReached { n_approx: None })
    );
    assert_eq!(r.first_overall_achieved, None);
}

#[test]
fn merge_sample_size_rejects_mismatched_cluster_atom() {
    let mk_part = |atom: usize| {
        let mut pr = ols_pr(100, vec![50, 25], vec![50, 25], 100);
        pr.n = 100;
        let mut r = ssr_grid(vec![pr]);
        r.cluster_atom = atom;
        r
    };
    let res = merge_sample_size_results(&wrap_ssr(vec![mk_part(1), mk_part(30)]));
    match res {
        Err(OrchestratorError::IncompatibleMerge(msg)) => {
            assert!(msg.contains("cluster_atom"), "message names the field: {msg}")
        }
        other => panic!("expected IncompatibleMerge, got {other:?}"),
    }
}

#[test]
fn merge_sample_size_rejects_mismatched_target_power() {
    let mk_part = |target_power: f64| {
        let mut r = ssr_grid(vec![ols_pr(100, vec![50, 25], vec![50, 25], 100)]);
        r.target_power = target_power;
        r
    };
    let res = merge_sample_size_results(&wrap_ssr(vec![mk_part(0.8), mk_part(0.9)]));
    match res {
        Err(OrchestratorError::IncompatibleMerge(msg)) => {
            assert!(msg.contains("target_power"), "message names the field: {msg}")
        }
        other => panic!("expected IncompatibleMerge, got {other:?}"),
    }
}

#[test]
fn merge_rejects_mismatched_n() {
    let a = ScenarioResult {
        scenarios: vec![("s".into(), {
            let mut pr = ols_pr(10, vec![5, 5], vec![5, 5], 10);
            pr.n = 100;
            pr
        })],
    };
    let b = ScenarioResult {
        scenarios: vec![("s".into(), {
            let mut pr = ols_pr(10, vec![5, 5], vec![5, 5], 10);
            pr.n = 200;
            pr
        })],
    };
    let r = merge_power_results(&[a, b]);
    assert!(matches!(r, Err(OrchestratorError::IncompatibleMerge(_))));
}

/// Histograms are pooled elementwise; overall CI recomputed from pooled counts
/// (not averaged). ols_pr has target_indices=[0,1], so n_buckets=3.
#[test]
fn merge_pools_histogram_elementwise_and_recomputes_overall_ci() {
    let mut a = ols_pr(4, vec![3, 2], vec![3, 2], 4);
    let mut b = ols_pr(4, vec![1, 4], vec![1, 4], 4);
    a.success_count_histogram_uncorrected = vec![1, 1, 2];
    a.success_count_histogram_corrected = vec![1, 1, 2];
    b.success_count_histogram_uncorrected = vec![0, 2, 2];
    b.success_count_histogram_corrected = vec![0, 2, 2];
    a.overall_significant_rate = Some(3.0 / 4.0);
    a.overall_significant_count = 3;
    b.overall_significant_rate = Some(2.0 / 4.0);
    b.overall_significant_count = 2;

    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;

    assert_eq!(pr.success_count_histogram_uncorrected, vec![1, 3, 4]);
    assert_eq!(pr.success_count_histogram_corrected, vec![1, 3, 4]);
    assert_eq!(pr.overall_significant_ci, Some(wilson_ci(5, 8)));
    // Rate recomputed from pooled count / pooled n_sims (5/8), not averaged.
    assert_eq!(pr.overall_significant_rate, Some(5.0 / 8.0));
}

/// overall_significant_ci is None when no overall test was run
/// (overall_significant_rate == None in all parts).
#[test]
fn merge_overall_ci_none_when_no_overall_test() {
    // ols_pr defaults: overall_significant_rate = None, overall_significant_count = 0
    let a = ols_pr(4, vec![3, 2], vec![3, 2], 4);
    let b = ols_pr(4, vec![1, 4], vec![1, 4], 4);
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    assert_eq!(merged.scenarios[0].1.overall_significant_ci, None);
}

/// Joint histogram spanning post-hoc contrasts (len > n_targets+1) must pool
/// elementwise, not be rejected. ols_pr has target_indices=[0,1]; with 3 post-hoc
/// contrasts the histogram is n_targets(2) + total_contrasts(3) + 1 = 6 buckets.
#[test]
fn merge_pools_posthoc_spanning_histogram() {
    let mut a = ols_pr(4, vec![3, 2], vec![3, 2], 4);
    let mut b = ols_pr(4, vec![1, 4], vec![1, 4], 4);
    a.success_count_histogram_uncorrected = vec![1, 0, 1, 1, 1, 0]; // sums to 4
    a.success_count_histogram_corrected = vec![1, 0, 1, 1, 1, 0];
    b.success_count_histogram_uncorrected = vec![0, 1, 1, 0, 1, 1]; // sums to 4
    b.success_count_histogram_corrected = vec![0, 1, 1, 0, 1, 1];
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;
    assert_eq!(
        pr.success_count_histogram_uncorrected,
        vec![1, 1, 2, 1, 2, 1]
    );
    assert_eq!(pr.success_count_histogram_corrected, vec![1, 1, 2, 1, 2, 1]);
    assert_eq!(
        pr.success_count_histogram_uncorrected.iter().sum::<u64>(),
        8
    );
}

/// A nonempty histogram with the wrong length (not 0 and not n_buckets) must be
/// rejected loudly — silent zeros from skipping would corrupt pooled counts.
/// ols_pr has target_indices=[0,1], so n_buckets=3; length 2 is wrong.
#[test]
fn merge_rejects_nonempty_wrong_length_histogram() {
    let mut a = ols_pr(4, vec![3, 2], vec![3, 2], 4);
    let mut b = ols_pr(4, vec![1, 4], vec![1, 4], 4);
    a.success_count_histogram_uncorrected = vec![1, 1, 2]; // correct length 3
    a.success_count_histogram_corrected = vec![1, 1, 2];
    b.success_count_histogram_uncorrected = vec![1, 2]; // WRONG length 2 (nonempty)
    b.success_count_histogram_corrected = vec![1, 2];
    assert!(matches!(
        merge_power_results(&wrap(vec![a, b])),
        Err(OrchestratorError::IncompatibleMerge(_))
    ));
}

/// Helper: build a PowerResult with one posthoc block (3 contrasts).
fn pr_with_posthoc(n_sims: u64, counts_unc: Vec<u64>, counts_cor: Vec<u64>) -> PowerResult {
    let mut pr = ols_pr(n_sims, vec![3, 2], vec![3, 2], n_sims);
    pr.posthoc = vec![PosthocPower {
        n_levels: 3,
        power_uncorrected: counts_unc
            .iter()
            .map(|&k| k as f64 / n_sims as f64)
            .collect(),
        power_corrected: counts_cor
            .iter()
            .map(|&k| k as f64 / n_sims as f64)
            .collect(),
        ci_uncorrected: vec![Ci { lo: 0.0, hi: 1.0 }; counts_unc.len()],
        ci_corrected: vec![Ci { lo: 0.0, hi: 1.0 }; counts_cor.len()],
        success_counts_uncorrected: counts_unc,
        success_counts_corrected: counts_cor,
    }];
    pr
}

/// Merge pools posthoc success counts elementwise and recomputes power/CI.
/// Two parts (n_sims=4 each): counts_unc [3,0,1] + [2,1,1] = [5,1,2]; n_sims_total=8.
#[test]
fn merge_sums_posthoc_success_counts() {
    let a = pr_with_posthoc(4, vec![3, 0, 1], vec![2, 0, 0]);
    let b = pr_with_posthoc(4, vec![2, 1, 1], vec![1, 1, 0]);
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;
    assert_eq!(pr.posthoc.len(), 1);
    let ph = &pr.posthoc[0];
    assert_eq!(ph.success_counts_uncorrected, vec![5, 1, 2]);
    assert_eq!(ph.success_counts_corrected, vec![3, 1, 0]);
    assert!(
        (ph.power_uncorrected[0] - 5.0 / 8.0).abs() < 1e-12,
        "power_unc[0]"
    );
    assert_eq!(ph.power_uncorrected[1], 1.0 / 8.0);
    assert_eq!(ph.power_corrected[2], 0.0);
    assert_eq!(ph.ci_uncorrected[0], wilson_ci(5, 8));
    assert_eq!(ph.n_levels, 3);
}

#[test]
fn merge_rejects_zero_n_sims() {
    // All parts carry n_sims=0. Without the guard, every rate field is NaN/inf.
    let zero = PowerResult {
        n: 100,
        n_sims: 0,
        target_indices: vec![0],
        contrast_pairs: vec![],
        power_uncorrected: vec![0.0],
        power_corrected: vec![0.0],
        ci_uncorrected: vec![Ci { lo: 0.0, hi: 1.0 }],
        ci_corrected: vec![Ci { lo: 0.0, hi: 1.0 }],
        convergence_rate: 1.0,
        boundary_hit: vec![],
        estimator_extras: EstimatorExtras::Ols {},
        overall_significant_rate: None,
        success_counts_uncorrected: vec![0],
        success_counts_corrected: vec![0],
        convergence_count: 0,
        overall_significant_count: 0,
        overall_significant_ci: None,
        success_count_histogram_uncorrected: vec![],
        success_count_histogram_corrected: vec![],
        grid_warnings: vec![],
        posthoc: vec![],
        factor_exclusion_counts: vec![],
        factor_separation_counts: vec![],
    };
    let parts = vec![
        ScenarioResult {
            scenarios: vec![("s".into(), zero.clone())],
        },
        ScenarioResult {
            scenarios: vec![("s".into(), zero)],
        },
    ];
    assert!(matches!(
        merge_power_results(&parts),
        Err(OrchestratorError::IncompatibleMerge(_))
    ));
}

/// Merge pools factor_exclusion_counts and factor_separation_counts elementwise.
/// Two parts: exclusion_counts [3,0]/[2,1] → [5,1]; mismatch → IncompatibleMerge.
#[test]
fn merge_pools_factor_exclusion_counts() {
    let mut a = ols_pr(10, vec![8, 6], vec![7, 5], 10);
    let mut b = ols_pr(10, vec![7, 5], vec![6, 4], 10);
    a.factor_exclusion_counts = vec![3, 0];
    a.factor_separation_counts = vec![1, 2];
    b.factor_exclusion_counts = vec![2, 1];
    b.factor_separation_counts = vec![0, 3];

    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;
    assert_eq!(pr.factor_exclusion_counts, vec![5, 1]);
    assert_eq!(pr.factor_separation_counts, vec![1, 5]);
}

#[test]
fn merge_rejects_mismatched_factor_exclusion_counts_length() {
    let mut a = ols_pr(10, vec![8, 6], vec![7, 5], 10);
    let mut b = ols_pr(10, vec![7, 5], vec![6, 4], 10);
    a.factor_exclusion_counts = vec![3, 0];
    a.factor_separation_counts = vec![1, 2];
    b.factor_exclusion_counts = vec![2]; // WRONG length
    b.factor_separation_counts = vec![0, 3];

    assert!(matches!(
        merge_power_results(&wrap(vec![a, b])),
        Err(OrchestratorError::IncompatibleMerge(_))
    ));
}

/// Asymmetric empty-vs-populated factor counts (an older payload meeting a
/// new one) must be rejected, not silently pooled — unlike the histogram gap
/// below, the strict length guard covers this case.
#[test]
fn merge_rejects_asymmetric_empty_factor_counts() {
    let mut a = ols_pr(10, vec![8, 6], vec![7, 5], 10);
    let b = ols_pr(10, vec![7, 5], vec![6, 4], 10);
    a.factor_exclusion_counts = vec![3, 0];
    a.factor_separation_counts = vec![0, 0];
    // b: factor_* remain vec![] (ols_pr default) — older payload shape.

    assert!(matches!(
        merge_power_results(&wrap(vec![a, b])),
        Err(OrchestratorError::IncompatibleMerge(_))
    ));
}

/// Fixtures built with `mle_pr` model zero bh==2 (hard-fail) boundary hits, so every
/// boundary hit is a pinned convergence (bh==1) and `boundary_hits` coincides with
/// `singular_count`. Callers needing bh==2 cases must set `boundary_hits` independently.
fn mle_pr(n_sims: u64, singular_count: u64, singular_n: u64) -> PowerResult {
    PowerResult {
        n: 100,
        n_sims,
        target_indices: vec![0],
        contrast_pairs: vec![],
        power_uncorrected: vec![0.5],
        power_corrected: vec![0.5],
        ci_uncorrected: vec![Ci { lo: 0.0, hi: 1.0 }],
        ci_corrected: vec![Ci { lo: 0.0, hi: 1.0 }],
        convergence_rate: singular_n as f64 / n_sims as f64,
        boundary_hit: vec![0; n_sims as usize],
        estimator_extras: EstimatorExtras::Mle {
            tau_estimate: 0.0,
            boundary_hits: singular_count, // bh==1 only — see helper doc comment above
            joint_uncorrected_rate: 0.0,
            joint_corrected_rate: 0.0,
            tau_sum: 0.0,
            tau_n: 0,
            joint_uncorrected_count: 0,
            joint_corrected_count: 0,
            singular_fit_rate: if singular_n > 0 {
                singular_count as f64 / singular_n as f64
            } else {
                0.0
            },
            singular_count,
            singular_n,
            boundary_rate_per_component: vec![],
            boundary_component_counts: vec![],
        },
        overall_significant_rate: None,
        success_counts_uncorrected: vec![n_sims / 2],
        success_counts_corrected: vec![n_sims / 2],
        convergence_count: singular_n,
        overall_significant_count: 0,
        overall_significant_ci: None,
        success_count_histogram_uncorrected: vec![],
        success_count_histogram_corrected: vec![],
        grid_warnings: vec![],
        posthoc: vec![],
        factor_exclusion_counts: vec![],
        factor_separation_counts: vec![],
    }
}

/// Mle extras: singular_count / singular_n pool by summation; singular_fit_rate
/// is recomputed from the pooled Σ/Σ (not averaged). Parts: (1,3) + (2,4) → (3,7), rate=3/7.
#[test]
fn merge_mle_extras_singular_fit_pools_counts_and_recomputes_rate() {
    let a = mle_pr(4, 1, 3);
    let b = mle_pr(4, 2, 4);
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let pr = &merged.scenarios[0].1;
    let EstimatorExtras::Mle {
        singular_count,
        singular_n,
        singular_fit_rate,
        ..
    } = &pr.estimator_extras
    else {
        panic!("expected Mle extras");
    };
    assert_eq!(*singular_count, 3);
    assert_eq!(*singular_n, 7);
    assert!(
        (*singular_fit_rate - 3.0 / 7.0).abs() < 1e-12,
        "rate={singular_fit_rate}"
    );
}

/// Glm extras carry three independently-pooled fields, each a Σ/Σ recomputed
/// from per-part sums (not averaged): `baseline_prob_realized` = Σsum/Σn,
/// `singular_fit_rate` = Σcount/Σn, `tau_squared_hat_mean` = Σsum/Σn. The
/// per-part rate/realized/mean fields are ignored by merge and recomputed.
fn glm_pr(
    n_sims: u64,
    baseline_sum: f64,
    baseline_n: u64,
    singular_count: u64,
    singular_n: u64,
    tau_sum: f64,
    tau_n: u64,
) -> PowerResult {
    let mut pr = mle_pr(n_sims, singular_count, singular_n);
    pr.estimator_extras = EstimatorExtras::Glm {
        baseline_prob_realized: if baseline_n > 0 {
            baseline_sum / baseline_n as f64
        } else {
            0.0
        },
        baseline_prob_sum: baseline_sum,
        baseline_prob_n: baseline_n,
        singular_fit_rate: if singular_n > 0 {
            singular_count as f64 / singular_n as f64
        } else {
            0.0
        },
        singular_count,
        singular_n,
        tau_squared_hat_mean: if tau_n > 0 {
            tau_sum / tau_n as f64
        } else {
            f64::NAN
        },
        tau_squared_hat_sum: tau_sum,
        tau_squared_hat_n: tau_n,
    };
    pr
}

/// Glm merge pools all three Σ/Σ diagnostics. Parts:
/// baseline (2.0,4)+(4.0,6) → 6.0/10 = 0.6; singular (1,3)+(2,4) → 3/7;
/// tau (1.5,2)+(2.5,3) → 4.0/5 = 0.8. Asserting the recomputed means *and* the
/// pooled sums/counts pins the accumulation operators and the divide.
#[test]
fn merge_glm_extras_pools_baseline_singular_tau() {
    let a = glm_pr(4, 2.0, 4, 1, 3, 1.5, 2);
    let b = glm_pr(4, 4.0, 6, 2, 4, 2.5, 3);
    let merged = merge_power_results(&wrap(vec![a, b])).unwrap();
    let EstimatorExtras::Glm {
        baseline_prob_realized,
        baseline_prob_sum,
        baseline_prob_n,
        singular_fit_rate,
        singular_count,
        singular_n,
        tau_squared_hat_mean,
        tau_squared_hat_sum,
        tau_squared_hat_n,
    } = &merged.scenarios[0].1.estimator_extras
    else {
        panic!("expected Glm extras");
    };
    assert_eq!((*baseline_prob_sum, *baseline_prob_n), (6.0, 10));
    assert!((*baseline_prob_realized - 0.6).abs() < 1e-12, "realized={baseline_prob_realized}");
    assert_eq!((*singular_count, *singular_n), (3, 7));
    assert!((*singular_fit_rate - 3.0 / 7.0).abs() < 1e-12, "rate={singular_fit_rate}");
    assert_eq!((*tau_squared_hat_sum, *tau_squared_hat_n), (4.0, 5));
    assert!((*tau_squared_hat_mean - 0.8).abs() < 1e-12, "tau_mean={tau_squared_hat_mean}");
}

/// A part built without the per-sim counters (empty histogram) must NOT merge with
/// a populated one — that asymmetry silently breaks the bucket-sum invariant
/// (Σ buckets == n_sims), so the merge rejects it. (Was a documented tolerant gap.)
#[test]
fn merge_rejects_asymmetric_empty_histogram() {
    let mut a = ols_pr(4, vec![3, 2], vec![3, 2], 4);
    let b = ols_pr(4, vec![1, 1], vec![1, 1], 4);
    a.success_count_histogram_uncorrected = vec![1, 1, 2];
    a.success_count_histogram_corrected = vec![1, 1, 2];
    // b: success_count_histogram_* remains vec![] (ols_pr default)
    assert!(matches!(
        merge_power_results(&wrap(vec![a, b])),
        Err(OrchestratorError::IncompatibleMerge(_))
    ));
}
