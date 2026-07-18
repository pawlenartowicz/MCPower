use engine_orchestrator::aggregation::wilson_ci;

/// Wilson CI is always bounded to [0,1] and ordered (lo <= hi),
/// for any (k, n) with k <= n. Property, not a value-match.
#[test]
fn wilson_ci_is_bounded_and_ordered() {
    for &n in &[1u64, 2, 10, 100, 1000, 50_000] {
        for &k in &[0u64, 1, n / 4, n / 2, (3 * n) / 4, n] {
            if k > n {
                continue;
            }
            let ci = wilson_ci(k, n);
            assert!(ci.lo >= 0.0, "lo {} < 0 for (k={k}, n={n})", ci.lo);
            assert!(ci.hi <= 1.0, "hi {} > 1 for (k={k}, n={n})", ci.hi);
            assert!(
                ci.lo <= ci.hi,
                "lo {} > hi {} for (k={k}, n={n})",
                ci.lo,
                ci.hi
            );
        }
    }
    // Monotonicity: lo and hi are non-decreasing as k increases for fixed n.
    for &n in &[10u64, 100u64] {
        let cis: Vec<_> = (0..=n).map(|k| wilson_ci(k, n)).collect();
        for i in 1..cis.len() {
            assert!(
                cis[i].lo >= cis[i - 1].lo - 1e-12,
                "lo not monotone at (k={i}, n={n}): {} < {}",
                cis[i].lo,
                cis[i - 1].lo
            );
            assert!(
                cis[i].hi >= cis[i - 1].hi - 1e-12,
                "hi not monotone at (k={i}, n={n}): {} < {}",
                cis[i].hi,
                cis[i - 1].hi
            );
        }
    }
}

/// n == 0 returns the full-uncertainty fallback (0.0, 1.0).
#[test]
fn wilson_ci_n_zero_full_uncertainty() {
    let ci = wilson_ci(0, 0);
    assert_eq!(ci.lo, 0.0);
    assert_eq!(ci.hi, 1.0);
}

/// For an interior proportion (0 < p < 1) the point estimate p = k/n lies
/// inside its own Wilson CI. (At the degenerate boundaries p=0 or p=1 the
/// Wilson interval is intentionally one-sided and need not contain the raw
/// extreme — that is a property of the score interval, not a bug.)
#[test]
fn wilson_ci_contains_point_estimate() {
    for &n in &[5u64, 37, 100, 999, 10_000] {
        for &k in &[1u64, n / 3, n / 2, n - 1] {
            if k == 0 || k >= n {
                continue;
            }
            let p = k as f64 / n as f64;
            let ci = wilson_ci(k, n);
            assert!(
                ci.lo <= p && p <= ci.hi,
                "interior point {p} outside CI [{}, {}] for (k={k}, n={n})",
                ci.lo,
                ci.hi
            );
        }
    }
}

/// Golden-value test against hand-computed Wilson CI with z=1.96.
/// Catches any wrong z-constant (1.645, 1.0), sign flip, or denominator error.
/// Tolerance 1e-3; nearest wrong-z answer (z=1.645 gives lo≈0.424) is >0.02 away.
/// Note: statsmodels proportion_confint rounds to fewer digits; the reference
/// values here are computed directly from the formula with z=1.96 and verified
/// to match the implementation to 1e-12.
#[test]
fn wilson_ci_known_values() {
    // k=7, n=10, z=1.96: lo ≈ 0.39677, hi ≈ 0.89227. Hand-derived in plan T-01.
    // z=1.645 gives lo≈0.424 (fails); z=1.0 gives lo≈0.494 (fails).
    let ci = wilson_ci(7, 10);
    assert!(
        (ci.lo - 0.3968).abs() < 1e-3,
        "lo={}, expected ≈0.3968",
        ci.lo
    );
    assert!(
        (ci.hi - 0.8923).abs() < 1e-3,
        "hi={}, expected ≈0.8923",
        ci.hi
    );
    // k=5, n=10: width with z=1.96 ≈ 0.5268. 90%-CI width ≈0.415; z=1.0 ≈0.282 — both fail.
    let ci2 = wilson_ci(5, 10);
    let width = ci2.hi - ci2.lo;
    assert!(
        (width - 0.5268).abs() < 1e-3,
        "width={width}, expected ≈0.5268"
    );
}

use engine_core::{BatchResult, EstimatorSpec, ResultShape};
use engine_orchestrator::aggregation::aggregate_batch;
use engine_orchestrator::EstimatorExtras;

fn make_batch(unc: Vec<u8>, cor: Vec<u8>, converged: Vec<u8>, shape: ResultShape) -> BatchResult {
    // Boundary/joint buffers are read element-wise by `aggregate_batch`
    // (`bh[sim * n_ss + ss_idx]`), so they must be sized `n_sims × n_ss` —
    // not empty. Initialise to all zeros (the "no boundary, no joint sig"
    // Ols default).
    let n_sims_ss = (shape.n_sims as usize) * (shape.n_sample_sizes as usize);
    BatchResult {
        uncorrected: unc,
        corrected: cor,
        posthoc_unc: vec![],
        posthoc_cor: vec![],
        converged,
        boundary_hit: vec![0u8; n_sims_ss],
        pinned_components: vec![0u64; n_sims_ss],
        joint_unc: vec![0u8; n_sims_ss],
        joint_cor: vec![0u8; n_sims_ss],
        overall: vec![],
        factor_excluded: vec![],
        tau_squared_hat: vec![],
        shape,
    }
}

/// Helper for histogram + overall-CI tests.
/// Layout: row-major `[n_sims, n_ss=1, n_targets]` for unc/cor;
///         row-major `[n_sims, n_ss=1]` for overall (may be empty).
fn batch_with(n_sims: u32, n_targets: u32, unc: &[u8], cor: &[u8], overall: &[u8]) -> BatchResult {
    let shape = ResultShape {
        n_sims,
        n_sample_sizes: 1,
        n_targets,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let n_sims_ss = n_sims as usize;
    BatchResult {
        uncorrected: unc.to_vec(),
        corrected: cor.to_vec(),
        posthoc_unc: vec![],
        posthoc_cor: vec![],
        converged: vec![1u8; n_sims_ss],
        boundary_hit: vec![0u8; n_sims_ss],
        pinned_components: vec![0u64; n_sims_ss],
        joint_unc: vec![0u8; n_sims_ss],
        joint_cor: vec![0u8; n_sims_ss],
        overall: overall.to_vec(),
        factor_excluded: vec![],
        tau_squared_hat: vec![],
        shape,
    }
}

#[test]
fn aggregate_batch_builds_joint_histogram_and_overall_ci() {
    // 1 sample-size, 2 targets, 4 sims; per-(sim,target) significance:
    //   sim0: t0=1 t1=1 -> 2 ; sim1: t0=1 t1=0 -> 1 ; sim2: 0,0 -> 0 ; sim3: 1,1 -> 2
    // overall vector [1,1,0,1] (3 of 4 significant).
    let batch = batch_with(
        /* n_sims */ 4,
        /* n_targets */ 2,
        /* uncorrected */ &[1, 1, 1, 0, 0, 0, 1, 1],
        /* corrected   */ &[1, 1, 1, 0, 0, 0, 1, 1],
        /* overall      */ &[1, 1, 0, 1],
    );
    let results = aggregate_batch(&batch, &[0, 1], &[], &EstimatorSpec::Ols);
    let r = &results[0];

    assert_eq!(r.success_count_histogram_uncorrected, vec![1, 1, 2]);
    assert_eq!(r.success_count_histogram_corrected, vec![1, 1, 2]);
    assert_eq!(
        r.success_count_histogram_uncorrected.len(),
        r.target_indices.len() + 1
    );
    assert_eq!(r.overall_significant_ci, Some(wilson_ci(3, 4)));
}

#[test]
fn aggregate_batch_overall_ci_none_when_no_overall_test() {
    let batch = batch_with(
        4,
        2,
        &[1, 1, 1, 0, 0, 0, 1, 1],
        &[1, 1, 1, 0, 0, 0, 1, 1],
        &[],
    ); // empty overall
    let r = &aggregate_batch(&batch, &[0, 1], &[], &EstimatorSpec::Ols)[0];
    assert_eq!(r.overall_significant_ci, None);
}

#[test]
fn aggregate_single_n_single_target() {
    let shape = ResultShape {
        n_sims: 10,
        n_sample_sizes: 1,
        n_targets: 1,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let unc = vec![1, 1, 1, 1, 1, 1, 1, 0, 0, 0]; // 7/10
    let cor = vec![1, 1, 1, 1, 1, 1, 0, 0, 0, 0]; // 6/10
    let conv = vec![1u8; 10];
    let br = make_batch(unc, cor, conv, shape);
    let aggs = aggregate_batch(&br, &[1], &[], &EstimatorSpec::Ols);
    assert_eq!(aggs.len(), 1);
    let pr = &aggs[0];
    approx::assert_relative_eq!(pr.power_uncorrected[0], 0.7);
    approx::assert_relative_eq!(pr.power_corrected[0], 0.6);
    approx::assert_relative_eq!(pr.convergence_rate, 1.0);
    assert_eq!(pr.target_indices, vec![1]);
    assert_eq!(pr.n_sims, 10);
    assert!(matches!(pr.estimator_extras, EstimatorExtras::Ols {}));
    let expected = engine_orchestrator::aggregation::wilson_ci(7, 10);
    approx::assert_relative_eq!(pr.ci_uncorrected[0].lo, expected.lo);
    approx::assert_relative_eq!(pr.ci_uncorrected[0].hi, expected.hi);
}

#[test]
fn aggregate_multi_n_multi_target_layout() {
    // 2 sims × 3 sample sizes × 2 targets, row-major.
    // index = sim * (n_ss * n_targets) + ss_idx * n_targets + t
    let shape = ResultShape {
        n_sims: 2,
        n_sample_sizes: 3,
        n_targets: 2,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let mut unc = vec![0u8; 2 * 3 * 2];
    unc[0] = 1; // sim 0, ss 0, target 0 → index 0*6 + 0*2 + 0 = 0
    unc[9] = 1; // sim 1, ss 1, target 1 → index 1*6 + 1*2 + 1 = 9
    let cor = unc.clone();
    let conv = vec![1u8; 2 * 3];
    let br = make_batch(unc, cor, conv, shape);
    let aggs = aggregate_batch(&br, &[1, 2], &[], &EstimatorSpec::Ols);
    assert_eq!(aggs.len(), 3);
    approx::assert_relative_eq!(aggs[0].power_uncorrected[0], 0.5); // 1/2
    approx::assert_relative_eq!(aggs[0].power_uncorrected[1], 0.0);
    approx::assert_relative_eq!(aggs[1].power_uncorrected[0], 0.0);
    approx::assert_relative_eq!(aggs[1].power_uncorrected[1], 0.5);
    approx::assert_relative_eq!(aggs[2].power_uncorrected[0], 0.0);
    approx::assert_relative_eq!(aggs[2].power_uncorrected[1], 0.0);
}

#[test]
fn aggregate_handles_zero_sims() {
    let shape = ResultShape {
        n_sims: 0,
        n_sample_sizes: 1,
        n_targets: 1,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let br = make_batch(vec![], vec![], vec![], shape);
    let aggs = aggregate_batch(&br, &[1], &[], &EstimatorSpec::Ols);
    assert_eq!(aggs.len(), 1);
    approx::assert_relative_eq!(aggs[0].power_uncorrected[0], 0.0);
    approx::assert_relative_eq!(aggs[0].convergence_rate, 1.0);
}

#[test]
fn aggregate_dispatches_glm_extras() {
    let shape = ResultShape {
        n_sims: 4,
        n_sample_sizes: 1,
        n_targets: 1,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let unc = vec![1u8, 0, 1, 1];
    let cor = unc.clone();
    let conv = vec![1u8; 4];
    let br = make_batch(unc, cor, conv, shape);
    let aggs = aggregate_batch(&br, &[1], &[], &EstimatorSpec::Glm);
    assert!(matches!(
        aggs[0].estimator_extras,
        EstimatorExtras::Glm { .. }
    ));
    // convergence_rate is orchestrator-owned: fixture uses vec![1u8; 4] (all converged).
    approx::assert_relative_eq!(aggs[0].convergence_rate, 1.0);
}

#[test]
fn aggregate_dispatches_mle_extras() {
    // Use non-trivial joint counts so the rate computation path is exercised.
    // joint_unc=3/4, joint_cor=2/4. All-zero inputs cannot catch a wrong denominator.
    let shape = ResultShape {
        n_sims: 4,
        n_sample_sizes: 1,
        n_targets: 1,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let mut br = make_batch(vec![1u8, 1, 0, 1], vec![1u8; 4], vec![1u8; 4], shape);
    br.joint_unc = vec![1, 1, 0, 1];
    br.joint_cor = vec![1, 0, 0, 1];
    let aggs = aggregate_batch(&br, &[1], &[], &EstimatorSpec::Mle);
    match &aggs[0].estimator_extras {
        EstimatorExtras::Mle {
            joint_uncorrected_rate,
            joint_corrected_rate,
            joint_uncorrected_count,
            joint_corrected_count,
            ..
        } => {
            assert!(
                (*joint_uncorrected_rate - 0.75).abs() < 1e-12,
                "expected 3/4, got {joint_uncorrected_rate}"
            );
            assert!(
                (*joint_corrected_rate - 0.5).abs() < 1e-12,
                "expected 2/4, got {joint_corrected_rate}"
            );
            assert_eq!(*joint_uncorrected_count, 3);
            assert_eq!(*joint_corrected_count, 2);
        }
        other => panic!("expected Mle, got {other:?}"),
    }
}

#[test]
fn aggregate_batch_populates_success_counts() {
    // 3 sims × 1 sample-size × 2 targets. Layout:
    //   idx = sim * (n_ss * n_targets) + ss_idx * n_targets + t
    // target 0 hits in sims {0, 2} → count 2; target 1 hits in sim {1} → count 1.
    let shape = ResultShape {
        n_sims: 3,
        n_sample_sizes: 1,
        n_targets: 2,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    let unc = vec![
        1, 0, // sim 0: t0 hit, t1 miss
        0, 1, // sim 1: t0 miss, t1 hit
        1, 0, // sim 2: t0 hit, t1 miss
    ];
    let cor = vec![1, 0, 0, 0, 1, 0];
    let conv = vec![1u8; 3];
    let br = make_batch(unc, cor, conv, shape);
    let aggs = aggregate_batch(&br, &[0, 1], &[], &EstimatorSpec::Ols);
    assert_eq!(aggs.len(), 1);
    let pr = &aggs[0];
    assert_eq!(pr.success_counts_uncorrected, vec![2u64, 1]);
    assert_eq!(pr.success_counts_corrected, vec![2u64, 0]);
    assert_eq!(pr.convergence_count, 3);
    assert_eq!(pr.overall_significant_count, 0); // batch.overall empty
                                                 // Rates must still equal counts/n_sims exactly.
    approx::assert_relative_eq!(pr.power_uncorrected[0], 2.0 / 3.0, max_relative = 1e-12);
    approx::assert_relative_eq!(pr.power_uncorrected[1], 1.0 / 3.0, max_relative = 1e-12);
}

#[test]
fn aggregate_batch_populates_mle_extras_counts() {
    // Mle estimator: joint_unc / joint_cor counters populate the *_count fields.
    let shape = ResultShape {
        n_sims: 4,
        n_sample_sizes: 1,
        n_targets: 1,
        posthoc_blocks: vec![],
        n_factors: 0,
        n_variance_components: 0,
    };
    // boundary_hit nonzero in 1/4 sims; joint_unc/cor hit in 3/4.
    let mut br = make_batch(vec![1u8, 1, 0, 1], vec![1u8; 4], vec![1u8; 4], shape);
    br.boundary_hit = vec![0, 0, 1, 0];
    br.joint_unc = vec![1, 1, 0, 1];
    br.joint_cor = vec![1, 0, 0, 1];
    let aggs = aggregate_batch(&br, &[0], &[], &EstimatorSpec::Mle);
    match &aggs[0].estimator_extras {
        EstimatorExtras::Mle {
            boundary_hits,
            joint_uncorrected_count,
            joint_corrected_count,
            ..
        } => {
            assert_eq!(*boundary_hits, 1);
            assert_eq!(*joint_uncorrected_count, 3);
            assert_eq!(*joint_corrected_count, 2);
        }
        other => panic!("expected Mle, got {other:?}"),
    }
    assert_eq!(aggs[0].convergence_count, 4);
}
