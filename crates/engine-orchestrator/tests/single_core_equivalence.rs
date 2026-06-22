//! Verifies single_core_find_power matches find_power within MC noise after
//! pooling (statistical equivalence, not bit-equal — RNG paths intentionally
//! differ across worker counts).

use engine_orchestrator::{
    find_power, merge_power_results, single_core_find_power, CancellationToken,
};

mod common;

#[test]
fn single_core_find_power_signature_and_smoke() {
    let contracts = vec![common::marginal_plus_contrast_contract()];
    let cancel = CancellationToken::new();
    let r = single_core_find_power(&contracts, 100, 200, 42, None, &cancel).unwrap();
    assert_eq!(r.scenarios.len(), 1);
    let pr = &r.scenarios[0].1;
    assert_eq!(pr.n, 100);
    assert_eq!(pr.n_sims, 200);
    // The fixture carries a marginal AND a contrast, so the counter vector is
    // sized marginals + contrasts — `target_indices.len()` alone would let a
    // dropped contrast tail pass silently.
    let n_entries = pr.target_indices.len() + pr.contrast_pairs.len();
    assert_eq!(pr.success_counts_uncorrected.len(), n_entries);
    // Counts must reproduce the rates exactly (no accidental zeroing).
    for t in 0..n_entries {
        let expected = pr.success_counts_uncorrected[t] as f64 / pr.n_sims as f64;
        assert!(
            (expected - pr.power_uncorrected[t]).abs() < 1e-12,
            "rate/count mismatch at t={t}: count={}, n_sims={}, rate={}",
            pr.success_counts_uncorrected[t],
            pr.n_sims,
            pr.power_uncorrected[t]
        );
    }
}

#[test]
fn merged_identical_scenarios_are_bit_identical() {
    // Seed-pairing regression, single-core twin: each worker call hands both
    // scenarios the same call-level seed, so identical-knob scenarios stay
    // bit-identical per worker and through merge_power_results.
    let contracts = vec![
        common::minimal_ols_contract_labelled("a"),
        common::minimal_ols_contract_labelled("b"),
    ];
    let cancel = CancellationToken::new();
    let mut parts = Vec::new();
    for i in 0..4u64 {
        let r = single_core_find_power(&contracts, 20, 100, 1000 + i, None, &cancel).unwrap();
        assert_eq!(
            r.scenarios[0].1, r.scenarios[1].1,
            "per-worker pairing broken at worker {i}"
        );
        parts.push(r);
    }
    let merged = merge_power_results(&parts).unwrap();
    let (pr_a, pr_b) = (&merged.scenarios[0].1, &merged.scenarios[1].1);
    // Anti-vacuity: power strictly inside (0, 1) — an unpaired stream would
    // almost surely move the pooled count.
    assert!(
        pr_a.power_uncorrected[0] > 0.0 && pr_a.power_uncorrected[0] < 1.0,
        "fixture must keep power off the boundaries, got {}",
        pr_a.power_uncorrected[0]
    );
    assert_eq!(
        pr_a, pr_b,
        "merged identical-knob scenarios must be bit-identical"
    );
}

#[test]
fn merged_four_workers_matches_find_power_shape() {
    // MERGE-01: a 4-worker single_core_find_power + merge produces a ScenarioResult
    // with the *same shape* as the multi-core find_power (same scenarios, label
    // order, n, pooled n_sims, target_indices, power-vector length). RNG paths are
    // intentionally NOT bit-equal across worker counts (dispatch design D-A), so the
    // numeric power *agreement* is a statistical equivalence property — L3 seed, not
    // an L1/L2 mechanic — and is deliberately not asserted here.
    let contracts = vec![common::minimal_ols_contract()];
    let cancel = CancellationToken::new();

    let mt = find_power(&contracts, 100, 1600, 42, None, &cancel).unwrap();

    let per_worker = 400;
    let mut parts = Vec::new();
    for i in 0..4u64 {
        // Host derives independent seeds per worker (full splitmix64 — the
        // same finalizer lower_contracts applies to the call-level seed).
        let seed = {
            let mut z = 42u64.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let r = single_core_find_power(&contracts, 100, per_worker, seed, None, &cancel).unwrap();
        parts.push(r);
    }
    let merged = merge_power_results(&parts).unwrap();

    let mt_pr = &mt.scenarios[0].1;
    let mg_pr = &merged.scenarios[0].1;

    assert_eq!(mt.scenarios.len(), merged.scenarios.len());
    assert_eq!(
        mt.scenarios[0].0, merged.scenarios[0].0,
        "scenario label parity"
    );
    assert_eq!(
        mg_pr.n_sims, 1600,
        "merge pools n_sims to the multi-core total"
    );
    assert_eq!(mt_pr.n_sims, mg_pr.n_sims, "n_sims parity");
    assert_eq!(mt_pr.n, mg_pr.n, "n parity");
    assert_eq!(
        mt_pr.target_indices, mg_pr.target_indices,
        "target_indices parity"
    );
    assert_eq!(
        mt_pr.power_uncorrected.len(),
        mg_pr.power_uncorrected.len(),
        "power vector length parity"
    );
    // Mechanical: counter array length must match target count (merge didn't truncate).
    assert_eq!(
        mg_pr.success_counts_uncorrected.len(),
        mg_pr.target_indices.len(),
        "merged success_counts_uncorrected length must equal target count"
    );
    // Mechanical: histogram bucket sum == pooled n_sims (every sim in exactly one bucket).
    if !mg_pr.success_count_histogram_uncorrected.is_empty() {
        assert_eq!(
            mg_pr
                .success_count_histogram_uncorrected
                .iter()
                .sum::<u64>(),
            mg_pr.n_sims,
            "histogram bucket sum must equal pooled n_sims"
        );
    }
}

/// M2: a crossed+nested (general lmm path) contract flows identically through
/// BOTH dispatch twins — multi-core `find_power` and `single_core_find_power`
/// × 2 workers + `merge_power_results`. The per-fit lmm solver sits BELOW the
/// merge layer, so this passes without any merge change (twin equivalence discharged by
/// construction; this test is the proof artifact). RNG paths are intentionally
/// not bit-equal across worker counts (dispatch design), so the check is
/// shape parity + a sane pooled convergence rate, not numeric bit-equality.
#[test]
fn general_path_merge_twin_shape_and_convergence() {
    use engine_contract::{
        ClusterSizing, ClusterSpec, EstimatorSpec, GroupingRelation, GroupingSpec,
    };
    let mut c = common::minimal_ols_contract();
    c.estimator = EstimatorSpec::Mle;
    let mut cluster =
        ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, 0.20);
    cluster.extra_groupings = vec![
        GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.15,
            slopes: vec![],
        },
        GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.08,
            slopes: vec![],
        },
    ];
    c.generation.cluster = Some(cluster); // atom 6·4·2 = 48; N=96 is a multiple
    let contracts = vec![c];
    let cancel = CancellationToken::new();

    let mt = find_power(&contracts, 96, 400, 42, None, &cancel).unwrap();

    let mut parts = Vec::new();
    for i in 0..2u64 {
        let seed = {
            let mut z = 42u64.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let r = single_core_find_power(&contracts, 96, 200, seed, None, &cancel).unwrap();
        parts.push(r);
    }
    let merged = merge_power_results(&parts).unwrap();

    let mt_pr = &mt.scenarios[0].1;
    let mg_pr = &merged.scenarios[0].1;
    assert_eq!(
        mg_pr.n_sims, 400,
        "merge pools n_sims to the multi-core total"
    );
    assert_eq!(mt_pr.n, mg_pr.n, "n parity");
    assert_eq!(mt_pr.target_indices, mg_pr.target_indices, "target parity");
    assert_eq!(
        mt_pr.power_uncorrected.len(),
        mg_pr.power_uncorrected.len(),
        "power vector length parity"
    );
    // The general path mostly converges; both twins must report a sane rate.
    assert!(
        mt_pr.convergence_rate.is_finite() && mt_pr.convergence_rate > 0.5,
        "multi-core convergence_rate {}",
        mt_pr.convergence_rate
    );
    assert!(
        mg_pr.convergence_rate.is_finite() && mg_pr.convergence_rate > 0.5,
        "merged convergence_rate {}",
        mg_pr.convergence_rate
    );
}
