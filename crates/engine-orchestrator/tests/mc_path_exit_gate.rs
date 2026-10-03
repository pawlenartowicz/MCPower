//! Monte-Carlo path gates for the GLM / GLMM / LMM estimators through
//! `find_power` and the single-core + merge path: same-seed reproducibility of
//! an unclustered logit run, and the estimator-extras wiring (including the
//! NaN placeholders) through both dispatch twins. Thread-count invariance is
//! owned by engine-core (`run_batch_bit_identical_1_vs_n_threads`,
//! `tests/warmstart_determinism.rs`).

mod common;

use common::{lme_spec, logit_spec};
use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec, OutcomeKind};
use engine_orchestrator::{
    find_power, merge_power_results, single_core_find_power, CancellationToken, EstimatorExtras,
};
use engine_spec_builder::build_contract;

const N: usize = 200;
const N_SIMS: usize = 800;
const BASE_SEED: u64 = 2137;

/// Two `find_power` calls with identical inputs (unclustered logit, IRLS path)
/// must be bit-identical. Compared as msgpack bytes because the Glm extras
/// carry a NaN placeholder (`baseline_prob_realized`) and `NaN != NaN` would
/// fail a direct `assert_eq!`; msgpack encodes NaN to a fixed bit pattern.
#[test]
fn find_power_logit_is_bit_identical_at_fixed_seed() {
    let contracts = build_contract(&logit_spec(), OutcomeKind::Binary, None, None, -0.5, vec![])
        .expect("build_contract glm");
    let cancel = CancellationToken::new();
    let r1 = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let r2 = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let (_, pr) = &r1.scenarios[0];
    assert!(
        matches!(pr.estimator_extras, EstimatorExtras::Glm { .. }),
        "expected Glm estimator_extras, got {:?}",
        pr.estimator_extras
    );
    assert_eq!(
        rmp_serde::to_vec(&r1).expect("encode r1"),
        rmp_serde::to_vec(&r2).expect("encode r2"),
        "same-seed logit find_power must be bit-identical"
    );
}

/// GLMM end-to-end: a clustered-logit contract (Glm + random intercept) routes
/// through `fit_on` and surfaces `EstimatorExtras::Glm` with a valid
/// `singular_fit_rate` and a finite `tau_squared_hat_mean` (the engine half of
/// the Laplace-bias warning). Guards the batch.rs dispatch + the
/// `from_batch` Glm aggregation wiring.
#[test]
fn find_power_clustered_logit_surfaces_glm_extras() {
    let contracts = build_contract(
        &logit_spec(),
        OutcomeKind::Binary,
        None,
        None, // Binary ⇒ estimator defaults to Glm; cluster present ⇒ GLMM path
        -0.5,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract clustered glm");

    // Smaller workload than the same-seed gate above: GLMM runs a joint
    // BOBYQA per (sim, N), so 64 sims at N=120 keeps this fast in debug while
    // still converging enough fits to populate the extras.
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, 120, 64, BASE_SEED, None, &cancel).expect("find_power ok");
    let (_, pr) = &r.scenarios[0];
    let EstimatorExtras::Glm {
        singular_fit_rate,
        singular_n,
        tau_squared_hat_mean,
        tau_squared_hat_n,
        ..
    } = &pr.estimator_extras
    else {
        panic!(
            "expected Glm estimator_extras, got {:?}",
            pr.estimator_extras
        );
    };
    assert!(*singular_n > 0, "some GLMM fits must converge");
    // ICC=0.3 small clusters (20 clusters at N=120) pin a non-zero share of
    // converged fits at the variance-component boundary — a real signal, not the
    // tautological `[0,1]` range. Empirically ≈0.17 for this fixture.
    assert!(
        *singular_fit_rate > 0.0,
        "ICC=0.3/N=120 geometry must pin some fits singular: {singular_fit_rate}"
    );
    assert!(*tau_squared_hat_n > 0, "converged fits must contribute τ̂²");
    assert!(
        tau_squared_hat_mean.is_finite(),
        "converged ⇒ finite mean τ̂², got {tau_squared_hat_mean}"
    );
}

/// EP-3 dispatch-twin — GLMM extras through the single-core + merge path.
///
/// `find_power_clustered_logit_surfaces_glm_extras` (above) exercises
/// `EstimatorExtras::Glm` only through the multi-core path. The WASM worker
/// path runs `single_core_find_power` per worker and calls `merge_power_results`
/// on the main thread; this test exercises that distinct path.
///
/// Divergence risk: `baseline_prob_realized` is always `NaN` in a raw
/// `from_batch` result (the kernel does not yet surface per-sim baseline
/// probabilities), so each single-core part carries `NaN`. The merge path
/// recomputes it from pooled counters: `if n > 0 { sum / n } else { 0.0 }`.
/// With `baseline_prob_n == 0` across all parts the result is `0.0` — finite,
/// not `NaN`. This test pins that NaN-pre / finite-post transition so a future
/// bug that leaves `baseline_prob_realized` NaN after merge is caught.
#[test]
fn ep3_glmm_extras_through_single_core_merge_path() {
    let contracts = build_contract(
        &logit_spec(),
        OutcomeKind::Binary,
        None,
        None,
        -0.5,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract clustered glm");

    let cancel = CancellationToken::new();

    // Run two single-core workers; keep n_sims small so GLMM stays fast in
    // debug while still converging enough fits to populate the extras.
    let mut parts = Vec::new();
    for i in 0..2u64 {
        let seed = {
            let mut z = BASE_SEED.wrapping_add(i.wrapping_mul(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let r = single_core_find_power(&contracts, 120, 32, seed, None, &cancel).expect("sc ok");
        // Invariant: baseline_prob_realized is NaN in every raw part because
        // the kernel does not surface per-sim baseline probabilities — the
        // field carries the sentinel until merge pools and recomputes it.
        let EstimatorExtras::Glm {
            baseline_prob_realized,
            ..
        } = &r.scenarios[0].1.estimator_extras
        else {
            panic!("expected Glm extras in part {i}");
        };
        assert!(
            baseline_prob_realized.is_nan(),
            "pre-merge baseline_prob_realized must be NaN in part {i}, got {baseline_prob_realized}"
        );
        parts.push(r);
    }

    let merged = merge_power_results(&parts).expect("merge ok");
    let (_, pr) = &merged.scenarios[0];
    let EstimatorExtras::Glm {
        baseline_prob_realized,
        singular_fit_rate,
        singular_n,
        tau_squared_hat_mean,
        tau_squared_hat_n,
        ..
    } = &pr.estimator_extras
    else {
        panic!(
            "expected Glm estimator_extras after merge, got {:?}",
            pr.estimator_extras
        );
    };
    // baseline_prob_realized must be finite after merge (0.0 when baseline_prob_n
    // stays 0 — the kernel does not yet populate it, so the merge falls through
    // to the else branch). Either way it must NOT be NaN.
    assert!(
        !baseline_prob_realized.is_nan(),
        "post-merge baseline_prob_realized must be finite, got {baseline_prob_realized}"
    );
    assert!(
        *singular_n > 0,
        "some GLMM fits must converge across workers"
    );
    // Same ICC=0.3/N=120 geometry, pooled across two single-core workers: the
    // boundary-pin share stays non-zero (empirically ≈0.27) — pins the real
    // signal rather than the tautological `[0,1]` range.
    assert!(
        *singular_fit_rate > 0.0,
        "merged ICC=0.3/N=120 geometry must pin some fits singular: {singular_fit_rate}"
    );
    assert!(*tau_squared_hat_n > 0, "converged fits must contribute τ̂²");
    assert!(
        tau_squared_hat_mean.is_finite(),
        "converged ⇒ finite merged mean τ̂², got {tau_squared_hat_mean}"
    );
}

/// Mle `tau_estimate` sentinel transition through the single-core + merge path.
///
/// `from_batch` emits `tau_estimate: NaN` (the kernel does not surface per-sim
/// τ̂), so every raw single-core part carries NaN. The merge path recomputes
/// `if tau_n > 0 { tau_sum / tau_n } else { 0.0 }`; with `tau_n == 0` across all
/// parts the merged sentinel is EXACTLY `0.0` — finite, not NaN. Pins that
/// NaN-pre / 0.0-post transition so flipping the merge else-branch to NaN is
/// caught. Mirrors `ep3_glmm_extras_through_single_core_merge_path`.
#[test]
fn mle_tau_estimate_nan_pre_zero_post_through_merge() {
    let contracts = build_contract(
        &lme_spec(),
        OutcomeKind::Continuous,
        None,
        Some(EstimatorSpec::Mle),
        0.0,
        vec![ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        }],
    )
    .expect("build_contract mle");

    let cancel = CancellationToken::new();
    let mut parts = Vec::new();
    for i in 0..2u64 {
        let r = single_core_find_power(&contracts, 120, 32, BASE_SEED + i, None, &cancel)
            .expect("sc ok");
        let EstimatorExtras::Mle { tau_estimate, .. } = &r.scenarios[0].1.estimator_extras else {
            panic!("expected Mle extras in part {i}");
        };
        assert!(
            tau_estimate.is_nan(),
            "pre-merge tau_estimate must be NaN in part {i}, got {tau_estimate}"
        );
        parts.push(r);
    }

    let merged = merge_power_results(&parts).expect("merge ok");
    let EstimatorExtras::Mle { tau_estimate, .. } = &merged.scenarios[0].1.estimator_extras else {
        panic!("expected Mle extras after merge");
    };
    // tau_n stays 0 (kernel does not populate per-sim τ̂), so the else-branch
    // produces exactly 0.0 — not NaN.
    assert_eq!(
        *tau_estimate, 0.0,
        "post-merge tau_estimate must be exactly 0.0, got {tau_estimate}"
    );
}
