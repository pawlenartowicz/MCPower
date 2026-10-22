//! Monte-Carlo path gates for the GLM / GLMM / LMM estimators through
//! `find_power` and the single-core + merge path: same-seed reproducibility of
//! an unclustered logit run, the realized event rate against its analytic
//! value, and the estimator-extras wiring (including the NaN-when-empty
//! fields) through both dispatch twins. Thread-count invariance is
//! owned by engine-core (`run_batch_bit_identical_1_vs_n_threads`,
//! `tests/warmstart_determinism.rs`).

mod common;

use common::{lme_spec, logit_spec};
use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec, OutcomeKind};
use engine_orchestrator::{
    CancellationToken, EstimatorExtras, PowerResult, ScenarioResult, find_power,
    merge_power_results, single_core_find_power,
};
use engine_spec_builder::{EffectAssignment, LinearSpec, PredictorSpec, VarKind, build_contract};

const N: usize = 200;
const N_SIMS: usize = 800;
const BASE_SEED: u64 = 2137;

/// Two `find_power` calls with identical inputs (unclustered logit, IRLS path)
/// must be bit-identical. Compared as msgpack bytes because the unclustered Glm
/// extras carry a NaN `tau_squared_hat_mean` and `NaN != NaN` would fail a
/// direct `assert_eq!`; msgpack encodes NaN to a fixed bit pattern.
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
/// Each part carries a finite realized event rate (`baseline_prob_realized`,
/// Σ/n over its sims). The merge recomputes it from the pooled Σ/n, so merging
/// one part must reproduce that part's value bit for bit, and merging two must
/// give the count-weighted pool — the same value a JSON wire hop must keep.
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
        parts.push(r);
    }
    let part_baseline = |r: &ScenarioResult<PowerResult>| match &r.scenarios[0].1.estimator_extras {
        EstimatorExtras::Glm {
            baseline_prob_realized,
            baseline_prob_sum,
            baseline_prob_n,
            ..
        } => (
            *baseline_prob_realized,
            *baseline_prob_sum,
            *baseline_prob_n,
        ),
        other => panic!("expected Glm extras, got {other:?}"),
    };
    let (rate0, sum0, n0) = part_baseline(&parts[0]);
    let (_, sum1, n1) = part_baseline(&parts[1]);
    // Every sim of a Binary outcome contributes its event rate, converged or not.
    assert_eq!((n0, n1), (32, 32));
    assert!(
        rate0 > 0.0 && rate0 < 1.0,
        "part 0 event rate must be a finite proportion, got {rate0}"
    );

    let alone = merge_power_results(&parts[..1]).expect("merge of one part ok");
    assert_eq!(
        part_baseline(&alone).0.to_bits(),
        rate0.to_bits(),
        "merging one part must reproduce its event rate"
    );

    let merged = merge_power_results(&parts).expect("merge ok");
    let (merged_rate, _, merged_n) = part_baseline(&merged);
    assert_eq!(merged_n, n0 + n1);
    assert!(
        (merged_rate - (sum0 + sum1) / (n0 + n1) as f64).abs() < 1e-15,
        "merged event rate must be the pooled Σ/n, got {merged_rate}"
    );
    let json = serde_json::to_string(&merged).expect("merged encodes");
    let back: ScenarioResult<PowerResult> = serde_json::from_str(&json).expect("merged decodes");
    assert_eq!(
        part_baseline(&back).0.to_bits(),
        merged_rate.to_bits(),
        "event rate must survive the JSON wire"
    );
    let (_, pr) = &merged.scenarios[0];
    let EstimatorExtras::Glm {
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

/// Mle `tau_estimate` through the single-core + merge path.
///
/// The kernel surfaces no per-sim τ̂ on the Mle path, so `from_batch` emits
/// `tau_estimate: NaN` with `tau_n == 0`. The merge recomputes
/// `tau_sum / tau_n` only when `tau_n > 0`, so merging those parts must keep
/// NaN (a merged 0.0 would read as a real estimate), and the NaN must survive
/// the JSON wire. Mirrors `ep3_glmm_extras_through_single_core_merge_path`.
#[test]
fn mle_tau_estimate_stays_nan_through_merge() {
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
    let json = serde_json::to_string(&merged).expect("merged encodes");
    let back: ScenarioResult<PowerResult> = serde_json::from_str(&json).expect("merged decodes");
    for r in [&merged, &back] {
        let EstimatorExtras::Mle {
            tau_estimate,
            tau_n,
            ..
        } = &r.scenarios[0].1.estimator_extras
        else {
            panic!("expected Mle extras after merge");
        };
        assert_eq!(*tau_n, 0);
        assert!(
            tau_estimate.is_nan(),
            "post-merge tau_estimate must stay NaN, got {tau_estimate}"
        );
    }
}

/// One standardized normal predictor `x1` with log-odds slope `beta` and
/// baseline log-odds `intercept`; unclustered, so the estimator is the IRLS GLM.
fn one_predictor_logit(beta: f64, intercept: f64) -> Vec<engine_contract::SimulationContract> {
    let spec = LinearSpec {
        formula: "y = x1".into(),
        predictors: vec![PredictorSpec {
            name: "x1".into(),
            pinned: false,
            kind: VarKind::Normal,
        }],
        effects: vec![EffectAssignment {
            name: "x1".into(),
            size: beta,
        }],
        targets: vec!["x1".into()],
        ..lme_spec()
    };
    build_contract(&spec, OutcomeKind::Binary, None, None, intercept, vec![])
        .expect("build_contract one-predictor logit")
}

fn event_rate(r: &ScenarioResult<PowerResult>) -> (f64, u64) {
    match &r.scenarios[0].1.estimator_extras {
        EstimatorExtras::Glm {
            baseline_prob_realized,
            baseline_prob_n,
            ..
        } => (*baseline_prob_realized, *baseline_prob_n),
        other => panic!("expected Glm extras, got {other:?}"),
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// β = 0 under optimistic: every row is Bernoulli(baseline), so the realized
/// event rate over N·n_sims iid rows sits within Monte-Carlo error of the
/// requested baseline (4 SE = 0.003 at 160,000 rows).
#[test]
fn event_rate_matches_baseline_when_beta_is_zero() {
    let baseline = 0.10;
    let contracts = one_predictor_logit(0.0, logit(baseline));
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let (rate, n) = event_rate(&r);
    assert_eq!(n, N_SIMS as u64);
    let se = (baseline * (1.0 - baseline) / (N * N_SIMS) as f64).sqrt();
    assert!(
        (rate - baseline).abs() < 4.0 * se,
        "event rate {rate} vs baseline {baseline} (4 SE = {})",
        4.0 * se
    );
}

/// β = 0.916 (OR 2.5) on a standardized predictor at baseline 0.10: the rows are
/// iid Bernoulli(p̄) with p̄ = E[expit(β₀ + βx)], x ~ N(0, 1), so the realized
/// event rate matches that integral (trapezoid on [−10, 10]) within 4 SE. The
/// single-core twin over two merged parts must agree with it too.
#[test]
fn event_rate_matches_marginal_integral_for_nonzero_beta() {
    let (baseline, beta) = (0.10, 0.916);
    let b0 = logit(baseline);
    let steps = 20_000;
    let h = 20.0 / steps as f64;
    let integrand = |x: f64| {
        let p = 1.0 / (1.0 + (-(b0 + beta * x)).exp());
        p * (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    };
    let p_bar = (0..=steps)
        .map(|k| {
            let w = if k == 0 || k == steps { 0.5 } else { 1.0 };
            w * integrand(-10.0 + k as f64 * h)
        })
        .sum::<f64>()
        * h;
    // The slope lifts the marginal rate well clear of the baseline, so the
    // gate below cannot pass on the β = 0 value.
    assert!(p_bar > baseline + 0.02, "p̄ = {p_bar}");
    let se = (p_bar * (1.0 - p_bar) / (N * N_SIMS) as f64).sqrt();

    let contracts = one_predictor_logit(beta, b0);
    let cancel = CancellationToken::new();
    let r = find_power(&contracts, N, N_SIMS, BASE_SEED, None, &cancel).expect("find_power ok");
    let (rate, _) = event_rate(&r);
    assert!(
        (rate - p_bar).abs() < 4.0 * se,
        "event rate {rate} vs E[expit] {p_bar} (4 SE = {})",
        4.0 * se
    );

    let parts: Vec<_> = (0..2u64)
        .map(|i| {
            single_core_find_power(&contracts, N, N_SIMS / 2, BASE_SEED + i, None, &cancel)
                .expect("sc ok")
        })
        .collect();
    let (merged_rate, merged_n) = event_rate(&merge_power_results(&parts).expect("merge ok"));
    assert_eq!(merged_n, N_SIMS as u64);
    assert!(
        (merged_rate - p_bar).abs() < 4.0 * se,
        "merged event rate {merged_rate} vs E[expit] {p_bar}"
    );
}
