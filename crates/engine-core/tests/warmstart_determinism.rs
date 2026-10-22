//! Determinism guard for the grid-sequential θ̂ warm-start (Phase 2).
//!
//! The warm-start carries the previous (smaller-N) grid point's converged θ̂ into
//! the next point's mixed-model fit. It is safe ONLY because the carry is walked
//! in fixed ascending-N order by the single worker that owns each sim — it never
//! crosses sims, so work-stealing order cannot perturb it. This test pins that
//! guarantee: the same mixed spec + seed must produce a byte-identical
//! `BatchResult` whether run on several rayon workers (`run_batch`) or on a single
//! thread with no rayon (`run_batch_st`). A regression that let the carry leak
//! across sims (the rejected cross-sim accumulator) would diverge here.
//!
//! Covers both mixed arms: a Gaussian LMM (`Mle`) and a clustered logistic GLMM
//! (`Glm` + cluster) — the two paths that feed the carry. Each arm also runs
//! on a shape where glmm's gradient search takes the pilot's curvature hint
//! (a one-slope LMM, n_θ = 3; a crossed-intercept GLMM, n_θ = 2), so the
//! hints are pinned as thread-count-invariant too.

mod common;

use engine_contract::{ColumnId, GroupingRelation, GroupingSpec, SlopeTerm};
use engine_core::batch::{curvature_hint_parts, run_batch, run_batch_st, set_n_threads};
use engine_core::spec::{
    BatchResult, ClusterSizing, ClusterSpec, Distribution, EngineError, EstimatorSpec, OutcomeKind,
    SimulationSpec,
};

const BASE_SEED: u64 = 2137;
const N_SIMS: u32 = 64;
// Ascending multi-point grid so the carry actually fires (each point but the
// first warm-starts from its predecessor).
const GRID: [u32; 4] = [60, 120, 180, 240];

/// Random-intercept mixed spec over `k` continuous predictors, τ² = 0.25, 20
/// clusters. `estimator`/`outcome_kind` select the LMM vs clustered-GLMM path.
fn mixed_spec(estimator: EstimatorSpec, outcome_kind: OutcomeKind) -> SimulationSpec {
    let k = 3usize;
    let mut correlation = vec![0.0; k * k];
    for i in 0..k {
        correlation[i * k + i] = 1.0;
    }
    let mut effect_sizes = vec![0.0; 1 + k];
    effect_sizes[1] = 0.4;
    effect_sizes[2] = 0.4;
    SimulationSpec {
        n_non_factor: k as u32,
        correlation,
        var_types: vec![Distribution::Normal; k],
        var_params: vec![0.0; k],
        effect_sizes,
        target_indices: vec![1, 2],
        outcome_kind,
        estimator,
        max_failed_fraction: 0.5,
        cluster: Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
        }),
        ..common::base_spec()
    }
}

/// Byte-identity of two `BatchResult`s. f64 vectors compare on `to_bits` so NaN
/// (the unconverged/unclustered sentinel) matches NaN bit-for-bit.
fn assert_identical(a: &BatchResult, b: &BatchResult, label: &str) {
    assert_eq!(a.uncorrected, b.uncorrected, "{label}: uncorrected");
    assert_eq!(a.corrected, b.corrected, "{label}: corrected");
    assert_eq!(a.posthoc_unc, b.posthoc_unc, "{label}: posthoc_unc");
    assert_eq!(a.posthoc_cor, b.posthoc_cor, "{label}: posthoc_cor");
    assert_eq!(a.converged, b.converged, "{label}: converged");
    assert_eq!(a.boundary_hit, b.boundary_hit, "{label}: boundary_hit");
    assert_eq!(
        a.pinned_components, b.pinned_components,
        "{label}: pinned_components"
    );
    assert_eq!(a.joint_unc, b.joint_unc, "{label}: joint_unc");
    assert_eq!(a.joint_cor, b.joint_cor, "{label}: joint_cor");
    assert_eq!(a.overall, b.overall, "{label}: overall");
    assert_eq!(
        a.factor_excluded, b.factor_excluded,
        "{label}: factor_excluded"
    );
    let a_tau: Vec<u64> = a.tau_squared_hat.iter().map(|x| x.to_bits()).collect();
    let b_tau: Vec<u64> = b.tau_squared_hat.iter().map(|x| x.to_bits()).collect();
    assert_eq!(a_tau, b_tau, "{label}: tau_squared_hat (bit-identical)");
}

/// The two shapes where glmm's gradient search takes the pilot's curvature
/// hint: a one-slope LMM (n_θ = 3) and a crossed-intercept GLMM (n_θ = 2).
fn hinted_shapes() -> [(SimulationSpec, &'static str); 2] {
    let mut slope_lmm = mixed_spec(EstimatorSpec::Mle, OutcomeKind::Continuous);
    slope_lmm.cluster.as_mut().unwrap().slopes = vec![SlopeTerm {
        column: ColumnId(0),
        variance: 0.10,
        corr_with_intercept: 0.0,
        corr_with: vec![],
    }];
    // x_full col 1 = x1 (col 0 = intercept).
    slope_lmm.cluster_slope_design_cols = vec![1];
    let mut crossed_glmm = mixed_spec(EstimatorSpec::Glm, OutcomeKind::Binary);
    crossed_glmm.cluster.as_mut().unwrap().extra_groupings = vec![GroupingSpec {
        relation: GroupingRelation::Crossed { n_clusters: 8 },
        tau_squared: 0.25,
        slopes: vec![],
    }];
    [
        (slope_lmm, "slope LMM (hinted)"),
        (crossed_glmm, "crossed GLMM (hinted)"),
    ]
}

fn tau_bits(r: &BatchResult) -> Vec<u64> {
    r.tau_squared_hat.iter().map(|x| x.to_bits()).collect()
}

/// Multi-thread (`run_batch`, several rayon workers) vs single-thread
/// (`run_batch_st`, no rayon) must be byte-identical for both mixed arms.
#[test]
fn warm_start_is_thread_count_invariant() {
    // Force a multi-worker pool so `run_batch` genuinely interleaves sims across
    // threads (a 1-core CI box would otherwise make this a no-op). Must run before
    // any `run_batch` in this (dedicated) test binary — the pool is set once.
    set_n_threads(4).expect("set thread pool for the determinism test");

    let [slope_lmm, crossed_glmm] = hinted_shapes();
    for (spec, label) in [
        (
            mixed_spec(EstimatorSpec::Mle, OutcomeKind::Continuous),
            "LMM",
        ),
        (mixed_spec(EstimatorSpec::Glm, OutcomeKind::Binary), "GLMM"),
        slope_lmm,
        crossed_glmm,
    ] {
        let parallel =
            run_batch(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch (parallel)");
        let serial =
            run_batch_st(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch_st (serial)");
        assert_identical(&parallel, &serial, label);

        // A second parallel run must match the first (re-entrancy / carry-reset).
        let parallel2 =
            run_batch(&spec, &GRID, N_SIMS, BASE_SEED, None).expect("run_batch (parallel #2)");
        assert_identical(&parallel, &parallel2, label);
    }
}

/// Hints computed by `curvature_hint_parts` and handed back on the spec give
/// the same batch as the engine's own pilot, on a grid and at one N. At one N
/// the slope LMM has no hint (no `truth_start`), and the empty vector must
/// still match.
#[test]
fn supplied_hints_match_own_pilot() {
    for (spec, label) in hinted_shapes() {
        for sizes in [&GRID[..], &[240][..]] {
            let parts = curvature_hint_parts(&spec, sizes, BASE_SEED).unwrap();
            if sizes.len() > 1 {
                // Guards against a vacuous comparison.
                assert!(
                    parts.iter().any(Option::is_some),
                    "{label}: pilot gave no hint"
                );
            }
            let own = run_batch_st(&spec, sizes, N_SIMS, BASE_SEED, None).unwrap();
            let mut hinted = spec.clone();
            hinted.curvature_hints = Some(parts);
            let given = run_batch_st(&hinted, sizes, N_SIMS, BASE_SEED, None).unwrap();
            assert_identical(&own, &given, label);
        }
    }
}

/// `Some([])` skips the pilot. A hinted search ends at a slightly different
/// point than an unhinted one, so τ̂² differs in the low bits; equal bits
/// would mean the engine ran its own pilot anyway.
#[test]
fn supplied_empty_hints_skip_the_pilot() {
    let [_, (spec, _)] = hinted_shapes();
    let own = run_batch_st(&spec, &GRID, N_SIMS, BASE_SEED, None).unwrap();
    let mut none = spec.clone();
    none.curvature_hints = Some(vec![]);
    let given = run_batch_st(&none, &GRID, N_SIMS, BASE_SEED, None).unwrap();
    assert_ne!(tau_bits(&own), tau_bits(&given));
}

/// A part glmm's `from_parts` rejects (here `units = 0`) acts as no hint for
/// its grid point, with no panic.
#[test]
fn rejected_hint_parts_act_as_no_hint() {
    let [_, (spec, label)] = hinted_shapes();
    let mut parts = curvature_hint_parts(&spec, &GRID, BASE_SEED).unwrap();
    assert!(
        parts.iter().any(Option::is_some),
        "{label}: pilot gave no hint"
    );
    for p in parts.iter_mut().flatten() {
        p.units = 0.0;
    }
    let mut rejected = spec.clone();
    rejected.curvature_hints = Some(parts);
    let mut empty = spec.clone();
    empty.curvature_hints = Some(vec![None; GRID.len()]);
    let a = run_batch_st(&rejected, &GRID, N_SIMS, BASE_SEED, None).unwrap();
    let b = run_batch_st(&empty, &GRID, N_SIMS, BASE_SEED, None).unwrap();
    assert_identical(&a, &b, label);
}

#[test]
fn hint_length_mismatch_is_an_error() {
    let [_, (mut spec, _)] = hinted_shapes();
    spec.curvature_hints = Some(vec![None; 3]);
    let err = run_batch_st(&spec, &GRID, 2, BASE_SEED, None).unwrap_err();
    assert!(matches!(err, EngineError::InvalidSpec(_)), "{err:?}");
    spec.curvature_hints = Some(vec![]);
    assert!(run_batch_st(&spec, &GRID, 2, BASE_SEED, None).is_ok());
}
