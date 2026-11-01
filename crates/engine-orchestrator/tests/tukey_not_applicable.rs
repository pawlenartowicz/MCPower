//! Under Tukey HSD a target outside every factor (the intercept, a continuous
//! or binary predictor, an interaction) has no corrected power. The engine
//! flags it once, in `PowerResult.corrected_not_applicable`, and reports NaN
//! corrected power and CI for it, through both dispatch twins and the merge.

use engine_contract::{CorrectionMethod, OutcomeKind, SimulationContract};
use engine_orchestrator::{
    ByValue, CancellationToken, GridMode, PowerResult, SampleSizeMethod, find_power,
    find_sample_size, merge_power_results, merge_sample_size_results, single_core_find_power,
    single_core_find_sample_size,
};
use engine_spec_builder::build_contract;
use engine_spec_builder::input::{
    EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec, ResidualSpec, VarKind,
};

/// `y = x + b + f + x:b`: continuous `x`, binary `b`, 3-level factor `f`,
/// and a continuous-by-binary interaction. Targets in this order:
/// x, b, f[mid], f[high], x:b.
fn contract(correction: CorrectionMethod) -> Vec<SimulationContract> {
    let effect = |name: &str, size: f64| EffectAssignment {
        name: name.into(),
        size,
    };
    let spec = LinearSpec {
        formula: "y = x + b + f + x:b".into(),
        predictors: vec![
            PredictorSpec {
                name: "x".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "b".into(),
                pinned: false,
                kind: VarKind::Binary { proportion: 0.5 },
            },
            PredictorSpec {
                name: "f".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["low".into(), "mid".into(), "high".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "low".into(),
                    sampled_proportions: None,
                },
            },
        ],
        effects: vec![
            effect("x", 0.3),
            effect("b", 0.4),
            effect("f[mid]", 0.3),
            effect("f[high]", 0.5),
            effect("x:b", 0.2),
        ],
        correlations: vec![],
        alpha: 0.05,
        correction,
        targets: vec![
            "x".into(),
            "b".into(),
            "f[mid]".into(),
            "f[high]".into(),
            "x:b".into(),
        ],
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: ResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![],
        upload: None,
        cluster_level_vars: vec![],
        wald_se: Default::default(),
        df_method: Default::default(),
        nagq: 1,
    };
    build_contract(&spec, OutcomeKind::Continuous, None, None, 0.0, vec![]).expect("build contract")
}

/// x, b and x:b are outside the factor; f[mid] and f[high] are its levels.
const FLAGS: [bool; 5] = [true, true, false, false, true];
const N: usize = 150;
const N_SIMS: usize = 200;
const SEED: u64 = 2137;

fn run_tukey() -> PowerResult {
    let cancel = CancellationToken::new();
    let r = find_power(
        &contract(CorrectionMethod::TukeyHsd),
        N,
        N_SIMS,
        SEED,
        None,
        &cancel,
    )
    .expect("find_power");
    r.scenarios[0].1.clone()
}

/// NaN-aware equality of two corrected power vectors.
fn same_power(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits())
}

#[test]
fn tukey_flags_exactly_the_non_factor_targets() {
    let pr = run_tukey();
    assert_eq!(pr.corrected_not_applicable, FLAGS.to_vec());
    for (t, &na) in FLAGS.iter().enumerate() {
        let (p, ci) = (pr.power_corrected[t], pr.ci_corrected[t]);
        if na {
            assert!(p.is_nan(), "target {t}: corrected power {p}");
            assert!(ci.lo.is_nan() && ci.hi.is_nan(), "target {t}: CI {ci:?}");
            assert_eq!(pr.success_counts_corrected[t], 0, "target {t}");
        } else {
            // A factor level keeps its Tukey-corrected rate: the pooled count
            // over n_sims, finite and no larger than the uncorrected rate.
            let k = pr.success_counts_corrected[t];
            assert!(k > 0, "anti-vacuity: f level {t} rejects at N={N}");
            assert_eq!(p, k as f64 / N_SIMS as f64, "target {t}");
            assert!(p <= pr.power_uncorrected[t], "target {t}");
            assert!(ci.lo.is_finite() && ci.hi.is_finite(), "target {t}");
        }
        // The uncorrected side is untouched.
        assert!(pr.power_uncorrected[t].is_finite(), "target {t}");
    }
}

#[test]
fn holm_flags_nothing() {
    let cancel = CancellationToken::new();
    let r = find_power(
        &contract(CorrectionMethod::Holm),
        N,
        N_SIMS,
        SEED,
        None,
        &cancel,
    )
    .expect("find_power");
    let pr = &r.scenarios[0].1;
    assert!(pr.corrected_not_applicable.is_empty());
    assert!(pr.power_corrected.iter().all(|p| p.is_finite()));
    assert!(
        pr.success_counts_corrected[0] > 0,
        "anti-vacuity: x rejects"
    );
}

/// The WASM path: one single-core worker per seed, each result through JSON
/// (serde_json writes NaN as `null`), then `merge_power_results`. A single
/// part at the native seed reproduces `find_power` exactly; two parts keep
/// the flags and the NaN entries.
#[test]
fn single_core_merge_agrees_with_native_on_flags_and_values() {
    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let worker = |seed: u64| {
        let part = single_core_find_power(&contracts, N, N_SIMS, seed, None, &cancel)
            .expect("single_core_find_power");
        let json = serde_json::to_string(&part).expect("serialize");
        serde_json::from_str(&json).expect("deserialize worker JSON")
    };

    let native = run_tukey();
    let one = merge_power_results(&[worker(SEED)]).expect("merge");
    let merged = &one.scenarios[0].1;
    assert_eq!(
        merged.corrected_not_applicable,
        native.corrected_not_applicable
    );
    assert!(same_power(&merged.power_corrected, &native.power_corrected));
    assert_eq!(merged.power_uncorrected, native.power_uncorrected);
    assert_eq!(
        merged.success_counts_corrected,
        native.success_counts_corrected
    );

    let two = merge_power_results(&[worker(SEED), worker(SEED + 1)]).expect("merge");
    let pooled = &two.scenarios[0].1;
    assert_eq!(pooled.corrected_not_applicable, FLAGS.to_vec());
    for (t, &na) in FLAGS.iter().enumerate() {
        assert_eq!(pooled.power_corrected[t].is_nan(), na, "target {t}");
        assert_eq!(pooled.ci_corrected[t].lo.is_nan(), na, "target {t}");
    }
}

#[test]
fn merge_refuses_parts_with_different_flags() {
    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let a = single_core_find_power(&contracts, N, 50, SEED, None, &cancel).expect("run");
    let mut b = a.clone();
    b.scenarios[0].1.corrected_not_applicable[0] = false;
    let err = merge_power_results(&[a, b]).expect_err("mismatched flags must not merge");
    assert_eq!(
        err.to_string(),
        "cannot merge incompatible results: corrected_not_applicable mismatch across parts"
    );
}

/// A flagged target has no corrected crossing: `first_achieved` and `fitted`
/// are `None` for it on both twins and after the merge, while the factor
/// levels keep a crossing.
#[test]
fn sample_size_gives_no_corrected_crossing_for_a_flagged_target() {
    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(100),
        mode: GridMode::Linear,
    };
    let native = find_sample_size(
        &contracts,
        0.8,
        (100, 600),
        150,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("find_sample_size");
    let part = single_core_find_sample_size(
        &contracts,
        0.8,
        (100, 600),
        150,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("single_core_find_sample_size");
    let json = serde_json::to_string(&part).expect("serialize");
    let part = serde_json::from_str(&json).expect("deserialize worker JSON");
    let merged = merge_sample_size_results(&[part]).expect("merge");

    for ssr in [&native.scenarios[0].1, &merged.scenarios[0].1] {
        for (t, &na) in FLAGS.iter().enumerate() {
            if na {
                assert_eq!(ssr.first_achieved[t], None, "target {t}");
                assert_eq!(ssr.fitted[t], None, "target {t}");
            } else {
                assert!(ssr.fitted[t].is_some(), "target {t}");
            }
        }
        // f[high] (effect 0.5) crosses 0.8 inside the grid.
        assert!(
            ssr.first_achieved[3].is_some(),
            "anti-vacuity: f[high] crosses"
        );
        for pr in &ssr.grid_or_trace {
            assert_eq!(pr.corrected_not_applicable, FLAGS.to_vec());
        }
    }
    assert_eq!(
        native.scenarios[0].1.fitted, merged.scenarios[0].1.fitted,
        "native and merge agree"
    );
}

#[test]
fn merge_refuses_flags_of_the_wrong_length() {
    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let mut part = single_core_find_power(&contracts, N, 50, SEED, None, &cancel).expect("run");
    part.scenarios[0].1.corrected_not_applicable.pop();
    let err = merge_power_results(&[part]).expect_err("short flags must not merge");
    assert_eq!(
        err.to_string(),
        "cannot merge incompatible results: corrected_not_applicable length does not match \
         target_indices + contrast_pairs"
    );
}

/// The corrected joint view is not applicable when any target is flagged:
/// no corrected histogram, no corrected joint crossing, on both twins and
/// after the merge. The uncorrected histogram is still there.
#[test]
fn flagged_target_makes_the_corrected_joint_view_not_applicable() {
    let pr = run_tukey();
    assert!(pr.success_count_histogram_corrected.is_empty());
    assert_eq!(
        pr.success_count_histogram_uncorrected.len(),
        FLAGS.len() + 1
    );
    assert_eq!(
        pr.success_count_histogram_uncorrected.iter().sum::<u64>(),
        N_SIMS as u64
    );

    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(100),
        mode: GridMode::Linear,
    };
    let native = find_sample_size(
        &contracts,
        0.8,
        (100, 400),
        100,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("find_sample_size");
    let part = single_core_find_sample_size(
        &contracts,
        0.8,
        (100, 400),
        100,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("single_core_find_sample_size");
    let merged = merge_sample_size_results(&[part.clone(), part]).expect("merge");
    for ssr in [&native.scenarios[0].1, &merged.scenarios[0].1] {
        assert!(ssr.first_joint_achieved.is_empty());
        assert!(ssr.fitted_joint.is_empty());
        for pr in &ssr.grid_or_trace {
            assert!(pr.success_count_histogram_corrected.is_empty());
            assert!(!pr.success_count_histogram_uncorrected.is_empty());
        }
    }

    // Holm on the same model keeps the corrected joint view.
    let holm = find_sample_size(
        &contract(CorrectionMethod::Holm),
        0.8,
        (100, 400),
        100,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("find_sample_size");
    assert_eq!(holm.scenarios[0].1.first_joint_achieved.len(), FLAGS.len());
}

/// Guard: every grid point of a sample-size part must carry the flags of
/// grid point 0, since the derivations read them off point 0.
#[test]
fn sample_size_merge_refuses_grid_points_with_different_flags() {
    let contracts = contract(CorrectionMethod::TukeyHsd);
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(100),
        mode: GridMode::Linear,
    };
    let mut part = single_core_find_sample_size(
        &contracts,
        0.8,
        (100, 300),
        50,
        method,
        1,
        SEED,
        None,
        &cancel,
    )
    .expect("single_core_find_sample_size");
    // Point 1 flags x:b off instead of on; its corrected histogram stays
    // empty, so only the grid-point flag guard can catch it.
    part.scenarios[0].1.grid_or_trace[1].corrected_not_applicable =
        vec![true, true, false, false, false];
    let err = merge_sample_size_results(&[part]).expect_err("mismatched grid-point flags");
    assert_eq!(
        err.to_string(),
        "cannot merge incompatible results: grid point 1 of scenario 0 has different \
         corrected_not_applicable flags than grid point 0"
    );
}
