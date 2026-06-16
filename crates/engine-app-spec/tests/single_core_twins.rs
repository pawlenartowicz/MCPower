//! Single-core run twins (WASM dispatch path): shape, merge-pooling, and pre-cancel.
//! Mirrors `driver_smoke.rs`; the merge test is the dispatch-twins L2 gate
//! (`run_single_core_* ×N + merge` produces a valid pooled result).

use engine_app_spec::{
    run_find_power, run_single_core_find_power, run_single_core_find_sample_size, AdapterError,
    AppSpec, EffectSize, LinearSpec, NullEmitter, ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    merge_power_results, ByValue, CancellationToken, GridMode, SampleSizeMethod,
};

fn sample_linear_spec() -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into(), "x2".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric { name: "x1".into(), distribution: Default::default(), pinned: false },
            VarType::Numeric { name: "x2".into(), distribution: Default::default(), pinned: false },
        ],
        effects: vec![
            EffectSize {
                name: "x1".into(),
                value: 0.3,
            },
            EffectSize {
                name: "x2".into(),
                value: 0.2,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 128,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    })
}

#[test]
fn run_single_core_find_power_smoke() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let result = run_single_core_find_power(&spec, 80, 800, 11, &NullEmitter, &cancel).expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, power) = &result.scenarios[0];
    assert_eq!(power.target_indices.len(), power.power_uncorrected.len());
    assert!(!power.target_indices.is_empty());
    assert!(!power.success_counts_uncorrected.is_empty());
}

#[test]
fn single_core_parts_merge_to_full_run() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let part_a = run_single_core_find_power(&spec, 80, 800, 11, &NullEmitter, &cancel).expect("a");
    let part_b = run_single_core_find_power(&spec, 80, 800, 22, &NullEmitter, &cancel).expect("b");
    let merged = merge_power_results(&[part_a, part_b]).expect("merge");
    assert_eq!(merged.scenarios.len(), 1);
    let (_label, power) = &merged.scenarios[0];
    assert_eq!(power.n_sims, 1600);
    for &p in &power.power_uncorrected {
        assert!((0.0..=1.0).contains(&p), "pooled power out of range: {p}");
    }
}

#[test]
fn run_single_core_find_sample_size_smoke() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(20),
        mode: GridMode::Linear,
    };
    let result =
        run_single_core_find_sample_size(&spec, (40, 200), method, 800, 11, &NullEmitter, &cancel)
            .expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, ssr) = &result.scenarios[0];
    assert!(!ssr.grid_or_trace.is_empty());
}

#[test]
fn run_single_core_find_power_errors_when_pre_cancelled() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let res = run_single_core_find_power(&spec, 80, 800, 11, &NullEmitter, &cancel);
    assert!(matches!(res, Err(AdapterError::Orchestrator(_))));
}

/// RULE-4 twin invariant: single-core power is statistically equivalent to
/// native multi-core power for the same spec + seed.
/// The fixture uses n_sims=128. At n_sims=128, σ ≈ 0.044 per estimate;
/// the ±0.15 band is ~3.4σ — appropriate for this cheap fixture.
/// (If n_sims is bumped in the fixture, tighten this tolerance accordingly.)
#[test]
fn single_core_merged_matches_native_find_power() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();

    // Native multi-core baseline: reads n_sims and seed from spec (128, 2137).
    let native = run_find_power(&spec, 80, &NullEmitter, &cancel).expect("native ok");
    // Single-core path: same seed, same n_sims, different dispatch (no rayon fan-out).
    let single =
        run_single_core_find_power(&spec, 80, 128, 2137, &NullEmitter, &cancel).expect("single ok");

    let (_, native_pw) = &native.scenarios[0];
    let (_, single_pw) = &single.scenarios[0];
    assert_eq!(
        native_pw.power_uncorrected.len(),
        single_pw.power_uncorrected.len(),
        "target count mismatch between native and single-core paths"
    );
    for (i, (&np, &sp)) in native_pw
        .power_uncorrected
        .iter()
        .zip(single_pw.power_uncorrected.iter())
        .enumerate()
    {
        assert!(
            (np - sp).abs() <= 0.15,
            "target {i}: native power {np:.4} vs single-core {sp:.4} differ by more than 0.15 \
             (expected statistical equivalence at n_sims={}/{})",
            native_pw.n_sims,
            single_pw.n_sims,
        );
    }
}
