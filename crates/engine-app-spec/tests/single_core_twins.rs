//! Single-core run twins (WASM dispatch path): shape, merge-pooling, and pre-cancel.
//! Mirrors `driver_smoke.rs`; the merge test is the dispatch-twins L2 gate
//! (`run_single_core_* ×N + merge` produces a valid pooled result).

use engine_app_spec::{
    run_find_power, run_single_core_find_power, run_single_core_find_sample_size, AdapterError,
    AppSpec, EffectSize, LinearSpec, NullEmitter, ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    merge_power_results, merge_sample_size_results, ByValue, CancellationToken, GridMode,
    SampleSizeMethod, SampleSizeResult,
};

fn sample_linear_spec() -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into(), "x2".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric {
                name: "x1".into(),
                distribution: Default::default(),
                pinned: false,
            },
            VarType::Numeric {
                name: "x2".into(),
                distribution: Default::default(),
                pinned: false,
            },
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

/// EP-3 — GAP 3: `merge_sample_size_results` (the WASM SSR pooling path) was
/// never called by any test in this crate. Unlike `merge_power_results`, it
/// RE-COMPUTES the crossing fit from pooled counts rather than summing fit
/// outputs — distinct semantics; a regression here (e.g. panicking grid-mismatch
/// check or wrong `first_achieved` lengths) would silently break the WASM
/// multi-worker sample-size path.
///
/// Two single-core SSR runs with different seeds are pooled; the assertions
/// verify the pooled structure, not the exact crossing values (which are
/// statistically irreproducible at n_sims=128).
#[test]
fn single_core_sample_size_parts_merge_via_merge_sample_size_results() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let method_a = SampleSizeMethod::Grid {
        by: ByValue::Fixed(20),
        mode: GridMode::Linear,
    };
    let method_b = SampleSizeMethod::Grid {
        by: ByValue::Fixed(20),
        mode: GridMode::Linear,
    };
    // Two workers: same grid, different seeds, same n_sims.
    let part_a = run_single_core_find_sample_size(
        &spec,
        (40, 200),
        method_a,
        128,
        11,
        &NullEmitter,
        &cancel,
    )
    .expect("part_a");
    let part_b = run_single_core_find_sample_size(
        &spec,
        (40, 200),
        method_b,
        128,
        22,
        &NullEmitter,
        &cancel,
    )
    .expect("part_b");

    let merged = merge_sample_size_results(&[part_a, part_b]).expect("merge ok");
    assert_eq!(merged.scenarios.len(), 1);
    let (_label, ssr): &(String, SampleSizeResult) = &merged.scenarios[0];

    // Grid structure is preserved across merge.
    assert!(
        ssr.grid_or_trace.len() >= 2,
        "merged grid must span multiple N"
    );
    let first_n = ssr.grid_or_trace[0].n;
    assert_eq!(first_n, 40, "first grid point is the lower bound");

    // Each grid point pooled both workers: n_sims doubles.
    for pt in &ssr.grid_or_trace {
        assert_eq!(
            pt.n_sims, 256,
            "each pooled grid point at N={} must carry n_sims=256 (128+128), got {}",
            pt.n, pt.n_sims,
        );
    }

    // `first_achieved` and `fitted` lengths must equal the power-vector length
    // (one slot per marginal target + contrasts; the crossing arrays are
    // parallel to `grid_or_trace[*].power_uncorrected`).
    let n_targets = ssr.grid_or_trace[0].power_uncorrected.len();
    assert_eq!(
        ssr.first_achieved.len(),
        n_targets,
        "first_achieved length must equal power-vector length"
    );
    assert_eq!(
        ssr.fitted.len(),
        n_targets,
        "fitted length must equal power-vector length"
    );
}

/// Twin invariant: single-core power is statistically equivalent to
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
