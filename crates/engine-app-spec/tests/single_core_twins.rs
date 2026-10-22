//! Single-core run twins (WASM dispatch path): shape, merge-pooling, and pre-cancel.
//! Mirrors `driver_smoke.rs`; the merge test is the dispatch-twins L2 gate
//! (`run_single_core_* ×N + merge` produces a valid pooled result).

use engine_app_spec::{
    AdapterError, AppSpec, EffectSize, LinearSpec, NullEmitter, ParsedFormula, TestSelection,
    VarType, run_find_power, run_single_core_find_power, run_single_core_find_sample_size,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    ByValue, CancellationToken, GridMode, OrchestratorError, SampleSizeMethod, SampleSizeResult,
    ScenarioResult, merge_power_results, merge_sample_size_results,
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
fn single_core_parts_merge_to_full_run() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let part_a =
        run_single_core_find_power(&spec, 80, 800, 11, None, &NullEmitter, &cancel).expect("a");
    let part_b =
        run_single_core_find_power(&spec, 80, 800, 22, None, &NullEmitter, &cancel).expect("b");
    let counts_a = part_a.scenarios[0].1.success_counts_uncorrected.clone();
    let counts_b = part_b.scenarios[0].1.success_counts_uncorrected.clone();
    assert_eq!(
        counts_a.len(),
        part_a.scenarios[0].1.target_indices.len(),
        "one success count per target in each part"
    );
    // Non-empty guard: the zip/compare checks below pass vacuously on empty
    // vectors. The spec targets x1 and x2.
    assert_eq!(counts_a.len(), 2, "two targets (x1, x2)");
    let merged = merge_power_results(&[part_a, part_b]).expect("merge");
    assert_eq!(merged.scenarios.len(), 1);
    let (_label, power) = &merged.scenarios[0];
    assert_eq!(power.n_sims, 1600);
    // Pooling sums the per-part success counts and re-derives power from them.
    let pooled: Vec<u64> = counts_a.iter().zip(&counts_b).map(|(a, b)| a + b).collect();
    assert_eq!(power.success_counts_uncorrected, pooled);
    for (&p, &k) in power.power_uncorrected.iter().zip(&pooled) {
        assert_eq!(p, k as f64 / 1600.0, "pooled power = pooled count / n_sims");
        // Anti-no-op floor: β ∈ {0.3, 0.2} at N=80 sits well above alpha
        // (seeded, so deterministic).
        assert!(p > 0.3, "pooled power should be well above chance, got {p}");
    }
}

#[test]
fn run_single_core_find_power_errors_when_pre_cancelled() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let res = run_single_core_find_power(&spec, 80, 800, 11, None, &NullEmitter, &cancel);
    assert!(
        matches!(
            res,
            Err(AdapterError::Orchestrator(
                OrchestratorError::Cancelled { .. }
            ))
        ),
        "a pre-cancelled token must surface as Cancelled, got {:?}",
        res.err()
    );
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
        None,
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
        None,
        &NullEmitter,
        &cancel,
    )
    .expect("part_b");

    // Each worker hands its part to the main thread as JSON: the crossing
    // fields (`fitted`, `fitted_joint`, `cluster_atom`) must survive that wire.
    let json = serde_json::to_string(&part_a).expect("serializes");
    let part_a: ScenarioResult<SampleSizeResult> =
        serde_json::from_str(&json).expect("round-trips through the JSON wire");
    let wire_ssr = &part_a.scenarios[0].1;
    assert!(
        !wire_ssr.fitted.is_empty(),
        "fitted must be non-empty for a 2-target spec"
    );
    assert_eq!(wire_ssr.fitted.len(), wire_ssr.first_achieved.len());
    assert_eq!(
        wire_ssr.fitted_joint.len(),
        wire_ssr.first_joint_achieved.len()
    );
    assert_eq!(wire_ssr.cluster_atom, 1);

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

/// Twin invariant at the app-spec driver layer: both drivers pass the same
/// seed and n_sims to the orchestrator's `find_power` / `single_core_find_power`
/// twins, which are bit-identical at a fixed seed, so the driver outputs must be
/// equal. A driver that mis-plumbs the seed or n_sims breaks the equality.
#[test]
fn single_core_matches_native_find_power() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();

    // Native multi-core baseline: reads n_sims and seed from spec (128, 2137).
    let native = run_find_power(&spec, 80, &NullEmitter, &cancel).expect("native ok");
    // Single-core path: same seed, same n_sims, different dispatch (no rayon fan-out).
    let single = run_single_core_find_power(&spec, 80, 128, 2137, None, &NullEmitter, &cancel)
        .expect("single ok");

    assert_eq!(
        native.scenarios.len(),
        1,
        "no scenarios → one baseline result"
    );
    let (_, native_pw) = &native.scenarios[0];
    let (_, single_pw) = &single.scenarios[0];
    assert_eq!(
        native_pw.target_indices.len(),
        native_pw.power_uncorrected.len(),
        "native path: one power entry per target"
    );
    assert!(
        !native_pw.power_uncorrected.is_empty(),
        "native path must report at least one target"
    );
    assert_eq!(native_pw.n_sims, single_pw.n_sims);
    assert_eq!(
        native_pw.success_counts_uncorrected, single_pw.success_counts_uncorrected,
        "same seed + n_sims: twin drivers must agree exactly"
    );
    assert_eq!(native_pw.power_uncorrected, single_pw.power_uncorrected);
}
