//! Native (non-wasm32) round-trip tests for the engine-wasm JSON contract.
//! Exercises the same serde + engine call sequence that the wasm-bindgen exports
//! perform, without the JS boundary glue (which requires `wasm-pack test --node`).
//! Mirrors `wasm_roundtrip.rs` which is dead in `cargo test --workspace` due to
//! its `#![cfg(target_arch = "wasm32")]` gate.
//!
//! These tests are NOT gated by target arch and run under standard `cargo test`.
//! The JSON const is adapted from the same fixture in `wasm_roundtrip.rs`;
//! `test_formula` is omitted (it has `#[serde(default)]` so it round-trips fine).

use engine_app_spec::{
    parse_formula, run_single_core_find_power, run_single_core_find_sample_size, AppSpec,
    EffectSize, LinearSpec, LogitSpec, NullEmitter, ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    merge_power_results, merge_sample_size_results, ByValue, CancellationToken, EstimatorExtras,
    GridMode, OrchestratorError, PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};

/// The same linear spec JSON used in `wasm_roundtrip.rs`. `test_formula` is
/// absent (serde default = None) — this is the current AppSpec serde shape.
const LINEAR_SPEC_JSON: &str = r#"{"family":"linear","parsed_formula":{"outcome":"y","predictors":["x1","x2"],"interaction_terms":[]},"var_types":[{"kind":"numeric","name":"x1"},{"kind":"numeric","name":"x2"}],"effects":[{"name":"x1","value":0.3},{"name":"x2","value":0.2}],"correlations":null,"alpha":0.05,"target_power":0.8,"n_sims":128,"seed":2137,"tests":{"kind":"all"},"correction":"none","scenarios":[],"csv":null,"report_overall":false,"contrasts":[]}"#;

/// Mirrors `wasm_roundtrip::find_power_returns_scenario_result_json`:
/// JSON spec → run_single_core_find_power → JSON result → parse → assert shape + values.
#[test]
fn find_power_json_round_trips_natively() {
    let spec: AppSpec = serde_json::from_str(LINEAR_SPEC_JSON).expect("spec deserializes");
    let cancel = CancellationToken::new();
    let result =
        run_single_core_find_power(&spec, 80, 200, 11, &NullEmitter, &cancel).expect("run ok");
    let json = serde_json::to_string(&result).expect("serializes");
    let back: ScenarioResult<PowerResult> = serde_json::from_str(&json).expect("round-trips");
    let (_, power) = &back.scenarios[0];
    assert_eq!(power.n_sims, 200, "n_sims should match the call argument");
    assert!(!power.power_uncorrected.is_empty(), "power values present");
    for &p in &power.power_uncorrected {
        // Discriminating floor: β ∈ {0.3, 0.2} at n=80 (n_sims=200, seed=11)
        // should clear > 0.3 for both targets; a no-op estimator would return 0.
        // Flag for central verification: floor confirmed by analytic expectation
        // (β=0.2 at n=80 → power ≈ 0.55), not by a seeded cargo run here.
        assert!(
            p > 0.3,
            "power should be well above chance for β≥0.2 at n=80, got {p}"
        );
    }
}

/// Mirrors `wasm_roundtrip::merge_pools_two_power_parts` natively.
/// Calls the orchestrator merge directly (not via the wasm-bindgen export).
#[test]
fn merge_power_results_pools_two_parts_natively() {
    let spec: AppSpec = serde_json::from_str(LINEAR_SPEC_JSON).expect("spec deserializes");
    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 200, 11, &NullEmitter, &cancel).expect("part ok");
    let merged = merge_power_results(&[part.clone(), part]).expect("merge ok");
    let (_, power) = &merged.scenarios[0];
    assert_eq!(power.n_sims, 400, "merged n_sims should sum the two parts");
}

/// Sample-size twin of `merge_power_results_pools_two_parts_natively`:
/// JSON round-trip through the engine-wasm wire shape, then the orchestrator
/// merge — covering the fields the model-based crossing added (`fitted`,
/// `fitted_joint`, `cluster_atom`), which must survive serde and be
/// recomputed from the pooled counts at merge.
#[test]
fn merge_sample_size_results_round_trips_and_pools_natively() {
    let spec: AppSpec = serde_json::from_str(LINEAR_SPEC_JSON).expect("spec deserializes");
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(50),
        mode: GridMode::Linear,
    };
    let part_a =
        run_single_core_find_sample_size(&spec, (50, 200), method, 100, 11, &NullEmitter, &cancel)
            .expect("part a ok");
    let part_b =
        run_single_core_find_sample_size(&spec, (50, 200), method, 100, 12, &NullEmitter, &cancel)
            .expect("part b ok");

    // The wasm worker hands each part over the JS boundary as JSON — prove
    // the new fields survive that wire shape.
    let json = serde_json::to_string(&part_a).expect("serializes");
    let back: ScenarioResult<SampleSizeResult> = serde_json::from_str(&json).expect("round-trips");
    let ssr = &back.scenarios[0].1;
    // Non-trivial crossing-fit check: 2 targets (x1, x2) → fitted must be
    // non-empty; a broken isotonic fitter that skips the computation would
    // return an empty vec while the shape-equality check still passed.
    assert!(
        !ssr.fitted.is_empty(),
        "fitted must be non-empty for a 2-target spec"
    );
    assert_eq!(ssr.fitted.len(), ssr.first_achieved.len());
    assert_eq!(ssr.fitted_joint.len(), ssr.first_joint_achieved.len());
    assert_eq!(ssr.cluster_atom, 1);

    let merged = merge_sample_size_results(&[back, part_b]).expect("merge ok");
    let merged_ssr = &merged.scenarios[0].1;
    assert_eq!(
        merged_ssr.grid_or_trace[0].n_sims, 200,
        "merged per-N n_sims sums the two parts"
    );
    assert_eq!(
        merged_ssr.fitted.len(),
        merged_ssr.first_achieved.len(),
        "merge recomputes fitted from pooled counts"
    );
}

/// Mirrors `wasm_roundtrip::parse_formula_round_trips` natively.
#[test]
fn parse_formula_json_round_trips_natively() {
    let parsed = parse_formula("y ~ x1 + x2").expect("parse ok");
    let json = serde_json::to_string(&parsed).expect("serializes");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
    assert!(v.get("dependent").is_some(), "dependent field present");
    assert!(v.get("predictors").is_some(), "predictors field present");
}

/// EP-1: power/ci/count vectors are sized `target_indices.len() + contrast_pairs.len()`.
///
/// Every other fixture in this file uses a two-predictor / no-contrast spec, so
/// the EP-1 length invariant holds only trivially. This test carries a real
/// contrast through the entire WASM path — JSON wire round-trip AND
/// `merge_power_results` — so a mis-sizing in either the adapter or the merge
/// would surface here rather than only at the orchestrator's merge-mismatch gate.
///
/// Design: 3-level factor `treatment` (levels "1"/"2"/"3", reference "1").
///   - 1 Marginal target  → `treatment[2]`
///   - 1 Contrast target  → `treatment[2]` vs `treatment[3]` (both non-reference)
///
/// Expected: `power_uncorrected.len() == 2`, `contrast_pairs.len() == 1`,
/// `success_counts_uncorrected.len() == 2`.
#[test]
fn contrast_bearing_spec_ep1_vector_lengths_survive_wire_and_merge() {
    // 3-level factor, reference level "1": dummies "2" and "3" become design
    // columns 1 and 2. The contrast pair ("treatment[2]", "treatment[3]")
    // maps to Contrast { positive: 1, negative: 2 } — both sides are
    // non-reference, so the assembler emits a true contrast, not a marginal.
    let spec = AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["treatment".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Factor {
            name: "treatment".into(),
            factor_n_levels: 3,
            factor_proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
        }],
        effects: vec![
            EffectSize {
                name: "treatment[2]".into(),
                value: 0.3,
            },
            EffectSize {
                name: "treatment[3]".into(),
                value: 0.5,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 128,
        seed: 2137,
        // One explicit marginal target + one contrast pair → total 2 result slots.
        tests: TestSelection::Effects {
            names: vec!["treatment[2]".into()],
        },
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts: vec![("treatment[2]".into(), "treatment[3]".into())],
        test_formula: None,
        outcome_options: None,
    });

    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 200, 11, &NullEmitter, &cancel).expect("run ok");

    // Wire round-trip: same path the WASM worker takes handing results to the
    // main thread over the JS boundary.
    let json = serde_json::to_string(&part).expect("serializes");
    let back: ScenarioResult<PowerResult> =
        serde_json::from_str(&json).expect("round-trips through JSON wire");

    let (_, power) = &back.scenarios[0];

    // EP-1: one marginal + one contrast → both result vectors must be length 2.
    assert_eq!(
        power.contrast_pairs.len(),
        1,
        "expected exactly 1 contrast pair"
    );
    assert_eq!(
        power.power_uncorrected.len(),
        power.target_indices.len() + power.contrast_pairs.len(),
        "EP-1: power_uncorrected must be sized target_indices + contrast_pairs"
    );
    assert_eq!(
        power.success_counts_uncorrected.len(),
        power.power_uncorrected.len(),
        "success_counts_uncorrected must parallel power_uncorrected"
    );

    // Merge path: same path the WASM main thread takes when aggregating two workers.
    let merged = merge_power_results(&[back.clone(), back]).expect("merge ok");
    let (_, merged_power) = &merged.scenarios[0];

    assert_eq!(
        merged_power.contrast_pairs.len(),
        1,
        "merge must preserve contrast_pairs"
    );
    assert_eq!(
        merged_power.power_uncorrected.len(),
        merged_power.target_indices.len() + merged_power.contrast_pairs.len(),
        "EP-1 must hold in the merged result"
    );
    assert_eq!(
        merged_power.n_sims, 400,
        "merged n_sims should sum the two parts"
    );
}

/// Merge-rejection: two parts with mismatched scenario labels must return
/// `OrchestratorError::IncompatibleMerge`, not panic or silently succeed.
///
/// Induced by running one part, cloning it, and overwriting the clone's label
/// before calling `merge_power_results` — the minimal perturbation that triggers
/// the label guard in `merge_one_power_result`'s outer loop.
#[test]
fn merge_power_results_rejects_mismatched_scenario_label() {
    let spec: AppSpec = serde_json::from_str(LINEAR_SPEC_JSON).expect("spec deserializes");
    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 64, 11, &NullEmitter, &cancel).expect("run ok");

    // Give the second "part" a different scenario label — both run the same
    // spec so target_indices agree; the label mismatch alone triggers the guard.
    let mut part_b = part.clone();
    part_b.scenarios[0].0 = "different_label".into();

    let result = merge_power_results(&[part, part_b]);
    assert!(
        result.is_err(),
        "merge must reject parts with mismatched scenario labels"
    );
    let OrchestratorError::IncompatibleMerge(_) = result.unwrap_err() else {
        panic!("error must be IncompatibleMerge");
    };
}

/// EP-2: NaN-bearing GLM extras survive the JSON wire and `merge_power_results`.
///
/// `serde_json` renders `f64::NAN` as JSON `null` and rejects `null` back as an
/// `f64` unless the field uses the `nan_tolerant` deserializer. Any logit result
/// from a single-core (WASM worker) run carries exactly two structural NaN fields:
///   - `baseline_prob_realized` — computed as `sum/n`, but per-worker `n == 0`
///     (the accumulator is filled only at `merge_power_results` time, not in the
///     single-core path's `EstimatorExtras::from_batch`)
///   - `tau_squared_hat_mean` — `tau_sum/tau_n`, with `tau_n == 0` for any
///     unclustered logit spec (no Laplace random-effect estimates)
///
/// The production WASM path is: worker calls `run_single_core_find_power` →
/// `serde_json::to_string` → postMessage → main thread `serde_json::from_str` →
/// `merge_power_results`. This test pins that `nan_tolerant` guards those fields,
/// so removing the guard later breaks this test rather than silently crashing a
/// live browser merge.
#[test]
fn nan_bearing_extras_survive_json_wire_and_merge() {
    // A minimal unclustered logit spec — `baseline_probability` anchors the
    // GLM intercept; effects are small so the run doesn't take long.
    let spec = AppSpec::Logit(LogitSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into()],
            interaction_terms: vec![],
        },
        var_types: vec![VarType::Numeric {
            name: "x1".into(),
            distribution: Default::default(),
            pinned: false,
        }],
        effects: vec![EffectSize {
            name: "x1".into(),
            value: 0.3,
        }],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 64,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        wald_se: Default::default(),
        link: Default::default(),
        agq: 1,
        scenarios: vec![],
        csv: None,
        baseline_probability: 0.3,
        test_formula: None,
        outcome_options: None,
    });

    let cancel = CancellationToken::new();
    let part =
        run_single_core_find_power(&spec, 80, 64, 11, &NullEmitter, &cancel).expect("run ok");

    // The single-core result must carry GLM extras with NaN in both structural
    // placeholders: baseline_prob_realized (accumulator only filled at merge,
    // never in from_batch) and tau_squared_hat_mean (tau_n==0, no RE estimates).
    let (_, pr) = &part.scenarios[0];
    let EstimatorExtras::Glm {
        baseline_prob_realized,
        tau_squared_hat_mean,
        ..
    } = &pr.estimator_extras
    else {
        panic!("logit run must yield Glm extras");
    };
    assert!(
        baseline_prob_realized.is_nan(),
        "baseline_prob_realized must be NaN before merge (n==0 in from_batch)"
    );
    assert!(
        tau_squared_hat_mean.is_nan(),
        "tau_squared_hat_mean must be NaN for unclustered logit (tau_n==0)"
    );

    // JSON wire: serde_json encodes NaN as null; nan_tolerant must decode it back.
    let json = serde_json::to_string(&part).expect("serializes");
    assert!(
        json.contains("null"),
        "NaN must appear as JSON null before the wire: {json}"
    );
    let back: ScenarioResult<PowerResult> =
        serde_json::from_str(&json).expect("EP-2: JSON null→NaN decode must not fail");

    let EstimatorExtras::Glm {
        baseline_prob_realized: bp_back,
        tau_squared_hat_mean: tau_back,
        ..
    } = &back.scenarios[0].1.estimator_extras
    else {
        panic!("extras must remain Glm after JSON round-trip");
    };
    assert!(
        bp_back.is_nan(),
        "baseline_prob_realized must survive JSON null→NaN"
    );
    assert!(
        tau_back.is_nan(),
        "tau_squared_hat_mean must survive JSON null→NaN"
    );

    // Merge path: two identical worker parts merged by the WASM main thread.
    // The merge must succeed (not panic/error on the NaN extras) and the merged
    // result's extras must still be Glm (NaN fields pool to 0.0/n==0 = NaN again).
    let merged = merge_power_results(&[back.clone(), back])
        .expect("EP-2: merge must not fail on NaN extras");
    let (_, merged_pr) = &merged.scenarios[0];
    assert!(
        matches!(merged_pr.estimator_extras, EstimatorExtras::Glm { .. }),
        "merged extras must remain Glm"
    );
    assert_eq!(
        merged_pr.n_sims, 128,
        "merged n_sims must sum the two parts"
    );
}
