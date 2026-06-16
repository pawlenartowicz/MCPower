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
    NullEmitter,
};
use engine_orchestrator::{
    merge_power_results, merge_sample_size_results, ByValue, CancellationToken, GridMode,
    PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
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
        assert!((0.0..=1.0).contains(&p), "power out of range: {p}");
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
    let part_a = run_single_core_find_sample_size(
        &spec,
        (50, 200),
        method,
        100,
        11,
        &NullEmitter,
        &cancel,
    )
    .expect("part a ok");
    let part_b = run_single_core_find_sample_size(
        &spec,
        (50, 200),
        method,
        100,
        12,
        &NullEmitter,
        &cancel,
    )
    .expect("part b ok");

    // The wasm worker hands each part over the JS boundary as JSON — prove
    // the new fields survive that wire shape.
    let json = serde_json::to_string(&part_a).expect("serializes");
    let back: ScenarioResult<SampleSizeResult> =
        serde_json::from_str(&json).expect("round-trips");
    let ssr = &back.scenarios[0].1;
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
