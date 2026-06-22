//! engine-wasm marshalling round-trips (wasm-bindgen-test).
//! Asserts JSON in → JSON out for the run/merge/parse exports — not numerics.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
// Run under Node via `wasm-pack test --node` (no headless browser in CI/dev here).
// Omitting `wasm_bindgen_test_configure!(run_in_browser)` selects the Node runner.

const LINEAR_SPEC_JSON: &str = r#"{"family":"linear","parsed_formula":{"outcome":"y","predictors":["x1","x2"],"interaction_terms":[]},"var_types":[{"kind":"numeric","name":"x1"},{"kind":"numeric","name":"x2"}],"effects":[{"name":"x1","value":0.3},{"name":"x2","value":0.2}],"correlations":null,"alpha":0.05,"target_power":0.8,"n_sims":128,"seed":2137,"tests":{"kind":"all"},"correction":"none","scenarios":[],"csv":null,"report_overall":false,"contrasts":[]}"#;

/// Minimal unclustered logit spec that produces NaN-bearing GLM extras when run
/// on a single worker. `baseline_prob_realized` and `tau_squared_hat_mean` are
/// NaN in the single-core result (both have zero-count denominators in
/// `from_batch`); they must survive `serde_json` null encoding through the real
/// `engine_wasm::find_power` → `engine_wasm::merge_power_results` boundary.
/// Mirrors `native_roundtrip::nan_bearing_extras_survive_json_wire_and_merge`
/// but exercises the actual wasm-bindgen export wrappers.
const LOGIT_SPEC_JSON: &str = r#"{"family":"logit","parsed_formula":{"outcome":"y","predictors":["x1"],"interaction_terms":[]},"var_types":[{"kind":"numeric","name":"x1"}],"effects":[{"name":"x1","value":0.3}],"correlations":null,"alpha":0.05,"target_power":0.8,"n_sims":64,"seed":2137,"tests":{"kind":"all"},"correction":"none","scenarios":[],"csv":null,"baseline_probability":0.3,"report_overall":false,"contrasts":[]}"#;

#[wasm_bindgen_test]
fn find_power_returns_scenario_result_json() {
    let out = engine_wasm::find_power(LINEAR_SPEC_JSON, 80, 200, 11, None);
    let out = match out {
        Ok(s) => s,
        Err(_) => panic!("find_power errored"),
    };
    let v: serde_json::Value = serde_json::from_str(&out).expect("valid json");
    // Cross the real wasm-bindgen boundary and confirm the contract SHAPE + values
    // survive it — not merely that a "scenarios" key exists. This is the field-
    // preservation path behind the historical NaN crash; mirrors the native bar in
    // native_roundtrip::find_power_json_round_trips_natively.
    let scenarios = v
        .get("scenarios")
        .and_then(|s| s.as_array())
        .expect("scenarios array");
    assert_eq!(scenarios.len(), 1, "one scenario");
    let power = &scenarios[0].as_array().expect("(label, PowerResult) tuple")[1];
    assert_eq!(
        power.get("n_sims").and_then(|n| n.as_u64()),
        Some(200),
        "n_sims must survive the boundary"
    );
    let pu = power
        .get("power_uncorrected")
        .and_then(|p| p.as_array())
        .expect("power_uncorrected array");
    assert!(!pu.is_empty(), "power values present");
    assert!(
        pu.iter().all(|p| p
            .as_f64()
            .map(|x| (0.0..=1.0).contains(&x))
            .unwrap_or(false)),
        "all power values in [0,1]"
    );
}

#[wasm_bindgen_test]
fn parse_formula_round_trips() {
    let out = match engine_wasm::parse_formula("y ~ x1 + x2") {
        Ok(s) => s,
        Err(_) => panic!("parse errored"),
    };
    let v: serde_json::Value = serde_json::from_str(&out).expect("valid json");
    assert!(v.get("dependent").is_some());
    assert!(v.get("predictors").is_some());
}

#[wasm_bindgen_test]
fn merge_pools_two_power_parts() {
    let part = match engine_wasm::find_power(LINEAR_SPEC_JSON, 80, 200, 11, None) {
        Ok(s) => s,
        Err(_) => panic!("part errored"),
    };
    let parts_json = format!("[{part},{part}]");
    let merged = match engine_wasm::merge_power_results(&parts_json) {
        Ok(s) => s,
        Err(_) => panic!("merge errored"),
    };
    let v: serde_json::Value = serde_json::from_str(&merged).expect("json");
    let n_sims = v["scenarios"][0][1]["n_sims"].as_u64().expect("n_sims");
    assert_eq!(n_sims, 400);
}

/// EP-2 via the real wasm-bindgen exports: NaN-bearing GLM extras survive the
/// JSON wire through `engine_wasm::find_power` and `engine_wasm::merge_power_results`.
///
/// A wrong or missing `nan_tolerant` serde attribute on the exported wrapper's
/// result type would cause `merge_power_results` to error on the null→f64
/// decode — an error that the native path (which bypasses serde_json round-trips
/// inside the wasm-bindgen glue) cannot catch. Running under `wasm-pack test
/// --node` exercises the real export boundary.
///
/// Mirrors `native_roundtrip::nan_bearing_extras_survive_json_wire_and_merge`.
#[wasm_bindgen_test]
fn nan_bearing_extras_survive_wasm_json_wire_and_merge() {
    // Single-core logit run: the worker calls find_power and sends JSON back
    // to the main thread. baseline_prob_realized and tau_squared_hat_mean are
    // NaN in a per-worker result (zero-count denominators in from_batch) →
    // serde_json encodes them as JSON null.
    let part = engine_wasm::find_power(LOGIT_SPEC_JSON, 80, 64, 11, None)
        .expect("find_power must succeed for a valid logit spec");

    // The JSON produced by find_power must contain "null" for the NaN fields.
    assert!(
        part.contains("null"),
        "single-core logit result must contain JSON null for NaN extras: {part}"
    );

    // The main thread merge call receives a JSON array of two worker parts.
    // If nan_tolerant is missing on any field, this deserialise step inside
    // merge_power_results will fail with a serde error.
    let parts_json = format!("[{part},{part}]");
    let merged = engine_wasm::merge_power_results(&parts_json)
        .expect("EP-2: merge_power_results must not error on NaN-bearing GLM extras");

    let v: serde_json::Value = serde_json::from_str(&merged).expect("merged result is valid JSON");
    let n_sims = v["scenarios"][0][1]["n_sims"]
        .as_u64()
        .expect("n_sims present");
    assert_eq!(n_sims, 128, "merged n_sims must sum the two 64-sim parts");

    // Confirm GLM extras are still present after the merge (not silently dropped).
    let extras = &v["scenarios"][0][1]["estimator_extras"];
    assert_eq!(
        extras.get("estimator").and_then(|e| e.as_str()),
        Some("glm"),
        "merged extras must remain GLM"
    );
}
