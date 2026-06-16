//! engine-wasm marshalling round-trips (wasm-bindgen-test).
//! Asserts JSON in → JSON out for the run/merge/parse exports — not numerics.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
// Run under Node via `wasm-pack test --node` (no headless browser in CI/dev here).
// Omitting `wasm_bindgen_test_configure!(run_in_browser)` selects the Node runner.

const LINEAR_SPEC_JSON: &str = r#"{"family":"linear","parsed_formula":{"outcome":"y","predictors":["x1","x2"],"interaction_terms":[]},"var_types":[{"kind":"numeric","name":"x1"},{"kind":"numeric","name":"x2"}],"effects":[{"name":"x1","value":0.3},{"name":"x2","value":0.2}],"correlations":null,"alpha":0.05,"target_power":0.8,"n_sims":128,"seed":2137,"tests":{"kind":"all"},"correction":"none","scenarios":[],"csv":null,"report_overall":false,"contrasts":[]}"#;

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
    let power = &scenarios[0]
        .as_array()
        .expect("(label, PowerResult) tuple")[1];
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
        pu.iter()
            .all(|p| p.as_f64().map(|x| (0.0..=1.0).contains(&x)).unwrap_or(false)),
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
