use engine_app_spec::AppSpec;
use serde_json::json;

#[test]
fn logit_appspec_round_trips_through_json() {
    let raw = json!({
        "family": "logit",
        "parsed_formula": {
            "outcome": "y",
            "predictors": ["x1", "x2"],
            "interaction_terms": [["x1", "x2"]]
        },
        "var_types": [
            { "kind": "numeric", "name": "x1" },
            { "kind": "binary",  "name": "x2", "binary_proportion": 0.4 }
        ],
        "effects": [
            { "name": "x1", "value": 0.3 },
            { "name": "x2", "value": 0.2 },
            { "name": "x1:x2", "value": 0.1 }
        ],
        "correlations": null,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_sims": 1000,
        "seed": 2137,
        "tests": { "kind": "all" },
        "correction": "none",
        "scenarios": [],
        "csv": null,
        "baseline_probability": 0.3
    });
    let spec: AppSpec = serde_json::from_value(raw.clone()).expect("decode");
    let re = serde_json::to_value(&spec).expect("encode");
    assert_eq!(re, raw);
}

#[test]
fn family_tag_is_lowercase_logit() {
    let raw = json!({
        "family": "logit",
        "parsed_formula": { "outcome": "y", "predictors": ["x1"], "interaction_terms": [] },
        "var_types": [{ "kind": "numeric", "name": "x1" }],
        "effects": [{ "name": "x1", "value": 0.3 }],
        "correlations": null,
        "alpha": 0.05, "target_power": 0.8, "n_sims": 1000, "seed": 2137,
        "tests": { "kind": "all" }, "correction": "none",
        "scenarios": [], "csv": null,
        "baseline_probability": 0.5
    });
    let spec: AppSpec = serde_json::from_value(raw).unwrap();
    let s = serde_json::to_string(&spec).unwrap();
    assert!(s.contains(r#""family":"logit""#), "got: {s}");
}
