use engine_app_spec::AppSpec;
use serde_json::json;

#[test]
fn linear_appspec_round_trips_through_json() {
    let raw = json!({
        "family": "linear",
        "parsed_formula": {
            "outcome": "y",
            "predictors": ["x1", "x2"],
            "interaction_terms": [["x1", "x2"]]
        },
        "var_types": [
            { "kind": "numeric", "name": "x1" },
            { "kind": "binary", "name": "x2", "binary_proportion": 0.4 }
        ],
        "effects": [
            { "name": "x1", "value": 0.3 },
            { "name": "x2", "value": 0.2 },
            { "name": "x1:x2", "value": 0.1 }
        ],
        "correlations": null,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_sims": 1600,
        "seed": 2137,
        "tests": { "kind": "all" },
        "correction": "none",
        "scenarios": [],
        "csv": null,
        "report_overall": false,
        "contrasts": []
    });
    let spec: AppSpec = serde_json::from_value(raw.clone()).expect("decode");
    let re = serde_json::to_value(&spec).expect("encode");
    assert_eq!(re, raw);
}

// The UI-overhaul wire fields round-trip with the exact JSON shapes the app's
// TS adapter emits: numeric `distribution`, factor `factor_labels` /
// `sampled_proportions`, `outcome_options`, and named extra groupings.
#[test]
fn ui_overhaul_fields_round_trip_through_json() {
    let raw = json!({
        "family": "linear",
        "parsed_formula": {
            "outcome": "y",
            "predictors": ["x1", "g"],
            "interaction_terms": []
        },
        "var_types": [
            { "kind": "numeric", "name": "x1", "distribution": "right_skewed" },
            { "kind": "factor", "name": "g", "factor_n_levels": 3,
              "factor_proportions": [0.4, 0.3, 0.3], "factor_reference": 1,
              "factor_labels": ["Europe", "Japan", "USA"],
              "sampled_proportions": true }
        ],
        "effects": [
            { "name": "x1", "value": 0.3 },
            { "name": "g[Europe]", "value": 0.2 },
            { "name": "g[USA]", "value": 0.2 }
        ],
        "correlations": null,
        "alpha": 0.05,
        "target_power": 0.8,
        "n_sims": 1600,
        "seed": 2137,
        "tests": { "kind": "all" },
        "correction": "none",
        "scenarios": [],
        "csv": null,
        "report_overall": false,
        "contrasts": [],
        "outcome_options": {
            "residual_distribution": "high_kurtosis",
            "heteroskedasticity_driver": "x1"
        }
    });
    let spec: AppSpec = serde_json::from_value(raw.clone()).expect("decode");
    let re = serde_json::to_value(&spec).expect("encode");
    assert_eq!(re, raw);
}

#[test]
fn named_extra_grouping_round_trips() {
    let raw = json!({
        "tau_squared": 0.1111111111111111,
        "relation": { "kind": "crossed", "n_clusters": 8 },
        "cluster_name": "item"
    });
    let g: engine_app_spec::app_spec::AppGroupingSpec =
        serde_json::from_value(raw.clone()).expect("decode");
    assert_eq!(g.cluster_name.as_deref(), Some("item"));
    let re = serde_json::to_value(&g).expect("encode");
    assert_eq!(re, raw);
}
