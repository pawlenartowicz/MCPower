//! Serialization-shape properties for the random-effect / formula types.
//!
//! Exercises the hand-written `Serialize` impls on `RandomEffect` and `Term`
//! (formula.rs): a broken impl — wrong tag, dropped field, wrong field name —
//! fails these assertions.

use engine_spec_builder::{ParsedFormula, RandomEffect, Term};

#[test]
fn random_effect_intercept_serializes_with_kind_group_parent() {
    let parent = RandomEffect::Intercept {
        group: "A".into(),
        parent: None,
    };
    let child = RandomEffect::Intercept {
        group: "A:B".into(),
        parent: Some("A".into()),
    };
    let v_parent = serde_json::to_value(&parent).unwrap();
    assert_eq!(v_parent["kind"], "intercept");
    assert_eq!(v_parent["group"], "A");
    assert!(v_parent["parent"].is_null(), "parent: None → null");

    let v_child = serde_json::to_value(&child).unwrap();
    assert_eq!(v_child["kind"], "intercept");
    assert_eq!(v_child["group"], "A:B");
    assert_eq!(v_child["parent"], "A");
}

#[test]
fn random_effect_slope_serializes_with_var_list() {
    let slope = RandomEffect::Slope {
        group: "g".into(),
        vars: vec!["x".into(), "y".into()],
    };
    let v = serde_json::to_value(&slope).unwrap();
    assert_eq!(v["kind"], "slope");
    assert_eq!(v["group"], "g");
    // Var list preserved in order.
    assert_eq!(v["vars"], serde_json::json!(["x", "y"]));
}

#[test]
fn parsed_formula_serializes_terms_and_random_effects() {
    let p = ParsedFormula {
        dependent: "y".into(),
        predictors: vec!["x1".into()],
        terms: vec![
            Term::Main { name: "x1".into() },
            Term::Interaction {
                vars: vec!["x1".into(), "x2".into()],
            },
        ],
        random_effects: vec![],
    };
    let v = serde_json::to_value(&p).unwrap();
    // Main and Interaction terms carry distinct tags.
    assert_eq!(v["terms"][0]["kind"], "main");
    assert_eq!(v["terms"][0]["name"], "x1");
    assert_eq!(v["terms"][1]["kind"], "interaction");
    assert_eq!(v["terms"][1]["vars"], serde_json::json!(["x1", "x2"]));
    // random_effects field is present and empty.
    assert_eq!(v["random_effects"], serde_json::json!([]));
}
