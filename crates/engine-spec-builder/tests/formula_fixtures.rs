//! Drives the shared parser corpus (`configs/formula-fixtures/cases.json` and
//! `configs/formula-fixtures/assignments/*.json`) through this crate's parsers.
//!
//! Comparison mirrors `ports/py/tests/spec/test_formula_fixtures.py`:
//! - a positive formula row must equal `expected` exactly, compared against the
//!   serde-derived JSON of `ParsedFormula` (the shape the R bridge emits);
//! - a negative row (`error` key) must fail with a `SpecError` whose Display text
//!   (the message both host bridges raise) matches the `error` regex;
//! - an assignment row whose `expected` is exactly `{"error": regex}` must fail at
//!   the top level; any other row must equal `expected` exactly, using the
//!   `{"items": [{"key", "value"}], "errors": [Display]}` projection the
//!   engine-py / engine-r bridges build.

use std::path::{Path, PathBuf};

use engine_spec_builder::{
    AssignmentKey, AssignmentKind, AssignmentValue, Assignments, KnownNames, VarTypeParams,
    parse_assignments, parse_formula,
};
use regex::Regex;
use serde_json::{Value, json};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/formula-fixtures")
}

fn read_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn error_matches(pattern: &str, message: &str) -> bool {
    Regex::new(pattern)
        .unwrap_or_else(|e| panic!("bad error regex {pattern:?}: {e}"))
        .is_match(message)
}

#[test]
fn formula_cases_match_corpus() {
    let corpus = read_json(&corpus_dir().join("cases.json"));
    let cases = corpus["cases"]
        .as_array()
        .expect("cases.json: `cases` array");
    assert!(!cases.is_empty(), "cases.json has no cases");

    let mut failures = Vec::new();
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let formula = case["formula"].as_str().expect("case formula");
        match (case.get("error"), parse_formula(formula)) {
            (Some(pattern), Err(e)) => {
                let pattern = pattern.as_str().expect("error regex");
                if !error_matches(pattern, &e.to_string()) {
                    failures.push(format!("{id}: error {e} does not match /{pattern}/"));
                }
            }
            (Some(_), Ok(parsed)) => {
                failures.push(format!("{id}: expected an error, parsed {parsed:?}"));
            }
            (None, Ok(parsed)) => {
                let got = serde_json::to_value(&parsed).expect("ParsedFormula serializes");
                if got != case["expected"] {
                    failures.push(format!("{id}: got {got}, expected {}", case["expected"]));
                }
            }
            (None, Err(e)) => failures.push(format!("{id}: unexpected error {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn assignments_json(a: &Assignments) -> Value {
    let items: Vec<Value> = a
        .items
        .iter()
        .map(|(key, value)| {
            let key = match key {
                AssignmentKey::Name(n) => json!({ "name": n }),
                AssignmentKey::Pair(x, y) => json!({ "pair": [x, y] }),
            };
            let value = match value {
                AssignmentValue::VariableTypeWithParams { var_type, params } => {
                    let info = match params {
                        VarTypeParams::Bare => json!({ "type": var_type }),
                        VarTypeParams::Binary { proportion } => {
                            json!({ "type": var_type, "proportion": proportion })
                        }
                        VarTypeParams::Factor {
                            n_levels,
                            proportions,
                        } => json!({
                            "type": var_type,
                            "n_levels": n_levels,
                            "proportions": proportions,
                        }),
                    };
                    json!({ "variable_type": info })
                }
                AssignmentValue::Correlation(r) => json!({ "correlation": r }),
                AssignmentValue::Effect(b) => json!({ "effect": b }),
            };
            json!({ "key": key, "value": value })
        })
        .collect();
    let errors: Vec<String> = a.errors.iter().map(ToString::to_string).collect();
    json!({ "items": items, "errors": errors })
}

fn string_list(v: &Value, field: &str) -> Vec<String> {
    v[field]
        .as_array()
        .unwrap_or_else(|| panic!("known.{field} must be an array"))
        .iter()
        .map(|s| s.as_str().expect("known names are strings").to_string())
        .collect()
}

#[test]
fn assignment_cases_match_corpus() {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(corpus_dir().join("assignments"))
        .expect("assignments dir")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no assignment fixtures found");

    let mut failures = Vec::new();
    for path in &paths {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let fixture = read_json(path);
        let input = fixture["input"].as_str().expect("input");
        let kind = match fixture["kind"].as_str().expect("kind") {
            "variable_type" => AssignmentKind::VariableType,
            "correlation" => AssignmentKind::Correlation,
            "effect" => AssignmentKind::Effect,
            other => panic!("{stem}: unknown kind {other:?}"),
        };
        let predictors = string_list(&fixture["known"], "predictors");
        let interaction_terms = string_list(&fixture["known"], "interaction_terms");
        let known = KnownNames {
            predictors: &predictors,
            interaction_terms: &interaction_terms,
        };
        let expected = &fixture["expected"];
        let top_level_error = expected
            .as_object()
            .filter(|o| o.len() == 1)
            .and_then(|o| o.get("error"))
            .and_then(Value::as_str);
        match (top_level_error, parse_assignments(input, kind, &known)) {
            (Some(pattern), Err(e)) => {
                if !error_matches(pattern, &e.to_string()) {
                    failures.push(format!("{stem}: error {e} does not match /{pattern}/"));
                }
            }
            (Some(_), Ok(a)) => failures.push(format!(
                "{stem}: expected a top-level error, got {}",
                assignments_json(&a)
            )),
            (None, Ok(a)) => {
                let got = assignments_json(&a);
                if &got != expected {
                    failures.push(format!("{stem}: got {got}, expected {expected}"));
                }
            }
            (None, Err(e)) => failures.push(format!("{stem}: unexpected error {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
