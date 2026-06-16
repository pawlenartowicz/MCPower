//! Drives canonical-suite.json through engine-app-spec's host-agnostic parser,
//! normalizes to the port-neutral canonical shape, and asserts equality with the
//! shared `expected`. Mirrors ports/py/tests/test_canonical_suite.py.

use std::path::PathBuf;

use engine_app_spec::{parse_formula, FormulaParse, RandomEffectJson, TermJson};
use serde::Deserialize;

#[derive(Deserialize)]
struct Suite {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    formula: String,
    #[serde(default)]
    expected: Option<Expected>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Expected {
    outcome: String,
    fixed_effects: Vec<String>,
    random_effects: Vec<String>,
}

fn canonical(p: &FormulaParse) -> Expected {
    let fixed = p
        .terms
        .iter()
        .map(|t| match t {
            TermJson::Main { name } => name.clone(),
            TermJson::Interaction { vars } => vars.join(":"),
        })
        .collect();
    let res = p
        .random_effects
        .iter()
        .map(|r| match r {
            RandomEffectJson::Intercept { group, .. } => format!("intercept|{group}"),
            RandomEffectJson::Slope { group, vars } => format!("slope({})|{group}", vars.join(",")),
        })
        .collect();
    Expected {
        outcome: p.dependent.clone(),
        fixed_effects: fixed,
        random_effects: res,
    }
}

fn suite_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/formula-fixtures/canonical-suite.json")
}

// `parse_formula` canonical-suite behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn canonical_suite_matches_engine() {
    let raw = std::fs::read_to_string(suite_path()).expect("read canonical-suite.json");
    let suite: Suite = serde_json::from_str(&raw).expect("parse suite json");
    for case in &suite.cases {
        match (&case.expected, &case.error) {
            (Some(expected), None) => {
                let parsed = parse_formula(&case.formula)
                    .unwrap_or_else(|e| panic!("case {} should parse, got {e}", case.id));
                assert_eq!(&canonical(&parsed), expected, "case {}", case.id);
            }
            (None, Some(err)) => {
                let res = parse_formula(&case.formula);
                let msg = res
                    .expect_err(&format!("case {} should error", case.id))
                    .to_string();
                assert!(
                    msg.contains(err.as_str()),
                    "case {}: {msg:?} !~ {err:?}",
                    case.id
                );
            }
            _ => panic!("case {} must have exactly one of expected/error", case.id),
        }
    }
}
