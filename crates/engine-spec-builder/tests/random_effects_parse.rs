//! Random-effect parse rows that the shared corpus in
//! `configs/formula-fixtures/cases.json` does not carry (driven by
//! `tests/formula_fixtures.rs`).

use engine_spec_builder::{parse_formula, RandomEffect, SpecError};

#[test]
fn intercept_suppression_rejected() {
    // The `(0+x|g)` / `(-1+x|g)` forms are corpus rows; these bare forms are not.
    for f in ["y ~ x + (0|g)", "y ~ x + (-1|g)"] {
        assert!(
            matches!(
                parse_formula(f),
                Err(SpecError::RandomInterceptSuppressionUnsupported)
            ),
            "expected suppression error for {f}, got {:?}",
            parse_formula(f)
        );
    }
}

#[test]
fn implicit_intercept_multivar_slope() {
    let p = parse_formula("y ~ x + z + (x+z|g)").unwrap();
    assert_eq!(
        p.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into(), "z".into()]
        }]
    );
}

#[test]
fn redundant_explicit_one_in_implicit_form_is_dropped() {
    // (x+1|g): the explicit '1' is a redundant intercept; vars = [x].
    let p = parse_formula("y ~ x + (x+1|g)").unwrap();
    assert_eq!(
        p.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
}
