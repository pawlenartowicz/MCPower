use engine_spec_builder::{parse_formula, RandomEffect, SpecError};

#[test]
fn parses_random_intercept() {
    let f = parse_formula("y ~ x + (1|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Intercept {
            group: "g".into(),
            parent: None,
        }]
    );
    assert_eq!(f.predictors, vec!["x".to_string()]); // group var is NOT a predictor
}

#[test]
fn parses_random_slope_single_var() {
    let f = parse_formula("y ~ x + (1+x|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()],
        }]
    );
}

#[test]
fn parses_random_slope_multi_var() {
    let f = parse_formula("y ~ x + z + (1+x+z|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into(), "z".into()],
        }]
    );
}

#[test]
fn parses_nested_intercept_expands_to_pair() {
    let f = parse_formula("y ~ x + (1|A/B)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![
            RandomEffect::Intercept {
                group: "A".into(),
                parent: None,
            },
            RandomEffect::Intercept {
                group: "A:B".into(),
                parent: Some("A".into()),
            },
        ]
    );
}

#[test]
fn duplicate_grouping_var_errors() {
    let err = parse_formula("y ~ x + (1|g) + (1|g)").unwrap_err();
    assert!(matches!(err, SpecError::DuplicateGroupingVar { .. }));
}

#[test]
fn rhs_after_re_extraction_has_clean_plusses() {
    // "+ (1|g)" gets stripped; "x + +" must not leak.
    let f = parse_formula("y ~ x + (1|g)").unwrap();
    assert_eq!(f.terms.len(), 1); // only "x" — RE term doesn't appear in `terms`
    assert_eq!(
        f.terms[0],
        engine_spec_builder::Term::Main { name: "x".into() }
    );
}

#[test]
fn intercept_suppression_rejected() {
    for f in [
        "y ~ x + (0+x|g)",
        "y ~ x + (-1+x|g)",
        "y ~ x + (0|g)",
        "y ~ x + (-1|g)",
    ] {
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
fn implicit_intercept_slope_equals_explicit() {
    let imp = parse_formula("y ~ x + (x|g)").unwrap();
    let exp = parse_formula("y ~ x + (1+x|g)").unwrap();
    assert_eq!(imp.random_effects, exp.random_effects);
    assert_eq!(
        imp.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
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

#[test]
fn explicit_intercept_forms_unchanged() {
    // Regression: the digit-led patterns parse exactly as before.
    assert_eq!(
        parse_formula("y ~ (1|g)").unwrap().random_effects,
        vec![RandomEffect::Intercept {
            group: "g".into(),
            parent: None
        }]
    );
    assert_eq!(
        parse_formula("y ~ x + (1+x|g)").unwrap().random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
}
