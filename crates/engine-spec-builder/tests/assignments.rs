use engine_spec_builder::{
    parse_assignments, AssignmentKey, AssignmentKind, AssignmentValue, KnownNames, VarTypeParams,
};

fn known_owned() -> (Vec<String>, Vec<String>) {
    (vec!["x1".into(), "x2".into(), "x3".into()], vec![])
}

#[test]
fn parses_effect_assignments() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("x1=0.5, x2=-0.3", AssignmentKind::Effect, &k).unwrap();
    assert!(res.errors.is_empty());
    assert_eq!(res.items.len(), 2);
    assert!(matches!(&res.items[0].0, AssignmentKey::Name(n) if n == "x1"));
    assert!(matches!(res.items[0].1, AssignmentValue::Effect(v) if (v - 0.5).abs() < 1e-12));
    assert!(matches!(&res.items[1].0, AssignmentKey::Name(n) if n == "x2"));
    assert!(matches!(res.items[1].1, AssignmentValue::Effect(v) if (v - (-0.3)).abs() < 1e-12));
}

#[test]
fn parses_correlation_pairs_sorted_lex() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("corr(x2,x1)=0.4", AssignmentKind::Correlation, &k).unwrap();
    assert!(res.errors.is_empty());
    assert_eq!(res.items.len(), 1);
    assert!(matches!(&res.items[0].0, AssignmentKey::Pair(a, b) if a == "x1" && b == "x2"));
    assert!(matches!(res.items[0].1, AssignmentValue::Correlation(v) if (v - 0.4).abs() < 1e-12));
}

#[test]
fn unknown_name_collected_as_soft_error_not_top_level() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("xnone=0.5, x1=0.3", AssignmentKind::Effect, &k).unwrap();
    assert_eq!(res.items.len(), 1);
    assert_eq!(res.errors.len(), 1);
    assert!(matches!(&res.items[0].0, AssignmentKey::Name(n) if n == "x1"));
}

#[test]
fn malformed_input_returns_top_level_err() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let err = parse_assignments("nonsense_no_equals", AssignmentKind::Effect, &k).unwrap_err();
    // Typed match: string scraping can't catch a renamed or restructured variant.
    assert!(
        matches!(
            err,
            engine_spec_builder::SpecError::MalformedAssignment { .. }
        ),
        "expected MalformedAssignment, got {err:?}"
    );
}

#[test]
fn variable_type_known_value() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("x1=binary, x2=normal", AssignmentKind::VariableType, &k).unwrap();
    assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    assert_eq!(res.items.len(), 2);
    // Bare binary fills the default proportion 0.5; bare normal is Bare.
    assert!(
        matches!(&res.items[0].1, AssignmentValue::VariableTypeWithParams { var_type, params }
        if var_type == "binary" && matches!(params, VarTypeParams::Binary { proportion } if (*proportion - 0.5).abs() < 1e-12))
    );
    assert!(
        matches!(&res.items[1].1, AssignmentValue::VariableTypeWithParams { var_type, params }
        if var_type == "normal" && matches!(params, VarTypeParams::Bare))
    );
}

#[test]
fn variable_type_tuple_factor_proportions() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res =
        parse_assignments("x1=(factor,0.2,0.3,0.5)", AssignmentKind::VariableType, &k).unwrap();
    assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    assert_eq!(res.items.len(), 1);
    assert!(
        matches!(&res.items[0].1, AssignmentValue::VariableTypeWithParams { var_type, params }
        if var_type == "factor"
        && matches!(params, VarTypeParams::Factor { n_levels, proportions }
            if *n_levels == 3 && (proportions[0] - 0.2).abs() < 1e-12 && (proportions[2] - 0.5).abs() < 1e-12))
    );
}

#[test]
fn variable_type_tuple_binary_proportion() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("x1=(binary,0.3)", AssignmentKind::VariableType, &k).unwrap();
    assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    assert!(
        matches!(&res.items[0].1, AssignmentValue::VariableTypeWithParams { var_type, params }
        if var_type == "binary" && matches!(params, VarTypeParams::Binary { proportion } if (*proportion - 0.3).abs() < 1e-12))
    );
}

#[test]
fn variable_type_unknown_value_collected_soft() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("x1=bogus, x2=binary", AssignmentKind::VariableType, &k).unwrap();
    assert_eq!(res.items.len(), 1);
    assert_eq!(res.errors.len(), 1);
}

#[test]
fn empty_input_returns_empty_assignments() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("   ", AssignmentKind::Effect, &k).unwrap();
    assert!(res.items.is_empty());
    assert!(res.errors.is_empty());
}

#[test]
fn quoted_variable_type_value_accepted() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("x1=\"binary\"", AssignmentKind::VariableType, &k).unwrap();
    assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    assert_eq!(res.items.len(), 1);
    assert!(
        matches!(&res.items[0].1, AssignmentValue::VariableTypeWithParams { var_type, .. } if var_type == "binary")
    );
}

#[test]
fn correlation_with_whitespace_inside_parens() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let res = parse_assignments("corr( x1 , x2 ) = 0.25", AssignmentKind::Correlation, &k).unwrap();
    assert!(res.errors.is_empty(), "errors: {:?}", res.errors);
    assert_eq!(res.items.len(), 1);
    assert!(matches!(&res.items[0].0, AssignmentKey::Pair(a, b) if a == "x1" && b == "x2"));
}

// Unbalanced parentheses in the top-level comma split cannot be resolved,
// so the parser must surface a top-level MalformedAssignment (not a soft error).
#[test]
fn unbalanced_parens_returns_top_level_err() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let err = parse_assignments("corr(x1,x2=0.4", AssignmentKind::Correlation, &k).unwrap_err();
    assert!(
        matches!(
            err,
            engine_spec_builder::SpecError::MalformedAssignment { .. }
        ),
        "expected MalformedAssignment for unbalanced parens, got {err:?}"
    );
}

#[test]
fn self_pair_correlation_returns_descriptive_error() {
    let (predictors, interaction_terms) = known_owned();
    let k = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let result = parse_assignments("corr(x1,x1)=0.5", AssignmentKind::Correlation, &k);
    // Per Python parity (parsers.py), self-correlation must surface as a
    // truthful error — not "unknown name x1" which is a lie.
    match result {
        Ok(parsed) => {
            // If errors collected as soft (current implementation returns Err at top level,
            // so this branch should not fire).
            assert!(parsed.errors.iter().any(|e| matches!(e, engine_spec_builder::SpecError::MalformedAssignment { input } if input.contains("cannot correlate"))),
                "expected MalformedAssignment about self-correlation in collected errors");
        }
        Err(err) => {
            assert!(
                matches!(&err, engine_spec_builder::SpecError::MalformedAssignment { input } if input.contains("cannot correlate")),
                "expected MalformedAssignment about self-correlation, got {err:?}",
            );
        }
    }
}
