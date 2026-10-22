//! Assignment-grammar edge cases outside the shared corpus
//! (`configs/formula-fixtures/assignments/`, driven by `tests/formula_fixtures.rs`).

use engine_spec_builder::{
    AssignmentKey, AssignmentKind, AssignmentValue, KnownNames, SpecError, VarTypeParams,
    parse_assignments,
};

fn known_owned() -> (Vec<String>, Vec<String>) {
    (vec!["x1".into(), "x2".into(), "x3".into()], vec![])
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
        matches!(err, SpecError::MalformedAssignment { .. }),
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
    // Self-correlation is a per-item (soft) error that says what is wrong, not
    // "unknown name x1"; the item is dropped and the call itself succeeds.
    let parsed = parse_assignments("corr(x1,x1)=0.5", AssignmentKind::Correlation, &k).unwrap();
    assert!(parsed.items.is_empty(), "items: {:?}", parsed.items);
    assert!(
        matches!(&parsed.errors[..], [SpecError::MalformedAssignment { input }] if input.contains("cannot correlate")),
        "errors: {:?}",
        parsed.errors
    );
}
