use engine_contract::{fixtures::example1_simple_ols, validate, ColumnSpec};

#[test]
fn free_validate_passes_on_example1() {
    validate(&example1_simple_ols()).expect("example1 must validate");
}

/// The free `validate` function must run the same checks as
/// `SimulationContract::validate`, not short-circuit to `Ok` — prove it forwards by
/// rejecting a known-invalid contract (a Resampled column with no uploaded frame).
/// The positive-only test above passes even against a `fn validate(_) { Ok(()) }` stub.
#[test]
fn free_validate_forwards_and_rejects_invalid() {
    let mut c = example1_simple_ols();
    c.generation.columns[0] = ColumnSpec::Resampled { frame_column: 0 };
    assert!(validate(&c).is_err(), "free validate must reject the bad contract");
    assert!(c.validate().is_err(), "method validate must reject it too");
}
