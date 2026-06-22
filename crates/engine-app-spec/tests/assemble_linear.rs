use engine_app_spec::{AppSpec, EffectSize, LinearSpec, ParsedFormula, TestSelection, VarType};
use engine_contract::CorrectionMethod;
use serde::Serialize;

fn sample_linear_spec() -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into(), "x2".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric {
                name: "x1".into(),
                distribution: Default::default(),
                pinned: false,
            },
            VarType::Binary {
                name: "x2".into(),
                binary_proportion: 0.4,
            },
        ],
        effects: vec![
            EffectSize {
                name: "x1".into(),
                value: 0.5,
            },
            EffectSize {
                name: "x2".into(),
                value: 0.3,
            },
        ],
        // Correlation is continuous-only by design; x2 is binary, so the
        // fixture carries no predictor correlation (a binary-involving pair is
        // rejected at the spec builder). Continuous↔continuous correlation
        // assembly is covered in engine-spec-builder's pipeline tests.
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 1000,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        report_overall: false,
        contrasts: vec![],
        test_formula: None,
        outcome_options: None,
    })
}

// `assemble_spec` Linear behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn assemble_linear_returns_one_contract_when_no_scenario() {
    let spec = sample_linear_spec();
    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble should succeed");
    assert_eq!(
        contracts.len(),
        1,
        "exactly one contract expected with empty scenarios"
    );
    // Sanity: the contract was assembled (has non-zero coefficients)
    let contract = &contracts[0];
    assert!(
        !contract.outcome.coefficients.is_empty(),
        "assembled contract should have outcome coefficients"
    );
}

#[test]
fn assemble_linear_forwards_report_overall_true() {
    // When the AppSpec sets report_overall = true, the assembled contract
    // must contain a TestTarget::Joint covering all non-intercept positions
    // (the contract adapter then routes this to SimulationSpec.report_overall).
    let AppSpec::Linear(mut linear) = sample_linear_spec() else {
        panic!("expected AppSpec::Linear");
    };
    linear.report_overall = true;
    let spec = AppSpec::Linear(linear);
    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble");
    let targets = &contracts[0].test.targets;
    let has_omnibus = targets.iter().any(|t| {
        matches!(t, engine_contract::TestTarget::Joint { terms } if terms.iter().all(|n| *n >= 1))
    });
    assert!(
        has_omnibus,
        "expected a Joint omnibus target; got {targets:?}"
    );
}

#[test]
fn assemble_linear_forwards_report_overall_false() {
    // Default sample spec has report_overall = false → no Joint emitted.
    let spec = sample_linear_spec();
    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble");
    let targets = &contracts[0].test.targets;
    assert!(
        targets
            .iter()
            .all(|t| matches!(t, engine_contract::TestTarget::Marginal { .. })),
        "expected no Joint target when report_overall is false; got {targets:?}"
    );
}

#[test]
fn assemble_linear_forwards_test_formula_as_reduced_design() {
    // Generate from y = x1 + x2 but fit only x1: the assembled contract must
    // carry a separate design_test with fewer terms than design_generation.
    // A dropped/ignored test_formula leaves design_test None and fails this.
    let AppSpec::Linear(mut linear) = sample_linear_spec() else {
        panic!("expected AppSpec::Linear");
    };
    linear.test_formula = Some("y = x1".into());
    let spec = AppSpec::Linear(linear);
    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble");
    let contract = &contracts[0];
    let design_test = contract
        .design_test
        .as_ref()
        .expect("design_test must be Some when test_formula is set");
    assert!(
        design_test.terms.len() < contract.design_generation.terms.len(),
        "reduced test model must have fewer terms than generation: test={}, generation={}",
        design_test.terms.len(),
        contract.design_generation.terms.len()
    );
}

#[test]
fn assemble_linear_propagates_alpha_and_targets() {
    let spec = sample_linear_spec();
    let contracts = engine_app_spec::assemble_spec(&spec).expect("assemble should succeed");
    let contract = &contracts[0];
    // alpha is preserved
    assert_eq!(contract.test.alpha, 0.05);
    // With TestSelection::All + 2 predictors, there should be target entries
    assert!(
        !contract.test.targets.is_empty(),
        "assembled contract should have test targets"
    );
}

// ── Snapshot test ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AssembleSummary {
    n_contracts: usize,
    target_count: usize,
    coefficient_count: usize,
    generation_column_count: usize,
    alpha: f64,
    correction_code: i32,
}

fn build_summary(contracts: &[engine_contract::SimulationContract]) -> AssembleSummary {
    let c = &contracts[0];
    AssembleSummary {
        n_contracts: contracts.len(),
        target_count: c.test.targets.len(),
        coefficient_count: c.outcome.coefficients.len(),
        generation_column_count: c.generation.columns.len(),
        alpha: c.test.alpha,
        correction_code: c.test.correction.code(),
    }
}

#[test]
fn assemble_linear_matches_golden_projection() {
    let spec = sample_linear_spec();
    let contracts = engine_app_spec::assemble_spec(&spec).unwrap();
    let summary = build_summary(&contracts);
    insta::assert_json_snapshot!(summary);
}
