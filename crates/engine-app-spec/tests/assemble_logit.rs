use engine_app_spec::{AppSpec, EffectSize, LogitSpec, ParsedFormula, TestSelection, VarType};
use engine_contract::{CorrectionMethod, EstimatorSpec, OutcomeKind};
use serde::Serialize;

fn sample_logit_spec(baseline_probability: f64) -> AppSpec {
    AppSpec::Logit(LogitSpec {
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
        // Correlation is continuous-only by design; x2 is binary, so the fixture
        // carries no predictor correlation (a binary-involving pair is rejected
        // at the spec builder).
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 1000,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        wald_se: Default::default(),
        scenarios: vec![],
        csv: None,
        baseline_probability,
        test_formula: None,
        outcome_options: None,
    })
}

// `assemble_spec` Logit behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn assemble_logit_returns_one_contract_with_family_logit() {
    let spec = sample_logit_spec(0.3);
    let contracts = engine_app_spec::assemble_spec(&spec).unwrap();
    assert_eq!(contracts.len(), 1);
    assert_eq!(contracts[0].outcome.kind, OutcomeKind::Binary);
    assert_eq!(contracts[0].estimator, EstimatorSpec::Glm);
}

#[test]
fn assemble_logit_sets_intercept_to_logit_of_baseline() {
    let spec = sample_logit_spec(0.3);
    let contracts = engine_app_spec::assemble_spec(&spec).unwrap();
    let expected = (0.3_f64 / 0.7_f64).ln();
    let got = contracts[0].outcome.intercept;
    assert!(
        (got - expected).abs() < 1e-12,
        "intercept = {got}, expected {expected}"
    );
}

#[test]
fn assemble_logit_rejects_baseline_at_or_below_zero() {
    for bad in [-0.1_f64, 0.0_f64] {
        let spec = sample_logit_spec(bad);
        let err = engine_app_spec::assemble_spec(&spec).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("baseline_probability") || msg.contains("baseline probability"),
            "got error {msg:?} for baseline={bad}"
        );
    }
}

#[test]
fn assemble_logit_rejects_baseline_at_or_above_one() {
    for bad in [1.0_f64, 1.1_f64] {
        let spec = sample_logit_spec(bad);
        let err = engine_app_spec::assemble_spec(&spec).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("baseline_probability") || msg.contains("baseline probability"),
            "got error {msg:?} for baseline={bad}"
        );
    }
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
fn assemble_logit_matches_golden_projection() {
    let spec = sample_logit_spec(0.3);
    let contracts = engine_app_spec::assemble_spec(&spec).unwrap();
    let summary = build_summary(&contracts);
    insta::assert_json_snapshot!(summary);
}
