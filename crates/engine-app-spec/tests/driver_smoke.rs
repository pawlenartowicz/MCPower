use engine_app_spec::{
    run_find_power, run_find_sample_size, AdapterError, AppSpec, EffectSize, LinearSpec, LogitSpec,
    NullEmitter, ParsedFormula, TestSelection, VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    ByValue, CancellationToken, EstimatorExtras, GridMode, SampleSizeMethod,
};

fn sample_linear_spec() -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into(), "x2".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric { name: "x1".into(), distribution: Default::default(), pinned: false },
            VarType::Numeric { name: "x2".into(), distribution: Default::default(), pinned: false },
        ],
        effects: vec![
            EffectSize {
                name: "x1".into(),
                value: 0.3,
            },
            EffectSize {
                name: "x2".into(),
                value: 0.2,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 128,
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

fn sample_logit_spec_for_driver() -> AppSpec {
    AppSpec::Logit(LogitSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["x1".into(), "x2".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Numeric { name: "x1".into(), distribution: Default::default(), pinned: false },
            VarType::Numeric { name: "x2".into(), distribution: Default::default(), pinned: false },
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
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 128,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        scenarios: vec![],
        csv: None,
        baseline_probability: 0.3,
        test_formula: None,
        outcome_options: None,
    })
}

#[test]
fn run_find_power_smoke_logit() {
    let spec = sample_logit_spec_for_driver();
    let cancel = CancellationToken::new();
    let result = run_find_power(&spec, 80, &NullEmitter, &cancel).expect("driver succeeds");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, power) = &result.scenarios[0];
    assert_eq!(power.target_indices.len(), power.power_uncorrected.len());
    assert!(matches!(
        power.estimator_extras,
        EstimatorExtras::Glm { .. }
    ));
}

// `run_find_power` driver behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn run_find_power_smoke() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let result = run_find_power(&spec, 80, &NullEmitter, &cancel).expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, power) = &result.scenarios[0];
    assert_eq!(power.target_indices.len(), power.power_uncorrected.len());
    assert!(!power.target_indices.is_empty());
}

// `run_find_sample_size` driver behavior is shared by the Tauri and WASM ports; the WASM
// single-core path mirrors this when it lands.
#[test]
fn run_find_sample_size_smoke() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    let method = SampleSizeMethod::Grid {
        by: ByValue::Fixed(20),
        mode: GridMode::Linear,
    };
    let result = run_find_sample_size(&spec, (40, 200), method, &NullEmitter, &cancel).expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, size_result) = &result.scenarios[0];
    // Grid (40..=200 by 20) ⇒ multiple points; pin structure, not just non-empty —
    // each point ran real sims and carries one power entry per target.
    assert!(size_result.grid_or_trace.len() >= 2, "grid spans multiple N");
    let first = &size_result.grid_or_trace[0];
    assert_eq!(first.n, 40, "first grid point is the lower bound");
    assert!(first.n_sims > 0, "grid point ran sims");
    assert_eq!(
        first.power_uncorrected.len(),
        first.target_indices.len(),
        "one power entry per target"
    );
}

#[test]
fn run_find_power_returns_error_when_token_pre_cancelled() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let res = run_find_power(&spec, 80, &NullEmitter, &cancel);
    assert!(matches!(res, Err(AdapterError::Orchestrator(_))));
}
