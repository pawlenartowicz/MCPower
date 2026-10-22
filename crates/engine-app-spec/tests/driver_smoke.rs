use engine_app_spec::{
    AdapterError, AppSpec, EffectSize, LinearSpec, LogitSpec, NullEmitter, ParsedFormula,
    TestSelection, VarType, run_find_power, run_find_sample_size, run_single_core_find_power,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::{
    ByValue, CancellationToken, EstimatorExtras, GridMode, OrchestratorError, SampleSizeMethod,
};

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
            VarType::Numeric {
                name: "x2".into(),
                distribution: Default::default(),
                pinned: false,
            },
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
            VarType::Numeric {
                name: "x1".into(),
                distribution: Default::default(),
                pinned: false,
            },
            VarType::Numeric {
                name: "x2".into(),
                distribution: Default::default(),
                pinned: false,
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
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 128,
        seed: 2137,
        tests: TestSelection::All,
        correction: CorrectionMethod::None,
        wald_se: Default::default(),
        scenarios: vec![],
        csv: None,
        baseline_probability: 0.3,
        link: Default::default(),
        test_formula: None,
        outcome_options: None,
        agq: 1,
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

/// EP-3 twin — GAP 1: `run_find_power_smoke_logit` above exercises `run_find_power`
/// only; the WASM worker path dispatches through `run_single_core_find_power` and
/// was never exercised for the Logit family. Divergence risk: Glm estimator
/// extras mis-tagged or absent on the single-core path.
#[test]
fn run_single_core_find_power_smoke_logit() {
    let spec = sample_logit_spec_for_driver();
    let cancel = CancellationToken::new();
    let result =
        run_single_core_find_power(&spec, 80, 128, 2137, None, &NullEmitter, &cancel).expect("ok");
    assert_eq!(result.scenarios.len(), 1);
    let (_label, power) = &result.scenarios[0];
    assert_eq!(
        power.target_indices.len(),
        power.power_uncorrected.len(),
        "one power entry per target"
    );
    assert!(
        matches!(power.estimator_extras, EstimatorExtras::Glm { .. }),
        "Logit family must tag EstimatorExtras::Glm on the single-core path; \
         got: {:?}",
        power.estimator_extras,
    );
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
    // Grid 40..=200 by 20: every point ran the spec's n_sims and carries one
    // power entry per target.
    let ns: Vec<usize> = size_result.grid_or_trace.iter().map(|p| p.n).collect();
    assert_eq!(ns, (40..=200).step_by(20).collect::<Vec<_>>());
    for pt in &size_result.grid_or_trace {
        assert_eq!(pt.n_sims, 128, "N={} must run the spec's n_sims", pt.n);
        assert_eq!(
            pt.power_uncorrected.len(),
            pt.target_indices.len(),
            "one power entry per target"
        );
    }
}

#[test]
fn run_find_power_returns_error_when_token_pre_cancelled() {
    let spec = sample_linear_spec();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let res = run_find_power(&spec, 80, &NullEmitter, &cancel);
    assert!(
        matches!(
            res,
            Err(AdapterError::Orchestrator(
                OrchestratorError::Cancelled { .. }
            ))
        ),
        "a pre-cancelled token must surface as Cancelled, got {:?}",
        res.err()
    );
}
