//! The ANOVA UI now emits AppSpec::Linear. This guards the Linear features that
//! the ANOVA layer relies on: a factor predictor with per-level effects, factor
//! contrast pairs, and tukey_hsd correction — end to end through the orchestrator.

use engine_app_spec::{
    run_find_power, AppSpec, EffectSize, LinearSpec, NullEmitter, ParsedFormula, TestSelection,
    VarType,
};
use engine_contract::CorrectionMethod;
use engine_orchestrator::CancellationToken;

fn linear_factor_spec() -> AppSpec {
    AppSpec::Linear(LinearSpec {
        parsed_formula: ParsedFormula {
            outcome: "y".into(),
            predictors: vec!["F1".into(), "cov1".into()],
            interaction_terms: vec![],
        },
        var_types: vec![
            VarType::Factor {
                name: "F1".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
            },
            VarType::Numeric { name: "cov1".into(), distribution: Default::default(), pinned: false },
        ],
        effects: vec![
            EffectSize {
                name: "F1[2]".into(),
                value: 0.4,
            },
            EffectSize {
                name: "F1[3]".into(),
                value: 0.4,
            },
            EffectSize {
                name: "cov1".into(),
                value: 0.3,
            },
        ],
        correlations: None,
        alpha: 0.05,
        target_power: 0.8,
        n_sims: 100,
        seed: 2137,
        tests: TestSelection::Effects {
            names: vec!["F1[2]".into(), "F1[3]".into()],
        },
        correction: CorrectionMethod::TukeyHsd,
        scenarios: vec![],
        csv: None,
        report_overall: true,
        contrasts: vec![("F1[2]".into(), "F1[3]".into())],
        test_formula: None,
        outcome_options: None,
    })
}

#[test]
fn linear_factor_with_contrasts_and_tukey_runs() {
    let cancel = CancellationToken::default();
    let result = run_find_power(&linear_factor_spec(), 200, &NullEmitter, &cancel)
        .expect("linear factor + contrasts + tukey should run");
    // Returns a well-formed single-scenario ScenarioResult<PowerResult>.
    // The contrast+tukey path emits power entries for the marginal targets *plus*
    // the contrast pairs, so power_uncorrected is longer than target_indices — both
    // must be non-empty with corrected/uncorrected the same length, and every power
    // is a rate in [0,1]. A run emitting an empty or malformed PowerResult fails.
    assert_eq!(result.scenarios.len(), 1);
    let (_, pr) = &result.scenarios[0];
    assert!(!pr.target_indices.is_empty(), "marginal targets present");
    assert!(!pr.power_uncorrected.is_empty(), "power vector present");
    assert_eq!(
        pr.power_uncorrected.len(),
        pr.power_corrected.len(),
        "uncorrected/corrected power vectors must have equal length"
    );
    assert!(
        pr.power_uncorrected
            .iter()
            .all(|&p| (0.0..=1.0).contains(&p)),
        "all uncorrected power values must be rates in [0,1]: {:?}",
        pr.power_uncorrected
    );
}
