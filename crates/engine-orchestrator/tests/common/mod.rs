// Shared fixture for orchestrator integration tests.
//
// Builds a `SimulationContract` for a simple 1-predictor OLS run.
// Trimmed to a single marginal test target so tests stay minimal.

// Not every integration-test binary exercises every shared fixture helper.
#![allow(dead_code)]

use engine_contract::ids::ColumnId;
use engine_contract::test_spec::CorrectionMethod;
use engine_contract::{
    ColumnSpec, Correlations, DesignSpec, DesignTerm, EstimatorSpec, GenerationSpec, OutcomeKind,
    OutcomeSpec, ResidualDist, ResidualSpec, ScenarioPerturbations, SimulationContract,
    SyntheticKind, TestSpec, TestTarget,
};

use engine_spec_builder::input::Correction as BuilderCorrection;
use engine_spec_builder::{
    EffectAssignment, HeteroskedasticityInput, LinearSpec, PredictorSpec,
    ResidualSpec as BuilderResidualSpec, VarKind,
};

pub fn minimal_ols_contract() -> SimulationContract {
    let scenario = ScenarioPerturbations {
        name: "optimistic".into(),
        ..ScenarioPerturbations::default()
    };
    SimulationContract {
        generation: GenerationSpec {
            columns: vec![ColumnSpec::Synthetic {
                kind: SyntheticKind::Normal,
                pinned: false,
            }],
            correlations: Correlations::Identity,
            cluster: None,
            uploaded_frame: None,
            cluster_level_columns: vec![],
        },
        design_generation: DesignSpec {
            terms: vec![
                DesignTerm::Const,
                DesignTerm::Direct {
                    column: ColumnId(0),
                },
            ],
        },
        outcome: OutcomeSpec {
            kind: OutcomeKind::Continuous,
            intercept: 0.0,
            coefficients: vec![0.0, 0.5],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: false,
            },
            heteroskedasticity_driver: None,
        },
        design_test: None,
        estimator: EstimatorSpec::Ols,
        wald_se: Default::default(),
        test: TestSpec {
            targets: vec![TestTarget::Marginal { term: 1 }],
            correction: CorrectionMethod::None,
            alpha: 0.05,
        },
        posthoc: vec![],
        scenario,
        max_failed_fraction: 0.05,
    }
}

/// Convenience wrapper for the common pattern of a single optimistic contract
/// with a custom scenario label.
pub fn minimal_ols_contract_labelled(label: &str) -> SimulationContract {
    let mut c = minimal_ols_contract();
    c.scenario.name = label.into();
    c
}

/// OLS contract carrying BOTH a marginal target and a pairwise contrast, so the
/// result vectors must be sized `target_indices.len() + contrast_pairs.len()`.
/// Exists to exercise the EP-1 length-asymmetry invariant end-to-end through a
/// real `run_batch` — the escaped-bug class survived because no orchestrator run
/// ever carried a contrast. The contrast's `negative` side is a real `Direct`
/// term (not `Const`), so the adapter keeps it a contrast rather than collapsing
/// it to a `Marginal`.
pub fn marginal_plus_contrast_contract() -> SimulationContract {
    let mut c = minimal_ols_contract();
    c.generation.columns.push(ColumnSpec::Synthetic {
        kind: SyntheticKind::Normal,
        pinned: false,
    });
    c.design_generation.terms.push(DesignTerm::Direct {
        column: ColumnId(1),
    });
    c.outcome.coefficients = vec![0.0, 0.5, 0.4];
    c.test.targets = vec![
        TestTarget::Marginal { term: 1 },
        TestTarget::Contrast {
            positive: 1,
            negative: 2,
        },
    ];
    c
}

/// Logit integration-test fixture.
///
/// Three continuous predictors + one binary predictor; intercept -0.5 is
/// supplied at `build_contract` time. The spec itself is outcome-agnostic;
/// `build_contract(&logit_spec(), OutcomeKind::Binary, None, -0.5, vec![])` produces a
/// Glm `SimulationContract`.
pub fn logit_spec() -> LinearSpec {
    LinearSpec {
        formula: "y = x1 + x2 + x3 + b".into(),
        predictors: vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x2".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x3".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "b".into(),
                pinned: false,
                kind: VarKind::Binary { proportion: 0.5 },
            },
        ],
        effects: vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.3,
            },
            EffectAssignment {
                name: "x2".into(),
                size: 0.2,
            },
            EffectAssignment {
                name: "x3".into(),
                size: 0.1,
            },
            EffectAssignment {
                name: "b".into(),
                size: 0.4,
            },
        ],
        correlations: vec![],
        alpha: 0.05,
        correction: BuilderCorrection::None,
        targets: vec!["x1".into()],
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: BuilderResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![],
        upload: None,
        cluster_level_vars: vec![],
        wald_se: Default::default(),
    }
}

/// LME integration-test fixture.
///
/// Two continuous predictors only; cluster structure is supplied at
/// `build_contract` time:
/// `build_contract(&lme_spec(), OutcomeKind::Continuous, None, 0.0,
///     vec![ClusterSpec { n_clusters: 20, tau_squared: 0.3 }])`.
pub fn lme_spec() -> LinearSpec {
    LinearSpec {
        formula: "y = x1 + x2".into(),
        predictors: vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "x2".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
        ],
        effects: vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.3,
            },
            EffectAssignment {
                name: "x2".into(),
                size: 0.2,
            },
        ],
        correlations: vec![],
        alpha: 0.05,
        correction: BuilderCorrection::None,
        targets: vec!["x1".into()],
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: BuilderResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![],
        upload: None,
        cluster_level_vars: vec![],
        wald_se: Default::default(),
    }
}
