//! Test fixtures shared across unit and integration tests. Behind the
//! `test-fixtures` cargo feature so it never ships in release artefacts.

use crate::contract::SimulationContract;
use crate::design::{DesignSpec, DesignTerm};
use crate::estimator::EstimatorSpec;
use crate::generation::{ColumnSpec, Correlations, GenerationSpec, SyntheticKind};
use crate::ids::ColumnId;
use crate::outcome::{OutcomeKind, OutcomeSpec, ResidualDist, ResidualSpec};
use crate::scenarios::ScenarioPerturbations;
use crate::test_spec::{CorrectionMethod, TestSpec, TestTarget};

/// Canonical minimal OLS contract used by the roundtrip and validation tests.
pub fn example1_simple_ols() -> SimulationContract {
    SimulationContract {
        generation: GenerationSpec {
            columns: vec![
                ColumnSpec::Synthetic {
                    kind: SyntheticKind::Normal,
                    pinned: false,
                },
                ColumnSpec::Synthetic {
                    kind: SyntheticKind::Normal,
                    pinned: false,
                },
            ],
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
                DesignTerm::Direct {
                    column: ColumnId(1),
                },
            ],
        },
        outcome: OutcomeSpec {
            kind: OutcomeKind::Continuous,
            intercept: 0.0,
            coefficients: vec![0.0, 0.5, 0.3],
            residual: ResidualSpec {
                distribution: ResidualDist::Normal,
                pinned: false,
            },
            heteroskedasticity_driver: None,
            link: None,
        },
        design_test: None, // None = correctly-specified (uses design_generation)
        estimator: EstimatorSpec::Ols,
        wald_se: Default::default(),
        nagq: 1,
        test: TestSpec {
            targets: vec![
                TestTarget::Marginal { term: 1 },
                TestTarget::Marginal { term: 2 },
            ],
            correction: CorrectionMethod::None,
            alpha: 0.05,
        },
        posthoc: vec![],
        scenario: ScenarioPerturbations::default(),
        max_failed_fraction: 0.03,
    }
}
