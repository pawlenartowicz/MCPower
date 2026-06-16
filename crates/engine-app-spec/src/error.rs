//! `AdapterError` — all failure modes for the GUI adapter layer (spec validation, builder, orchestrator).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("invalid binary proportion for `{name}`: {value} (must be in [0,1])")]
    InvalidProportion { name: String, value: f64 },

    #[error("factor `{name}`: expected {expected} proportions, got {got}")]
    FactorLevelMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    #[error("factor `{name}`: expected {expected} level labels, got {got}")]
    FactorLabelMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    #[error("factor `{name}`: duplicate level label `{label}`")]
    DuplicateFactorLabel { name: String, label: String },

    #[error("heteroskedasticity driver `{0}` is not a non-factor predictor in the model")]
    UnknownHeteroskedasticityDriver(String),

    #[error("factor `{name}`: reference index {reference} is out of range for {n_levels} levels")]
    FactorReferenceOutOfRange {
        name: String,
        reference: u32,
        n_levels: usize,
    },

    #[error("failed to encode effect skeleton to JSON: {0}")]
    SkeletonEncode(String),

    #[error("correlation matrix names ({n_names}) does not match values rows ({n_rows})")]
    InvalidCorrelations { n_names: usize, n_rows: usize },

    #[error("builder returned no contracts")]
    EmptyContracts,

    #[error("invalid baseline_probability: {value} (must be strictly in (0, 1))")]
    BaselineProbabilityOutOfRange { value: f64 },

    #[error("unknown predictor name in random slope: {0}")]
    UnknownPredictor(String),

    #[error("get_effects_from_data: no csv data attached to this spec")]
    GetEffectsNoCsv,

    #[error("get_effects_from_data: outcome column {outcome:?} not found in csv columns")]
    GetEffectsOutcomeMissing { outcome: String },

    #[error("get_effects_from_data: predictor {name:?} is in the model but missing from csv")]
    GetEffectsPredictorMissing { name: String },

    #[error("get_effects_from_data: beta vector length {betas} != design columns {cols}")]
    GetEffectsBetaColumnMismatch { betas: usize, cols: usize },

    #[error(
        "get_effects_from_data: column {name:?} has {got} values but expected {expected} (n_rows)"
    )]
    GetEffectsColumnLength {
        name: String,
        expected: usize,
        got: usize,
    },

    /// Declared var_type class conflicts with the column's auto-detected class.
    /// Authoritative guard for both the App and WASM — prevents a meaningless
    /// ColumnId-keyed kernel error from reaching the user.
    #[error("Column '{name}' was detected as {detected} from your uploaded data; it can't be modeled as {declared}. Uploaded columns take their type from the data.")]
    UploadClassConflict {
        name: String,
        detected: String,
        declared: String,
    },

    #[error(transparent)]
    SpecBuilder(#[from] engine_spec_builder::SpecError),

    #[error(transparent)]
    Orchestrator(#[from] engine_orchestrator::OrchestratorError),
}

impl AdapterError {
    /// Host-presentation category for the GUI error surface. Returns
    /// `"cluster_setup"` for the cluster-vs-sample-size-grid configuration
    /// errors the app frames as fixable settings (dedicated card + guidance),
    /// `"generic"` for everything else.
    ///
    /// Matches the typed orchestrator variant, NOT the Display string:
    /// `ClusterGridSinglePoint`'s message contains no "cluster" word, so a
    /// frontend substring check would silently misclassify it.
    pub fn host_kind(&self) -> &'static str {
        use engine_orchestrator::OrchestratorError as OE;
        match self {
            AdapterError::Orchestrator(
                OE::InvalidClusterAtom
                | OE::ClusterGridEmpty { .. }
                | OE::ClusterGridSinglePoint { .. }
                | OE::MixedClusterAtoms { .. }
                | OE::ClusterSizeTooSmall { .. }
                | OE::ClusterTooFewAtN { .. },
            ) => "cluster_setup",
            _ => "generic",
        }
    }
}

#[cfg(test)]
mod host_kind_tests {
    use super::*;
    use engine_orchestrator::OrchestratorError as OE;

    #[test]
    fn cluster_grid_errors_are_cluster_setup() {
        let cases = [
            AdapterError::Orchestrator(OE::ClusterSizeTooSmall { got: 1, min: 2 }),
            AdapterError::Orchestrator(OE::ClusterTooFewAtN {
                n: 20,
                cluster_size: 20,
                got: 1,
                min: 2,
            }),
            AdapterError::Orchestrator(OE::ClusterGridEmpty {
                from: 10,
                to: 5,
                atom: 4,
            }),
            AdapterError::Orchestrator(OE::MixedClusterAtoms { a: 4, b: 6 }),
            AdapterError::Orchestrator(OE::InvalidClusterAtom),
        ];
        for e in &cases {
            assert_eq!(e.host_kind(), "cluster_setup", "{e}");
        }
    }

    #[test]
    fn cluster_grid_single_point_classifies_without_a_cluster_word() {
        // The variant that motivates typed matching over a substring check.
        let e = AdapterError::Orchestrator(OE::ClusterGridSinglePoint {
            from: 10,
            to: 20,
            atom: 10,
        });
        assert_eq!(e.host_kind(), "cluster_setup");
        assert!(!e.to_string().contains("cluster"));
    }

    #[test]
    fn other_errors_are_generic() {
        assert_eq!(AdapterError::EmptyContracts.host_kind(), "generic");
        assert_eq!(
            AdapterError::Orchestrator(OE::InvalidScenarios("x".into())).host_kind(),
            "generic",
        );
    }
}
