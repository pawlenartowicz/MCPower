//! `ContractError` — crate-level validation error vocabulary; downstream crates reuse its variants.

use thiserror::Error;

/// Validation failures from `SimulationContract::validate` and the config /
/// theme validators; the `#[error]` strings carry the detail.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ContractError {
    #[error("coefficients.len ({coeffs}) must equal design_generation.terms.len ({terms})")]
    CoefficientLengthMismatch { coeffs: usize, terms: usize },

    #[error("test target term index {term} out of range (design_test has {n_terms} terms)")]
    TestTargetTermOutOfRange { term: u32, n_terms: usize },

    #[error("invalid TestSpec: {0}")]
    InvalidTestSpec(&'static str),

    #[error("ColumnId({id}) out of range (generation has {n_columns} columns)")]
    ColumnIdOutOfRange { id: u32, n_columns: usize },

    #[error("correlation matrix has {got} entries, expected {expected} (n={n})")]
    CorrelationDimensionMismatch {
        got: usize,
        expected: usize,
        n: usize,
    },

    #[error("correlation references non-continuous column (ColumnId({id}))")]
    CorrelationOnNonContinuous { id: u32 },

    #[error("factor column has invalid proportions: {0}")]
    InvalidFactorProportions(&'static str),

    #[error("dummy level_index {level_index} out of range (column ColumnId({column}) has {n_levels} levels)")]
    DummyLevelOutOfRange {
        column: u32,
        level_index: u32,
        n_levels: u32,
    },

    #[error("Direct term references factor column ColumnId({id}); use DummyOf instead")]
    DirectOnFactor { id: u32 },

    #[error("interaction term has {n} component(s); needs at least 2")]
    InteractionTooFewComponents { n: usize },

    #[error("interaction component must be Direct or DummyOf (no Const or nested Interaction)")]
    InteractionBadComponent,

    #[error("heteroskedasticity references non-continuous column ColumnId({id})")]
    HeteroskedasticityOnFactor { id: u32 },

    #[error(
        "frame_column {frame_column} references uploaded_frame but it is None or has fewer columns"
    )]
    UploadedFrameMissing { frame_column: u32 },

    #[error("estimator Mle requires generation.cluster to be Some(_)")]
    MleWithoutCluster,

    #[error("estimator Glm requires outcome.kind == Binary")]
    GlmRequiresBinary,

    #[error("posthoc tests require estimator == Ols")]
    PosthocRequiresOls,

    #[error("scenario.lme requires estimator == Mle")]
    LmeScenarioRequiresMle,

    #[error("extra grouping RE width q_g = 1 + #slopes must be ≤ {max}; got q_g = {got}")]
    ExtraGroupingQTooLarge { got: u32, max: u32 },

    #[error("SlopeTerm.column {id} is a factor or non-continuous column; random slopes require a continuous predictor")]
    SlopeColumnNotContinuous { id: u32 },

    #[error("SlopeTerm.column {id} has no Direct fixed-effect term; a random slope requires its variable as a fixed effect")]
    SlopeColumnNotInDesign { id: u32 },

    #[error("SlopeTerm.variance must be finite and >= 0; got {got}")]
    SlopeVarianceInvalid { got: f64 },

    #[error("a slope correlation must be finite in [-1, 1]; got {got}")]
    SlopeCorrInvalid { got: f64 },

    #[error("a random slope requires cluster.tau_squared > 0 (the intercept variance it correlates with); got {got}")]
    SlopeInterceptVarianceMissing { got: f64 },

    #[error("slope {slope} corr_with must list correlations with all earlier slopes; expected {expected}, got {got}")]
    SlopeCorrWithLenInvalid { slope: u32, expected: u32, got: u32 },

    #[error("the random-effects covariance D = diag(τ)·R·diag(τ) is not positive semidefinite")]
    SlopeCovarianceNotPSD,

    #[error("GroupingRelation::Crossed requires ClusterSizing::FixedClusters (crossed factors with a fixed-size primary are not supported)")]
    CrossedRequiresFixedClusters,

    #[error("at most one NestedWithin grouping is supported by this engine version")]
    MultipleNestedUnsupported,

    #[error("NestedWithin requires cluster_size % n_per_parent == 0; got {cluster_size} % {n_per_parent}")]
    NestedSizeIndivisible {
        cluster_size: u32,
        n_per_parent: u32,
    },

    #[error("extra grouping count must be >= 2; got {got}")]
    GroupingCountTooSmall { got: u32 },

    #[error("extra grouping tau_squared must be finite and >= 0; got {got}")]
    GroupingTauSquaredInvalid { got: f64 },

    #[error("at most {max} extra groupings are supported by this engine version; got {got}")]
    TooManyExtraGroupings { got: u32, max: u32 },

    #[error("primary RE width q_p = 1 + #slopes must be ≤ {max}; got q_p = {got}")]
    TooManySlopes { got: u32, max: u32 },

    #[error("max_failed_fraction must be in [0.0, 1.0]; got {got}")]
    MaxFailedFractionOutOfRange { got: f64 },

    #[error("test.alpha must be in (0.0, 1.0); got {got}")]
    AlphaOutOfRange { got: f64 },

    /// Reserved for the adapter / engine entry (invariant 16). `validate()`
    /// never returns this; the contract crate ships it so downstream code can
    /// reuse the error vocabulary.
    #[error("correlation matrix is not positive semi-definite")]
    CorrelationNotPsd,

    #[error("invalid PosthocSpec: {0}")]
    InvalidPosthoc(&'static str),

    #[error("invalid scenarios.json: {0}")]
    InvalidScenariosConfig(String),

    #[error("invalid plot theme: {0}")]
    InvalidPlotTheme(String),

    #[error("invalid config.json: {0}")]
    InvalidConfig(String),

    #[error("scenario perturbation field out of range: {field} got {got}")]
    ScenarioPerturbationOutOfRange { field: &'static str, got: f64 },

    #[error("LME scenario perturbation field out of range: {field} got {got}")]
    LmePerturbationOutOfRange { field: &'static str, got: f64 },

    #[error("SyntheticKind::Binary {{ p }} must be finite and in [0.0, 1.0]; got {got}")]
    BinaryProbabilityOutOfRange { got: f64 },

    #[error("ResampledBinary {{ proportion }} must be finite and in [0.0, 1.0]; got {got}")]
    ResampledBinaryProportionOutOfRange { got: f64 },

    #[error(
        "ClusterSpec.tau_squared must be finite and >= 0 on an intercept-only primary; got {got}"
    )]
    InterceptOnlyTauSquaredInvalid { got: f64 },
}
