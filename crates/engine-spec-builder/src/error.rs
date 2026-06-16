//! All validation errors that the spec-builder can emit, covering formula syntax, predictor config, and correlation constraints.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpecError {
    #[error("formula is empty")]
    EmptyFormula,
    #[error("formula syntax error at position {pos}: {msg}")]
    FormulaSyntax { pos: usize, msg: String },
    #[error("unknown predictor in formula: '{name}'")]
    UnknownPredictor { name: String },
    #[error("test_formula references '{name}', which is not in the model formula; add '{name}' to the model formula and give it an effect (use 0 if it should not influence the generated data)")]
    TestFormulaPredictorMissing { name: String },
    #[error(
        "predictor '{name}' is not a factor; post-hoc pairwise tests require a factor predictor"
    )]
    NotAFactorPredictor { name: String },
    #[error("term removal with '-' is not supported")]
    TermRemovalUnsupported,
    #[error("duplicate grouping variable: {name}")]
    DuplicateGroupingVar { name: String },
    #[error("empty slope term for group {group}")]
    EmptySlopeTerm { group: String },
    #[error("random slopes are unsupported")]
    RandomSlopesUnsupported,
    #[error("a random slope requires a random intercept in this engine version; intercept suppression ('0 +' / '-1 +') in a random-effects term is not supported — write '(x | g)' or '(1 + x | g)'")]
    RandomInterceptSuppressionUnsupported,
    #[error("effect '{name}': effect_size must be finite, got {value}")]
    NonFiniteEffect { name: String, value: f64 },
    #[error("factor '{name}': must have {min}..={max} levels, got {got}")]
    FactorLevelCount {
        name: String,
        got: usize,
        min: usize,
        max: usize,
    },
    #[error("factor '{name}': proportions must sum to 1.0, got {sum}")]
    FactorProportionSum { name: String, sum: f64 },
    #[error("factor '{name}': proportions length {got} != levels length {expected}")]
    FactorProportionLengthMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("factor '{name}': reference level '{reference}' is not among `levels`")]
    FactorReferenceMissing { name: String, reference: String },
    #[error("factor '{name}': all proportions must be > 0")]
    FactorProportionNonPositive { name: String },
    #[error("correlation pair ({a}, {b}): value must be in [-1, 1], got {value}")]
    CorrelationOutOfRange { a: String, b: String, value: f64 },
    #[error("correlation refers to unknown variable: '{name}'")]
    CorrelationUnknownVar { name: String },
    #[error("correlation matrix is not positive semi-definite")]
    CorrelationNotPsd,
    #[error("correlation is only supported between continuous variables; '{name}' is binary (binary and factor variables are generated from their marginals)")]
    CorrelationNonContinuous { name: String },
    #[error("target '{name}' is not an effect in this model")]
    UnknownTarget { name: String },
    #[error(
        "unknown residual distribution: '{name}' (expected one of: normal, right_skewed, \
         left_skewed, high_kurtosis, uniform)"
    )]
    UnknownResidualDist { name: String },
    #[error("scenario '{name}': 'binary' is not supported in new_distributions — a swapped binary column would be a degenerate constant; allowed: normal, right_skewed, left_skewed, high_kurtosis, uniform")]
    ScenarioBinarySwapUnsupported { name: String },
    #[error("scenario '{name}': residual_dists contains a df-consuming distribution (high_kurtosis / right_skewed / left_skewed) with residual_change_prob > 0, but residual_df is {got}; set residual_df >= 3")]
    ScenarioResidualDfTooLow { name: String, got: f64 },
    #[error("effect count mismatch: model has {expected} effects, got {got}")]
    EffectCountMismatch { expected: usize, got: usize },
    #[error("internal contract validation failed: {0}")]
    InternalContractValidate(String),
    #[error("exactly one cluster spec is required for the Mle estimator; more than one cluster is unsupported")]
    ClusterFamilyMismatch,
    #[error("malformed assignment input: {input:?}")]
    MalformedAssignment { input: String },
    #[error("unknown assignment name: {name:?}")]
    UnknownAssignmentName { name: String },
    #[error("unknown variable type: {value:?}")]
    UnknownVariableType { value: String },
    #[error("contrast pair references unknown effect name: '{name}'")]
    UnknownContrastName { name: String },
    #[error("recovery design: modeled predictor '{name}' has no matching uploaded column")]
    RecoveryColumnMissing { name: String },
    #[error("recovery design: column '{name}' has {got} values, expected {expected}")]
    RecoveryColumnLength {
        name: String,
        expected: usize,
        got: usize,
    },
    /// A malformed `name=value` variable-type assignment. `message` carries the
    /// per-value diagnostic verbatim (e.g. "Factor cannot have more than 20
    /// levels"); the `{name}: {message}` shape matches the host wording.
    #[error("{name}: {message}")]
    InvalidVariableTypeValue { name: String, message: String },
}
