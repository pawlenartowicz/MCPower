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
    #[error(
        "cluster-level variable '{name}' is bound to uploaded data; the engine resamples it from the upload and cannot draw it once per cluster. Use a synthesized predictor instead"
    )]
    ClusterLevelUploaded { name: String },
    #[error(
        "test_formula references '{name}', which is not in the model formula; add '{name}' to the model formula and give it an effect (use 0 if it should not influence the generated data)"
    )]
    TestFormulaPredictorMissing { name: String },
    #[error(
        "test_formula leaves out '{name}', which has a random slope in the model; keep '{name}' in test_formula (the test model keeps the model's random effects, and a random slope needs its variable as a fixed effect)"
    )]
    TestFormulaDropsSlopeVariable { name: String },
    /// `term` is the first random-effect term of the test_formula, as written.
    #[error(
        "test_formula sets the fixed effects of the test model only; the test model keeps the model's random effects. Remove '{term}' from test_formula"
    )]
    TestFormulaRandomEffect { term: String },
    /// A post-hoc request (`'all-contrasts'`, alias `'all-posthoc'`) on a model that is not OLS.
    /// `model` names the model kind; `example` lists the factor's dummies.
    #[error(
        "post-hoc comparisons ('all-contrasts') are available for OLS models only; this model is {model}. Test the factor's dummies instead, e.g. target_test = '{example}'"
    )]
    PosthocRequiresOls { model: String, example: String },
    /// A named contrast (`'f[2] vs f[3]'`) on a model that is not OLS.
    #[error(
        "contrasts ('{contrast}') are available for OLS models only; this model is {model}. Test the factor's dummies instead, e.g. target_test = '{example}'"
    )]
    ContrastRequiresOls {
        contrast: String,
        model: String,
        example: String,
    },
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
    #[error(
        "a random slope requires a random intercept in this engine version; intercept suppression ('0 +' / '-1 +') in a random-effects term is not supported — write '(x | g)' or '(1 + x | g)'"
    )]
    RandomInterceptSuppressionUnsupported,
    #[error(
        "uncorrelated random effects ('||') are not supported; write '(1 + x | g)' and set the slope-intercept correlation to 0"
    )]
    UncorrelatedRandomEffectsUnsupported,
    /// The colon spelling of a nested intercept, `(1|a:b)`.
    #[error(
        "'(1|{parent}:{child})' is not supported; write '(1|{parent}/{child})', which gives {parent} and {child} within {parent} a random intercept each"
    )]
    NestedColonFormUnsupported { parent: String, child: String },
    /// A random slope inside a nested term, `(1 + x|a/b)` or `(x|a/b)`.
    /// `term` is the term in canonical spacing.
    #[error(
        "random slopes in a nested term ('{term}') are not supported in the formula; write '(1|{parent}/{child})' and, in Python or R, declare the slope with set_cluster('{parent}:{child}', random_slopes = ...)"
    )]
    NestedSlopeUnsupported {
        term: String,
        parent: String,
        child: String,
    },
    /// More than one nested term, `(1|a/b) + (1|c/d)`; the engine carries one
    /// nested grouping.
    #[error(
        "only one nested term is supported per model; the formula has '{first}' and '{second}'"
    )]
    MultipleNestedTermsUnsupported { first: String, second: String },
    /// A random-effect term whose grouping name the parser does not take.
    #[error(
        "the grouping name '{name}' in '{term}' is not supported; a grouping name takes ASCII letters, digits and '_' and does not start with a digit"
    )]
    GroupingNameUnsupported { term: String, name: String },
    /// A parenthesised term with '|' that no random-effect pattern matches.
    #[error(
        "unsupported random-effect term '{term}'; write a random intercept '(1|g)', a random slope '(1 + x|g)' or a nested intercept '(1|a/b)'"
    )]
    RandomEffectTermUnsupported { term: String },
    /// The named `df_method` test is not defined on this design (the
    /// contract's own refusal, raised here so every port reports it as a user
    /// error).
    #[error("{0}")]
    DfMethodNotDefined(engine_contract::ContractError),
    /// A contract rule the user's input broke
    /// (`ContractError::is_user_input`); the message is the contract's own.
    #[error("{0}")]
    ContractRule(engine_contract::ContractError),
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
    #[error("binary variable '{name}': proportion must be between 0 and 1, got {value}")]
    BinaryProportionOutOfRange { name: String, value: f64 },
    #[error("uploaded factor '{name}': level code {value} is not an integer in 0..{n_levels}")]
    FactorCodeOutOfRange {
        name: String,
        value: f64,
        n_levels: usize,
    },
    #[error("correlation pair ({a}, {b}): value must be in [-1, 1], got {value}")]
    CorrelationOutOfRange { a: String, b: String, value: f64 },
    #[error("correlation refers to unknown variable: '{name}'")]
    CorrelationUnknownVar { name: String },
    #[error("correlation matrix is not positive semi-definite")]
    CorrelationNotPsd,
    #[error(
        "correlation is only supported between continuous variables; '{name}' is binary (binary and factor variables are generated from their marginals)"
    )]
    CorrelationNonContinuous { name: String },
    #[error("target '{name}' is not an effect in this model")]
    UnknownTarget { name: String },
    #[error(
        "unknown residual distribution: '{name}' (expected one of: normal, right_skewed, \
         left_skewed, high_kurtosis, uniform)"
    )]
    UnknownResidualDist { name: String },
    #[error(
        "scenario '{name}': 'binary' is not supported in new_distributions — a swapped binary column would be a degenerate constant; allowed: normal, right_skewed, left_skewed, high_kurtosis, uniform"
    )]
    ScenarioBinarySwapUnsupported { name: String },
    #[error("effect count mismatch: model has {expected} effects, got {got}")]
    EffectCountMismatch { expected: usize, got: usize },
    /// A contract validation failure that only a port or builder bug can
    /// cause; user-input rules go to [`SpecError::ContractRule`].
    #[error("internal contract validation failed: {0}")]
    InternalContractValidate(String),
    #[error(
        "exactly one cluster spec is required for the Mle estimator; more than one cluster is unsupported"
    )]
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
