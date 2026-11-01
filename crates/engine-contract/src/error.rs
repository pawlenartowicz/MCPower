//! `ContractError` — crate-level validation error vocabulary; downstream crates reuse its variants.

use thiserror::Error;

/// Validation failures from `SimulationContract::validate` and the config
/// validator; the `#[error]` strings carry the detail.
///
/// Two classes, told apart by [`ContractError::is_user_input`]: rules a
/// user's input can break (their messages are written in user terms, since a
/// port shows them as they are), and wiring faults only a port bug can cause.
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

    #[error(
        "dummy level_index {level_index} out of range (column ColumnId({column}) has {n_levels} levels)"
    )]
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

    #[error(
        "estimator 'glm' needs a binary or count outcome; use family 'logit', 'probit' or \
         'poisson', or estimator 'ols' for a continuous outcome"
    )]
    GlmRequiresBinaryOrCount,

    #[error("outcome.link == Some(Probit) requires outcome.kind == Binary")]
    ProbitRequiresBinary,

    #[error("nagq must be odd and in 1..=25; got {got}")]
    NagqOutOfRange { got: u8 },

    #[error(
        "nagq > 1 requires an eligible AGQ shape (Binary/Count GLMM, clustered, \
         single grouping factor, ≤ 3 REs per group); got nagq = {got}"
    )]
    NagqIneligibleShape { got: u8 },

    #[error("post-hoc comparisons are available for OLS models only")]
    PosthocRequiresOls,

    #[error("scenario.lme requires a clustered design (generation.cluster)")]
    LmeScenarioRequiresMle,

    /// `got` and `max` count the random effects of the grouping (intercept
    /// plus slopes); the message counts slopes.
    #[error(
        "a crossed or nested grouping takes at most {} random slopes; got {}",
        .max.saturating_sub(1),
        .got.saturating_sub(1)
    )]
    ExtraGroupingQTooLarge { got: u32, max: u32 },

    #[error(
        "SlopeTerm.column {id} is a factor or non-continuous column; random slopes require a continuous predictor"
    )]
    SlopeColumnNotContinuous { id: u32 },

    #[error(
        "SlopeTerm.column {id} has no Direct fixed-effect term; a random slope requires its variable as a fixed effect"
    )]
    SlopeColumnNotInDesign { id: u32 },

    #[error("a variable has two random slopes on one grouping; list each slope variable once")]
    SlopeColumnRepeated { id: u32 },

    #[error("a random-slope variance must be finite and >= 0; got {got}")]
    SlopeVarianceInvalid { got: f64 },

    #[error("a random-slope correlation must be finite and in [-1, 1]; got {got}")]
    SlopeCorrInvalid { got: f64 },

    #[error(
        "a random slope needs a random-intercept variance above 0 to correlate with; set ICC \
         (tau_squared for a Poisson model) above 0 for its grouping; got an intercept variance \
         of {got}"
    )]
    SlopeInterceptVarianceMissing { got: f64 },

    /// `slope` is the 0-based slope index; the message counts from 1.
    #[error(
        "random slope {} gives {got} corr_with value(s); it needs {expected}, one correlation \
         with each earlier random slope of the same grouping",
        .slope + 1
    )]
    SlopeCorrWithLenInvalid { slope: u32, expected: u32, got: u32 },

    #[error(
        "the random-slope correlations are not a valid correlation matrix (not positive \
         semidefinite); move the slope-intercept correlations, or any correlations between \
         slopes, toward 0"
    )]
    SlopeCovarianceNotPSD,

    #[error(
        "a crossed grouping needs the primary grouping sized by n_clusters; a primary sized by \
         cluster_size is not supported with crossed groupings"
    )]
    CrossedRequiresFixedClusters,

    #[error(
        "only one nested grouping such as '(1|a/b)' is supported per model; this model has more \
         than one"
    )]
    MultipleNestedUnsupported,

    // No '%' in this message, by house style for engine messages. The R
    // bridge escapes one anyway (`r_error` in engine-r).
    #[error(
        "a nested grouping needs the primary cluster_size to be a multiple of n_per_parent; got \
         cluster_size {cluster_size} and n_per_parent {n_per_parent}"
    )]
    NestedSizeIndivisible {
        cluster_size: u32,
        n_per_parent: u32,
    },

    #[error(
        "a crossed or nested grouping needs at least 2 levels (n_clusters or n_per_parent); got \
         {got}"
    )]
    GroupingCountTooSmall { got: u32 },

    #[error("extra grouping tau_squared must be finite and >= 0; got {got}")]
    GroupingTauSquaredInvalid { got: f64 },

    #[error(
        "at most {max} crossed or nested groupings besides the primary grouping are supported; \
         got {got}"
    )]
    TooManyExtraGroupings { got: u32, max: u32 },

    /// `got` and `max` count the primary random effects (intercept plus
    /// slopes); the message counts slopes.
    #[error(
        "a grouping takes at most {} random slopes; got {}",
        .max.saturating_sub(1),
        .got.saturating_sub(1)
    )]
    TooManySlopes { got: u32, max: u32 },

    #[error("max_failed_fraction must be in [0.0, 1.0]; got {got}")]
    MaxFailedFractionOutOfRange { got: f64 },

    #[error("alpha must be between 0 and 1 (exclusive); got {got}")]
    AlphaOutOfRange { got: f64 },

    /// Reserved for the adapter / engine entry (invariant 16). `validate()`
    /// never returns this; the contract crate ships it so downstream code can
    /// reuse the error vocabulary.
    #[error("correlation matrix is not positive semi-definite")]
    CorrelationNotPsd,

    #[error("invalid PosthocSpec: {0}")]
    InvalidPosthoc(&'static str),

    #[error("invalid config.json: {0}")]
    InvalidConfig(String),

    #[error(
        "scenario setting '{field}' must be {}; got {got}",
        scenario_field_range(field)
    )]
    ScenarioPerturbationOutOfRange { field: &'static str, got: f64 },

    #[error("scenario setting '{field}' must be finite and >= 0; got {got}")]
    LmePerturbationOutOfRange { field: &'static str, got: f64 },

    #[error("SyntheticKind::Binary {{ p }} must be finite and in [0.0, 1.0]; got {got}")]
    BinaryProbabilityOutOfRange { got: f64 },

    #[error("ResampledBinary {{ proportion }} must be finite and in [0.0, 1.0]; got {got}")]
    ResampledBinaryProportionOutOfRange { got: f64 },

    #[error(
        "ClusterSpec.tau_squared must be finite and >= 0 on an intercept-only primary; got {got}"
    )]
    InterceptOnlyTauSquaredInvalid { got: f64 },

    /// `df_method = between_within` on crossed or nested groupings; `lmm`
    /// picks the alternatives the message names.
    #[error(
        "df_method 'between_within' needs a single grouping factor: with crossed or nested \
         groupings the residual df have no between-within split. {}",
        between_within_alternatives(*.lmm)
    )]
    BetweenWithinNeedsSingleGrouping { lmm: bool },

    #[error(
        "df_method 'satterthwaite' needs a continuous outcome: a binary or count mixed model \
         has no Satterthwaite df. Use 'auto' (the default), which tests it at between-within \
         df with a single grouping factor and against Wald z otherwise, or 'asymptotic' \
         (Wald z)"
    )]
    SatterthwaiteNeedsLmm,

    /// A `FixedClusters` primary with fewer than 2 clusters. The text starts
    /// with the ports' own `set_cluster` message; the R upload path reaches it
    /// with one distinct group in the data.
    #[error(
        "n_clusters must be an integer >= 2, got {got} (a mixed model needs at least 2 \
         clusters; uploaded data needs at least 2 distinct groups in the grouping column)"
    )]
    PrimaryClusterCountTooSmall { got: u32 },
}

impl ContractError {
    /// True for a rule a user's input can break, false for a wiring fault
    /// only a port bug can cause. The spec builder reports the first class as
    /// a user error with this message as it is, and the second as an internal
    /// error. Every variant is classified here; a variant that a host check
    /// always catches first stays internal. The notes say where each
    /// user-input rule is reached from.
    pub fn is_user_input(&self) -> bool {
        match self {
            // Python and R: one intercept correlation over several slopes, or
            // R corr_with values that do not fit together.
            Self::SlopeCovarianceNotPSD
            // R: corr_with missing or of the wrong length.
            | Self::SlopeCorrWithLenInvalid { .. }
            // Python slope_intercept_corr outside [-1, 1]; R corr_with.
            | Self::SlopeCorrInvalid { .. }
            // Python slope_variance < 0.
            | Self::SlopeVarianceInvalid { .. }
            // Python and R: ICC = 0 on a grouping with a random slope.
            | Self::SlopeInterceptVarianceMissing { .. }
            // Python and R: one variable listed twice in a slope term and in
            // set_cluster.
            | Self::SlopeColumnRepeated { .. }
            // Python and R: estimator = "glm" on a continuous outcome.
            | Self::GlmRequiresBinaryOrCount
            // Python and R: more slopes than the solver carries.
            | Self::TooManySlopes { .. }
            | Self::ExtraGroupingQTooLarge { .. }
            | Self::TooManyExtraGroupings { .. }
            // Python and R: a crossed grouping beside a primary sized by
            // cluster_size; a nested grouping whose n_per_parent does not
            // divide the primary cluster_size.
            | Self::CrossedRequiresFixedClusters
            | Self::NestedSizeIndivisible { .. }
            // A direct contract with two nested terms of different parents;
            // the formula parser refuses `(1|a/b) + (1|c/d)` first.
            | Self::MultipleNestedUnsupported
            // The app sends a nested n_per_parent without a host check
            // (Python and R check counts first).
            | Self::GroupingCountTooSmall { .. }
            // R get_effects_from_data with one distinct group in the data
            // (the ports' set_cluster checks n_clusters first); a direct
            // contract.
            | Self::PrimaryClusterCountTooSmall { .. }
            // Python and R set_alpha; the app checks only that alpha is finite.
            | Self::AlphaOutOfRange { .. }
            // Python and R set_scenario_configs.
            | Self::ScenarioPerturbationOutOfRange { .. }
            | Self::LmePerturbationOutOfRange { .. }
            // A post-hoc request on a mixed or GLM model; the spec builder
            // refuses it first with a message naming the request.
            | Self::PosthocRequiresOls
            // A named df_method the design does not admit; the spec builder
            // refuses it first (`SpecError::DfMethodNotDefined`).
            | Self::BetweenWithinNeedsSingleGrouping { .. }
            | Self::SatterthwaiteNeedsLmm => true,

            // Wiring faults: shape and index rules every port satisfies by
            // construction, or rules a host or the spec builder checks first
            // (agq, max_failed_fraction, binary proportions, the predictor
            // correlation matrix, ICC, slope columns).
            Self::CoefficientLengthMismatch { .. }
            | Self::TestTargetTermOutOfRange { .. }
            | Self::InvalidTestSpec(_)
            | Self::ColumnIdOutOfRange { .. }
            | Self::CorrelationDimensionMismatch { .. }
            | Self::CorrelationOnNonContinuous { .. }
            | Self::InvalidFactorProportions(_)
            | Self::DummyLevelOutOfRange { .. }
            | Self::DirectOnFactor { .. }
            | Self::InteractionTooFewComponents { .. }
            | Self::InteractionBadComponent
            | Self::HeteroskedasticityOnFactor { .. }
            | Self::UploadedFrameMissing { .. }
            | Self::MleWithoutCluster
            | Self::ProbitRequiresBinary
            | Self::NagqOutOfRange { .. }
            | Self::NagqIneligibleShape { .. }
            | Self::LmeScenarioRequiresMle
            | Self::SlopeColumnNotContinuous { .. }
            | Self::SlopeColumnNotInDesign { .. }
            | Self::GroupingTauSquaredInvalid { .. }
            | Self::MaxFailedFractionOutOfRange { .. }
            | Self::CorrelationNotPsd
            | Self::InvalidPosthoc(_)
            | Self::InvalidConfig(_)
            | Self::BinaryProbabilityOutOfRange { .. }
            | Self::ResampledBinaryProportionOutOfRange { .. }
            | Self::InterceptOnlyTauSquaredInvalid { .. } => false,
        }
    }
}

/// The valid range of a `ScenarioPerturbations` field, for its message.
fn scenario_field_range(field: &str) -> &'static str {
    if field == "distribution_change_prob" {
        "in [0, 1]"
    } else {
        "finite and >= 0"
    }
}

fn between_within_alternatives(lmm: bool) -> &'static str {
    if lmm {
        "Use 'auto' (the default) or 'satterthwaite', which test this model at Satterthwaite \
         df, or 'asymptotic' (Wald z)"
    } else {
        "Use 'auto' (the default) or 'asymptotic', which test this binary or count model \
         against Wald z"
    }
}

#[cfg(test)]
mod tests {
    //! The exhaustive match in `is_user_input` is the compile-time half: a
    //! new variant does not build until it is classified. These tests pin the
    //! class of each user-input variant and that its message, which ports
    //! show as it is, names no contract internals.
    use super::ContractError;

    /// Contract-internal names a user-facing message must not carry.
    const INTERNAL_NAMES: [&str; 9] = [
        "SlopeTerm",
        "ColumnId",
        "ClusterSpec",
        "ClusterSizing",
        "GroupingRelation",
        "NestedWithin",
        "q_p",
        "τ",
        "test.alpha",
    ];

    fn assert_user_input(e: ContractError) {
        assert!(e.is_user_input(), "{e:?} must be classified as user input");
        let msg = e.to_string();
        for name in INTERNAL_NAMES {
            assert!(!msg.contains(name), "{e:?} message names '{name}': {msg}");
        }
    }

    macro_rules! user_input_tests {
        ($($name:ident: $variant:expr;)*) => {
            $(
                #[test]
                fn $name() {
                    assert_user_input($variant);
                }
            )*
        };
    }

    user_input_tests! {
        slope_covariance_not_psd: ContractError::SlopeCovarianceNotPSD;
        slope_column_repeated: ContractError::SlopeColumnRepeated { id: 0 };
        glm_requires_binary_or_count: ContractError::GlmRequiresBinaryOrCount;
        slope_corr_with_len_invalid:
            ContractError::SlopeCorrWithLenInvalid { slope: 1, expected: 1, got: 0 };
        slope_corr_invalid: ContractError::SlopeCorrInvalid { got: 1.5 };
        slope_variance_invalid: ContractError::SlopeVarianceInvalid { got: -1.0 };
        slope_intercept_variance_missing:
            ContractError::SlopeInterceptVarianceMissing { got: 0.0 };
        too_many_slopes: ContractError::TooManySlopes { got: 9, max: 8 };
        extra_grouping_q_too_large: ContractError::ExtraGroupingQTooLarge { got: 5, max: 4 };
        too_many_extra_groupings: ContractError::TooManyExtraGroupings { got: 7, max: 6 };
        crossed_requires_fixed_clusters: ContractError::CrossedRequiresFixedClusters;
        multiple_nested_unsupported: ContractError::MultipleNestedUnsupported;
        nested_size_indivisible:
            ContractError::NestedSizeIndivisible { cluster_size: 25, n_per_parent: 4 };
        grouping_count_too_small: ContractError::GroupingCountTooSmall { got: 1 };
        primary_cluster_count_too_small: ContractError::PrimaryClusterCountTooSmall { got: 1 };
        alpha_out_of_range: ContractError::AlphaOutOfRange { got: 1.5 };
        scenario_perturbation_out_of_range: ContractError::ScenarioPerturbationOutOfRange {
            field: "heterogeneity",
            got: -1.0,
        };
        lme_perturbation_out_of_range: ContractError::LmePerturbationOutOfRange {
            field: "icc_noise_sd",
            got: -1.0,
        };
        posthoc_requires_ols: ContractError::PosthocRequiresOls;
        between_within_needs_single_grouping:
            ContractError::BetweenWithinNeedsSingleGrouping { lmm: true };
        satterthwaite_needs_lmm: ContractError::SatterthwaiteNeedsLmm;
    }

    // Slope counts in the messages exclude the random intercept.
    #[test]
    fn slope_limits_count_slopes_not_random_effects() {
        assert_eq!(
            ContractError::TooManySlopes { got: 9, max: 8 }.to_string(),
            "a grouping takes at most 7 random slopes; got 8"
        );
        assert_eq!(
            ContractError::ExtraGroupingQTooLarge { got: 5, max: 4 }.to_string(),
            "a crossed or nested grouping takes at most 3 random slopes; got 4"
        );
    }

    // House style: engine messages carry no '%'. The R bridge escapes one
    // anyway (`r_error` in engine-r), so this guards style, not a crash.
    #[test]
    fn nested_size_message_has_no_percent_sign() {
        let msg = ContractError::NestedSizeIndivisible {
            cluster_size: 25,
            n_per_parent: 4,
        }
        .to_string();
        assert!(!msg.contains('%'), "{msg}");
    }

    #[test]
    fn wiring_fault_is_not_user_input() {
        assert!(
            !ContractError::CoefficientLengthMismatch {
                coeffs: 2,
                terms: 3
            }
            .is_user_input()
        );
    }
}
