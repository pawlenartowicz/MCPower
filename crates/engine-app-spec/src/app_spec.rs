//! `AppSpec` tagged-enum state model (`Linear`/`Logit`/`Mixed`) mirrored by the app's TS `AppSpec`.

use engine_contract::CorrectionMethod;
use engine_spec_builder::input::{ScenarioInput, UploadColumn, UploadMode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "lowercase")]
pub enum AppSpec {
    Linear(LinearSpec),
    Logit(LogitSpec),
    Mixed(MixedSpec),
}

impl AppSpec {
    /// The configured target power, used for the find_sample_size threshold
    /// line and the find_power target rule in the emitted plot specs.
    pub fn target_power(&self) -> f64 {
        match self {
            AppSpec::Linear(s) => s.target_power,
            AppSpec::Logit(s) => s.target_power,
            AppSpec::Mixed(s) => s.target_power,
        }
    }

    /// Whether a multiple-testing correction is active. `true` when the
    /// correction is anything other than `None`; `false` when `None`.
    /// Used by plot helpers to select corrected vs uncorrected power arrays.
    pub fn is_corrected(&self) -> bool {
        let c = match self {
            AppSpec::Linear(s) => s.correction,
            AppSpec::Logit(s) => s.correction,
            AppSpec::Mixed(s) => s.correction,
        };
        c != engine_contract::CorrectionMethod::None
    }
}

/// GUI state for a logistic-regression power run. `baseline_probability` is converted
/// to a log-odds intercept by the assembler before passing to the contract builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitSpec {
    pub parsed_formula: ParsedFormula,
    pub var_types: Vec<VarType>,
    pub effects: Vec<EffectSize>,
    pub correlations: Option<CorrelationMatrix>,
    pub alpha: f64,
    pub target_power: f64,
    pub n_sims: u64,
    pub seed: u64,
    pub tests: TestSelection,
    pub correction: CorrectionMethod,
    /// Scenarios to fan out, mirroring `engine_spec_builder::ScenarioInput`
    /// (host-projected from the Scenarios toggle). Empty → one baseline contract
    /// (toggle off), identical to the pre-fan-out behavior.
    #[serde(default)]
    pub scenarios: Vec<ScenarioInput>,
    pub csv: Option<CsvData>,
    /// Baseline (intercept) probability for the logistic model. The assembler
    /// converts this to the corresponding log-odds intercept passed to
    /// `BuilderLogitSpec`. Required — no meaningful default exists.
    pub baseline_probability: f64,
    /// Optional misspecified test model (see `LinearSpec.test_formula`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_formula: Option<String>,
    /// Outcome-level generation knobs; `None` = builder defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_options: Option<OutcomeOptions>,
}

/// Cluster dimension knob: the user fixes exactly one; the other is derived
/// from the sample size at run time. Internally tagged so it serialises as the
/// TS `{ kind: 'n_clusters' | 'cluster_size', value: number }` discriminated union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClusterDim {
    NClusters { value: u32 },
    ClusterSize { value: u32 },
}

/// One extra grouping factor in UI-layer terms, mirroring
/// `engine_contract::GroupingSpec` (sans slopes — the UI does not expose slopes
/// on secondary groupings; the engine rejects them). `tau_squared` is the RE
/// variance entered directly (the host already converts ICC → τ² for the primary
/// grouping; secondaries take τ² straight from the UI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppGroupingSpec {
    pub tau_squared: f64,
    pub relation: AppGroupingRelation,
    /// Display name of the grouping factor (from its formula term). Carried
    /// for host-side uses (script generation, labelling); the contract's
    /// `GroupingSpec` is index-based and ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_name: Option<String>,
}

/// Mirrors `engine_contract::GroupingRelation` (externally tagged there; this
/// app-layer copy is internally tagged on `kind` so it serialises as the TS
/// `{ kind: 'crossed' | 'nested_within', … }` discriminated union the adapter emits).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AppGroupingRelation {
    Crossed { n_clusters: u32 },
    NestedWithin { n_per_parent: u32 },
}

/// One random slope on the primary grouping factor, in UI-layer terms.
/// `predictor_name` names a predictor in `parsed_formula.predictors`; the
/// assembler resolves it to a generation `ColumnId` by its position among the
/// NON-FACTOR predictors (mirrors the Python/R ports). `slope_variance` and
/// `slope_intercept_corr` pass straight into `SlopeTerm`; `corr_with` is always
/// empty (slope↔slope correlations are not exposed in this pass).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSlopeTerm {
    pub predictor_name: String,
    pub slope_variance: f64,
    pub slope_intercept_corr: f64,
}

/// Outcome distribution for a mixed model. Defaults to `Gaussian` (Continuous + Mle),
/// keeping all existing mixed specs deserialising unchanged. `Binary` activates
/// `OutcomeKind::Binary + EstimatorSpec::Glm`; `baseline_probability` becomes a
/// log-odds intercept and ICC is scaled by π²/3 to the latent log-odds variance.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MixedOutcome {
    #[default]
    Gaussian,
    Binary {
        baseline_probability: f64,
    },
}

/// Linear fields (mirrors `LinearSpec`) plus cluster configuration.
/// `icc` is converted to `tau_squared` in `assemble_mixed`; `cluster_name`
/// mirrors the formula's `(1|name)` term (read-only in the UI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedSpec {
    pub parsed_formula: ParsedFormula,
    pub var_types: Vec<VarType>,
    pub effects: Vec<EffectSize>,
    pub correlations: Option<CorrelationMatrix>,
    pub alpha: f64,
    pub target_power: f64,
    pub n_sims: u64,
    pub seed: u64,
    pub tests: TestSelection,
    pub correction: CorrectionMethod,
    /// Scenarios to fan out, mirroring `engine_spec_builder::ScenarioInput`
    /// (host-projected from the Scenarios toggle). Empty → one baseline contract
    /// (toggle off), identical to the pre-fan-out behavior.
    #[serde(default)]
    pub scenarios: Vec<ScenarioInput>,
    pub csv: Option<CsvData>,
    #[serde(default)]
    pub report_overall: bool,
    #[serde(default)]
    pub contrasts: Vec<(String, String)>,
    /// Optional misspecified test model (see `LinearSpec.test_formula`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_formula: Option<String>,
    pub cluster_name: String,
    pub icc: f64,
    pub cluster_dim: ClusterDim,
    /// Predictor names that are constant within each cluster (cluster-level
    /// covariates). Forwarded to `BuilderLinearSpec.cluster_level_vars`; empty = none.
    #[serde(default)]
    pub cluster_level_vars: Vec<String>,
    /// Extra grouping factors (crossed or nested), in formula order after the
    /// primary. Empty → single-grouping design (today's default).
    #[serde(default)]
    pub extra_groupings: Vec<AppGroupingSpec>,
    /// Random slopes on the primary grouping factor. Empty → intercept-only RE.
    #[serde(default)]
    pub slopes: Vec<AppSlopeTerm>,
    /// Outcome distribution. `Gaussian` (default) → Continuous + Mle; `Binary` →
    /// Binary + Glm with a logit intercept and latent-scale τ². `#[serde(default)]`
    /// keeps existing mixed specs deserialising as Gaussian — back-compat.
    #[serde(default)]
    pub outcome: MixedOutcome,
    /// Outcome-level generation knobs; `None` = builder defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_options: Option<OutcomeOptions>,
}

/// GUI state for an OLS power run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSpec {
    pub parsed_formula: ParsedFormula,
    pub var_types: Vec<VarType>,
    pub effects: Vec<EffectSize>,
    pub correlations: Option<CorrelationMatrix>,
    pub alpha: f64,
    pub target_power: f64,
    pub n_sims: u64,
    pub seed: u64,
    pub tests: TestSelection,
    pub correction: CorrectionMethod,
    /// Scenarios to fan out, mirroring `engine_spec_builder::ScenarioInput`
    /// (host-projected from the Scenarios toggle). Empty → one baseline contract
    /// (toggle off), identical to the pre-fan-out behavior.
    #[serde(default)]
    pub scenarios: Vec<ScenarioInput>,
    pub csv: Option<CsvData>,
    /// v1-parity omnibus: OLS F-test / Logit LRT vs intercept-only. The
    /// assembler forwards this to `BuilderLinearSpec.report_overall`, which
    /// the contract adapter routes to `SimulationSpec.report_overall`.
    #[serde(default)]
    pub report_overall: bool,
    /// Pairwise contrast pairs as `(positive_name, negative_name)` in
    /// effect-name notation (e.g. `"treatment[B]"`, `"treatment[C]"`).
    #[serde(default)]
    pub contrasts: Vec<(String, String)>,
    /// Optional misspecified test model: data is generated from the full
    /// formula but only these terms are fitted/tested. Omitted/empty → fit the
    /// full model. Forwarded to `BuilderLinearSpec.test_formula`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_formula: Option<String>,
    /// Outcome-level generation knobs; `None` = builder defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_options: Option<OutcomeOptions>,
}

/// Decomposed formula: outcome name, flat predictor list, and interaction term groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFormula {
    pub outcome: String,
    pub predictors: Vec<String>,
    #[serde(default)]
    pub interaction_terms: Vec<Vec<String>>,
}

/// Synthetic distribution for a numeric (continuous) predictor. Maps 1:1 onto
/// the parameter-free `VarKind` continuous variants in `engine-spec-builder`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericDistribution {
    #[default]
    Normal,
    RightSkewed,
    LeftSkewed,
    HighKurtosis,
    Uniform,
}

/// Per-predictor distribution declaration from the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum VarType {
    Numeric {
        name: String,
        /// Synthetic distribution. `#[serde(default)]` = Normal, so payloads
        /// from before the knob deserialise unchanged; the default also skips
        /// serialization so neutral wires stay byte-identical to pre-knob ones.
        #[serde(default, skip_serializing_if = "NumericDistribution::is_normal")]
        distribution: NumericDistribution,
        /// `true` = the user explicitly chose `distribution` (incl. explicit
        /// normal) — scenario distribution swaps leave the column alone.
        /// "Neutral" on this wire keys on unpinned-default, NOT on the value:
        /// a pinned explicit normal must be sent (pinned=true survives the
        /// distribution field's skip rule above).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pinned: bool,
    },
    Binary {
        name: String,
        binary_proportion: f64,
    },
    /// A factor predictor. `factor_reference` is the **0-based index** of the
    /// baseline level (dropped from dummy expansion) within the factor's level
    /// list. `factor_labels` are the user's display labels, parallel to
    /// `factor_proportions`; when empty the engine falls back to the legacy
    /// `1..=factor_n_levels` names. Labels are load-bearing, not cosmetic: the
    /// engine derives effect names (`name[label]`) from them, and the host's
    /// effect assignments use the same labels — they must agree.
    Factor {
        name: String,
        factor_n_levels: u32,
        factor_proportions: Vec<f64>,
        #[serde(default)]
        factor_reference: u32,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        factor_labels: Vec<String>,
        /// Per-factor proportion-sampling override: `None` inherits the
        /// scenario `sampled_factor_proportions`; `Some(true)` draws shares
        /// multinomially each run; `Some(false)` allocates exactly.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sampled_proportions: Option<bool>,
    },
}

impl NumericDistribution {
    /// serde `skip_serializing_if` helper — keeps neutral wires identical to
    /// pre-knob payloads.
    fn is_normal(&self) -> bool {
        *self == NumericDistribution::Normal
    }
}

/// Outcome-level *structural* knobs from the GUI's Model "More options"
/// dialog. Magnitudes (λ, heterogeneity) are scenario-only and have no wire
/// here. An absent struct (the wire default) reproduces the default
/// behaviour exactly: unpinned normal residual, lp-driven heteroskedasticity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeOptions {
    /// Residual distribution, canonical spec-builder vocabulary: `"normal"` |
    /// `"right_skewed"` | `"left_skewed"` | `"high_kurtosis"` | `"uniform"`.
    /// `None` = unpinned default (normal; scenarios may swap it). `Some` =
    /// the user explicitly chose a shape → pinned; an explicit `"normal"`
    /// must be sent (neutral keys on unpinned-default, not on the value). A
    /// pinned `high_kurtosis` takes its df from the active scenario.
    #[serde(default)]
    pub residual_distribution: Option<String>,
    /// Heteroskedasticity driver predictor name (must be a non-factor
    /// predictor). `None` = the linear predictor Xβ drives the variance.
    /// The variance ratio λ comes from the active scenario.
    #[serde(default)]
    pub heteroskedasticity_driver: Option<String>,
}

/// Named effect size: predictor label and its standardized coefficient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectSize {
    pub name: String,
    pub value: f64,
}

/// Predictor correlation matrix: `names[i]` labels row/column `i` of the square `values` grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub names: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

/// Which effects the GUI targets for power: all (omnibus), a named subset, or named contrasts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TestSelection {
    All,
    Effects { names: Vec<String> },
    Contrasts { names: Vec<String> },
}

/// Uploaded CSV/dataframe data attached to a spec. Mirrors the serde shape of
/// `engine_spec_builder::input::UploadInput` so the IPC/TS side can use the
/// same wire type without a mapping layer. `UploadColumn` and `UploadMode` are
/// re-exported from spec-builder; host detection determines `col_type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvData {
    pub mode: UploadMode,
    pub n_rows: u32,
    pub columns: Vec<UploadColumn>,
}

#[cfg(test)]
mod target_power_tests {
    use super::*;

    #[test]
    fn target_power_reads_each_variant() {
        // Build the smallest valid spec per variant via serde from JSON so the
        // test does not depend on the full struct field set.
        let linear: AppSpec = serde_json::from_str(
            r#"{"family":"linear","target_power":0.8,
                "parsed_formula":{"outcome":"y","predictors":["x"],"interaction_terms":[]},
                "var_types":[{"kind":"numeric","name":"x"}],
                "effects":[{"name":"x","value":0.3}],
                "correlations":null,"alpha":0.05,"n_sims":10,"seed":1,
                "tests":{"kind":"all"},"correction":"none",
                "csv":null,"report_overall":true,"contrasts":[]}"#,
        )
        .expect("linear spec deserializes");
        assert_eq!(linear.target_power(), 0.8);
    }
}
