//! Serde-deserializable input types for the spec-builder (LinearSpec, PredictorSpec, ScenarioInput, etc.); no validation logic here.
//!
//! `ScenarioInput.new_distributions` and `residual_dists` arrive as integer codes pre-encoded by the host; no name→code translation here.
//! `LinearSpec.predictors` order drives factor-expansion order — see `variables.rs` (column-ordering invariant home).

use serde::{Deserialize, Serialize};

/// Top-level input for a Linear (OLS) power analysis.
///
/// Order discipline:
///   * `predictors` is the order in which factor expansion sees variables
///     and therefore the order in which dummies appear in the output spec.
///   * `effects` map effect *names* (e.g. `"x1"`, `"x1:x2"`, `"group[3]"`)
///     to coefficients. The builder validates that every parsed effect has
///     an entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearSpec {
    /// R-style formula: `"y = x1 + x2 + x1:x2"` or `"y ~ x1 + x2"`.
    pub formula: String,
    /// Predictor declarations; their order drives factor-expansion (dummy) order.
    pub predictors: Vec<PredictorSpec>,
    /// Effect-name → coefficient assignments; every parsed effect needs an entry.
    pub effects: Vec<EffectAssignment>,
    /// User-supplied correlation pairs; missing pairs default to 0.0.
    /// Pairs reference *non-factor* predictor names only.
    pub correlations: Vec<CorrelationPair>,
    /// Per-test significance level (e.g. 0.05).
    pub alpha: f64,
    /// Multiple-comparison correction applied across targets.
    pub correction: Correction,
    /// Effect names to report power for. Use `"overall"` to expand to all
    /// fixed effects after factor expansion.
    pub targets: Vec<String>,
    pub heteroskedasticity: HeteroskedasticityInput,
    pub residual: ResidualSpec,
    /// Post-batch convergence threshold; 0.0..=1.0.
    pub max_failed_fraction: f64,
    /// Scenarios to run. Empty → builder emits a single "optimistic" baseline.
    pub scenarios: Vec<ScenarioInput>,
    /// Optional misspecified test model. When `None`, `design_test` mirrors
    /// `design_generation` (the standard case). When `Some(formula)`, the
    /// engine fits only the listed terms while β generation still uses the
    /// full generation formula — useful for studying omitted-variable bias.
    /// The test formula must reference predictors already declared in
    /// `predictors`; interactions are not supported.
    #[serde(default)]
    pub test_formula: Option<String>,
    /// v1-parity omnibus: when `true`, the builder emits an additional
    /// `TestTarget::Joint { terms: <all non-intercept positions of design_test> }`
    /// that the contract adapter routes to `SimulationSpec.report_overall`
    /// (OLS F-test / Logit LRT vs intercept-only). LME doesn't yet ship a
    /// joint Wald-χ² omnibus channel, so this is a no-op there.
    #[serde(default)]
    pub report_overall: bool,
    /// Pairwise contrast pairs, each `(positive_effect_name, negative_effect_name)`.
    ///
    /// Example: `("treatment[B]", "treatment[C]")` emits a
    /// `TestTarget::Contrast { positive, negative }` in the contract. If
    /// either name is the factor's reference level (i.e. absent from the
    /// expanded effect list), the builder collapses the pair to
    /// `TestTarget::Marginal` on the non-reference term instead.
    ///
    /// Layered on top of whatever `targets` selects — both are emitted.
    #[serde(default)]
    pub contrast_pairs: Vec<(String, String)>,
    /// Post-hoc pairwise requests, one per factor. Empty = no post-hoc.
    #[serde(default)]
    pub posthoc_requests: Vec<PosthocRequest>,
    /// Optional uploaded predictor data. When present, matched predictor columns
    /// are drawn from the frame rather than synthesised.
    #[serde(default)]
    pub upload: Option<UploadInput>,
    /// Predictor names generated constant within each cluster (the cluster-level
    /// / design-effect covariates). Resolved against the predictor table to
    /// `GenerationSpec.cluster_level_columns`. Empty ⇒ every predictor varies
    /// per row. Names must be real predictors (validated host-side).
    #[serde(default)]
    pub cluster_level_vars: Vec<String>,
    /// Standard-error flavour for the clustered-binary/count GLMM estimator.
    /// `Rx` (1.1.0 default, fastmode) is the Schur speed knob; `Hessian` uses
    /// the per-fit FD-Hessian SE (lme4 `use.hessian = TRUE`, the accurate/AGQ
    /// opt-in). No-op for OLS/LMM and unclustered GLM.
    #[serde(default)]
    pub wald_se: engine_contract::WaldSe,
    /// Reference distribution for mixed-model fixed-effect Wald tests. No-op
    /// for `Ols` and unclustered `Glm`.
    #[serde(default)]
    pub df_method: engine_contract::DfMethod,
    /// AGQ node count for the GLMM likelihood (1 = Laplace, the default).
    /// Host-side eligibility (Binary/Count GLMM, single grouping, ≤ 3 REs, odd
    /// k ≤ 25) is checked before this reaches the engine; the contract backstop
    /// (invariant 25) re-enforces it.
    #[serde(default = "default_nagq_input")]
    pub nagq: u8,
}

/// Serde default for `LinearSpec.nagq` — Laplace.
fn default_nagq_input() -> u8 {
    1
}

/// One all-pairwise post-hoc request for a single factor.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosthocRequest {
    /// Factor predictor name to run all-pairwise post-hoc on.
    pub factor: String,
    /// Post-hoc alpha; `None` → model alpha. (No user knob yet — always None.)
    #[serde(default)]
    pub posthoc_alpha: Option<f64>,
}

/// One declared predictor: its `name` and distribution/factor `kind`.
#[derive(Debug, Clone, Deserialize)]
pub struct PredictorSpec {
    /// Predictor name as written in the formula.
    pub name: String,
    /// Distribution or factor specification (serde-flattened `kind` tag).
    #[serde(flatten)]
    pub kind: VarKind,
    /// `true` = the user explicitly chose this distribution (incl. explicit
    /// normal) — scenario distribution swaps leave the column alone. Only
    /// meaningful for continuous synthetic kinds; ignored for factors.
    #[serde(default)]
    pub pinned: bool,
}

/// A predictor's data-generating kind: a synthetic distribution or a `Factor`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VarKind {
    Normal,
    Binary {
        proportion: f64,
    },
    RightSkewed,
    LeftSkewed,
    HighKurtosis,
    Uniform,
    Factor {
        levels: Vec<String>,
        proportions: Vec<f64>,
        /// Reference level (omitted from dummy expansion). Must appear in `levels`.
        reference: String,
        /// Per-factor proportion-sampling override: `None`/absent inherits the
        /// scenario `sampled_factor_proportions`; `true` sampled, `false` exact.
        #[serde(default)]
        sampled_proportions: Option<bool>,
    },
}

/// One effect-name → coefficient-size assignment.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectAssignment {
    pub name: String,
    pub size: f64,
}

/// A pairwise correlation `value` between non-factor predictors `a` and `b`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelationPair {
    pub a: String,
    pub b: String,
    pub value: f64,
}

/// The multiple-comparison correction method. Re-exported from
/// `engine_contract` so the LinearSpec wire (`correction: "benjamini_hochberg"`)
/// and the contract share a single enum — both use
/// `#[serde(rename_all = "snake_case")]` with identical variants, so the wire
/// strings (`"none"`, `"bonferroni"`, `"holm"`, `"benjamini_hochberg"`) are
/// unchanged.
pub use engine_contract::CorrectionMethod as Correction;

/// Mirrors Python's `_heteroskedasticity` dict shape (model.py). `driver_var_index`
/// references a non-factor predictor by position in the non-factor block
/// (`SimulationSpec.var_types` / `var_params`), or is `None` for the linear
/// predictor Xβ. The variance ratio λ is scenario-only — there is no model
/// pin. The `None` default is scenario-driven, lp-based het.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeteroskedasticityInput {
    #[serde(default)]
    pub driver_var_index: Option<u32>,
}

/// Residual error-term distribution, and whether the user pinned it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualSpec {
    /// Canonical name: "normal", "right_skewed", "left_skewed",
    /// "high_kurtosis", or "uniform" (the `RESIDUAL_CODES` table). No df —
    /// `high_kurtosis` takes its df from the active scenario's `residual_df`.
    pub distribution: String,
    /// `true` = the user explicitly chose the distribution (incl. explicit
    /// "normal") — scenario residual swaps leave it alone.
    #[serde(default)]
    pub pinned: bool,
}

impl Default for ResidualSpec {
    fn default() -> Self {
        Self {
            distribution: "normal".into(),
            pinned: false,
        }
    }
}

/// Faithfulness ladder over the uploaded predictor matrix X.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadMode {
    None,
    Partial,
    Strict,
}

/// Host-detected column type. Detection is host-side; this carries the verdict + level names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadColumnType {
    Continuous,
    Binary,
    Factor,
}

/// One uploaded column. `values` are raw (un-standardized) numerics; for a
/// factor, `values` are level codes 0..k-1 and `labels[code]` is the display name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadColumn {
    pub name: String,
    pub col_type: UploadColumnType,
    pub values: Vec<f64>,
    #[serde(default)]
    pub labels: Vec<String>,
}

/// An uploaded predictor frame: faithfulness `mode`, row count, and typed columns.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadInput {
    pub mode: UploadMode,
    pub n_rows: u32,
    pub columns: Vec<UploadColumn>,
}

/// Mirrors Python's `build_scenario_dict` output (spec/spec_builder.py), which has
/// already encoded the two distribution lists to integer codes via the engine's
/// `DIST_CODES` (for `new_distributions`) and `RESIDUAL_CODES`
/// (for `residual_dists`) tables. The builder copies these integers straight into
/// `ScenarioPerturbations` — no name → code translation happens in Rust.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioInput {
    pub name: String,
    /// Scenario contribution to the β-jitter SD (additive with the model baseline); 0.0 = off.
    pub heterogeneity: f64,
    /// Residual variance ratio λ; 1.0 = homoskedastic.
    pub heteroskedasticity_ratio: f64,
    /// SD of noise added to the predictor correlations; 0.0 = off.
    pub correlation_noise_sd: f64,
    /// Probability a predictor's distribution is swapped to a `new_distributions` draw; 0.0 = off.
    pub distribution_change_prob: f64,
    /// Candidate predictor-distribution integer codes for the swap (host-encoded).
    pub new_distributions: Vec<i32>,
    /// Probability the residual distribution is swapped to a `residual_dists` draw; 0.0 = off.
    pub residual_change_prob: f64,
    /// Candidate residual-distribution integer codes (host-encoded).
    pub residual_dists: Vec<i32>,
    /// Degrees of freedom for heavy-tailed (`high_kurtosis`) residual draws.
    pub residual_df: f64,
    /// Factor-proportion sampling (scenario knob); `false` (exact) when absent.
    #[serde(default)]
    pub sampled_factor_proportions: bool,
    /// Scenario assumption: truth-seeded fitter start vs. cold (blind) start;
    /// `false` (cold start) when absent.
    #[serde(default)]
    pub truth_start: bool,
    /// Random-effect distribution code (RE_DIST_CODES space): 0=normal,
    /// 1=heavy_tailed (t kernel). The RE knob keeps its own
    /// normal/heavy_tailed vocabulary — it is NOT the residual-pool space.
    /// Default 0 (normal/Gaussian REs).
    #[serde(default)]
    pub random_effect_dist: i32,
    /// Degrees of freedom for the RE distribution when `random_effect_dist` == 1 (t). 0.0 = unset.
    #[serde(default)]
    pub random_effect_df: f64,
    /// SD of per-block ICC jitter. 0.0 = off (no ICC noise).
    #[serde(default)]
    pub icc_noise_sd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_input_sampled_factor_proportions_defaults_false_when_absent() {
        // Hosts that predate the knob omit the key; serde must fill `false` (exact).
        let json = r#"{
            "name": "optimistic",
            "heterogeneity": 0.0,
            "heteroskedasticity_ratio": 1.0,
            "correlation_noise_sd": 0.0,
            "distribution_change_prob": 0.0,
            "new_distributions": [],
            "residual_change_prob": 0.0,
            "residual_dists": [],
            "residual_df": 0.0
        }"#;
        let s: ScenarioInput = serde_json::from_str(json).unwrap();
        assert!(!s.sampled_factor_proportions);
    }

    #[test]
    fn linear_spec_deserializes_posthoc_requests() {
        // Minimal host-style LinearSpec JSON plus posthoc_requests; an absent
        // `posthoc_alpha` deserializes to None.
        let json = r#"{
            "formula": "y = x1 + x2",
            "predictors": [
                {"name": "x1", "kind": "normal"},
                {"name": "x2", "kind": "binary", "proportion": 0.3}
            ],
            "effects": [
                {"name": "x1", "size": 0.5},
                {"name": "x2", "size": 0.3}
            ],
            "correlations": [],
            "alpha": 0.05,
            "correction": "bonferroni",
            "targets": ["overall"],
            "heteroskedasticity": {},
            "residual": {"distribution": "normal"},
            "max_failed_fraction": 0.1,
            "scenarios": [],
            "posthoc_requests": [{"factor": "dose_group"}]
        }"#;
        let spec: LinearSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.posthoc_requests.len(), 1);
        assert_eq!(spec.posthoc_requests[0].factor, "dose_group");
        assert_eq!(spec.posthoc_requests[0].posthoc_alpha, None);
    }

    #[test]
    fn unknown_keys_are_rejected_in_linear_spec_and_nested_structs() {
        let base = |extra_top: &str, residual: &str| {
            format!(
                r#"{{
                "formula": "y = x1",
                "predictors": [{{"name": "x1", "kind": "normal"}}],
                "effects": [{{"name": "x1", "size": 0.5}}],
                "correlations": [],
                "alpha": 0.05,
                "correction": "none",
                "targets": ["x1"],
                {extra_top}
                "heteroskedasticity": {{}},
                "residual": {residual},
                "max_failed_fraction": 0.1,
                "scenarios": []
            }}"#
            )
        };
        let ok = base("", r#"{"distribution": "normal"}"#);
        assert!(serde_json::from_str::<LinearSpec>(&ok).is_ok());

        let top = base(r#""heterogeneity": 0.0,"#, r#"{"distribution": "normal"}"#);
        let err = serde_json::from_str::<LinearSpec>(&top).unwrap_err();
        assert!(
            err.to_string().contains("unknown field `heterogeneity`"),
            "{err}"
        );

        let nested = base("", r#"{"distribution": "normal", "df": 0.0}"#);
        let err = serde_json::from_str::<LinearSpec>(&nested).unwrap_err();
        assert!(err.to_string().contains("unknown field `df`"), "{err}");
    }
}
