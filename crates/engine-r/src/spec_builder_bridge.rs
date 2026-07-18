//! extendr bindings for spec building and formula/assignment parsing; mirrors engine-py's spec_builder_bridge — change together.
use extendr_api::error::Result;
use extendr_api::prelude::*;
use serde::Serialize;

/// Build the whole `Vec<SimulationContract>` as a single msgpack blob,
/// parametrising outcome_kind + estimator + intercept + clusters at the call
/// site.  Mirrors `engine-py/src/spec_builder_bridge.rs` `build_contract_from_spec`.
///
/// Returns a named List with elements:
///   * `names`     — character vector of scenario names
///   * `contracts` — raw vector (msgpack-encoded `Vec<SimulationContract>`)
///   * `skeleton`  — JSON string of the index-only `EffectSkeleton` (β-column
///                   aligned), so the R frontend names results from it + its own
///                   label store instead of re-deriving the factor layout.
///
/// `outcome_kind` is one of `"continuous"` | `"binary"` | `"count"`.
/// `link` is `"canonical"` (or `""`) for the canonical link, or `"probit"` to
/// override a binary outcome to the probit link.
/// `estimator`    is one of `"ols"` | `"glm"` | `"mle"`.
#[extendr]
pub fn build_contract_from_spec(
    json: &str,
    outcome_kind: &str,
    link: &str,
    estimator: &str,
    intercept: f64,
    clusters_json: &str,
) -> Result<List> {
    let spec: engine_spec_builder::LinearSpec = serde_json::from_str(json)
        .map_err(|e| Error::Other(format!("malformed LinearSpec JSON: {e}")))?;
    let outcome = match outcome_kind {
        "continuous" => engine_contract::OutcomeKind::Continuous,
        "binary" => engine_contract::OutcomeKind::Binary,
        "count" => engine_contract::OutcomeKind::Count,
        other => return Err(Error::Other(format!("unknown outcome_kind {other:?}"))),
    };
    let link = match link {
        "" | "canonical" => None,
        "probit" => Some(engine_contract::LinkKind::Probit),
        other => return Err(Error::Other(format!("unknown link {other:?}"))),
    };
    let est = match estimator {
        "ols" => engine_contract::EstimatorSpec::Ols,
        "glm" => engine_contract::EstimatorSpec::Glm,
        "mle" => engine_contract::EstimatorSpec::Mle,
        other => return Err(Error::Other(format!("unknown estimator {other:?}"))),
    };
    let clusters: Vec<engine_contract::ClusterSpec> = serde_json::from_str(clusters_json)
        .map_err(|e| Error::Other(format!("malformed clusters_json: {e}")))?;
    let (contracts, skeleton) = engine_spec_builder::build_contract_with_skeleton(
        &spec,
        outcome,
        link,
        Some(est),
        intercept,
        clusters,
    )
    .map_err(|e| Error::Other(format!("{e}")))?;
    let names: Vec<String> = contracts.iter().map(|c| c.scenario.name.clone()).collect();
    let bytes = rmp_serde::to_vec_named(&contracts)
        .map_err(|e| Error::Other(format!("msgpack encode failed: {e}")))?;
    let skeleton_json = serde_json::to_string(&skeleton)
        .map_err(|e| Error::Other(format!("skeleton encode failed: {e}")))?;
    Ok(list!(
        names = names,
        contracts = Raw::from_bytes(&bytes),
        skeleton = skeleton_json
    ))
}

/// Synthetic-distribution name → integer code table as a JSON object string.
/// R parses and caches it (`zzz.R`); single source of truth.
#[extendr]
pub fn dist_codes() -> String {
    engine_spec_builder::dist_codes_json()
}

/// Residual-distribution name → integer code table as a JSON object string.
/// See [`dist_codes`].
#[extendr]
pub fn residual_codes() -> String {
    engine_spec_builder::residual_codes_json()
}

/// Random-effect distribution name → integer code table as a JSON object
/// string (normal/heavy_tailed — the RE knob's own vocabulary). See
/// [`dist_codes`].
#[extendr]
pub fn re_dist_codes() -> String {
    engine_spec_builder::re_dist_codes_json()
}

/// Build the canonical effect-recovery design from a `LinearSpec` JSON whose
/// `upload.columns` carry the typed (coded) upload columns. Returns a named list
/// `list(design_flat=<dbl>, semantic_names=<chr>, ncol=<int>)`; the host feeds
/// `design_flat` to `debug_load_data`. Single-sources the design assembly.
#[extendr]
pub fn build_recovery_design(spec_json: &str) -> Result<List> {
    let spec: engine_spec_builder::LinearSpec = serde_json::from_str(spec_json)
        .map_err(|e| Error::Other(format!("invalid LinearSpec JSON: {e}")))?;
    let parsed = engine_spec_builder::parse_formula(&spec.formula)
        .map_err(|e| Error::Other(format!("{e}")))?;
    let table = engine_spec_builder::build_predictor_table(&parsed, &spec.predictors)
        .map_err(|e| Error::Other(format!("{e}")))?;
    let upload = spec.upload.as_ref().ok_or_else(|| {
        Error::Other("build_recovery_design: LinearSpec has no upload block".into())
    })?;
    let rd =
        engine_spec_builder::build_recovery_design(&table, &upload.columns, upload.n_rows as usize)
            .map_err(|e| Error::Other(format!("{e}")))?;
    Ok(list!(
        design_flat = rd.design_flat,
        semantic_names = rd.semantic_names,
        ncol = rd.ncol as i32
    ))
}

/// Z-score a slice with population SD (ddof=0) via the shared engine helper —
/// the single source the host uses to scale the outcome for OLS recovery.
#[extendr]
pub fn standardize_continuous(values: Vec<f64>) -> Vec<f64> {
    engine_spec_builder::standardize_continuous(&values)
}

/// Split a comma-separated assignment string at top-level commas (parens
/// suppress splitting), trimming each segment and dropping empties. The shared
/// splitter the host correlation/effect normalisers bind to. Errors on
/// unbalanced parentheses.
#[extendr]
pub fn split_assignments(input: &str) -> Result<Vec<String>> {
    engine_spec_builder::split_assignments(input).map_err(|e| Error::Other(format!("{e}")))
}

// ── Parser bridge ─────────────────────────────────────────────────────────────

use engine_spec_builder::{
    parse_assignments as rust_parse_assignments, parse_formula as rust_parse_formula,
    AssignmentKey, AssignmentKind, AssignmentValue, KnownNames, VarTypeParams,
};

/// Serializable wrapper for a single assignment item — mirrors engine-py's
/// `{"key": {"name": str} | {"pair": [str, str]}, "value": {"variable_type"|"correlation"|"effect": ...}}`.
#[derive(Serialize)]
struct AssignmentItemJson {
    key: AssignmentKeyJson,
    value: AssignmentValueJson,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AssignmentKeyJson {
    Name { name: String },
    Pair { pair: [String; 2] },
}

/// The legacy variable-type info dict the host's set_variable_type consumes.
#[derive(Serialize)]
struct VarTypeInfoJson {
    #[serde(rename = "type")]
    var_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    proportion: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_levels: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proportions: Option<Vec<f64>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AssignmentValueJson {
    VariableType { variable_type: String },
    VariableTypeParams { variable_type: VarTypeInfoJson },
    Correlation { correlation: f64 },
    Effect { effect: f64 },
}

/// Serializable envelope for the full Assignments result.
#[derive(Serialize)]
struct AssignmentsJson {
    items: Vec<AssignmentItemJson>,
    errors: Vec<String>,
}

/// Parse a model formula string into a JSON string.
///
/// Returns a JSON object:
///   `{"dependent": str, "predictors": [str], "terms": [...], "random_effects": [...]}`
/// `terms` items have shape `{"kind": "main", "name": str}` or `{"kind": "interaction", "vars": [str]}`.
/// `random_effects` items have shape `{"kind": "intercept", "group": str, "parent": str|null}` or
/// `{"kind": "slope", "group": str, "vars": [str]}`.
///
/// Raises an error on syntax errors or unsupported constructs.
/// Mirrors `engine-py/src/spec_builder_bridge.rs` `parse_formula`.
#[extendr]
pub fn parse_formula(input: &str) -> Result<String> {
    let parsed = rust_parse_formula(input).map_err(|e| Error::Other(format!("{e}")))?;
    serde_json::to_string(&parsed)
        .map_err(|e| Error::Other(format!("JSON serialization failed: {e}")))
}

/// Parse an assignment string (variable types / correlations / effects) into a JSON string.
///
/// `kind` is one of `"variable_type"`, `"correlation"`, `"effect"`.
/// `known_json` is a JSON object with keys `"predictors": [str]` and `"interaction_terms": [str]`.
///
/// Returns a JSON object `{"items": [{"key": ..., "value": ...}, ...], "errors": [str, ...]}`.
/// `key` is `{"name": str}` or `{"pair": [str, str]}`.
/// `value` is `{"variable_type": str}` | `{"correlation": f64}` | `{"effect": f64}`.
/// Per-item failures appear in `errors`; only top-level malformed input raises an error.
/// Mirrors `engine-py/src/spec_builder_bridge.rs` `parse_assignments`.
#[extendr]
pub fn parse_assignments(input: &str, kind: &str, known_json: &str) -> Result<String> {
    let kind_enum = match kind {
        "variable_type" => AssignmentKind::VariableType,
        "correlation" => AssignmentKind::Correlation,
        "effect" => AssignmentKind::Effect,
        other => return Err(Error::Other(format!("unknown kind: {other}"))),
    };
    // Deserialize the known-names JSON: {"predictors": [...], "interaction_terms": [...]}
    let known_raw: serde_json::Value = serde_json::from_str(known_json)
        .map_err(|e| Error::Other(format!("malformed known_json: {e}")))?;
    let predictors: Vec<String> = known_raw["predictors"]
        .as_array()
        .ok_or_else(|| Error::Other("known_json missing 'predictors' array".to_string()))?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| Error::Other("predictors must be strings".to_string()))
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let interaction_terms: Vec<String> = known_raw["interaction_terms"]
        .as_array()
        .ok_or_else(|| Error::Other("known_json missing 'interaction_terms' array".to_string()))?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| Error::Other("interaction_terms must be strings".to_string()))
                .map(|s| s.to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let known_struct = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let parsed = rust_parse_assignments(input, kind_enum, &known_struct)
        .map_err(|e| Error::Other(format!("{e}")))?;

    let items: Vec<AssignmentItemJson> = parsed
        .items
        .iter()
        .map(|(k, v)| {
            let key = match k {
                AssignmentKey::Name(n) => AssignmentKeyJson::Name { name: n.clone() },
                AssignmentKey::Pair(a, b) => AssignmentKeyJson::Pair {
                    pair: [a.clone(), b.clone()],
                },
            };
            let value = match v {
                AssignmentValue::VariableType(t) => AssignmentValueJson::VariableType {
                    variable_type: t.clone(),
                },
                AssignmentValue::VariableTypeWithParams { var_type, params } => {
                    let info = match params {
                        VarTypeParams::Bare => VarTypeInfoJson {
                            var_type: var_type.clone(),
                            proportion: None,
                            n_levels: None,
                            proportions: None,
                        },
                        VarTypeParams::Binary { proportion } => VarTypeInfoJson {
                            var_type: var_type.clone(),
                            proportion: Some(*proportion),
                            n_levels: None,
                            proportions: None,
                        },
                        VarTypeParams::Factor {
                            n_levels,
                            proportions,
                        } => VarTypeInfoJson {
                            var_type: var_type.clone(),
                            proportion: None,
                            n_levels: Some(*n_levels),
                            proportions: Some(proportions.clone()),
                        },
                    };
                    AssignmentValueJson::VariableTypeParams {
                        variable_type: info,
                    }
                }
                AssignmentValue::Correlation(c) => {
                    AssignmentValueJson::Correlation { correlation: *c }
                }
                AssignmentValue::Effect(e) => AssignmentValueJson::Effect { effect: *e },
            };
            AssignmentItemJson { key, value }
        })
        .collect();
    let errors: Vec<String> = parsed.errors.iter().map(|e| e.to_string()).collect();
    let envelope = AssignmentsJson { items, errors };
    serde_json::to_string(&envelope)
        .map_err(|e| Error::Other(format!("JSON serialization failed: {e}")))
}

extendr_module! {
    mod spec_builder_bridge;
    fn build_contract_from_spec;
    fn parse_formula;
    fn parse_assignments;
    fn dist_codes;
    fn residual_codes;
    fn re_dist_codes;
    fn build_recovery_design;
    fn standardize_continuous;
    fn split_assignments;
}
