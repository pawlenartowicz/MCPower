//! Bridge `engine-spec-builder` to Python: contract construction, formula parsing, and assignment parsing.
//! Mirror: `engine-r/src/spec_builder_bridge.rs` — change together.

use engine_spec_builder::{build_contract_with_skeleton, LinearSpec, SpecError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Build msgpack bytes of the `SimulationContract` itself (no adapter pass).
/// Test-only back-door (gated behind the `test-bridge` Cargo feature) — the
/// production wheel never ships this symbol so the raw contract shape stays
/// out of the public surface.
#[cfg(feature = "test-bridge")]
#[pyfunction]
pub fn build_contract_from_json(json: &str) -> PyResult<Vec<(String, Py<PyBytes>)>> {
    use engine_spec_builder::build_linear_contract;
    let spec: LinearSpec = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid LinearSpec JSON: {e}")))?;
    let contracts = build_linear_contract(&spec).map_err(map_spec_error)?;

    Python::attach(|py| {
        let mut out = Vec::with_capacity(contracts.len());
        for c in contracts {
            let name = c.scenario.name.clone();
            let bytes = rmp_serde::to_vec_named(&c)
                .map_err(|e| PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;
            out.push((name, PyBytes::new(py, &bytes).into()));
        }
        Ok(out)
    })
}

/// Build the whole `Vec<SimulationContract>` as a single msgpack blob,
/// parametrising outcome_kind + estimator + intercept + clusters at the call
/// site. Replaces the legacy `build_contract_specs_from_json` path.
///
/// Returns `(scenario_names, contracts_msgpack, effect_skeleton_json)` so
/// Python can correlate results without re-decoding the contracts blob, and
/// name results (factor labels, interactions) from the index-only skeleton
/// instead of re-deriving the factor-expansion layout. The skeleton is JSON
/// (`Vec<EffectDescriptor>`); it is identical across scenarios, so it is
/// returned once per build.
/// `outcome_kind` is one of `"continuous"` | `"binary"`.
/// `estimator` is one of `"ols"` | `"glm"` | `"mle"`.
///
#[pyfunction]
pub fn build_contract_from_spec(
    json: &str,
    outcome_kind: &str,
    estimator: &str,
    intercept: f64,
    clusters_json: &str,
) -> PyResult<(Vec<String>, Py<PyBytes>, String)> {
    let spec: LinearSpec = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid LinearSpec JSON: {e}")))?;
    let outcome_kind = match outcome_kind {
        "continuous" => engine_contract::OutcomeKind::Continuous,
        "binary" => engine_contract::OutcomeKind::Binary,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown outcome_kind {other:?}"
            )))
        }
    };
    let estimator = match estimator {
        "ols" => engine_contract::EstimatorSpec::Ols,
        "glm" => engine_contract::EstimatorSpec::Glm,
        "mle" => engine_contract::EstimatorSpec::Mle,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown estimator {other:?}"
            )))
        }
    };
    let clusters: Vec<engine_contract::ClusterSpec> = serde_json::from_str(clusters_json)
        .map_err(|e| PyValueError::new_err(format!("invalid clusters JSON: {e}")))?;
    let (contracts, skeleton) =
        build_contract_with_skeleton(&spec, outcome_kind, Some(estimator), intercept, clusters)
            .map_err(map_spec_error)?;
    let names: Vec<String> = contracts.iter().map(|c| c.scenario.name.clone()).collect();
    let bytes = rmp_serde::to_vec_named(&contracts)
        .map_err(|e| PyRuntimeError::new_err(format!("msgpack encode: {e}")))?;
    let skeleton_json = serde_json::to_string(&skeleton)
        .map_err(|e| PyRuntimeError::new_err(format!("skeleton encode: {e}")))?;
    Python::attach(|py| Ok((names, PyBytes::new(py, &bytes).into(), skeleton_json)))
}

/// Synthetic-distribution name → integer code table as a JSON object string.
/// Python parses and caches it (`mcpower/config.py`); single source of truth.
#[pyfunction]
pub fn dist_codes() -> String {
    engine_spec_builder::dist_codes_json()
}

/// Residual-distribution name → integer code table as a JSON object string.
/// See [`dist_codes`].
#[pyfunction]
pub fn residual_codes() -> String {
    engine_spec_builder::residual_codes_json()
}

/// Random-effect distribution name → integer code table as a JSON object
/// string (normal/heavy_tailed — the RE knob's own vocabulary). See
/// [`dist_codes`].
#[pyfunction]
pub fn re_dist_codes() -> String {
    engine_spec_builder::re_dist_codes_json()
}

/// Build the canonical effect-recovery design from a `LinearSpec` JSON whose
/// `upload.columns` carry the typed (coded) upload columns. Returns
/// `(design_flat, semantic_names, ncol)`; the host feeds `design_flat` to
/// `fit_uploaded_data`. Single-sources the design assembly with the Tauri/R ports.
#[pyfunction]
pub fn build_recovery_design(spec_json: &str) -> PyResult<(Vec<f64>, Vec<String>, usize)> {
    let spec: LinearSpec = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("invalid LinearSpec JSON: {e}")))?;
    let parsed = engine_spec_builder::parse_formula(&spec.formula).map_err(map_spec_error)?;
    let table = engine_spec_builder::build_predictor_table(&parsed, &spec.predictors)
        .map_err(map_spec_error)?;
    let upload = spec.upload.as_ref().ok_or_else(|| {
        PyValueError::new_err("build_recovery_design: LinearSpec has no upload block")
    })?;
    let rd =
        engine_spec_builder::build_recovery_design(&table, &upload.columns, upload.n_rows as usize)
            .map_err(map_spec_error)?;
    Ok((rd.design_flat, rd.semantic_names, rd.ncol))
}

/// Z-score a slice with population SD (ddof=0) via the shared engine helper —
/// the single source the host uses to scale the outcome for OLS recovery.
#[pyfunction]
pub fn standardize_continuous(values: Vec<f64>) -> Vec<f64> {
    engine_spec_builder::standardize_continuous(&values)
}

/// Split a comma-separated assignment string at top-level commas (parens
/// suppress splitting), trimming each segment and dropping empties. The shared
/// splitter the host correlation/effect normalisers bind to. Raises on
/// unbalanced parentheses.
#[pyfunction]
pub fn split_assignments(input: &str) -> PyResult<Vec<String>> {
    engine_spec_builder::split_assignments(input).map_err(map_spec_error)
}

// ── Parser bridge ─────────────────────────────────────────────────────────────

use engine_spec_builder::{
    parse_assignments as rust_parse_assignments, parse_formula as rust_parse_formula,
    AssignmentKey, AssignmentKind, AssignmentValue, KnownNames, RandomEffect, Term, VarTypeParams,
};
use pyo3::types::PyDict;

/// Parse a model formula string into a Python dict.
///
/// Returns `{"dependent": str, "predictors": [str], "terms": [...], "random_effects": [...]}`.
/// `terms` items have shape `{"kind": "main", "name": str}` or `{"kind": "interaction", "vars": [str]}`.
/// `random_effects` items have shape `{"kind": "intercept", "group": str, "parent": str|None}` or
/// `{"kind": "slope", "group": str, "vars": [str]}`.
///
/// Raises `ValueError` on syntax errors or unsupported constructs.
#[pyfunction]
pub fn parse_formula(py: Python<'_>, input: &str) -> PyResult<Py<PyAny>> {
    let parsed = rust_parse_formula(input).map_err(map_spec_error)?;
    Ok(parsed_formula_to_pydict(py, &parsed)?.into_any().unbind())
}

/// Parse an assignment string (variable types / correlations / effects) into a Python dict.
///
/// `kind` is one of `"variable_type"`, `"correlation"`, `"effect"`.
/// `known` is a dict with keys `"predictors": [str]` and `"interaction_terms": [str]`.
///
/// Returns `{"items": [{"key": ..., "value": ...}, ...], "errors": [str, ...]}`.
/// `key` is `{"name": str}` or `{"pair": [str, str]}`. `value` is `{"variable_type"|"correlation"|"effect": ...}`.
/// Per-item failures appear in `errors`; only top-level malformed input raises `ValueError`.
#[pyfunction]
pub fn parse_assignments(
    py: Python<'_>,
    input: &str,
    kind: &str,
    known: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let kind_enum = match kind {
        "variable_type" => AssignmentKind::VariableType,
        "correlation" => AssignmentKind::Correlation,
        "effect" => AssignmentKind::Effect,
        other => return Err(PyValueError::new_err(format!("unknown kind: {other}"))),
    };
    let predictors: Vec<String> = known
        .get_item("predictors")?
        .ok_or_else(|| PyValueError::new_err("missing 'predictors' key"))?
        .extract()?;
    let interaction_terms: Vec<String> = known
        .get_item("interaction_terms")?
        .ok_or_else(|| PyValueError::new_err("missing 'interaction_terms' key"))?
        .extract()?;
    let known_struct = KnownNames {
        predictors: &predictors,
        interaction_terms: &interaction_terms,
    };
    let parsed = rust_parse_assignments(input, kind_enum, &known_struct).map_err(map_spec_error)?;
    Ok(assignments_to_pydict(py, &parsed)?.into_any().unbind())
}

fn parsed_formula_to_pydict<'py>(
    py: Python<'py>,
    p: &engine_spec_builder::ParsedFormula,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("dependent", &p.dependent)?;
    d.set_item("predictors", &p.predictors)?;
    let terms: Vec<Bound<'py, PyDict>> = p
        .terms
        .iter()
        .map(|t| term_to_pydict(py, t))
        .collect::<Result<_, _>>()?;
    d.set_item("terms", terms)?;
    let res: Vec<Bound<'py, PyDict>> = p
        .random_effects
        .iter()
        .map(|r| random_effect_to_pydict(py, r))
        .collect::<Result<_, _>>()?;
    d.set_item("random_effects", res)?;
    Ok(d)
}

fn term_to_pydict<'py>(py: Python<'py>, t: &Term) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match t {
        Term::Main { name } => {
            d.set_item("kind", "main")?;
            d.set_item("name", name)?;
        }
        Term::Interaction { vars } => {
            d.set_item("kind", "interaction")?;
            d.set_item("vars", vars)?;
        }
    }
    Ok(d)
}

fn random_effect_to_pydict<'py>(py: Python<'py>, r: &RandomEffect) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match r {
        RandomEffect::Intercept { group, parent } => {
            d.set_item("kind", "intercept")?;
            d.set_item("group", group)?;
            // Option<String> → Python str or None
            match parent {
                Some(p) => d.set_item("parent", p)?,
                None => d.set_item("parent", py.None())?,
            }
        }
        RandomEffect::Slope { group, vars } => {
            d.set_item("kind", "slope")?;
            d.set_item("group", group)?;
            d.set_item("vars", vars)?;
        }
    }
    Ok(d)
}

fn assignments_to_pydict<'py>(
    py: Python<'py>,
    a: &engine_spec_builder::Assignments,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let items: Vec<Bound<'py, PyDict>> = a
        .items
        .iter()
        .map(|(k, v)| assignment_item_to_pydict(py, k, v))
        .collect::<Result<_, _>>()?;
    d.set_item("items", items)?;
    let errors: Vec<String> = a.errors.iter().map(|e| e.to_string()).collect();
    d.set_item("errors", errors)?;
    Ok(d)
}

fn assignment_item_to_pydict<'py>(
    py: Python<'py>,
    key: &AssignmentKey,
    value: &AssignmentValue,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let key_d = PyDict::new(py);
    match key {
        AssignmentKey::Name(n) => {
            key_d.set_item("name", n)?;
        }
        AssignmentKey::Pair(a, b) => {
            key_d.set_item("pair", vec![a, b])?;
        }
    }
    d.set_item("key", key_d)?;
    let val_d = PyDict::new(py);
    match value {
        AssignmentValue::VariableType(v) => {
            val_d.set_item("variable_type", v)?;
        }
        AssignmentValue::VariableTypeWithParams { var_type, params } => {
            // Emit the legacy info dict the host's set_variable_type consumes:
            // {"type": ..., ["proportion"] | ["n_levels", "proportions"]}.
            let info = PyDict::new(py);
            info.set_item("type", var_type)?;
            match params {
                VarTypeParams::Bare => {}
                VarTypeParams::Binary { proportion } => {
                    info.set_item("proportion", *proportion)?;
                }
                VarTypeParams::Factor {
                    n_levels,
                    proportions,
                } => {
                    info.set_item("n_levels", *n_levels)?;
                    info.set_item("proportions", proportions.clone())?;
                }
            }
            val_d.set_item("variable_type", info)?;
        }
        AssignmentValue::Correlation(v) => {
            val_d.set_item("correlation", *v)?;
        }
        AssignmentValue::Effect(v) => {
            val_d.set_item("effect", *v)?;
        }
    }
    d.set_item("value", val_d)?;
    Ok(d)
}

fn map_spec_error(e: SpecError) -> PyErr {
    // Every SpecError variant maps to PyValueError except InternalContractValidate
    // (a builder/adapter bug from the host's perspective).
    match e {
        SpecError::EmptyFormula
        | SpecError::FormulaSyntax { .. }
        | SpecError::UnknownPredictor { .. }
        | SpecError::TermRemovalUnsupported
        | SpecError::DuplicateGroupingVar { .. }
        | SpecError::EmptySlopeTerm { .. }
        | SpecError::RandomSlopesUnsupported
        | SpecError::RandomInterceptSuppressionUnsupported
        | SpecError::NonFiniteEffect { .. }
        | SpecError::FactorLevelCount { .. }
        | SpecError::FactorProportionSum { .. }
        | SpecError::FactorProportionLengthMismatch { .. }
        | SpecError::FactorReferenceMissing { .. }
        | SpecError::FactorProportionNonPositive { .. }
        | SpecError::CorrelationOutOfRange { .. }
        | SpecError::CorrelationUnknownVar { .. }
        | SpecError::CorrelationNotPsd
        | SpecError::CorrelationNonContinuous { .. }
        | SpecError::UnknownTarget { .. }
        | SpecError::UnknownResidualDist { .. }
        | SpecError::ScenarioBinarySwapUnsupported { .. }
        | SpecError::ScenarioResidualDfTooLow { .. }
        | SpecError::EffectCountMismatch { .. }
        | SpecError::ClusterFamilyMismatch
        | SpecError::MalformedAssignment { .. }
        | SpecError::UnknownAssignmentName { .. }
        | SpecError::UnknownVariableType { .. }
        | SpecError::UnknownContrastName { .. }
        | SpecError::TestFormulaPredictorMissing { .. }
        | SpecError::RecoveryColumnMissing { .. }
        | SpecError::RecoveryColumnLength { .. }
        | SpecError::InvalidVariableTypeValue { .. }
        | SpecError::NotAFactorPredictor { .. } => PyValueError::new_err(e.to_string()),
        SpecError::InternalContractValidate(_) => PyRuntimeError::new_err(e.to_string()),
    }
}
