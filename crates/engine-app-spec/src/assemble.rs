//! `assemble_spec` — run-path-free projector from `AppSpec` GUI state to `Vec<SimulationContract>`; reuses `engine-spec-builder`.

use engine_contract::SimulationContract;
use engine_spec_builder::input::{CorrelationPair, UploadColumn, UploadColumnType, UploadInput};
use engine_spec_builder::{
    build_contract_with_skeleton, EffectAssignment, EffectSkeleton, HeteroskedasticityInput,
    LinearSpec as BuilderLinearSpec, PredictorSpec, ResidualSpec, VarKind,
};

use engine_contract::CorrectionMethod;

use crate::app_spec::{
    AppGroupingRelation, AppGroupingSpec, AppSlopeTerm, AppSpec, ClusterDim, CorrelationMatrix,
    CsvData, EffectSize, LinearSpec, LogitSpec, MixedOutcome, MixedSpec, NumericDistribution,
    OutcomeOptions, ParsedFormula, TestSelection, VarType,
};
use crate::error::AdapterError;

/// Thin wrapper that drops the `EffectSkeleton` for the run path (which only
/// needs the contracts). The projection lives once in
/// [`assemble_spec_with_skeleton`].
pub fn assemble_spec(spec: &AppSpec) -> Result<Vec<SimulationContract>, AdapterError> {
    Ok(assemble_spec_with_skeleton(spec)?.0)
}

/// As [`assemble_spec`], but also returns the index-only [`EffectSkeleton`] so a
/// host (Tauri, WASM) can name results from it + its own label store. The
/// skeleton is identical across the returned contracts (one design per spec).
pub fn assemble_spec_with_skeleton(
    spec: &AppSpec,
) -> Result<(Vec<SimulationContract>, EffectSkeleton), AdapterError> {
    // Each assembler returns the builder's full contract Vec: empty `scenarios`
    // yields one baseline contract, a non-empty list yields one contract per
    // scenario. The run drivers pass the whole slice to the orchestrator, which
    // fans out N scenarios into a `ScenarioResult` of len N.
    match spec {
        AppSpec::Linear(linear) => assemble_linear(linear),
        AppSpec::Logit(logit) => assemble_logit(logit),
        AppSpec::Mixed(mixed) => assemble_mixed(mixed),
    }
}

/// JSON of the index-only [`EffectSkeleton`] for `spec` — the convenience both
/// GUI hosts call to name results without re-deriving the factor-expansion
/// layout. Mirrors the `skeleton` string the Py/R bridges return.
pub fn effect_skeleton_json(spec: &AppSpec) -> Result<String, AdapterError> {
    let (_, skeleton) = assemble_spec_with_skeleton(spec)?;
    serde_json::to_string(&skeleton).map_err(|e| AdapterError::SkeletonEncode(e.to_string()))
}

/// τ² = icc / (1 − icc), mirroring the Python port (`model.py`).
/// icc is validated (hard `[0, 1)` range + stability band) in `assemble_mixed`
/// before this is called, and by the Py/R hosts; guard the denominator anyway.
pub(crate) fn icc_to_tau_squared(icc: f64) -> f64 {
    let denom = 1.0 - icc;
    if denom > 0.0 {
        icc / denom
    } else {
        0.0
    }
}

/// Latent-scale logistic ICC→τ²: `τ² = icc/(1 − icc) · π²/3`. The random
/// intercept of a logistic GLMM lives on the log-odds scale, whose residual
/// variance is π²/3 (not 1); `icc_to_tau_squared` contributes the `icc/(1 − icc)`
/// factor. Shared with the Python and R ports.
pub(crate) fn icc_to_tau_squared_logit(icc: f64) -> f64 {
    icc_to_tau_squared(icc) * std::f64::consts::PI.powi(2) / 3.0
}

/// Logit (log-odds) of a probability `p ∈ (0, 1)`: `ln(p / (1 − p))`. Turns a
/// baseline probability into the GLM intercept; the caller guards `p`'s range.
/// Inverse of [`logistic`] — change the two together so the baseline round-trip
/// `logistic(logit(p)) == p` stays exact.
#[inline]
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Logistic (inverse-logit / expit) of a log-odds `x`: `1 / (1 + exp(−x))`.
/// Recovers a baseline probability from a fitted intercept, undoing [`logit`];
/// change the two together so the round-trip `logistic(logit(p)) == p` stays exact.
#[inline]
pub(crate) fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn assemble_mixed(
    spec: &MixedSpec,
) -> Result<(Vec<SimulationContract>, EffectSkeleton), AdapterError> {
    let mut builder_spec = project_to_builder_spec(
        &spec.parsed_formula,
        &spec.var_types,
        &spec.effects,
        &spec.correlations,
        spec.correction,
        &spec.tests,
        spec.alpha,
        spec.csv.as_ref(),
    )?;
    builder_spec.report_overall = spec.report_overall;
    builder_spec.contrast_pairs = spec.contrasts.clone();
    builder_spec.test_formula = spec.test_formula.clone();
    builder_spec.scenarios = spec.scenarios.clone();
    builder_spec.cluster_level_vars = spec.cluster_level_vars.clone();
    builder_spec.wald_se = spec.wald_se;
    apply_outcome_options(
        &mut builder_spec,
        spec.outcome_options.as_ref(),
        &spec.var_types,
    )?;

    let sizing = match spec.cluster_dim {
        ClusterDim::NClusters { value } => {
            engine_contract::ClusterSizing::FixedClusters { n_clusters: value }
        }
        ClusterDim::ClusterSize { value } => engine_contract::ClusterSizing::FixedSize {
            cluster_size: value,
        },
    };

    let slopes = resolve_slopes(&spec.var_types, &spec.slopes)?;
    let extra_groupings: Vec<engine_contract::GroupingSpec> =
        spec.extra_groupings.iter().map(to_grouping_spec).collect();

    // Validate the primary ICC here (the Python/R hosts validate it before
    // building their contracts; this is the Tauri/WASM path's equivalent).
    // Two tiers, identical semantics to those hosts: a hard (0, 1) range, then a
    // numerical-stability band read from the shared config. Without this, an
    // icc >= 1 reaches `icc_to_tau_squared` and silently yields tau_squared = 0,
    // i.e. a plausible-looking run with no cluster variance. Extra groupings
    // arrive as tau_squared (host-converted), so only the primary icc is checked.
    let [icc_lo, icc_hi] = engine_contract::config().limits.icc_stability;
    let icc = spec.icc;
    if !(0.0..1.0).contains(&icc) {
        return Err(AdapterError::IccOutOfRange { value: icc });
    }
    if icc != 0.0 && (icc < icc_lo || icc > icc_hi) {
        return Err(AdapterError::IccOutOfStabilityBand {
            value: icc,
            lo: icc_lo,
            hi: icc_hi,
        });
    }

    let (outcome_kind, estimator, intercept, tau_squared) = match &spec.outcome {
        MixedOutcome::Gaussian => (
            engine_contract::OutcomeKind::Continuous,
            Some(engine_contract::EstimatorSpec::Mle),
            0.0,
            icc_to_tau_squared(spec.icc),
        ),
        MixedOutcome::Binary {
            baseline_probability,
        } => {
            let p = *baseline_probability;
            if !(p > 0.0 && p < 1.0) {
                return Err(AdapterError::BaselineProbabilityOutOfRange { value: p });
            }
            (
                engine_contract::OutcomeKind::Binary,
                Some(engine_contract::EstimatorSpec::Glm),
                logit(p),
                icc_to_tau_squared_logit(spec.icc),
            )
        }
    };

    let built = build_contract_with_skeleton(
        &builder_spec,
        outcome_kind,
        estimator,
        intercept,
        vec![engine_contract::ClusterSpec {
            sizing,
            tau_squared,
            slopes,
            extra_groupings,
        }],
    )
    .map_err(AdapterError::from)?;
    Ok(built)
}

/// Convert a UI-layer [`AppGroupingSpec`] to the contract's `GroupingSpec`.
/// Slopes on secondary groupings are not exposed (the engine rejects them — see
/// the `engine-contract` grouping invariant), so they are always empty.
fn to_grouping_spec(g: &AppGroupingSpec) -> engine_contract::GroupingSpec {
    engine_contract::GroupingSpec {
        tau_squared: g.tau_squared,
        relation: match g.relation {
            AppGroupingRelation::Crossed { n_clusters } => {
                engine_contract::GroupingRelation::Crossed { n_clusters }
            }
            AppGroupingRelation::NestedWithin { n_per_parent } => {
                engine_contract::GroupingRelation::NestedWithin { n_per_parent }
            }
        },
        slopes: vec![],
    }
}

/// Resolve UI-layer slope predictor names to contract `SlopeTerm`s.
///
/// A slope's `ColumnId` is its position among the **non-factor** predictors, NOT
/// its raw position in `parsed_formula.predictors`: the engine orders non-factor
/// generation columns (Numeric AND Binary — both occupy a column) before factor
/// columns (`project_contract.rs`), so the raw formula position is off-by-one
/// whenever a factor precedes the slope predictor. Mirrors Python's
/// `non_factor_names.index(name)` and the R port — the cross-port
/// column-resolution tripwire test pins all three to agree.
fn resolve_slopes(
    var_types: &[VarType],
    slopes: &[AppSlopeTerm],
) -> Result<Vec<engine_contract::SlopeTerm>, AdapterError> {
    let non_factor_names: Vec<&str> = var_types
        .iter()
        .filter(|vt| !matches!(vt, VarType::Factor { .. }))
        .map(var_type_name)
        .collect();
    slopes
        .iter()
        .map(|s| {
            let col_idx = non_factor_names
                .iter()
                .position(|p| *p == s.predictor_name)
                .ok_or_else(|| AdapterError::UnknownPredictor(s.predictor_name.clone()))?;
            Ok(engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(col_idx as u32),
                variance: s.slope_variance,
                corr_with_intercept: s.slope_intercept_corr,
                corr_with: vec![],
            })
        })
        .collect()
}

fn assemble_linear(
    spec: &LinearSpec,
) -> Result<(Vec<SimulationContract>, EffectSkeleton), AdapterError> {
    let mut builder_spec = project_to_builder_spec(
        &spec.parsed_formula,
        &spec.var_types,
        &spec.effects,
        &spec.correlations,
        spec.correction,
        &spec.tests,
        spec.alpha,
        spec.csv.as_ref(),
    )?;
    builder_spec.report_overall = spec.report_overall;
    builder_spec.contrast_pairs = spec.contrasts.clone();
    builder_spec.test_formula = spec.test_formula.clone();
    builder_spec.scenarios = spec.scenarios.clone();
    apply_outcome_options(
        &mut builder_spec,
        spec.outcome_options.as_ref(),
        &spec.var_types,
    )?;

    // estimator None → build_contract default coupling: Continuous + no clusters → Ols.
    let built = build_contract_with_skeleton(
        &builder_spec,
        engine_contract::OutcomeKind::Continuous,
        None,
        0.0,
        vec![],
    )
    .map_err(AdapterError::from)?;
    Ok(built)
}

fn assemble_logit(
    spec: &LogitSpec,
) -> Result<(Vec<SimulationContract>, EffectSkeleton), AdapterError> {
    let p = spec.baseline_probability;
    if !(p > 0.0 && p < 1.0) {
        return Err(AdapterError::BaselineProbabilityOutOfRange { value: p });
    }
    let intercept = logit(p);

    let mut builder_spec = project_to_builder_spec(
        &spec.parsed_formula,
        &spec.var_types,
        &spec.effects,
        &spec.correlations,
        spec.correction,
        &spec.tests,
        spec.alpha,
        spec.csv.as_ref(),
    )?;
    builder_spec.test_formula = spec.test_formula.clone();
    builder_spec.scenarios = spec.scenarios.clone();
    builder_spec.wald_se = spec.wald_se;
    apply_outcome_options(
        &mut builder_spec,
        spec.outcome_options.as_ref(),
        &spec.var_types,
    )?;

    // estimator None → build_contract default coupling: Binary → Glm.
    let built = build_contract_with_skeleton(
        &builder_spec,
        engine_contract::OutcomeKind::Binary,
        None,
        intercept,
        vec![],
    )
    .map_err(AdapterError::from)?;
    Ok(built)
}

/// Convert a `CsvData` attachment to the spec-builder's `UploadInput` envelope.
fn csv_to_upload_input(csv: &CsvData) -> UploadInput {
    UploadInput {
        mode: csv.mode,
        n_rows: csv.n_rows,
        columns: csv.columns.clone(),
    }
}

/// Shared projection logic for Linear, Logit, and Mixed assemblers.
///
/// Builds a `BuilderLinearSpec` from the common fields shared between
/// `LinearSpec`, `LogitSpec`, and `MixedSpec`. The returned spec has
/// `report_overall: false` and `contrast_pairs: vec![]` as defaults; callers
/// that need different values (e.g. `assemble_linear`) must override them after
/// calling this helper.
#[allow(clippy::too_many_arguments)]
fn project_to_builder_spec(
    parsed_formula: &ParsedFormula,
    var_types: &[VarType],
    effects: &[EffectSize],
    correlations: &Option<CorrelationMatrix>,
    correction: CorrectionMethod,
    tests: &TestSelection,
    alpha: f64,
    csv: Option<&CsvData>,
) -> Result<BuilderLinearSpec, AdapterError> {
    // Target format: "outcome ~ pred1 + pred2 + pred1:pred2"
    let outcome = &parsed_formula.outcome;
    let mut terms: Vec<String> = parsed_formula.predictors.clone();
    for interaction in &parsed_formula.interaction_terms {
        terms.push(interaction.join(":"));
    }
    let formula = if terms.is_empty() {
        format!("{outcome} ~ 1")
    } else {
        format!("{outcome} ~ {}", terms.join(" + "))
    };

    // Preserve parsed_formula.predictors order; if no matching var_types
    // entry, default to VarKind::Normal.

    // Build a name → UploadColumn lookup once, used for the class-conflict
    // guard below (only when a csv is attached).
    let col_by_name: std::collections::HashMap<&str, &UploadColumn> = csv
        .map(|c| {
            c.columns
                .iter()
                .map(|col| (col.name.as_str(), col))
                .collect()
        })
        .unwrap_or_default();

    let predictors: Result<Vec<PredictorSpec>, AdapterError> = parsed_formula
        .predictors
        .iter()
        .map(|pred_name| {
            let var_type = var_types.iter().find(|vt| var_type_name(vt) == pred_name);
            let upload_col = col_by_name.get(pred_name.as_str()).copied();

            // Class-conflict guard: fires ONLY when the user *explicitly* declared
            // a class that clashes with a matched column's auto-detected class.
            // An undeclared predictor (`var_type == None`) is never a conflict —
            // the uploaded column's detected type is authoritative ("data wins"),
            // so we fall through to derive the kind from the detected type below.
            // This rejects an explicit mismatch with a named, friendly error
            // instead of letting a ColumnId-keyed kernel error reach the user.
            if let (Some(vt), Some(col)) = (var_type, upload_col) {
                let detected_class = col.col_type;
                let declared_class = var_type_to_col_class(Some(vt));
                if declared_class != detected_class {
                    return Err(AdapterError::UploadClassConflict {
                        name: pred_name.clone(),
                        detected: col_class_label(detected_class),
                        declared: col_class_label(declared_class),
                    });
                }
            }

            let kind = match var_type {
                // Undeclared predictor: if it matches an uploaded column, let the
                // column's detected type drive the build (data wins, matching the
                // Python/R ports) so a detected factor lands in the factor region
                // — otherwise an undeclared factor would build a `Direct` term and
                // trip `DirectOnFactor` downstream. `apply_upload` overwrites the
                // synthetic column spec with the data-backed variant; the synthetic
                // params here only need to pass pre-projection validation.
                None => match upload_col {
                    Some(col) => detected_var_kind(col),
                    None => VarKind::Normal,
                },

                Some(VarType::Numeric { distribution, .. }) => match distribution {
                    NumericDistribution::Normal => VarKind::Normal,
                    NumericDistribution::RightSkewed => VarKind::RightSkewed,
                    NumericDistribution::LeftSkewed => VarKind::LeftSkewed,
                    NumericDistribution::HighKurtosis => VarKind::HighKurtosis,
                    NumericDistribution::Uniform => VarKind::Uniform,
                },

                Some(VarType::Binary {
                    name,
                    binary_proportion: p,
                }) => {
                    if !(*p >= 0.0 && *p <= 1.0) {
                        return Err(AdapterError::InvalidProportion {
                            name: name.clone(),
                            value: *p,
                        });
                    }
                    VarKind::Binary { proportion: *p }
                }

                Some(VarType::Factor {
                    name,
                    factor_n_levels,
                    factor_proportions,
                    factor_reference,
                    factor_labels,
                    sampled_proportions,
                }) => {
                    let n = *factor_n_levels as usize;
                    if factor_proportions.len() != n {
                        return Err(AdapterError::FactorLevelMismatch {
                            name: name.clone(),
                            expected: n,
                            got: factor_proportions.len(),
                        });
                    }
                    if (*factor_reference as usize) >= n {
                        return Err(AdapterError::FactorReferenceOutOfRange {
                            name: name.clone(),
                            reference: *factor_reference,
                            n_levels: n,
                        });
                    }
                    // User labels become the engine's level names — and thereby
                    // the effect names (`name[label]`), which must match the
                    // host's effect assignments. Empty = legacy "1".."k".
                    let levels: Vec<String> = if factor_labels.is_empty() {
                        (1..=*factor_n_levels).map(|i| format!("{i}")).collect()
                    } else {
                        if factor_labels.len() != n {
                            return Err(AdapterError::FactorLabelMismatch {
                                name: name.clone(),
                                expected: n,
                                got: factor_labels.len(),
                            });
                        }
                        for (i, lbl) in factor_labels.iter().enumerate() {
                            if factor_labels[..i].contains(lbl) {
                                return Err(AdapterError::DuplicateFactorLabel {
                                    name: name.clone(),
                                    label: lbl.clone(),
                                });
                            }
                        }
                        factor_labels.clone()
                    };
                    let reference = levels[*factor_reference as usize].clone();
                    VarKind::Factor {
                        levels,
                        proportions: factor_proportions.clone(),
                        reference,
                        sampled_proportions: *sampled_proportions,
                    }
                }
            };

            // Pin rule: only an explicitly-declared Numeric carries the
            // user's pin; Binary/Factor/undeclared columns are never
            // swap-eligible anyway (non-normal kinds), so they stay false.
            let pinned = matches!(var_type, Some(VarType::Numeric { pinned: true, .. }));
            Ok(PredictorSpec {
                name: pred_name.clone(),
                pinned,
                kind,
            })
        })
        .collect();
    let predictors = predictors?;

    let effects: Vec<EffectAssignment> = effects
        .iter()
        .map(|e| EffectAssignment {
            name: e.name.clone(),
            size: e.value,
        })
        .collect();

    let correlations: Vec<CorrelationPair> = match correlations {
        None => vec![],
        Some(matrix) => {
            let n = matrix.names.len();
            if matrix.values.len() != n {
                return Err(AdapterError::InvalidCorrelations {
                    n_names: n,
                    n_rows: matrix.values.len(),
                });
            }
            for (i, row) in matrix.values.iter().enumerate() {
                if row.len() != n {
                    return Err(AdapterError::InvalidCorrelations {
                        n_names: n,
                        n_rows: row.len(),
                    });
                }
                let _ = i;
            }

            let mut pairs = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = matrix.values[i][j];
                    if v != 0.0 {
                        pairs.push(CorrelationPair {
                            a: matrix.names[i].clone(),
                            b: matrix.names[j].clone(),
                            value: v,
                        });
                    }
                }
            }
            pairs
        }
    };

    // The spec-builder re-exports `engine_contract::CorrectionMethod` directly,
    // so the contract correction passes straight through with no mapping.

    let targets: Vec<String> = match tests {
        TestSelection::All => vec!["overall".into()],
        TestSelection::Effects { names } => names.clone(),
        TestSelection::Contrasts { names } => names.clone(),
    };

    // `scenarios` defaults to empty here (one baseline contract); each assembler
    // overrides it with the AppSpec's projected scenario list before building.

    Ok(BuilderLinearSpec {
        formula,
        predictors,
        effects,
        correlations,
        alpha,
        correction,
        targets,
        heteroskedasticity: HeteroskedasticityInput::default(),
        residual: ResidualSpec::default(),
        max_failed_fraction: 0.1,
        scenarios: vec![],
        test_formula: None,
        // Defaults: callers override when needed (e.g. assemble_linear).
        report_overall: false,
        contrast_pairs: vec![],
        posthoc_requests: vec![],
        upload: csv.map(csv_to_upload_input),
        cluster_level_vars: vec![],
        wald_se: Default::default(),
    })
}

fn var_type_name(vt: &VarType) -> &str {
    match vt {
        VarType::Numeric { name, .. } => name,
        VarType::Binary { name, .. } => name,
        VarType::Factor { name, .. } => name,
    }
}

/// Apply the GUI's outcome-level structural knobs onto the builder spec.
/// `None` keeps the builder defaults (unpinned normal residual, lp-driven
/// scenario heteroskedasticity). A present `residual_distribution` is an
/// explicit user choice → pinned (incl. explicit "normal" — neutral keys on
/// unpinned-default, not on the value).
fn apply_outcome_options(
    builder_spec: &mut BuilderLinearSpec,
    options: Option<&OutcomeOptions>,
    var_types: &[VarType],
) -> Result<(), AdapterError> {
    let Some(o) = options else { return Ok(()) };
    if let Some(dist) = &o.residual_distribution {
        builder_spec.residual = ResidualSpec {
            distribution: dist.clone(),
            pinned: true,
        };
    }
    let driver_var_index = match &o.heteroskedasticity_driver {
        None => None,
        Some(name) => {
            // Same non-factor index space as random slopes (see resolve_slopes).
            let idx = var_types
                .iter()
                .filter(|vt| !matches!(vt, VarType::Factor { .. }))
                .position(|vt| var_type_name(vt) == name)
                .ok_or_else(|| AdapterError::UnknownHeteroskedasticityDriver(name.clone()))?;
            Some(idx as u32)
        }
    };
    if driver_var_index.is_some() {
        builder_spec.heteroskedasticity = HeteroskedasticityInput { driver_var_index };
    }
    Ok(())
}

/// Map a declared `VarType` (or absence → Numeric default) to the equivalent
/// `UploadColumnType` class for conflict detection.
/// `VarType::Numeric` (continuous distribution) → `Continuous`
/// `VarType::Binary`                             → `Binary`
/// `VarType::Factor`                             → `Factor`
fn var_type_to_col_class(var_type: Option<&VarType>) -> UploadColumnType {
    match var_type {
        None | Some(VarType::Numeric { .. }) => UploadColumnType::Continuous,
        Some(VarType::Binary { .. }) => UploadColumnType::Binary,
        Some(VarType::Factor { .. }) => UploadColumnType::Factor,
    }
}

/// Derive a placeholder `VarKind` from a matched, *undeclared* upload column so
/// the detected type drives the build (factor → factor region, binary → binary)
/// — "data wins", matching the Python/R ports. `apply_upload` later overwrites
/// the synthetic column spec with the data-backed variant (`Resampled` /
/// `ResampledBinary` / `FactorFromFrame`), so the params chosen here only need to
/// satisfy `validate_pre_projection`; the actual proportions/levels come from the
/// uploaded data downstream.
///
/// Factor levels are `"1".."k"` (k = `labels.len()`), reference `"1"`, mirroring
/// the declared-`Factor` branch's level-naming convention; proportions are equal
/// with the final entry absorbing rounding so they sum to exactly 1.0.
fn detected_var_kind(col: &UploadColumn) -> VarKind {
    match col.col_type {
        UploadColumnType::Continuous => VarKind::Normal,
        UploadColumnType::Binary => VarKind::Binary { proportion: 0.5 },
        UploadColumnType::Factor => {
            let k = col.labels.len();
            let levels: Vec<String> = (1..=k).map(|i| format!("{i}")).collect();
            let mut proportions = vec![1.0 / k as f64; k];
            if let Some(last) = proportions.last_mut() {
                *last = 1.0 - (k - 1) as f64 / k as f64;
            }
            VarKind::Factor {
                levels,
                proportions,
                reference: "1".into(),
                sampled_proportions: None,
            }
        }
    }
}

/// Human-readable label for an `UploadColumnType`, used in the error message.
fn col_class_label(t: UploadColumnType) -> String {
    match t {
        UploadColumnType::Continuous => "continuous".into(),
        UploadColumnType::Binary => "binary".into(),
        UploadColumnType::Factor => "factor".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pf(predictors: Vec<&str>, interactions: Vec<Vec<&str>>) -> ParsedFormula {
        ParsedFormula {
            outcome: "y".into(),
            predictors: predictors.into_iter().map(String::from).collect(),
            interaction_terms: interactions
                .into_iter()
                .map(|v| v.into_iter().map(String::from).collect())
                .collect(),
        }
    }

    // Formula reconstruction preserves predictor order + interactions,
    // and an empty-predictor formula collapses to "{outcome} ~ 1".
    #[test]
    fn project_reconstructs_formula_in_predictor_order() {
        let spec = project_to_builder_spec(
            &pf(vec!["x1", "x2"], vec![vec!["x1", "x2"]]),
            &[],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        assert_eq!(spec.formula, "y ~ x1 + x2 + x1:x2");
    }

    #[test]
    fn project_empty_predictors_yields_intercept_only_formula() {
        let spec = project_to_builder_spec(
            &pf(vec![], vec![]),
            &[],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        assert_eq!(spec.formula, "y ~ 1");
    }

    // Only non-zero off-diagonal correlations become CorrelationPairs;
    // an all-zero matrix yields an empty pair list.
    #[test]
    fn project_emits_only_nonzero_correlation_pairs() {
        let corr = Some(CorrelationMatrix {
            names: vec!["a".into(), "b".into(), "c".into()],
            values: vec![
                vec![1.0, 0.3, 0.0],
                vec![0.3, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        });
        let spec = project_to_builder_spec(
            &pf(vec!["a", "b", "c"], vec![]),
            &[
                VarType::Numeric {
                    name: "a".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
                VarType::Numeric {
                    name: "b".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
                VarType::Numeric {
                    name: "c".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
            ],
            &[],
            &corr,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        assert_eq!(spec.correlations.len(), 1, "only the a,b=0.3 pair survives");
        assert_eq!(spec.correlations[0].a, "a");
        assert_eq!(spec.correlations[0].b, "b");
    }

    #[test]
    fn project_all_zero_correlation_matrix_yields_no_pairs() {
        let corr = Some(CorrelationMatrix {
            names: vec!["a".into(), "b".into()],
            values: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let spec = project_to_builder_spec(
            &pf(vec!["a", "b"], vec![]),
            &[
                VarType::Numeric {
                    name: "a".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
                VarType::Numeric {
                    name: "b".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
            ],
            &[],
            &corr,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        assert!(spec.correlations.is_empty());
    }

    // TestSelection maps All→["overall"], Effects→names, Contrasts→names.
    #[test]
    fn project_maps_test_selection_to_targets() {
        let mk = |tests: TestSelection| {
            project_to_builder_spec(
                &pf(vec!["x1"], vec![]),
                &[VarType::Numeric {
                    name: "x1".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                }],
                &[],
                &None,
                CorrectionMethod::None,
                &tests,
                0.05,
                None,
            )
            .expect("project ok")
            .targets
        };
        assert_eq!(mk(TestSelection::All), vec!["overall".to_string()]);
        assert_eq!(
            mk(TestSelection::Effects {
                names: vec!["x1".into(), "x2".into()]
            }),
            vec!["x1".to_string(), "x2".to_string()]
        );
        assert_eq!(
            mk(TestSelection::Contrasts {
                names: vec!["c1".into()]
            }),
            vec!["c1".to_string()]
        );
    }

    // The 0-based `factor_reference` index selects which `1..=k` level is the
    // dropped baseline: index 1 → level "2", with the full level list intact.
    #[test]
    fn factor_reference_index_selects_baseline_level() {
        let spec = project_to_builder_spec(
            &pf(vec!["g"], vec![]),
            &[VarType::Factor {
                name: "g".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 1,
                factor_labels: vec![],
                sampled_proportions: None,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        match &spec.predictors[0].kind {
            VarKind::Factor {
                levels, reference, ..
            } => {
                assert_eq!(
                    levels,
                    &vec!["1".to_string(), "2".to_string(), "3".to_string()]
                );
                assert_eq!(
                    reference, "2",
                    "factor_reference index 1 → baseline level \"2\""
                );
            }
            other => panic!("expected Factor, got {other:?}"),
        }
    }

    // A reference index beyond the level count is rejected with a clear error,
    // not a downstream kernel/contract failure.
    #[test]
    fn factor_reference_out_of_range_errors() {
        let err = project_to_builder_spec(
            &pf(vec!["g"], vec![]),
            &[VarType::Factor {
                name: "g".into(),
                factor_n_levels: 2,
                factor_proportions: vec![0.5, 0.5],
                factor_reference: 5,
                factor_labels: vec![],
                sampled_proportions: None,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                AdapterError::FactorReferenceOutOfRange {
                    reference: 5,
                    n_levels: 2,
                    ..
                }
            ),
            "expected FactorReferenceOutOfRange, got {err:?}"
        );
    }

    // ── UI-overhaul wiring: distributions, labels, sampled shares, outcome knobs ──

    #[test]
    fn numeric_distribution_maps_to_var_kind() {
        let mk = |d: NumericDistribution| {
            project_to_builder_spec(
                &pf(vec!["x"], vec![]),
                &[VarType::Numeric {
                    name: "x".into(),
                    pinned: false,
                    distribution: d,
                }],
                &[],
                &None,
                CorrectionMethod::None,
                &TestSelection::All,
                0.05,
                None,
            )
            .expect("project ok")
            .predictors[0]
                .kind
                .clone()
        };
        assert!(matches!(mk(NumericDistribution::Normal), VarKind::Normal));
        assert!(matches!(
            mk(NumericDistribution::RightSkewed),
            VarKind::RightSkewed
        ));
        assert!(matches!(
            mk(NumericDistribution::LeftSkewed),
            VarKind::LeftSkewed
        ));
        assert!(matches!(
            mk(NumericDistribution::HighKurtosis),
            VarKind::HighKurtosis
        ));
        assert!(matches!(mk(NumericDistribution::Uniform), VarKind::Uniform));
    }

    // User labels become the engine's level names (and thus its effect names);
    // the reference index selects among the labels, and the per-factor
    // sampled-shares override passes through.
    #[test]
    fn factor_labels_become_engine_levels_and_reference() {
        let spec = project_to_builder_spec(
            &pf(vec!["origin"], vec![]),
            &[VarType::Factor {
                name: "origin".into(),
                factor_n_levels: 3,
                factor_proportions: vec![0.4, 0.3, 0.3],
                factor_reference: 1,
                factor_labels: vec!["Europe".into(), "Japan".into(), "USA".into()],
                sampled_proportions: Some(true),
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        match &spec.predictors[0].kind {
            VarKind::Factor {
                levels,
                reference,
                sampled_proportions,
                ..
            } => {
                assert_eq!(
                    levels,
                    &vec!["Europe".to_string(), "Japan".to_string(), "USA".to_string()]
                );
                assert_eq!(reference, "Japan");
                assert_eq!(*sampled_proportions, Some(true));
            }
            other => panic!("expected Factor, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_factor_labels_rejected() {
        let err = project_to_builder_spec(
            &pf(vec!["g"], vec![]),
            &[VarType::Factor {
                name: "g".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 0,
                factor_labels: vec!["A".into(), "B".into(), "A".into()],
                sampled_proportions: None,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, AdapterError::DuplicateFactorLabel { ref label, .. } if label == "A"),
            "expected DuplicateFactorLabel, got {err:?}"
        );
    }

    // apply_outcome_options: residual/heterogeneity always apply; the
    // heteroskedasticity block resolves the driver among NON-FACTOR predictors
    // (same index space as random slopes).
    #[test]
    fn outcome_options_apply_to_builder_spec() {
        let var_types = [
            VarType::Factor {
                name: "f".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 0,
                factor_labels: vec![],
                sampled_proportions: None,
            },
            VarType::Numeric {
                name: "x".into(),
                pinned: false,
                distribution: NumericDistribution::Normal,
            },
        ];
        let mut builder_spec = project_to_builder_spec(
            &pf(vec!["f", "x"], vec![]),
            &var_types,
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        let opts = OutcomeOptions {
            residual_distribution: Some("high_kurtosis".into()),
            heteroskedasticity_driver: Some("x".into()),
        };
        apply_outcome_options(&mut builder_spec, Some(&opts), &var_types).expect("apply ok");
        assert_eq!(builder_spec.residual.distribution, "high_kurtosis");
        assert!(builder_spec.residual.pinned, "explicit choice must pin");
        // x is formula position 1 but non-factor position 0.
        assert_eq!(builder_spec.heteroskedasticity.driver_var_index, Some(0));
    }

    // An explicit "normal" is a pin, not a neutral value — it must land on
    // the builder spec as pinned (scenarios leave it alone).
    #[test]
    fn explicit_normal_residual_is_pinned() {
        let var_types = [VarType::Numeric {
            name: "x".into(),
            pinned: false,
            distribution: NumericDistribution::Normal,
        }];
        let mut builder_spec = project_to_builder_spec(
            &pf(vec!["x"], vec![]),
            &var_types,
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        let opts = OutcomeOptions {
            residual_distribution: Some("normal".into()),
            ..Default::default()
        };
        apply_outcome_options(&mut builder_spec, Some(&opts), &var_types).expect("apply ok");
        assert_eq!(builder_spec.residual.distribution, "normal");
        assert!(builder_spec.residual.pinned);
    }

    #[test]
    fn outcome_options_factor_driver_rejected() {
        let var_types = [VarType::Factor {
            name: "f".into(),
            factor_n_levels: 3,
            factor_proportions: vec![1.0 / 3.0; 3],
            factor_reference: 0,
            factor_labels: vec![],
            sampled_proportions: None,
        }];
        let mut builder_spec = project_to_builder_spec(
            &pf(vec!["f"], vec![]),
            &var_types,
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        let opts = OutcomeOptions {
            heteroskedasticity_driver: Some("f".into()),
            ..Default::default()
        };
        let err = apply_outcome_options(&mut builder_spec, Some(&opts), &var_types).unwrap_err();
        assert!(matches!(err, AdapterError::UnknownHeteroskedasticityDriver(ref n) if n == "f"));
    }

    // All-default options leave the scenario-driven heteroskedasticity block
    // untouched — a fixed λ would silently disable scenario λ perturbations.
    #[test]
    fn neutral_outcome_options_keep_scenario_driven_heteroskedasticity() {
        let var_types = [VarType::Numeric {
            name: "x".into(),
            pinned: false,
            distribution: NumericDistribution::Normal,
        }];
        let mut builder_spec = project_to_builder_spec(
            &pf(vec!["x"], vec![]),
            &var_types,
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None,
        )
        .expect("project ok");
        apply_outcome_options(
            &mut builder_spec,
            Some(&OutcomeOptions::default()),
            &var_types,
        )
        .expect("apply ok");
        assert_eq!(builder_spec.heteroskedasticity.driver_var_index, None);
        assert_eq!(builder_spec.residual.distribution, "normal");
        assert!(
            !builder_spec.residual.pinned,
            "absent dist must stay unpinned"
        );
    }

    // icc_to_tau_squared is positive for icc>0 and clamps to 0.0
    // (no NaN, no panic) when the denominator 1-icc is at or below zero.
    #[test]
    fn icc_to_tau_squared_positive_for_valid_icc() {
        assert!(icc_to_tau_squared(0.2) > 0.0);
        assert!(icc_to_tau_squared(0.5).is_finite());
    }

    #[test]
    fn icc_to_tau_squared_clamps_degenerate_denominator_to_zero() {
        assert_eq!(icc_to_tau_squared(1.0), 0.0);
        assert_eq!(icc_to_tau_squared(1.5), 0.0);
        assert!(icc_to_tau_squared(1.0).is_finite());
    }

    // ── Upload class-conflict guard ────────────────────────────────────────

    use engine_spec_builder::input::{UploadColumn, UploadColumnType, UploadMode};

    fn make_csv(col_name: &str, col_type: UploadColumnType) -> CsvData {
        CsvData {
            mode: UploadMode::Strict,
            n_rows: 3,
            columns: vec![UploadColumn {
                name: col_name.into(),
                col_type,
                values: vec![0.0, 1.0, 2.0],
                labels: vec![],
            }],
        }
    }

    // Declared binary, column detected continuous → UploadClassConflict naming the column.
    #[test]
    fn upload_class_conflict_binary_declared_continuous_detected() {
        let csv = make_csv("x1", UploadColumnType::Continuous);
        let err = project_to_builder_spec(
            &pf(vec!["x1"], vec![]),
            &[VarType::Binary {
                name: "x1".into(),
                binary_proportion: 0.5,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            Some(&csv),
        )
        .unwrap_err();
        match err {
            AdapterError::UploadClassConflict { ref name, .. } => {
                assert_eq!(name, "x1");
            }
            other => panic!("expected UploadClassConflict, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("x1"), "error message should name the column");
        assert!(msg.contains("continuous"), "should mention detected class");
        assert!(msg.contains("binary"), "should mention declared class");
    }

    // Declared factor, column detected binary → UploadClassConflict.
    #[test]
    fn upload_class_conflict_factor_declared_binary_detected() {
        let csv = make_csv("grp", UploadColumnType::Binary);
        let err = project_to_builder_spec(
            &pf(vec!["grp"], vec![]),
            &[VarType::Factor {
                name: "grp".into(),
                factor_n_levels: 2,
                factor_proportions: vec![0.5, 0.5],
                factor_reference: 0,
                factor_labels: vec![],
                sampled_proportions: None,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            Some(&csv),
        )
        .unwrap_err();
        match err {
            AdapterError::UploadClassConflict { ref name, .. } => {
                assert_eq!(name, "grp");
            }
            other => panic!("expected UploadClassConflict, got {other:?}"),
        }
    }

    // Declared binary, column detected binary → no error (matching class).
    #[test]
    fn upload_class_no_conflict_binary_matches_binary() {
        let csv = make_csv("x1", UploadColumnType::Binary);
        project_to_builder_spec(
            &pf(vec!["x1"], vec![]),
            &[VarType::Binary {
                name: "x1".into(),
                binary_proportion: 0.5,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            Some(&csv),
        )
        .expect("matching binary class should not conflict");
    }

    // Declared Numeric (continuous subtype), column detected continuous → no error.
    #[test]
    fn upload_class_no_conflict_numeric_declared_continuous_detected() {
        let csv = make_csv("x1", UploadColumnType::Continuous);
        project_to_builder_spec(
            &pf(vec!["x1"], vec![]),
            &[VarType::Numeric {
                name: "x1".into(),
                distribution: NumericDistribution::Normal,
                pinned: false,
            }],
            &[],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            Some(&csv),
        )
        .expect("Numeric declared against detected continuous should not conflict");
    }

    // ── Undeclared upload columns: detected type wins (data wins) ───────────

    use engine_contract::{ColumnSpec, DesignTerm, OutcomeKind};
    use engine_spec_builder::build_contract;

    /// 3-row binary CSV column with 0/1 values (empirical proportion ≈ 0.33).
    fn make_binary_csv(col_name: &str) -> CsvData {
        CsvData {
            mode: UploadMode::Strict,
            n_rows: 3,
            columns: vec![UploadColumn {
                name: col_name.into(),
                col_type: UploadColumnType::Binary,
                values: vec![0.0, 0.0, 1.0],
                labels: vec![],
            }],
        }
    }

    /// 3-row factor CSV column with level codes 0..2 and three labels.
    fn make_factor_csv(col_name: &str) -> CsvData {
        CsvData {
            mode: UploadMode::Strict,
            n_rows: 3,
            columns: vec![UploadColumn {
                name: col_name.into(),
                col_type: UploadColumnType::Factor,
                values: vec![0.0, 1.0, 2.0],
                labels: vec!["A".into(), "B".into(), "C".into()],
            }],
        }
    }

    /// Project an undeclared single-predictor formula against `csv`, supplying the
    /// matching effect assignments, then build the OLS contract. Returns the lone
    /// contract so callers can assert its built column/design shape.
    fn build_undeclared(csv: &CsvData, effects: &[EffectSize]) -> SimulationContract {
        let pred = csv.columns[0].name.clone();
        let builder_spec = project_to_builder_spec(
            &pf(vec![&pred], vec![]),
            &[], // no declared var_types — the undeclared path
            effects,
            &None,
            CorrectionMethod::None,
            &TestSelection::Effects {
                names: effects.iter().map(|e| e.name.clone()).collect(),
            },
            0.05,
            Some(csv),
        )
        .expect("undeclared upload column must project without a conflict error");
        build_contract(&builder_spec, OutcomeKind::Continuous, None, 0.0, vec![])
            .expect("undeclared upload column must build a valid contract")
            .into_iter()
            .next()
            .expect("one contract")
    }

    // Undeclared predictor + column detected Binary → no error, and it builds as
    // a data-backed binary column (`ResampledBinary`) with a `Direct` design term
    // (binary predictors are non-factors). NOT a synthetic continuous column.
    #[test]
    fn undeclared_binary_upload_builds_as_resampled_binary() {
        let csv = make_binary_csv("x1");
        let c = build_undeclared(
            &csv,
            &[EffectSize {
                name: "x1".into(),
                value: 0.3,
            }],
        );
        // Single non-factor column, backed by the uploaded binary data.
        assert_eq!(c.generation.columns.len(), 1);
        assert!(
            matches!(c.generation.columns[0], ColumnSpec::ResampledBinary { .. }),
            "undeclared binary upload must build a ResampledBinary column, got {:?}",
            c.generation.columns[0]
        );
        // design_generation: [Const, Direct{col 0}] — binary is a non-factor, so a
        // Direct term is correct (and does NOT trip DirectOnFactor).
        assert!(
            matches!(c.design_generation.terms[1], DesignTerm::Direct { .. }),
            "binary predictor must use a Direct design term, got {:?}",
            c.design_generation.terms[1]
        );
        c.validate().expect("contract must self-validate");
    }

    // Undeclared predictor + column detected Factor → no error, and it builds as a
    // data-backed factor column (`FactorFromFrame`) with `DummyOf` design terms
    // (NOT a `Direct` term that would later trip `DirectOnFactor`).
    #[test]
    fn undeclared_factor_upload_builds_as_factor_dummies() {
        let csv = make_factor_csv("grp");
        // 3-level factor → effects on the two non-reference dummies grp[2], grp[3].
        let c = build_undeclared(
            &csv,
            &[
                EffectSize {
                    name: "grp[2]".into(),
                    value: 0.3,
                },
                EffectSize {
                    name: "grp[3]".into(),
                    value: 0.5,
                },
            ],
        );
        // Single column placed in the factor region, backed by uploaded data.
        assert_eq!(c.generation.columns.len(), 1);
        assert!(
            matches!(
                c.generation.columns[0],
                ColumnSpec::FactorFromFrame { n_levels: 3, .. }
            ),
            "undeclared factor upload must build a FactorFromFrame(3) column, got {:?}",
            c.generation.columns[0]
        );
        // design_generation: [Const, DummyOf{col0,1}, DummyOf{col0,2}] — k-1 dummies
        // for a 3-level factor, and crucially NO Direct term referencing the factor
        // column (which would trip ContractError::DirectOnFactor).
        let n_dummyof = c
            .design_generation
            .terms
            .iter()
            .filter(|t| matches!(t, DesignTerm::DummyOf { .. }))
            .count();
        assert_eq!(n_dummyof, 2, "3-level factor must yield 2 DummyOf terms");
        assert!(
            !c.design_generation.terms.iter().any(|t| matches!(
                t,
                DesignTerm::Direct { column } if *column == engine_contract::ColumnId(0)
            )),
            "factor column 0 must NOT have a Direct term (would trip DirectOnFactor)"
        );
        c.validate().expect("contract must self-validate");
    }

    // ── Scenario fan-out (B3) ────────────────────────────────────────────────

    fn neutral_scenario(name: &str) -> engine_spec_builder::input::ScenarioInput {
        engine_spec_builder::input::ScenarioInput {
            name: name.into(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        }
    }

    fn linear_app_spec(scenarios: Vec<engine_spec_builder::input::ScenarioInput>) -> AppSpec {
        AppSpec::Linear(LinearSpec {
            parsed_formula: pf(vec!["x1"], vec![]),
            var_types: vec![VarType::Numeric {
                name: "x1".into(),
                distribution: NumericDistribution::Normal,
                pinned: false,
            }],
            effects: vec![EffectSize {
                name: "x1".into(),
                value: 0.3,
            }],
            correlations: None,
            alpha: 0.05,
            target_power: 0.8,
            n_sims: 100,
            seed: 2137,
            tests: TestSelection::All,
            correction: CorrectionMethod::None,
            scenarios,
            csv: None,
            report_overall: false,
            contrasts: vec![],
            test_formula: None,
            outcome_options: None,
        })
    }

    // A declared VarType::Factor with no upload (csv: None) must build as
    // FactorSynthetic — not Synthetic (continuous) or FactorFromFrame (upload-backed).
    // This pins the type mapping; a Factor→Continuous regression would only fail if
    // the snapshot count changed, so we assert the variant directly.
    #[test]
    fn declared_factor_no_upload_assembles_as_factor_synthetic() {
        use engine_contract::ColumnSpec;
        use engine_spec_builder::build_contract;
        let spec = project_to_builder_spec(
            &pf(vec!["g"], vec![]),
            &[VarType::Factor {
                name: "g".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 0,
                factor_labels: vec![],
                sampled_proportions: None,
            }],
            &[
                EffectSize {
                    name: "g[2]".into(),
                    value: 0.2,
                },
                EffectSize {
                    name: "g[3]".into(),
                    value: 0.4,
                },
            ],
            &None,
            CorrectionMethod::None,
            &TestSelection::All,
            0.05,
            None, // no upload — synthetic path
        )
        .expect("project ok");
        let contracts = build_contract(
            &spec,
            engine_contract::OutcomeKind::Continuous,
            None,
            0.0,
            vec![],
        )
        .expect("build ok");
        let c = contracts.into_iter().next().expect("one contract");
        assert_eq!(c.generation.columns.len(), 1);
        assert!(
            matches!(
                c.generation.columns[0],
                ColumnSpec::FactorSynthetic { n_levels: 3, .. }
            ),
            "declared Factor with no upload must build FactorSynthetic(3), got {:?}",
            c.generation.columns[0]
        );
    }

    #[test]
    fn n_scenarios_assemble_to_n_contracts() {
        let spec = linear_app_spec(vec![
            neutral_scenario("optimistic"),
            neutral_scenario("realistic"),
            neutral_scenario("doomer"),
        ]);
        let contracts = assemble_spec(&spec).expect("assemble ok");
        assert_eq!(contracts.len(), 3, "three scenarios → three contracts");
    }

    // ── Mixed growth fields: cluster_level_vars / extra_groupings / slopes ────

    use crate::app_spec::{
        AppGroupingRelation, AppGroupingSpec, AppSlopeTerm, ClusterDim, MixedOutcome, MixedSpec,
    };

    /// Minimal `y ~ x1 + x2 + (1|school)` Mixed spec with empty growth fields.
    /// Tests inject the one field under test, keeping the construction DRY.
    fn mixed_fixture() -> MixedSpec {
        MixedSpec {
            parsed_formula: pf(vec!["x1", "x2"], vec![]),
            var_types: vec![
                VarType::Numeric {
                    name: "x1".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
                VarType::Numeric {
                    name: "x2".into(),
                    distribution: NumericDistribution::Normal,
                    pinned: false,
                },
            ],
            // One effect per design column — the builder rejects an effect-count
            // mismatch (EffectCountMismatch).
            effects: vec![
                EffectSize {
                    name: "x1".into(),
                    value: 0.3,
                },
                EffectSize {
                    name: "x2".into(),
                    value: 0.2,
                },
            ],
            correlations: None,
            alpha: 0.05,
            target_power: 0.8,
            n_sims: 100,
            seed: 2137,
            tests: TestSelection::All,
            correction: CorrectionMethod::None,
            wald_se: Default::default(),
            scenarios: vec![],
            csv: None,
            report_overall: false,
            contrasts: vec![],
            test_formula: None,
            outcome_options: None,
            cluster_name: "school".into(),
            icc: 0.2,
            cluster_dim: ClusterDim::NClusters { value: 20 },
            cluster_level_vars: vec![],
            extra_groupings: vec![],
            slopes: vec![],
            outcome: MixedOutcome::Gaussian,
        }
    }

    // A non-empty cluster_level_vars threads through assemble_mixed →
    // project_to_builder_spec → build_contract_with_skeleton without error.
    // Detailed cluster-level semantics are covered in engine-spec-builder tests.
    #[test]
    fn assemble_mixed_forwards_cluster_level_vars() {
        let mut spec = mixed_fixture();
        spec.cluster_level_vars = vec!["x2".into()];
        let (contracts, _) =
            assemble_mixed(&spec).expect("assemble_mixed with cluster_level_vars must not error");
        // x2 must land as a cluster-level column, not be silently dropped (was: is_ok only).
        assert_eq!(
            contracts[0].generation.cluster_level_columns.len(),
            1,
            "x2 must forward as one cluster-level column"
        );
    }

    // A crossed extra grouping (no slopes) passes the engine's grouping
    // invariants and builds. (Slopes on extra groupings are rejected by the
    // engine, not extra groupings themselves.)
    #[test]
    fn assemble_mixed_forwards_extra_groupings() {
        let mut spec = mixed_fixture();
        spec.extra_groupings = vec![AppGroupingSpec {
            tau_squared: 0.1,
            cluster_name: None,
            relation: AppGroupingRelation::Crossed { n_clusters: 8 },
        }];
        let (contracts, _) =
            assemble_mixed(&spec).expect("assemble_mixed with extra_groupings must succeed");
        // The crossed grouping must land on the contract's cluster (was: is_ok only).
        let cluster = contracts[0]
            .generation
            .cluster
            .as_ref()
            .expect("mixed contract carries a cluster");
        assert_eq!(
            cluster.extra_groupings.len(),
            1,
            "crossed grouping must forward"
        );
    }

    // ICC validation mirrors the Py/R hosts: 0 is allowed (no clustering), the
    // shared-config stability band [0.05, 0.95] is enforced, and icc >= 1 / icc < 0
    // are rejected up front rather than silently collapsing to tau_squared = 0.
    #[test]
    fn assemble_mixed_rejects_out_of_band_icc() {
        let mut spec = mixed_fixture();

        // 0 = no clustering: must not be rejected by the ICC guards.
        spec.icc = 0.0;
        assert!(!matches!(
            assemble_mixed(&spec),
            Err(AdapterError::IccOutOfRange { .. } | AdapterError::IccOutOfStabilityBand { .. })
        ));

        // Below / above the stability band → band error.
        spec.icc = 0.02;
        assert!(matches!(
            assemble_mixed(&spec),
            Err(AdapterError::IccOutOfStabilityBand { .. })
        ));
        spec.icc = 0.97;
        assert!(matches!(
            assemble_mixed(&spec),
            Err(AdapterError::IccOutOfStabilityBand { .. })
        ));

        // Outside the hard (0, 1) range → range error.
        spec.icc = 1.5;
        assert!(matches!(
            assemble_mixed(&spec),
            Err(AdapterError::IccOutOfRange { .. })
        ));
        spec.icc = -0.1;
        assert!(matches!(
            assemble_mixed(&spec),
            Err(AdapterError::IccOutOfRange { .. })
        ));
    }

    // A random slope on a known predictor builds without error.
    #[test]
    fn assemble_mixed_forwards_slopes() {
        let mut spec = mixed_fixture();
        spec.slopes = vec![AppSlopeTerm {
            predictor_name: "x1".into(),
            slope_variance: 0.05,
            slope_intercept_corr: -0.2,
        }];
        let (contracts, _) =
            assemble_mixed(&spec).expect("assemble_mixed with slopes must not error");
        // The slope on x1 must land on the cluster with its variance (was: is_ok only).
        // The slope's ColumnId mapping is pinned separately by
        // slope_column_id_is_non_factor_position_not_formula_position.
        let cluster = contracts[0]
            .generation
            .cluster
            .as_ref()
            .expect("mixed contract carries a cluster");
        assert_eq!(cluster.slopes.len(), 1, "slope on x1 must forward");
        assert!(
            (cluster.slopes[0].variance - 0.05).abs() < 1e-12,
            "slope variance forwarded: {}",
            cluster.slopes[0].variance
        );
    }

    // A slope naming a predictor not in the model is a named adapter error, not
    // a downstream ColumnId-keyed kernel failure.
    #[test]
    fn assemble_mixed_unknown_slope_predictor_errors() {
        let mut spec = mixed_fixture();
        spec.slopes = vec![AppSlopeTerm {
            predictor_name: "nonexistent".into(),
            slope_variance: 0.05,
            slope_intercept_corr: 0.0,
        }];
        assert!(
            matches!(
                assemble_mixed(&spec),
                Err(AdapterError::UnknownPredictor(_))
            ),
            "unknown slope predictor must error"
        );
    }

    // Cross-port column-resolution tripwire (mirrors the Python and R ports):
    // y ~ f + x + (1 + x|g) with a FACTOR `f` declared BEFORE the continuous slope
    // var `x`. The raw formula position of x is 1, but its non-factor generation
    // column is 0 — the slope must bind to ColumnId(0). A regression here (binding
    // to formula position 1, the factor's column) would diverge from Py/R and trip
    // a slope-on-factor invariant.
    #[test]
    fn slope_column_id_is_non_factor_position_not_formula_position() {
        let mut spec = mixed_fixture();
        spec.parsed_formula = pf(vec!["f", "x"], vec![]);
        spec.var_types = vec![
            VarType::Factor {
                name: "f".into(),
                factor_n_levels: 3,
                factor_proportions: vec![1.0 / 3.0; 3],
                factor_reference: 0,
                factor_labels: vec![],
                sampled_proportions: None,
            },
            VarType::Numeric {
                name: "x".into(),
                distribution: NumericDistribution::Normal,
                pinned: false,
            },
        ];
        // One effect per design column: the two non-reference factor dummies
        // (f[2], f[3]) plus the continuous x.
        spec.effects = vec![
            EffectSize {
                name: "f[2]".into(),
                value: 0.2,
            },
            EffectSize {
                name: "f[3]".into(),
                value: 0.2,
            },
            EffectSize {
                name: "x".into(),
                value: 0.3,
            },
        ];
        spec.slopes = vec![AppSlopeTerm {
            predictor_name: "x".into(),
            slope_variance: 0.05,
            slope_intercept_corr: 0.0,
        }];
        let (contracts, _) = assemble_mixed(&spec).expect("assemble ok");
        let cluster = contracts[0]
            .generation
            .cluster
            .as_ref()
            .expect("mixed contract has a cluster");
        assert_eq!(
            cluster.slopes[0].column,
            engine_contract::ColumnId(0),
            "slope must bind to x's non-factor generation column (0), not its formula position (1)"
        );
    }

    // ── MixedOutcome: binary path sets Binary + Glm + logit intercept + latent τ² ──

    #[test]
    fn assemble_mixed_binary_outcome_sets_binary_glm_and_logit_scale_tau() {
        use engine_contract::{EstimatorSpec, OutcomeKind};

        let p = 0.3_f64;
        let icc = 0.2_f64;
        let mut spec = mixed_fixture();
        spec.icc = icc;
        spec.outcome = MixedOutcome::Binary {
            baseline_probability: p,
        };

        let (contracts, _) = assemble_mixed(&spec).expect("binary mixed assembles ok");
        let c = &contracts[0];

        assert_eq!(c.outcome.kind, OutcomeKind::Binary, "binary outcome");
        assert_eq!(c.estimator, EstimatorSpec::Glm, "binary mixed must use Glm");

        let expected_intercept = (p / (1.0 - p)).ln();
        assert!(
            (c.outcome.intercept - expected_intercept).abs() < 1e-10,
            "intercept = logit(p)"
        );

        let expected_tau = icc / (1.0 - icc) * std::f64::consts::PI.powi(2) / 3.0;
        let cluster = c.generation.cluster.as_ref().expect("cluster present");
        assert!(
            (cluster.tau_squared - expected_tau).abs() < 1e-10,
            "τ² latent logistic scale"
        );
    }

    #[test]
    fn assemble_mixed_gaussian_outcome_is_unchanged() {
        use engine_contract::{EstimatorSpec, OutcomeKind};

        let icc = 0.2_f64;
        let mut spec = mixed_fixture();
        spec.icc = icc;
        spec.outcome = MixedOutcome::Gaussian;

        let (contracts, _) = assemble_mixed(&spec).expect("gaussian mixed assembles ok");
        let c = &contracts[0];
        assert_eq!(c.outcome.kind, OutcomeKind::Continuous);
        assert_eq!(c.estimator, EstimatorSpec::Mle);
        assert!((c.outcome.intercept - 0.0).abs() < 1e-12);
        let expected_tau = icc / (1.0 - icc);
        let cluster = c.generation.cluster.as_ref().expect("cluster present");
        assert!((cluster.tau_squared - expected_tau).abs() < 1e-10);
    }

    // ── icc_to_tau_squared_logit unit tests ──

    #[test]
    fn icc_to_tau_squared_logit_scales_by_pi_squared_over_3() {
        let icc = 0.2_f64;
        let ratio = icc_to_tau_squared_logit(icc) / icc_to_tau_squared(icc);
        assert!((ratio - std::f64::consts::PI.powi(2) / 3.0).abs() < 1e-10);
    }

    #[test]
    fn icc_to_tau_squared_logit_clamps_degenerate_denominator() {
        assert_eq!(icc_to_tau_squared_logit(1.0), 0.0);
        assert!(icc_to_tau_squared_logit(1.5).is_finite());
    }

    // ── logit / logistic are exact inverses (baseline round-trip) ──

    #[test]
    fn logistic_is_inverse_of_logit() {
        // Baseline-from-data recovery (logistic) must exactly undo the forward
        // baseline→intercept conversion (logit); drift here corrupts the round-trip.
        for &p in &[0.01_f64, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let round_trip = logistic(logit(p));
            assert!(
                (round_trip - p).abs() < 1e-12,
                "logistic(logit({p})) = {round_trip}, expected {p}"
            );
        }
        // Spot-check the absolute scale: logit(0.5) == 0 → logistic(0) == 0.5.
        assert!((logistic(0.0) - 0.5).abs() < 1e-12);
    }

    // ── Task 3.3 passthrough: binary MixedSpec round-trips through assemble_spec ──

    #[test]
    fn binary_mixed_spec_round_trips_to_binary_glm_contract() {
        use crate::app_spec::AppSpec;
        use engine_contract::{EstimatorSpec, OutcomeKind};

        let p = 0.25_f64;
        let icc = 0.15_f64;
        let mut spec = mixed_fixture();
        spec.icc = icc;
        spec.outcome = MixedOutcome::Binary {
            baseline_probability: p,
        };

        let contracts = assemble_spec(&AppSpec::Mixed(spec)).expect("binary mixed assembles");
        assert_eq!(contracts.len(), 1);
        let c = &contracts[0];
        assert_eq!(c.outcome.kind, OutcomeKind::Binary);
        assert_eq!(c.estimator, EstimatorSpec::Glm);
        let expected_intercept = (p / (1.0 - p)).ln();
        assert!((c.outcome.intercept - expected_intercept).abs() < 1e-10);
        let expected_tau = icc_to_tau_squared_logit(icc);
        let cluster = c.generation.cluster.as_ref().expect("contract has cluster");
        assert!((cluster.tau_squared - expected_tau).abs() < 1e-10);
    }

    #[test]
    fn binary_mixed_spec_with_scenarios_assembles_to_n_contracts() {
        use crate::app_spec::AppSpec;

        let mut spec = mixed_fixture();
        spec.outcome = MixedOutcome::Binary {
            baseline_probability: 0.3,
        };
        // mixed_fixture has no scenarios, so this produces one baseline contract.
        let contracts = assemble_spec(&AppSpec::Mixed(spec)).expect("assembles ok");
        for c in &contracts {
            assert_eq!(c.outcome.kind, engine_contract::OutcomeKind::Binary);
        }
    }
}
