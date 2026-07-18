//! Projects a validated `LinearSpec` into `Vec<SimulationContract>`; the single
//! source of truth for the `LinearSpec → contract` mapping across all ports.

use engine_contract::{
    ColumnId, ColumnSpec, Correlations, DesignSpec, DesignTerm, EstimatorSpec, GenerationSpec,
    LmeScenarioPerturbations, OutcomeKind, OutcomeSpec, PosthocSpec, ResidualDist,
    ResidualSpec as ContractResidual, ScenarioPerturbations as ContractScenario,
    SimulationContract, SyntheticKind, TestSpec, TestTarget,
};

use crate::correlation::build_correlation_matrix;
use crate::error::SpecError;
use crate::formula::parse_formula;
use crate::input::{
    HeteroskedasticityInput, LinearSpec, PosthocRequest, ScenarioInput, UploadMode,
};
use crate::skeleton::{EffectDescriptor, EffectSkeleton};
use crate::targets::resolve_targets;
use crate::upload::{apply_upload, measure_correlations};
use crate::validate::{validate_effect_assignments, validate_pre_projection};
use crate::variables::{build_predictor_table, PredictorTable};

/// Thin wrapper that drops the `EffectSkeleton` for callers that only need the
/// contracts. The lowering lives once in `build_linear_contract_with_skeleton`.
pub fn build_linear_contract(spec: &LinearSpec) -> Result<Vec<SimulationContract>, SpecError> {
    Ok(build_linear_contract_with_skeleton(spec)?.0)
}

/// Project a validated `LinearSpec` into `Vec<SimulationContract>` **and** the
/// `EffectSkeleton` (β-column-aligned, single source of the result-naming
/// layout). The skeleton is built from the test design when a `test_formula`
/// trims the model — the same table `build_target_terms` resolves against — so
/// `skeleton[target_indices[i]]` names the right effect.
///
/// # Errors
/// Propagates errors from validation, formula parsing, effect/target/correlation resolution,
/// and internal contract validation (`InternalContractValidate`).
pub fn build_linear_contract_with_skeleton(
    spec: &LinearSpec,
) -> Result<(Vec<SimulationContract>, EffectSkeleton), SpecError> {
    validate_pre_projection(spec)?;
    let parsed = parse_formula(&spec.formula)?;
    let table = build_predictor_table(&parsed, &spec.predictors)?;
    validate_effect_assignments(spec, &table.effect_names)?;

    let columns = build_columns(&table);
    let cluster_level_columns = resolve_cluster_level_columns(&table, &spec.cluster_level_vars)?;
    let correlations = build_contract_correlations(&table, spec)?;
    let design = build_design(&table)?;
    let coefficients = build_coefficients(&table, spec)?;
    let (design_test, test_table) = build_design_test(spec, &table, &design)?;
    // design_test here is the generation-design clone when test_formula is None
    // (correctly-specified); the contract field is set to None separately at the struct literal.
    let target_terms = build_target_terms(spec, &table, test_table.as_ref(), &design_test)?;
    let posthoc = build_posthoc_specs(&spec.posthoc_requests, &table, &design_test)?;
    let heteroskedasticity_driver =
        build_contract_heteroskedasticity_driver(&spec.heteroskedasticity, &table)?;
    let residual = build_contract_residual(&spec.residual)?;

    let scenarios: Vec<ContractScenario> = if spec.scenarios.is_empty() {
        vec![ContractScenario {
            name: "optimistic".into(),
            ..Default::default()
        }]
    } else {
        spec.scenarios
            .iter()
            .map(scenario_to_contract)
            .collect::<Result<Vec<_>, SpecError>>()?
    };

    let predictor_names: Vec<String> = table
        .non_factor_names
        .iter()
        .chain(table.factor_names.iter())
        .cloned()
        .collect();

    let mut generation = GenerationSpec {
        columns,
        correlations,
        cluster: None,
        uploaded_frame: None,
        cluster_level_columns,
    };

    // Apply uploaded frame data when present.
    if let Some(upload) = &spec.upload {
        let non_factor_std = apply_upload(
            &mut generation,
            &predictor_names,
            table.non_factor_names.len(),
            upload,
        );
        // Partial mode: measure empirical Spearman rank correlations (converted to
        // the latent-Gaussian scale), then overlay explicit pairs.
        // Strict mode flows through as a bootstrap frame (apply_upload sets
        // `uploaded_frame.bootstrap = true`); it intentionally skips correlation
        // measurement — whole-row resampling preserves the joint distribution
        // directly, so a measured copula is neither needed nor applied.
        if matches!(upload.mode, UploadMode::Partial) {
            measure_correlations(
                &mut generation,
                &table.non_factor_names,
                &spec.correlations,
                &non_factor_std,
            )?;
        }
    }

    let template = SimulationContract {
        generation,
        design_generation: design,
        outcome: OutcomeSpec {
            kind: OutcomeKind::Continuous, // overwritten by build_contract
            intercept: 0.0,                // overwritten by build_contract
            coefficients,
            residual,
            heteroskedasticity_driver,
            link: None,
        },
        design_test: if spec.test_formula.is_some() {
            Some(design_test)
        } else {
            None
        },
        estimator: EstimatorSpec::Ols, // overwritten by build_contract
        wald_se: spec.wald_se,
        nagq: spec.nagq,
        test: TestSpec {
            targets: target_terms,
            correction: spec.correction,
            alpha: spec.alpha,
        },
        posthoc,
        scenario: ContractScenario::default(),
        max_failed_fraction: spec.max_failed_fraction,
    };

    let mut out = Vec::with_capacity(scenarios.len());
    for sc in scenarios {
        let mut c = template.clone();
        c.scenario = sc;
        c.validate_template()
            .map_err(|e| SpecError::InternalContractValidate(e.to_string()))?;
        out.push(c);
    }
    // The skeleton aligns with `target_indices` (which index `design_test`), so
    // build it from the test table when a `test_formula` is set, else the
    // generation table — exactly what `build_target_terms` resolves against.
    let skeleton = build_effect_skeleton(test_table.as_ref().unwrap_or(&table));
    Ok((out, skeleton))
}

/// Build `Vec<SimulationContract>` from a project input + per-call
/// outcome_kind, optional estimator (None = default coupling), intercept,
/// and (optional) clusters. Single source of truth for the contract surface;
/// no post-hoc patching from any host.
///
/// Default coupling (when `estimator` is None):
/// - `outcome_kind == Binary | Count` ⇒ `Glm`
/// - `!clusters.is_empty()`           ⇒ `Mle`
/// - else                             ⇒ `Ols`
///
/// Invariants:
/// - `estimator == Mle` ⇒ `clusters.len() == 1`.
/// - At most one ClusterSpec (random intercept only; random slopes are rejected).
///
/// Thin wrapper that drops the `EffectSkeleton` for callers that only need the
/// contracts. The lowering lives once in `build_contract_with_skeleton`.
///
/// # Errors
/// Same as [`build_contract_with_skeleton`].
pub fn build_contract(
    spec: &LinearSpec,
    outcome_kind: engine_contract::OutcomeKind,
    link: Option<engine_contract::LinkKind>,
    estimator: Option<engine_contract::EstimatorSpec>,
    intercept: f64,
    clusters: Vec<engine_contract::ClusterSpec>,
) -> Result<Vec<engine_contract::SimulationContract>, SpecError> {
    Ok(build_contract_with_skeleton(spec, outcome_kind, link, estimator, intercept, clusters)?.0)
}

/// As [`build_contract`], but also returns the `EffectSkeleton` so a host can
/// name results without re-deriving the factor-expansion layout. The skeleton
/// is identical across the returned contracts (one design per analysis).
///
/// # Errors
/// Propagates from validation, formula parsing, cluster/estimator coupling (`ClusterFamilyMismatch`),
/// random-slope rejection (`RandomSlopesUnsupported`), and internal contract validation.
pub fn build_contract_with_skeleton(
    spec: &LinearSpec,
    outcome_kind: engine_contract::OutcomeKind,
    link: Option<engine_contract::LinkKind>,
    estimator: Option<engine_contract::EstimatorSpec>,
    intercept: f64,
    clusters: Vec<engine_contract::ClusterSpec>,
) -> Result<(Vec<engine_contract::SimulationContract>, EffectSkeleton), SpecError> {
    use engine_contract::{EstimatorSpec, OutcomeKind};
    // Cluster cap: at most one cluster spec.
    if clusters.len() > 1 {
        return Err(SpecError::ClusterFamilyMismatch);
    }
    // Default coupling: derive the matching estimator when the caller didn't state one.
    // Binary (logit/probit) and Count (Poisson) fit via GLM/GLMM (Glm + cluster ⇒
    // GLMM by invariant_12_estimator_outcome_matrix); Continuous stays Mle (clustered LMM) or Ols.
    let estimator = estimator.unwrap_or(
        if matches!(outcome_kind, OutcomeKind::Binary | OutcomeKind::Count) {
            EstimatorSpec::Glm
        } else if !clusters.is_empty() {
            EstimatorSpec::Mle
        } else {
            EstimatorSpec::Ols
        },
    );
    // Mle must have exactly one cluster; everything else is gated by validate().
    if estimator == EstimatorSpec::Mle && clusters.len() != 1 {
        return Err(SpecError::ClusterFamilyMismatch);
    }
    let (mut contracts, skeleton) = build_linear_contract_with_skeleton(spec)?;

    // Only random intercepts are supported. Reject random-slope terms
    // explicitly so callers get a clear error instead of silently dropped
    // effects.
    // formula is re-parsed here for the random-effect walk; build_linear_contract's parse result is not returned.
    let parsed = parse_formula(&spec.formula)?;
    for re in &parsed.random_effects {
        if let crate::formula::RandomEffect::Slope { .. } = re {
            return Err(SpecError::RandomSlopesUnsupported);
        }
    }

    for c in &mut contracts {
        c.outcome.kind = outcome_kind;
        c.outcome.link = link;
        c.outcome.intercept = intercept;
        c.estimator = estimator;
        c.generation.cluster = clusters.first().cloned();
        c.validate()
            .map_err(|e| SpecError::InternalContractValidate(e.to_string()))?;
    }
    Ok((contracts, skeleton))
}

/// Resolve cluster-level predictor *names* to `ColumnId`s against the predictor
/// table. Non-factor names map to their 0-based position, factor names to
/// `n_non_factor + factor_index`, per the column-ordering invariant owned by
/// `variables.rs`. Mirrors the name resolution used by
/// `build_contract_heteroskedasticity` / `build_posthoc_specs`.
fn resolve_cluster_level_columns(
    table: &PredictorTable,
    names: &[String],
) -> Result<Vec<ColumnId>, SpecError> {
    let n_non_factor = table.non_factor_names.len() as u32;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if let Some(i) = table.non_factor_names.iter().position(|n| n == name) {
            out.push(ColumnId(i as u32));
        } else if let Some(fi) = table.factor_names.iter().position(|n| n == name) {
            out.push(ColumnId(n_non_factor + fi as u32));
        } else {
            return Err(SpecError::UnknownPredictor { name: name.clone() });
        }
    }
    Ok(out)
}

fn build_columns(table: &PredictorTable) -> Vec<ColumnSpec> {
    let mut cols: Vec<ColumnSpec> = Vec::new();
    for i in 0..table.non_factor_names.len() {
        let kind = match table.var_types[i] {
            0 => SyntheticKind::Normal,
            1 => SyntheticKind::Binary {
                p: table.var_params[i],
            },
            2 => SyntheticKind::RightSkewed,
            3 => SyntheticKind::LeftSkewed,
            4 => SyntheticKind::HighKurtosis,
            5 => SyntheticKind::Uniform,
            other => panic!("unhandled synthetic var_type code {other}"),
        };
        cols.push(ColumnSpec::Synthetic {
            kind,
            pinned: table.var_pinned.get(i).copied().unwrap_or(false),
        });
    }
    let mut prop_off = 0usize;
    for (factor_idx, &nl) in table.factor_n_levels.iter().enumerate() {
        let n_levels = nl.max(0) as u32;
        let take = n_levels as usize;
        let props = table.factor_proportions[prop_off..prop_off + take].to_vec();
        prop_off += take;
        cols.push(ColumnSpec::FactorSynthetic {
            n_levels,
            proportions: props,
            sampled_proportions: table.factor_sampled.get(factor_idx).copied().flatten(),
        });
    }
    cols
}

fn build_contract_correlations(
    table: &PredictorTable,
    spec: &LinearSpec,
) -> Result<Correlations, SpecError> {
    if spec.correlations.is_empty() {
        return Ok(Correlations::Identity);
    }
    // Correlation is continuous-only by design: reject any pair naming a binary
    // predictor (var_type code 1). Binary/factor variables are generated from
    // their marginals; their joint dependence is preserved only by strict-mode
    // data upload, not the synthetic/partial copula path. (Factors are not in
    // `non_factor_names` and already error as CorrelationUnknownVar.)
    for pair in &spec.correlations {
        for name in [&pair.a, &pair.b] {
            if let Some(idx) = table.non_factor_names.iter().position(|p| p == name) {
                if table.var_types[idx] == 1 {
                    return Err(SpecError::CorrelationNonContinuous { name: name.clone() });
                }
            }
        }
    }
    let flat = build_correlation_matrix(&table.non_factor_names, &spec.correlations)?;
    let n = table.non_factor_names.len();
    let continuous_columns: Vec<ColumnId> = (0..n as u32).map(ColumnId).collect();
    Ok(Correlations::Matrix {
        continuous_columns,
        values: flat,
    })
}

/// Resolve one interaction component `(predictor_name, level)` to a contract
/// `DesignTerm`, against the **generation** table's columns. Used by both
/// `build_design` (gen table == self) and `build_design_test` (remap by name).
fn component_to_design_term(
    gen_table: &PredictorTable,
    name: &str,
    level: &Option<String>,
) -> Result<DesignTerm, SpecError> {
    match level {
        None => {
            let idx = gen_table
                .non_factor_names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| SpecError::UnknownPredictor {
                    name: name.to_string(),
                })?;
            Ok(DesignTerm::Direct {
                column: ColumnId(idx as u32),
            })
        }
        Some(lvl) => {
            let gen_fi = gen_table
                .factor_names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| SpecError::UnknownPredictor {
                    name: name.to_string(),
                })?;
            let gen_column = ColumnId(gen_table.non_factor_names.len() as u32 + gen_fi as u32);
            let level_index = gen_table.factor_levels[gen_fi]
                .iter()
                .position(|l| l == lvl)
                .ok_or_else(|| SpecError::UnknownPredictor {
                    name: format!("{name}[{lvl}]"),
                })? as u32
                + 1;
            Ok(DesignTerm::DummyOf {
                column: gen_column,
                level_index,
            })
        }
    }
}

fn build_design(table: &PredictorTable) -> Result<DesignSpec, SpecError> {
    let mut terms = vec![DesignTerm::Const];
    for i in 0..table.non_factor_names.len() {
        terms.push(DesignTerm::Direct {
            column: ColumnId(i as u32),
        });
    }
    let factor_offset = table.non_factor_names.len() as u32;
    for (fi, &nl) in table.factor_n_levels.iter().enumerate() {
        let col = ColumnId(factor_offset + fi as u32);
        for level_index in 1..(nl.max(0) as u32) {
            terms.push(DesignTerm::DummyOf {
                column: col,
                level_index,
            });
        }
    }
    // Append one Interaction term per interaction effect, in effect order —
    // matching the order build_coefficients places betas.
    for comps in &table.interaction_components {
        if comps.is_empty() {
            continue;
        }
        let components = comps
            .iter()
            .map(|(name, level)| component_to_design_term(table, name, level))
            .collect::<Result<Vec<_>, _>>()?;
        terms.push(DesignTerm::Interaction { components });
    }
    Ok(DesignSpec { terms })
}

fn build_coefficients(table: &PredictorTable, spec: &LinearSpec) -> Result<Vec<f64>, SpecError> {
    let n_dummies: u32 = table
        .factor_n_levels
        .iter()
        .map(|n| (*n - 1).max(0) as u32)
        .sum();
    let n_main = 1 + table.non_factor_names.len() as u32 + n_dummies;
    let n_interactions = table
        .interaction_components
        .iter()
        .filter(|c| !c.is_empty())
        .count() as u32;
    let n_total = n_main + n_interactions;
    let mut coeffs = vec![0.0; n_total as usize];
    let assignment_by_name: std::collections::HashMap<&str, f64> = spec
        .effects
        .iter()
        .map(|e| (e.name.as_str(), e.size))
        .collect();
    let mut interaction_rank = 0u32;
    for ((name, cols), comps) in table
        .effect_names
        .iter()
        .zip(table.effect_columns.iter())
        .zip(table.interaction_components.iter())
    {
        let value = *assignment_by_name
            .get(name.as_str())
            .expect("validate_effect_assignments ensured presence");
        let col = if comps.is_empty() {
            cols[0] as usize
        } else {
            let c = (n_main + interaction_rank) as usize;
            interaction_rank += 1;
            c
        };
        if col >= coeffs.len() {
            return Err(SpecError::UnknownPredictor {
                name: format!("{name} (column {col} out of range)"),
            });
        }
        coeffs[col] = value;
    }
    Ok(coeffs)
}

/// Build `design_test` and return the test-side `PredictorTable` (None when
/// `test_formula` is absent — caller uses the generation table for target
/// resolution in that case).
///
/// When a `test_formula` is set, `design_test.terms` reference the SAME
/// `generation.columns` entries as `design_generation` — no columns are
/// re-emitted; each test term resolves to its generation column ID by name.
fn build_design_test(
    spec: &LinearSpec,
    gen_table: &PredictorTable,
    gen_design: &DesignSpec,
) -> Result<(DesignSpec, Option<PredictorTable>), SpecError> {
    let Some(formula) = &spec.test_formula else {
        return Ok((gen_design.clone(), None));
    };
    let parsed = parse_formula(formula)?;
    // An unknown predictor surfaced while building the TEST table is a
    // test_formula term that isn't in the model — re-tag it so the user is
    // told to add it to the model formula (effect may be 0), not just "unknown".
    let test_table = build_predictor_table(&parsed, &spec.predictors).map_err(|e| match e {
        SpecError::UnknownPredictor { name } => SpecError::TestFormulaPredictorMissing { name },
        other => other,
    })?;

    let n_gen_non_factor = gen_table.non_factor_names.len() as u32;
    let mut terms = vec![DesignTerm::Const];
    for name in &test_table.non_factor_names {
        let pos = gen_table
            .non_factor_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| SpecError::TestFormulaPredictorMissing { name: name.clone() })?;
        terms.push(DesignTerm::Direct {
            column: ColumnId(pos as u32),
        });
    }
    // Walk the test table's factors in test-formula declaration order; for
    // each one, locate its corresponding generation-factor index and emit a
    // DummyOf per non-reference level (in generation declaration order so
    // level_index matches what the engine expects).
    for (test_fi, fname) in test_table.factor_names.iter().enumerate() {
        let gen_fi = gen_table
            .factor_names
            .iter()
            .position(|n| n == fname)
            .ok_or_else(|| SpecError::TestFormulaPredictorMissing {
                name: fname.clone(),
            })?;
        let gen_column = ColumnId(n_gen_non_factor + gen_fi as u32);
        let gen_levels = &gen_table.factor_levels[gen_fi];
        let test_levels = &test_table.factor_levels[test_fi];
        for lvl in test_levels {
            let level_index = gen_levels.iter().position(|l| l == lvl).ok_or_else(|| {
                SpecError::UnknownPredictor {
                    name: format!("{fname}[{lvl}]"),
                }
            })? as u32
                + 1;
            terms.push(DesignTerm::DummyOf {
                column: gen_column,
                level_index,
            });
        }
    }
    // Interaction terms: resolve each test-formula interaction's components to
    // GENERATION column ids by name (mirrors the dummy remap above), in the
    // test table's effect order so target term positions line up.
    for comps in &test_table.interaction_components {
        if comps.is_empty() {
            continue;
        }
        let components = comps
            .iter()
            .map(|(name, level)| component_to_design_term(gen_table, name, level))
            .collect::<Result<Vec<_>, _>>()?;
        terms.push(DesignTerm::Interaction { components });
    }
    Ok((DesignSpec { terms }, Some(test_table)))
}

/// Design-test term position for every effect in `table`, interaction-aware.
/// Main effect → its design-matrix column (== term index, since
/// `effect_columns[i][0]` equals the term's position in the design spec for
/// Const/Direct/DummyOf terms). Interaction → its appended term index
/// `1 + n_non_factor + n_dummies + rank`, matching the emission order in
/// `build_design` / `build_design_test`.
fn effect_term_positions(table: &PredictorTable) -> Vec<u32> {
    let n_non_factor = table.non_factor_names.len() as u32;
    let n_dummies: u32 = table
        .factor_n_levels
        .iter()
        .map(|n| (*n - 1).max(0) as u32)
        .sum();
    let base = 1 + n_non_factor + n_dummies;
    let mut positions = Vec::with_capacity(table.effect_names.len());
    let mut interaction_rank = 0u32;
    for comps in &table.interaction_components {
        if comps.is_empty() {
            // Main effect or factor dummy: position IS the design-matrix column.
            let i = positions.len();
            positions.push(table.effect_columns[i][0]);
        } else {
            // Interaction: appended after all main-effect and dummy terms.
            positions.push(base + interaction_rank);
            interaction_rank += 1;
        }
    }
    positions
}

/// Assemble the β-column-aligned [`EffectSkeleton`] for `table`: index 0 is the
/// intercept and every other effect is placed at its `design_test` term
/// position, matching `target_indices`' 1-based space. The per-effect
/// descriptors are computed once during factor expansion
/// (`PredictorTable::effect_descriptors`); this only reorders them from
/// formula-term order into β-column order.
pub fn build_effect_skeleton(table: &PredictorTable) -> EffectSkeleton {
    let n_dummies: usize = table
        .factor_n_levels
        .iter()
        .map(|n| (*n - 1).max(0) as usize)
        .sum();
    let n_interactions = table
        .interaction_components
        .iter()
        .filter(|c| !c.is_empty())
        .count();
    let n_terms = 1 + table.non_factor_names.len() + n_dummies + n_interactions;
    let positions = effect_term_positions(table);
    // Index 0 is never targeted by an effect (positions are >= 1), so it stays
    // the intercept; the rest are overwritten in place.
    let mut skeleton: EffectSkeleton = vec![EffectDescriptor::Intercept; n_terms];
    for (i, &pos) in positions.iter().enumerate() {
        skeleton[pos as usize] = table.effect_descriptors[i].clone();
    }
    skeleton
}

fn build_target_terms(
    spec: &LinearSpec,
    gen_table: &PredictorTable,
    test_table: Option<&PredictorTable>,
    design_test: &engine_contract::DesignSpec,
) -> Result<Vec<TestTarget>, SpecError> {
    // `TestTarget::Marginal { term }` indexes into `design_test.terms`, not
    // into `generation.columns`. When `test_formula` is set, names resolve
    // against the test formula's effect_names (so users can't ask for power
    // on terms the test model doesn't fit) and the column index comes from
    // the test table's `effect_columns` — which mirror the test design's
    // term layout by construction (intercept at 0, non-factors next, then
    // dummies in declaration order). In the standard case (no test_formula)
    // `gen_table.effect_columns[i][0]` is identical to the term position
    // because `design_test == design_generation`.
    let table = test_table.unwrap_or(gen_table);
    let resolver_indices = resolve_targets(&spec.targets, &table.effect_names)?;
    let term_positions_by_effect = effect_term_positions(table);
    use std::collections::BTreeSet;
    let mut term_positions: BTreeSet<u32> = BTreeSet::new();
    for ti in &resolver_indices {
        let idx = (*ti as usize).saturating_sub(1);
        if let Some(&pos) = term_positions_by_effect.get(idx) {
            term_positions.insert(pos);
        }
    }
    // The "at least one target" check lives at the end of this function, after
    // the omnibus Joint is appended — checking only `overall` (no marginals, no
    // contrasts) is a legitimate analysis, so an empty `term_positions` here is
    // not yet an error.
    // `term_positions` is a set, so these marginals are already mutually unique;
    // seed the dedup tracker with them so a reference-collapsing contrast below
    // can't re-emit a marginal a coefficient test already requested.
    let mut seen_marginal: BTreeSet<u32> = term_positions.iter().copied().collect();
    let mut seen_contrast: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut out: Vec<TestTarget> = term_positions
        .into_iter()
        .map(|term| TestTarget::Marginal { term })
        .collect();

    // ── Contrast pairs ────────────────────────────────────────────────────────
    // Route (positive_name, negative_name) pairs.
    //
    // Resolution rules:
    //   - Name found in effect_names → term index (== effect_columns[i][0]).
    //   - Name not found but matches a known factor reference label
    //     (e.g. "group[A]" where "A" was the reference) → marks as the
    //     reference level; the pair collapses to Marginal on the other side.
    //   - Name not recognised at all → UnknownContrastName error.
    //
    // A reference level for factor F is any dummy name `F[r]` where `F[r]` is
    // NOT in effect_names but `F` IS in factor_names. All non-reference dummies
    // for F are in effect_names by construction; anything else is unknown.
    //
    // Dedup as we go (mirrors the contract invariant in engine-contract
    // `validate.rs::invariant_03`, but silently drops the redundant target
    // instead of erroring): ANOVA auto-populates every pairwise contrast, and a
    // reference-level pair collapses to the same Marginal a checked coefficient
    // test already produced — that overlap is a no-op duplication, not an error.
    for (pos_name, neg_name) in &spec.contrast_pairs {
        let pos_idx = resolve_contrast_name(pos_name, table)?;
        let neg_idx = resolve_contrast_name(neg_name, table)?;
        match (pos_idx, neg_idx) {
            // Both non-reference: emit a Contrast (canonical key dedups reversed pairs).
            (Some(p), Some(n)) => {
                if seen_contrast.insert((p.min(n), p.max(n))) {
                    out.push(TestTarget::Contrast {
                        positive: p,
                        negative: n,
                    });
                }
            }
            // One side is the reference (β=0); the contrast reduces to a Marginal
            // on the other side (t² is symmetric).
            (None, Some(term)) | (Some(term), None) => {
                if seen_marginal.insert(term) {
                    out.push(TestTarget::Marginal { term });
                }
            }
            // Both reference — degenerate pair, treat as a no-op (0 − 0 = 0).
            (None, None) => {}
        }
    }

    // Omnibus: emit one Joint covering every non-intercept position in
    // `design_test`. The contract adapter routes this to
    // `SimulationSpec.report_overall = true`. The omnibus is a first-class
    // target — a run that checks only `overall` (no marginals/contrasts) is
    // valid and lands here with an otherwise-empty `out`.
    if spec.report_overall {
        let joint_terms: Vec<u32> = design_test
            .terms
            .iter()
            .enumerate()
            .filter_map(|(i, t)| match t {
                engine_contract::DesignTerm::Const => None,
                _ => Some(i as u32),
            })
            .collect();
        if joint_terms.len() >= 2 {
            out.push(TestTarget::Joint { terms: joint_terms });
        }
    }

    // Single "at least one target" gate, after every kind of target (marginal,
    // contrast, omnibus) has had its chance to populate `out`. Posthoc requests
    // are carried separately on the spec, so they also satisfy it.
    if out.is_empty() && spec.posthoc_requests.is_empty() {
        return Err(SpecError::UnknownTarget {
            name: "(empty)".into(),
        });
    }
    Ok(out)
}

/// Resolve a contrast name to its term position in `design_test.terms`.
///
/// Returns `Some(term_idx)` if the name is a non-reference effect,
/// `None` if it is the factor's reference level (β=0 in dummy coding),
/// or an `UnknownContrastName` error if the name is not recognised at all.
///
/// Reference detection: a name `F[r]` where `F` is a known factor and `r`
/// equals that factor's declared reference level is treated as the reference.
/// Any other `F[x]` where x is not in the factor's level list → unknown name.
fn resolve_contrast_name(name: &str, table: &PredictorTable) -> Result<Option<u32>, SpecError> {
    // Check if it's a non-reference effect.
    if let Some(idx) = table.effect_names.iter().position(|n| n == name) {
        let term = effect_term_positions(table)[idx];
        return Ok(Some(term));
    }
    // Not a non-reference effect. Check if it is a reference-level dummy:
    // name must have shape `F[r]` where `F` is a known factor and `r` equals
    // that factor's declared reference level.
    if let Some(bracket_pos) = name.find('[') {
        if name.ends_with(']') {
            let factor_part = &name[..bracket_pos];
            let level_part = &name[bracket_pos + 1..name.len() - 1];
            if let Some(fi) = table.factor_names.iter().position(|f| f == factor_part) {
                if table.factor_references[fi] == level_part {
                    // Confirmed: this is the reference level for factor `fi`.
                    return Ok(None);
                }
                // Factor name matched but level is neither reference nor a
                // non-reference dummy → unrecognised level name.
                return Err(SpecError::UnknownContrastName {
                    name: name.to_owned(),
                });
            }
        }
    }
    Err(SpecError::UnknownContrastName {
        name: name.to_owned(),
    })
}

fn build_contract_heteroskedasticity_driver(
    h: &HeteroskedasticityInput,
    table: &PredictorTable,
) -> Result<Option<ColumnId>, SpecError> {
    if let Some(var_index) = h.driver_var_index {
        if (var_index as usize) >= table.non_factor_names.len() {
            return Err(SpecError::UnknownPredictor {
                name: format!("var_index={var_index}"),
            });
        }
    }
    // `driver_var_index` is a non-factor 0-based index; non-factors occupy the
    // first generation columns (formula order, factors after), so it maps
    // directly to the contract `ColumnId`. λ is scenario-only — no ratio here.
    Ok(h.driver_var_index.map(ColumnId))
}

fn build_contract_residual(r: &crate::input::ResidualSpec) -> Result<ContractResidual, SpecError> {
    // Canonical names only — the RESIDUAL_CODES table is the single source;
    // the v1 aliases (heavy_tailed/skewed/t) are gone (pre-1.0 break).
    let distribution = match r.distribution.as_str() {
        "normal" => ResidualDist::Normal,
        "right_skewed" => ResidualDist::RightSkewed,
        "left_skewed" => ResidualDist::LeftSkewed,
        "high_kurtosis" => ResidualDist::HighKurtosis,
        "uniform" => ResidualDist::Uniform,
        other => return Err(SpecError::UnknownResidualDist { name: other.into() }),
    };
    Ok(ContractResidual {
        distribution,
        pinned: r.pinned,
    })
}

/// Project `spec.posthoc_requests` into `Vec<PosthocSpec>` for the contract.
///
/// For each `PosthocRequest`:
/// 1. Looks up the named predictor in `table.factor_names` — errors if not a factor.
/// 2. Collects the `DummyOf` term indices for all non-reference levels of that
///    factor from `design_test.terms` (level_index 1..n_levels, in level order).
/// 3. Passes `posthoc_alpha` through unchanged.
fn build_posthoc_specs(
    requests: &[PosthocRequest],
    table: &PredictorTable,
    design_test: &DesignSpec,
) -> Result<Vec<PosthocSpec>, SpecError> {
    if requests.is_empty() {
        return Ok(vec![]);
    }
    let n_non_factor = table.non_factor_names.len() as u32;
    let mut out = Vec::with_capacity(requests.len());
    for req in requests {
        // Resolve the factor name → factor index and column id.
        let fi = table
            .factor_names
            .iter()
            .position(|n| n == &req.factor)
            .ok_or_else(|| {
                // Distinguish "valid continuous predictor" from "name doesn't exist at all".
                if table.non_factor_names.iter().any(|n| n == &req.factor) {
                    SpecError::NotAFactorPredictor {
                        name: req.factor.clone(),
                    }
                } else {
                    SpecError::UnknownPredictor {
                        name: req.factor.clone(),
                    }
                }
            })?;
        let factor_column = ColumnId(n_non_factor + fi as u32);
        // Collect term indices: walk design_test.terms looking for
        // DummyOf { column == factor_column }, in term order (= level order).
        let target_term_indices: Vec<u32> = design_test
            .terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| match term {
                DesignTerm::DummyOf { column, .. } if *column == factor_column => Some(i as u32),
                _ => None,
            })
            .collect();
        out.push(PosthocSpec {
            factor_column,
            target_term_indices,
            posthoc_alpha: req.posthoc_alpha,
        });
    }
    Ok(out)
}

/// FULL synthetic-distribution name → integer code table — the single source
/// the Python and R ports read (via the `dist_codes` host bridge) so neither
/// keeps a hand-maintained copy. Codes 0–5 are the [`SyntheticKind`] decode arms
/// in [`scenario_to_contract`] (kept in sync there; code 1 `binary` is a reject
/// arm for scenario pools — a swapped binary column would be degenerate — but
/// stays in the table for variable-type name lookups); 97/98/99 are the
/// uploaded-data sentinels routed through the host frame path (no decode arm,
/// so they cannot be derived from the arms — they live only here).
pub const DIST_CODES: &[(&str, i32)] = &[
    ("normal", 0),
    ("binary", 1),
    ("right_skewed", 2),
    ("left_skewed", 3),
    ("high_kurtosis", 4),
    ("uniform", 5),
    ("uploaded_factor", 97),
    ("uploaded_binary", 98),
    ("uploaded_data", 99),
];

/// FULL residual-distribution name → integer code table — the canonical five
/// parameterless shapes, sharing codes with [`DIST_CODES`] (one code space;
/// code 1 `binary` has no residual meaning and is absent). The codes match
/// the [`residual_dist_from_code`] decode arms. The v1 aliases
/// (`heavy_tailed`/`skewed`/`t`) are dropped — pre-1.0 break, accepted.
pub const RESIDUAL_CODES: &[(&str, i32)] = &[
    ("normal", 0),
    ("right_skewed", 2),
    ("left_skewed", 3),
    ("high_kurtosis", 4),
    ("uniform", 5),
];

/// Random-effect distribution name → code table. The RE knob keeps its own
/// `normal`/`heavy_tailed`/`right_skewed` vocabulary (it is not the
/// residual-pool space); `heavy_tailed` realizes the t kernel. Matches
/// [`re_dist_from_code`].
pub const RE_DIST_CODES: &[(&str, i32)] =
    &[("normal", 0), ("heavy_tailed", 1), ("right_skewed", 2)];

fn codes_to_json(table: &[(&str, i32)]) -> String {
    let map: std::collections::BTreeMap<&str, i32> = table.iter().copied().collect();
    serde_json::to_string(&map).expect("static code table serializes to JSON")
}

/// JSON object `{name: code}` of [`DIST_CODES`] for the host bridges (Python
/// `json.loads`, R `jsonlite::fromJSON`). Object key order is irrelevant — hosts
/// look up by name.
pub fn dist_codes_json() -> String {
    codes_to_json(DIST_CODES)
}

/// JSON object `{name: code}` of [`RESIDUAL_CODES`] — see [`dist_codes_json`].
pub fn residual_codes_json() -> String {
    codes_to_json(RESIDUAL_CODES)
}

/// JSON object `{name: code}` of [`RE_DIST_CODES`] — see [`dist_codes_json`].
pub fn re_dist_codes_json() -> String {
    codes_to_json(RE_DIST_CODES)
}

/// Decode a single residual-distribution code (the `RESIDUAL_CODES` space —
/// the canonical five, sharing codes with `DIST_CODES`) into a `ResidualDist`.
fn residual_dist_from_code(code: i32) -> Result<ResidualDist, SpecError> {
    match code {
        0 => Ok(ResidualDist::Normal),
        2 => Ok(ResidualDist::RightSkewed),
        3 => Ok(ResidualDist::LeftSkewed),
        4 => Ok(ResidualDist::HighKurtosis),
        5 => Ok(ResidualDist::Uniform),
        // `binary` (code 1) shares the DIST_CODES space but has no residual meaning
        // (RESIDUAL_CODES omits it) — reject it by name, not as an opaque
        // "residual_code=1".
        1 => Err(SpecError::UnknownResidualDist {
            name: "binary".into(),
        }),
        other => Err(SpecError::UnknownResidualDist {
            name: format!("residual_code={other}"),
        }),
    }
}

/// Decode a random-effect distribution code (the `RE_DIST_CODES` space:
/// 0=normal, 1=heavy_tailed, 2=right_skewed) into a `ResidualDist`.
/// `heavy_tailed` realizes the t kernel (`HighKurtosis`).
fn re_dist_from_code(code: i32) -> Result<ResidualDist, SpecError> {
    match code {
        0 => Ok(ResidualDist::Normal),
        1 => Ok(ResidualDist::HighKurtosis),
        2 => Ok(ResidualDist::RightSkewed),
        other => Err(SpecError::UnknownResidualDist {
            name: format!("re_dist_code={other}"),
        }),
    }
}

/// Decode integer-coded scenario fields into typed contract values.
///
/// The decode arms here must stay in sync with [`DIST_CODES`] (synthetic codes 0–5)
/// and [`RESIDUAL_CODES`] (residual codes 0–2). Those tables' docs cite this function;
/// this is the reciprocal coupling declaration.
fn scenario_to_contract(s: &ScenarioInput) -> Result<ContractScenario, SpecError> {
    let new_distributions = s
        .new_distributions
        .iter()
        .map(|&code| match code {
            0 => Ok(SyntheticKind::Normal),
            // `binary` (code 1) is rejected: the swap path carries no p, so a
            // swapped column would degenerate to a constant (singular X'X).
            // `normal` stays allowed — swap-to-normal is the identity, a
            // legitimate way to dilute the effective swap probability.
            1 => Err(SpecError::ScenarioBinarySwapUnsupported {
                name: s.name.clone(),
            }),
            2 => Ok(SyntheticKind::RightSkewed),
            3 => Ok(SyntheticKind::LeftSkewed),
            4 => Ok(SyntheticKind::HighKurtosis),
            5 => Ok(SyntheticKind::Uniform),
            other => Err(SpecError::UnknownResidualDist {
                name: format!("synthetic_code={other}"),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let residual_dists = s
        .residual_dists
        .iter()
        .map(|&code| residual_dist_from_code(code))
        .collect::<Result<Vec<_>, _>>()?;
    // LME RE knobs: emit only when a non-default knob is present, so an
    // all-Gaussian/zero-jitter scenario stays `lme: None` (and thus
    // optimistic-eligible). The RE code lives in the RE_DIST_CODES space
    // (normal/heavy_tailed), not the residual-pool space.
    let lme = if s.random_effect_dist != 0 || s.icc_noise_sd != 0.0 {
        Some(LmeScenarioPerturbations {
            random_effect_dist: re_dist_from_code(s.random_effect_dist)?,
            random_effect_df: s.random_effect_df,
            icc_noise_sd: s.icc_noise_sd,
        })
    } else {
        None
    };
    // An armed residual swap with a df-consuming pool entry must carry an
    // explicit df ≥ 3 — otherwise the kernel's df floor would silently turn
    // an unset df into t(3)/χ²(3), the heaviest shape. Normal and uniform
    // consume no df and stay exempt (the residual analog of the
    // swap-to-normal dilution case above).
    let df_consuming = residual_dists.iter().any(|d| {
        matches!(
            d,
            ResidualDist::HighKurtosis | ResidualDist::RightSkewed | ResidualDist::LeftSkewed
        )
    });
    if s.residual_change_prob > 0.0 && df_consuming && s.residual_df < 3.0 {
        return Err(SpecError::ScenarioResidualDfTooLow {
            name: s.name.clone(),
            got: s.residual_df,
        });
    }
    Ok(ContractScenario {
        name: s.name.clone(),
        heterogeneity: s.heterogeneity,
        heteroskedasticity_ratio: s.heteroskedasticity_ratio,
        correlation_noise_sd: s.correlation_noise_sd,
        distribution_change_prob: s.distribution_change_prob,
        new_distributions,
        residual_change_prob: s.residual_change_prob,
        residual_dists,
        residual_df: s.residual_df,
        sampled_factor_proportions: s.sampled_factor_proportions,
        truth_start: s.truth_start,
        lme,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::*;
    use engine_contract::{EstimatorSpec, OutcomeKind};

    /// scenario_to_contract decodes the RE-distribution code and emits
    /// `lme: Some(..)` only when a non-default RE knob is present (non-Normal
    /// dist OR non-zero ICC jitter); else `lme: None`.
    #[test]
    fn scenario_to_contract_wires_lme_knobs() {
        // Heavy-tailed REs + ICC jitter → Some, with code 1 (t) decoded.
        let sc = ScenarioInput {
            name: "realistic".into(),
            heterogeneity: 0.1,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 1, // t / heavy-tailed
            random_effect_df: 5.0,
            icc_noise_sd: 0.15,
        };
        let c = scenario_to_contract(&sc).unwrap();
        let lme = c.lme.expect("lme Some when RE knobs are non-default");
        assert_eq!(lme.random_effect_dist, ResidualDist::HighKurtosis);
        assert_eq!(lme.random_effect_df, 5.0);
        assert_eq!(lme.icc_noise_sd, 0.15);

        // Gaussian REs, zero ICC → None (keeps the run optimistic-eligible).
        let sc0 = ScenarioInput {
            name: "optimistic".into(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        };
        assert!(scenario_to_contract(&sc0).unwrap().lme.is_none());
    }

    fn simple_spec() -> LinearSpec {
        LinearSpec {
            formula: "y = x1 + x2".into(),
            predictors: vec![
                PredictorSpec {
                    name: "x1".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
                PredictorSpec {
                    name: "x2".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
            ],
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "x2".into(),
                    size: 0.3,
                },
            ],
            correlations: vec![],
            alpha: 0.05,
            correction: Correction::None,
            targets: vec!["overall".into()],
            heteroskedasticity: HeteroskedasticityInput::default(),
            residual: ResidualSpec::default(),
            max_failed_fraction: 0.1,
            scenarios: vec![],
            test_formula: None,
            report_overall: false,
            contrast_pairs: vec![],
            posthoc_requests: vec![],
            upload: None,
            cluster_level_vars: vec![],
            wald_se: Default::default(),
            nagq: 1,
        }
    }

    #[test]
    fn builds_contract_for_two_continuous_predictors() {
        let mut contracts = build_linear_contract(&simple_spec()).unwrap();
        assert_eq!(contracts.len(), 1);
        let c = contracts.pop().unwrap();
        assert_eq!(c.generation.columns.len(), 2);
        assert_eq!(c.design_generation.terms.len(), 3);
        assert_eq!(c.outcome.coefficients, vec![0.0, 0.5, 0.3]);
        // "overall" → two marginals only. Omnibus Joint is gated on
        // `spec.report_overall` — simple_spec leaves it false.
        assert_eq!(c.test.targets.len(), 2);
        assert!(matches!(
            c.test.targets.first(),
            Some(TestTarget::Marginal { term: 1 })
        ));
        c.validate().unwrap();
    }

    #[test]
    fn report_overall_appends_joint_with_all_non_intercept_terms() {
        let mut spec = simple_spec();
        spec.report_overall = true;
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // design_test has 3 terms (Const, x1, x2), so non-intercept positions
        // are {1, 2}. We expect: 2 Marginals + 1 Joint.
        assert_eq!(c.test.targets.len(), 3);
        let joint = c
            .test
            .targets
            .iter()
            .find_map(|t| match t {
                TestTarget::Joint { terms } => Some(terms.clone()),
                _ => None,
            })
            .expect("Joint emitted for report_overall");
        assert_eq!(joint, vec![1, 2]);
        c.validate().unwrap();
    }

    #[test]
    fn report_overall_alone_is_valid_with_no_marginals() {
        // Regression: checking only "overall" (empty per-effect target list, as the
        // app sends when only the omnibus box is ticked) used to be rejected with
        // `target '(empty)'` because the empty-target guard fired before the Joint
        // was appended. The omnibus is a first-class target — the run is valid and
        // emits a lone Joint.
        let mut spec = simple_spec();
        spec.targets = vec![]; // no per-effect/marginal/contrast tests
        spec.report_overall = true;
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        assert_eq!(c.test.targets.len(), 1);
        assert!(matches!(
            c.test.targets.first(),
            Some(TestTarget::Joint { .. })
        ));
        c.validate().unwrap();
    }

    #[test]
    fn no_targets_and_no_overall_still_errors_empty() {
        // The single end-of-function guard must still reject a run with nothing to
        // test: no marginals, no contrasts, no posthoc, and no omnibus.
        let mut spec = simple_spec();
        spec.targets = vec![];
        spec.report_overall = false;
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(
            matches!(&err, SpecError::UnknownTarget { name } if name == "(empty)"),
            "expected UnknownTarget '(empty)', got {err:?}"
        );
    }

    #[test]
    fn report_overall_with_test_formula_uses_design_test_terms() {
        // When test_formula trims the model, the omnibus Joint must cover the
        // *test* design's non-intercept positions, not the generation design's.
        let mut spec = simple_spec();
        spec.report_overall = true;
        spec.test_formula = Some("y = x1".into());
        spec.targets = vec!["x1".into()];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // design_test has 2 terms (Const, x1) → omnibus joint covers {1} only.
        // Joint requires >= 2 terms, so the spec-builder skips emission and
        // the contract has only the marginal.
        assert!(
            c.test
                .targets
                .iter()
                .all(|t| matches!(t, TestTarget::Marginal { .. })),
            "test_formula reducing to 1 non-intercept term should not emit a Joint"
        );
    }

    #[test]
    fn test_formula_drops_term_from_design_test_but_not_design_generation() {
        let mut spec = simple_spec();
        // Generation = "y = x1 + x2", test_formula = "y = x1" → design_test
        // has Const + Direct{0}; design_generation still has all three terms.
        spec.test_formula = Some("y = x1".into());
        spec.targets = vec!["x1".into()];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        assert_eq!(c.design_generation.terms.len(), 3);
        let dt = c
            .design_test
            .as_ref()
            .expect("design_test Some when test_formula set");
        assert_eq!(dt.terms.len(), 2);
        assert!(matches!(dt.terms[0], DesignTerm::Const));
        assert!(matches!(
            dt.terms[1],
            DesignTerm::Direct { column } if column == ColumnId(0)
        ));
        // β layout unchanged — coefficients still reference the full design.
        assert_eq!(c.outcome.coefficients, vec![0.0, 0.5, 0.3]);
    }

    #[test]
    fn test_formula_rejects_predictor_not_in_generation() {
        let mut spec = simple_spec();
        spec.test_formula = Some("y = z".into()); // z is not in spec.predictors
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(matches!(err, SpecError::TestFormulaPredictorMissing { .. }));
    }

    #[test]
    fn test_formula_with_factor_remaps_to_generation_column_ids() {
        let mut spec = simple_spec();
        spec.formula = "y = x1 + group".into();
        spec.predictors = vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    proportions: vec![0.4, 0.3, 0.3],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        spec.effects = vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.5,
            },
            EffectAssignment {
                name: "group[B]".into(),
                size: 0.2,
            },
            EffectAssignment {
                name: "group[C]".into(),
                size: 0.4,
            },
        ];
        spec.targets = vec!["group[B]".into(), "group[C]".into()];
        spec.test_formula = Some("y = group".into());
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // design_test: Const + DummyOf{group, 1} + DummyOf{group, 2}
        let dt = c
            .design_test
            .as_ref()
            .expect("design_test Some when test_formula set");
        assert_eq!(dt.terms.len(), 3);
        assert!(matches!(
            dt.terms[1],
            DesignTerm::DummyOf { column, level_index: 1 } if column == ColumnId(1)
        ));
        assert!(matches!(
            dt.terms[2],
            DesignTerm::DummyOf { column, level_index: 2 } if column == ColumnId(1)
        ));
        // Targets index into design_test.terms. For the test design
        // [Const, DummyOf{group,1}, DummyOf{group,2}], the dummies sit at
        // positions 1 and 2.
        let targets: Vec<_> = c
            .test
            .targets
            .iter()
            .map(|t| match t {
                TestTarget::Marginal { term } => *term,
                _ => panic!("unexpected joint"),
            })
            .collect();
        assert_eq!(targets, vec![1, 2]);
    }

    #[test]
    fn build_contract_accepts_outcome_kind_intercept_clusters() {
        use engine_contract::{ClusterSizing, ClusterSpec};
        let lspec = simple_spec();
        let c = build_contract(&lspec, OutcomeKind::Binary, None, None, -0.5, vec![])
            .expect("build_contract")
            .pop()
            .unwrap();
        assert_eq!(c.outcome.kind, OutcomeKind::Binary);
        assert_eq!(c.estimator, EstimatorSpec::Glm);
        assert!((c.outcome.intercept - (-0.5)).abs() < 1e-12);
        assert!(c.generation.cluster.is_none());

        let c_mle = build_contract(
            &lspec,
            OutcomeKind::Continuous,
            None,
            Some(EstimatorSpec::Mle),
            0.0,
            vec![ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
                tau_squared: 0.3,
                slopes: vec![],
                extra_groupings: vec![],
            }],
        )
        .expect("build_contract mle")
        .pop()
        .unwrap();
        assert_eq!(c_mle.estimator, EstimatorSpec::Mle);
        assert_eq!(
            c_mle.generation.cluster.as_ref().unwrap().sizing,
            ClusterSizing::FixedClusters { n_clusters: 20 }
        );
    }

    #[test]
    fn build_contract_rejects_cluster_count_gt_one() {
        use engine_contract::{ClusterSizing, ClusterSpec};
        let lspec = simple_spec();
        let err = build_contract(
            &lspec,
            OutcomeKind::Continuous,
            None,
            Some(EstimatorSpec::Mle),
            0.0,
            vec![
                ClusterSpec {
                    sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
                    tau_squared: 0.3,
                    slopes: vec![],
                    extra_groupings: vec![],
                },
                ClusterSpec {
                    sizing: ClusterSizing::FixedClusters { n_clusters: 30 },
                    tau_squared: 0.2,
                    slopes: vec![],
                    extra_groupings: vec![],
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(err, SpecError::ClusterFamilyMismatch));
    }

    #[test]
    fn build_contract_rejects_mle_without_cluster() {
        let lspec = simple_spec();
        let err = build_contract(
            &lspec,
            OutcomeKind::Continuous,
            None,
            Some(EstimatorSpec::Mle),
            0.0,
            vec![],
        )
        .unwrap_err();
        assert!(matches!(err, SpecError::ClusterFamilyMismatch));
    }

    #[test]
    fn build_contract_accepts_logit_with_scenario_heterogeneity() {
        // Gate-removal regression: binary + heterogeneity (β-jitter as
        // log-odds heterogeneity) is a supported combination. Heterogeneity
        // is scenario-only — assert the scenario knob lands on the wire.
        let mut spec = simple_spec();
        spec.scenarios = vec![ScenarioInput {
            name: "het".into(),
            heterogeneity: 0.2,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 0,
            random_effect_df: 0.0,
            icc_noise_sd: 0.0,
        }];
        let contracts = build_contract(&spec, OutcomeKind::Binary, None, None, -0.5, vec![])
            .expect("binary with scenario heterogeneity>0 should validate");
        assert_eq!(contracts[0].scenario.heterogeneity, 0.2);
    }

    #[test]
    fn build_contract_rejects_random_slopes() {
        let mut spec = simple_spec();
        spec.formula = "y ~ x1 + x2 + (1+x1|g)".into();
        let err =
            build_contract(&spec, OutcomeKind::Continuous, None, None, 0.0, vec![]).unwrap_err();
        assert!(matches!(err, SpecError::RandomSlopesUnsupported));
    }

    #[test]
    fn build_contract_rejects_implicit_random_slope() {
        // (x1|g) parses to a Slope (implicit-intercept form), so the builder
        // rejects it identically to (1+x1|g) — it no longer dies earlier as a
        // FormulaSyntax error. Mirrors build_contract_rejects_random_slopes.
        let mut spec = simple_spec();
        spec.formula = "y ~ x1 + x2 + (x1|g)".into();
        let err =
            build_contract(&spec, OutcomeKind::Continuous, None, None, 0.0, vec![]).unwrap_err();
        assert!(matches!(err, SpecError::RandomSlopesUnsupported));
    }

    #[test]
    fn build_contract_accepts_random_intercept() {
        let mut spec = simple_spec();
        spec.formula = "y ~ x1 + x2 + (1|g)".into();
        let c = build_contract(
            &spec,
            OutcomeKind::Continuous,
            None,
            Some(EstimatorSpec::Mle),
            0.0,
            vec![engine_contract::ClusterSpec {
                sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 20 },
                tau_squared: 0.3,
                slopes: vec![],
                extra_groupings: vec![],
            }],
        )
        .expect("intercept-only RE should be accepted")
        .pop()
        .unwrap();
        // Cluster survived onto the generation spec — a mutation that drops it
        // from `build_contract_with_skeleton` would pass the `expect` but fail here.
        assert!(
            c.generation.cluster.is_some(),
            "cluster must be set on generation"
        );
        // Estimator coupling: Mle was stated explicitly and must land on the wire.
        assert_eq!(c.estimator, EstimatorSpec::Mle);
    }

    #[test]
    fn factor_flip_layout_matches_design_matrix_columns() {
        let mut spec = simple_spec();
        spec.formula = "y = group + x1".into();
        spec.predictors = vec![
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["1".into(), "2".into(), "3".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "1".into(),
                    sampled_proportions: None,
                },
            },
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
        ];
        spec.effects = vec![
            EffectAssignment {
                name: "group[2]".into(),
                size: 0.3,
            },
            EffectAssignment {
                name: "group[3]".into(),
                size: 0.5,
            },
            EffectAssignment {
                name: "x1".into(),
                size: 0.4,
            },
        ];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // Engine layout: [intercept, x1, group[2], group[3]]
        assert_eq!(c.outcome.coefficients, vec![0.0, 0.4, 0.3, 0.5]);
    }

    // ── Contrast routing tests ────────────────────────────────────────────────

    /// Build a 3-level factor spec for contrast routing tests.
    /// Factor "group" has levels ["A", "B", "C"], reference "A".
    /// Effect names: ["group[B]", "group[C]"].
    fn contrast_spec_base() -> LinearSpec {
        LinearSpec {
            formula: "y = group".into(),
            predictors: vec![PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            }],
            effects: vec![
                EffectAssignment {
                    name: "group[B]".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "group[C]".into(),
                    size: 0.5,
                },
            ],
            correlations: vec![],
            alpha: 0.05,
            correction: Correction::None,
            targets: vec!["overall".into()],
            heteroskedasticity: HeteroskedasticityInput::default(),
            residual: ResidualSpec::default(),
            max_failed_fraction: 0.1,
            scenarios: vec![],
            test_formula: None,
            report_overall: false,
            contrast_pairs: vec![],
            posthoc_requests: vec![],
            upload: None,
            cluster_level_vars: vec![],
            wald_se: Default::default(),
            nagq: 1,
        }
    }

    /// nLevels-only placeholders (levels provided as "1", "2", ...)
    /// must emit `factor[1]`, `factor[2]`, NOT `factor[L1]`, `factor[L2]`.
    #[test]
    fn nlevel_placeholder_emits_bare_integer_names() {
        let spec = LinearSpec {
            formula: "y = group".into(),
            predictors: vec![PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["1".into(), "2".into(), "3".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "1".into(),
                    sampled_proportions: None,
                },
            }],
            effects: vec![
                EffectAssignment {
                    name: "group[2]".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "group[3]".into(),
                    size: 0.5,
                },
            ],
            correlations: vec![],
            alpha: 0.05,
            correction: Correction::None,
            targets: vec!["overall".into()],
            heteroskedasticity: HeteroskedasticityInput::default(),
            residual: ResidualSpec::default(),
            max_failed_fraction: 0.1,
            scenarios: vec![],
            test_formula: None,
            report_overall: false,
            contrast_pairs: vec![],
            posthoc_requests: vec![],
            upload: None,
            cluster_level_vars: vec![],
            wald_se: Default::default(),
            nagq: 1,
        };
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // group[2] and group[3] must resolve to the non-reference dummy
        // columns (term 1 and term 2; reference level "1" excluded). Asserting the
        // term indices catches a mutation that adds an "L" prefix or mis-routes the
        // effect to the wrong column — a count-only check would miss both.
        let mut marginals: Vec<u32> = c
            .test
            .targets
            .iter()
            .filter_map(|t| match t {
                TestTarget::Marginal { term } => Some(*term),
                _ => None,
            })
            .collect();
        marginals.sort_unstable();
        assert_eq!(marginals, vec![1, 2], "group[2]→term 1, group[3]→term 2");
    }

    /// A contrast pair where one side is the reference level collapses to
    /// a Marginal on the non-reference term.
    #[test]
    fn contrast_pair_with_reference_collapses_to_marginal() {
        let mut spec = contrast_spec_base();
        // "group[A]" is the reference level (not in effect_names). This pair
        // should collapse to Marginal { term = group[B]'s term index }.
        spec.targets = vec![];
        spec.contrast_pairs = vec![("group[B]".into(), "group[A]".into())];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        assert_eq!(c.test.targets.len(), 1);
        // Collapses to a Marginal on the non-reference side (group[B] →
        // term 1; reference "A" excluded). Pin the term, not just the variant.
        assert_eq!(
            c.test.targets[0],
            TestTarget::Marginal { term: 1 },
            "reference-side contrast must collapse to Marginal on group[B] (term 1)"
        );
    }

    /// A contrast pair where neither side is the reference level emits a
    /// Contrast variant.
    #[test]
    fn contrast_pair_non_reference_emits_contrast_variant() {
        let mut spec = contrast_spec_base();
        spec.targets = vec![];
        spec.contrast_pairs = vec![("group[B]".into(), "group[C]".into())];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        assert_eq!(c.test.targets.len(), 1);
        // Both-non-reference pair emits Contrast with the design term
        // indices (group[B] → 1 positive, group[C] → 2 negative). Pin the indices,
        // not just the variant — a swap or wrong column would otherwise pass.
        assert_eq!(
            c.test.targets[0],
            TestTarget::Contrast {
                positive: 1,
                negative: 2
            },
            "group[B] vs group[C] must map to Contrast{{positive:1, negative:2}}"
        );
    }

    /// An unrecognised name in a contrast pair must produce an
    /// UnknownContrastName error.
    #[test]
    fn unknown_contrast_name_errors() {
        let mut spec = contrast_spec_base();
        spec.targets = vec![];
        spec.contrast_pairs = vec![("group[B]".into(), "group[NONEXISTENT]".into())];
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(
            matches!(err, SpecError::UnknownContrastName { .. }),
            "expected UnknownContrastName, got: {err:?}"
        );
    }

    // Every emitted contract passes its own validate(); with multiple scenarios
    // each returned contract is individually valid (the builder validates
    // per-scenario, not just the first).
    #[test]
    fn every_emitted_contract_individually_validates() {
        let mut spec = simple_spec();
        spec.scenarios = vec![
            ScenarioInput {
                name: "optimistic".into(),
                heterogeneity: 0.0,
                heteroskedasticity_ratio: 1.0,
                correlation_noise_sd: 0.0,
                distribution_change_prob: 0.0,
                new_distributions: vec![],
                residual_change_prob: 0.0,
                residual_dists: vec![],
                residual_df: 0.0,
                sampled_factor_proportions: false,
                truth_start: false,
                random_effect_dist: 0,
                random_effect_df: 0.0,
                icc_noise_sd: 0.0,
            },
            ScenarioInput {
                name: "realistic".into(),
                heterogeneity: 0.1,
                heteroskedasticity_ratio: 2.0,
                correlation_noise_sd: 0.1,
                distribution_change_prob: 0.0,
                new_distributions: vec![],
                residual_change_prob: 0.0,
                residual_dists: vec![],
                residual_df: 0.0,
                sampled_factor_proportions: true,
                truth_start: false,
                random_effect_dist: 0,
                random_effect_df: 0.0,
                icc_noise_sd: 0.0,
            },
        ];
        let contracts = build_linear_contract(&spec).unwrap();
        assert_eq!(contracts.len(), 2);
        for c in &contracts {
            c.validate()
                .expect("each emitted contract must self-validate");
        }
    }

    // Empty correlation pairs yield Correlations::Identity; non-empty pairs
    // yield Correlations::Matrix whose continuous_columns list the non-factor
    // column IDs.
    #[test]
    fn empty_corr_is_identity_nonempty_is_matrix() {
        let c = build_linear_contract(&simple_spec())
            .unwrap()
            .pop()
            .unwrap();
        assert!(matches!(c.generation.correlations, Correlations::Identity));

        let mut spec = simple_spec();
        spec.correlations = vec![CorrelationPair {
            a: "x1".into(),
            b: "x2".into(),
            value: 0.3,
        }];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        match c.generation.correlations {
            Correlations::Matrix {
                continuous_columns, ..
            } => {
                assert_eq!(continuous_columns, vec![ColumnId(0), ColumnId(1)]);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    // A contrast pair where both sides are the reference level is a degenerate
    // no-op and emits no target entry.
    #[test]
    fn contrast_pair_both_reference_emits_no_target() {
        let mut spec = contrast_spec_base();
        // Keep one ordinary target so the build doesn't fail on empty targets,
        // then confirm the both-reference contrast adds nothing of its own.
        spec.targets = vec!["group[B]".into()];
        spec.contrast_pairs = vec![("group[A]".into(), "group[A]".into())];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // Only the single marginal from `targets`; the both-reference contrast
        // contributed no target.
        assert_eq!(c.test.targets.len(), 1);
        assert!(matches!(c.test.targets[0], TestTarget::Marginal { .. }));
    }

    // Contrast pairs and `targets` are additive — both are emitted into
    // test.targets, not mutually exclusive.
    #[test]
    fn contrast_pairs_and_targets_are_additive() {
        let mut spec = contrast_spec_base();
        spec.targets = vec!["group[B]".into()];
        spec.contrast_pairs = vec![("group[B]".into(), "group[C]".into())];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // One marginal from `targets` + one contrast from `contrast_pairs`.
        let n_marginal = c
            .test
            .targets
            .iter()
            .filter(|t| matches!(t, TestTarget::Marginal { .. }))
            .count();
        let n_contrast = c
            .test
            .targets
            .iter()
            .filter(|t| matches!(t, TestTarget::Contrast { .. }))
            .count();
        assert_eq!(n_marginal, 1, "marginal from targets must survive");
        assert_eq!(n_contrast, 1, "contrast from contrast_pairs must survive");
    }

    // The skeleton is β-column-aligned, NOT effect/formula-term order: a factor
    // sandwiched between two continuous predictors must have its dummies placed
    // AFTER both continuous columns (intercept, x1, x2, group[B], group[C]),
    // even though the formula order is x1, group, x2. Length must equal the
    // design's term count so `skeleton[target_index]` is always in range.
    #[test]
    fn build_effect_skeleton_is_beta_column_aligned() {
        let spec = LinearSpec {
            formula: "y = x1 + group + x2".into(),
            predictors: vec![
                PredictorSpec {
                    name: "x1".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
                PredictorSpec {
                    name: "group".into(),
                    pinned: false,
                    kind: VarKind::Factor {
                        levels: vec!["A".into(), "B".into(), "C".into()],
                        proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        reference: "A".into(),
                        sampled_proportions: None,
                    },
                },
                PredictorSpec {
                    name: "x2".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
            ],
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "group[B]".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "group[C]".into(),
                    size: 0.4,
                },
                EffectAssignment {
                    name: "x2".into(),
                    size: 0.2,
                },
            ],
            targets: vec!["overall".into()],
            ..simple_spec()
        };
        let (contracts, skeleton) = build_linear_contract_with_skeleton(&spec).unwrap();
        assert_eq!(
            skeleton,
            vec![
                EffectDescriptor::Intercept,
                EffectDescriptor::Continuous {
                    predictor: "x1".into()
                },
                EffectDescriptor::Continuous {
                    predictor: "x2".into()
                },
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 1
                },
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 2
                },
            ]
        );
        // One descriptor per β-column / design term.
        assert_eq!(skeleton.len(), contracts[0].design_generation.terms.len());
    }

    // When a test_formula trims the model, the skeleton is built from the TEST
    // design, so `skeleton[target_term]` names the fitted effect. Here the test
    // model keeps only `group`, so the skeleton is [Intercept, group[B], group[C]]
    // and the marginal targets (terms 1, 2) index straight into it.
    #[test]
    fn build_effect_skeleton_follows_test_formula_design() {
        let mut spec = simple_spec();
        spec.formula = "y = x1 + group".into();
        spec.predictors = vec![
            PredictorSpec {
                name: "x1".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    proportions: vec![0.4, 0.3, 0.3],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
        ];
        spec.effects = vec![
            EffectAssignment {
                name: "x1".into(),
                size: 0.5,
            },
            EffectAssignment {
                name: "group[B]".into(),
                size: 0.2,
            },
            EffectAssignment {
                name: "group[C]".into(),
                size: 0.4,
            },
        ];
        spec.targets = vec!["group[B]".into(), "group[C]".into()];
        spec.test_formula = Some("y = group".into());
        let (contracts, skeleton) = build_linear_contract_with_skeleton(&spec).unwrap();
        assert_eq!(
            skeleton,
            vec![
                EffectDescriptor::Intercept,
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 1
                },
                EffectDescriptor::FactorLevel {
                    factor: "group".into(),
                    level: 2
                },
            ]
        );
        // Marginal targets index design_test.terms == skeleton positions.
        for t in &contracts[0].test.targets {
            if let TestTarget::Marginal { term } = t {
                assert!((*term as usize) < skeleton.len());
                assert!(matches!(
                    skeleton[*term as usize],
                    EffectDescriptor::FactorLevel { .. }
                ));
            }
        }
    }

    #[test]
    fn build_design_appends_interaction_term() {
        let spec = LinearSpec {
            formula: "y = x1 + x2 + x1:x2".into(),
            predictors: vec![
                PredictorSpec {
                    name: "x1".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
                PredictorSpec {
                    name: "x2".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
            ],
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "x2".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "x1:x2".into(),
                    size: 0.2,
                },
            ],
            targets: vec!["x1:x2".into()],
            ..simple_spec()
        };
        let table = build_predictor_table(&parse_formula(&spec.formula).unwrap(), &spec.predictors)
            .unwrap();
        let design = build_design(&table).unwrap();
        // terms: [Const, Direct(0), Direct(1), Interaction([Direct(0),Direct(1)])]
        assert_eq!(design.terms.len(), 4);
        assert!(matches!(
            &design.terms[3],
            DesignTerm::Interaction { components }
                if components.len() == 2
                && matches!(components[0], DesignTerm::Direct { column } if column == ColumnId(0))
                && matches!(components[1], DesignTerm::Direct { column } if column == ColumnId(1))
        ));
    }

    /// Default coupling: estimator None + a cluster spec must yield Mle.
    #[test]
    fn build_coefficients_places_interaction_on_appended_column() {
        let spec = LinearSpec {
            formula: "y = x1 + x2 + x1:x2".into(),
            predictors: vec![
                PredictorSpec {
                    name: "x1".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
                PredictorSpec {
                    name: "x2".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
            ],
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "x2".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "x1:x2".into(),
                    size: 0.2,
                },
            ],
            targets: vec!["x1:x2".into()],
            ..simple_spec()
        };
        let table = build_predictor_table(&parse_formula(&spec.formula).unwrap(), &spec.predictors)
            .unwrap();
        let coeffs = build_coefficients(&table, &spec).unwrap();
        // [intercept, x1, x2, x1:x2]
        assert_eq!(coeffs, vec![0.0, 0.5, 0.3, 0.2]);
        // coefficients length must match design_generation.terms length
        assert_eq!(coeffs.len(), build_design(&table).unwrap().terms.len());
    }

    #[test]
    fn interaction_target_resolves_to_appended_term_position() {
        let spec = LinearSpec {
            formula: "y = x1 + x2 + x1:x2".into(),
            predictors: vec![
                PredictorSpec {
                    name: "x1".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
                PredictorSpec {
                    name: "x2".into(),
                    pinned: false,
                    kind: VarKind::Normal,
                },
            ],
            effects: vec![
                EffectAssignment {
                    name: "x1".into(),
                    size: 0.5,
                },
                EffectAssignment {
                    name: "x2".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "x1:x2".into(),
                    size: 0.2,
                },
            ],
            targets: vec!["x1:x2".into()],
            ..simple_spec()
        };
        let contracts = build_linear_contract(&spec).unwrap();
        let c = &contracts[0];
        // design_test is None (no test_formula); design_generation has 4 terms:
        // [Const(0), Direct(x1)=1, Direct(x2)=2, Interaction(x1:x2)=3]
        assert!(
            c.test
                .targets
                .iter()
                .any(|t| matches!(t, TestTarget::Marginal { term: 3 })),
            "x1:x2 target must resolve to term 3 (the appended Interaction term), got: {:?}",
            c.test.targets
        );
        // Confirm term 3 in design_generation is indeed an Interaction term.
        let dt = c.design_test.as_ref().unwrap_or(&c.design_generation);
        assert!(
            matches!(dt.terms[3], DesignTerm::Interaction { .. }),
            "term 3 must be an Interaction, got: {:?}",
            dt.terms[3]
        );
    }

    #[test]
    fn build_contract_none_estimator_with_cluster_defaults_to_mle() {
        use engine_contract::{ClusterSizing, ClusterSpec};
        let lspec = simple_spec();
        let c = build_contract(
            &lspec,
            OutcomeKind::Continuous,
            None,
            None,
            0.0,
            vec![ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
                tau_squared: 0.3,
                slopes: vec![],
                extra_groupings: vec![],
            }],
        )
        .expect("build_contract with None estimator + cluster")
        .pop()
        .unwrap();
        assert_eq!(
            c.estimator,
            EstimatorSpec::Mle,
            "expected Mle from default coupling when estimator is None and cluster is provided"
        );
    }

    // ── posthoc_requests projection tests ─────────────────────────────────────

    /// Build a one-way ANOVA spec for `y ~ dose_group` with 3 levels.
    /// Factor "dose_group" has levels ["low", "mid", "high"], reference "low".
    /// Effect names: ["dose_group[mid]", "dose_group[high]"].
    fn anova_3level_spec() -> LinearSpec {
        LinearSpec {
            formula: "y = dose_group".into(),
            predictors: vec![PredictorSpec {
                name: "dose_group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["low".into(), "mid".into(), "high".into()],
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    reference: "low".into(),
                    sampled_proportions: None,
                },
            }],
            effects: vec![
                EffectAssignment {
                    name: "dose_group[mid]".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "dose_group[high]".into(),
                    size: 0.5,
                },
            ],
            correlations: vec![],
            alpha: 0.05,
            correction: Correction::None,
            targets: vec!["overall".into()],
            heteroskedasticity: HeteroskedasticityInput::default(),
            residual: ResidualSpec::default(),
            max_failed_fraction: 0.1,
            scenarios: vec![],
            test_formula: None,
            report_overall: false,
            contrast_pairs: vec![],
            posthoc_requests: vec![PosthocRequest {
                factor: "dose_group".into(),
                posthoc_alpha: None,
            }],
            upload: None,
            cluster_level_vars: vec![],
            wald_se: Default::default(),
            nagq: 1,
        }
    }

    #[test]
    fn posthoc_requests_project_to_contract_posthoc() {
        let spec = anova_3level_spec();
        let contracts = build_linear_contract(&spec).unwrap();
        assert_eq!(
            contracts[0].posthoc.len(),
            1,
            "one block per requested factor"
        );
        assert_eq!(
            contracts[0].posthoc[0].target_term_indices.len(),
            2,
            "k-1 dummy terms for a 3-level factor"
        );
        // The posthoc_alpha flows through as None.
        assert_eq!(contracts[0].posthoc[0].posthoc_alpha, None);
        // Contract must also self-validate (invariant_17).
        contracts[0]
            .validate()
            .expect("posthoc contract must validate");
    }

    #[test]
    fn posthoc_empty_requests_keeps_posthoc_empty() {
        // No posthoc_requests → contract.posthoc must remain empty.
        let spec = contrast_spec_base(); // has posthoc_requests: vec![]
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        assert!(c.posthoc.is_empty());
    }

    #[test]
    fn posthoc_request_for_non_factor_errors() {
        // Requesting posthoc on a continuous predictor must produce NotAFactorPredictor,
        // not UnknownPredictor — the name is valid, it's just not a factor.
        let mut spec = simple_spec(); // x1, x2 are Normal (non-factor)
        spec.posthoc_requests = vec![PosthocRequest {
            factor: "x1".into(),
            posthoc_alpha: None,
        }];
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(
            matches!(err, SpecError::NotAFactorPredictor { ref name } if name == "x1"),
            "expected NotAFactorPredictor {{name: \"x1\"}}, got: {err:?}"
        );
        assert!(
            format!("{err}").contains("not a factor"),
            "error message must mention 'not a factor', got: {err}"
        );
    }

    #[test]
    fn posthoc_request_for_unknown_predictor_errors() {
        // A name that doesn't exist at all → UnknownPredictor.
        let mut spec = anova_3level_spec();
        spec.posthoc_requests = vec![PosthocRequest {
            factor: "nonexistent".into(),
            posthoc_alpha: None,
        }];
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(
            matches!(err, SpecError::UnknownPredictor { ref name } if name == "nonexistent"),
            "expected UnknownPredictor {{name: \"nonexistent\"}}, got: {err:?}"
        );
    }

    /// `cluster_level_vars` resolve to `generation.cluster_level_columns`:
    /// a non-factor name → its ColumnId (position in the non-factor block); a
    /// factor name → ColumnId(n_non_factor + factor_index). The factor comes
    /// FIRST in the formula ("y = group + x1") so positional resolution would
    /// yield the reversed ids — the assertion pins the non-factors-first layout.
    #[test]
    fn cluster_level_vars_resolve_to_columns() {
        let mut spec = simple_spec();
        spec.formula = "y = group + x1".into();
        spec.predictors = vec![
            PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into()],
                    proportions: vec![0.4, 0.3, 0.3],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            },
            PredictorSpec {
                name: "x1".into(),
                kind: VarKind::Normal,
                pinned: false,
            },
        ];
        spec.effects = vec![
            EffectAssignment {
                name: "group[B]".into(),
                size: 0.2,
            },
            EffectAssignment {
                name: "group[C]".into(),
                size: 0.4,
            },
            EffectAssignment {
                name: "x1".into(),
                size: 0.5,
            },
        ];
        spec.targets = vec!["x1".into()];
        spec.cluster_level_vars = vec!["x1".into(), "group".into()];
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        // Pins the build_columns ordering the resolver mirrors: non-factors
        // first ⇒ columns[0] is the continuous x1, columns[1] the factor.
        assert!(matches!(
            c.generation.columns[0],
            ColumnSpec::Synthetic { .. }
        ));
        assert!(matches!(
            c.generation.columns[1],
            ColumnSpec::FactorSynthetic { .. }
        ));
        assert_eq!(
            c.generation.cluster_level_columns,
            vec![ColumnId(0), ColumnId(1)]
        );
        c.validate().unwrap();
    }

    /// An unknown cluster-level name surfaces UnknownPredictor.
    #[test]
    fn cluster_level_vars_unknown_name_errors() {
        let mut spec = simple_spec();
        spec.cluster_level_vars = vec!["nope".into()];
        let err = build_linear_contract(&spec).unwrap_err();
        assert!(matches!(err, SpecError::UnknownPredictor { ref name } if name == "nope"));
    }

    #[test]
    fn posthoc_term_indices_point_at_dummyof_terms() {
        // Verify that the emitted term indices in posthoc actually reference
        // DummyOf terms for the factor's column.
        let spec = anova_3level_spec();
        let c = build_linear_contract(&spec).unwrap().pop().unwrap();
        let ph = &c.posthoc[0];
        let design = c.design_test.as_ref().unwrap_or(&c.design_generation);
        for &ti in &ph.target_term_indices {
            match &design.terms[ti as usize] {
                engine_contract::DesignTerm::DummyOf { column, .. } => {
                    assert_eq!(
                        *column, ph.factor_column,
                        "DummyOf must reference the posthoc factor_column"
                    );
                }
                other => panic!("expected DummyOf, got {other:?}"),
            }
        }
    }

    /// The synthetic codes 0–5 in `DIST_CODES`, the canonical residual codes
    /// in `RESIDUAL_CODES` (shared code space with `DIST_CODES`), and the RE
    /// codes in `RE_DIST_CODES` must agree with the decode arms. Co-located
    /// here so an arm change without a table change is caught.
    #[test]
    fn code_tables_match_decode_arms() {
        let dist_lookup = |name: &str| DIST_CODES.iter().find(|(n, _)| *n == name).map(|(_, c)| *c);
        for (name, code) in [
            ("normal", 0),
            ("binary", 1),
            ("right_skewed", 2),
            ("left_skewed", 3),
            ("high_kurtosis", 4),
            ("uniform", 5),
        ] {
            assert_eq!(
                dist_lookup(name),
                Some(code),
                "DIST_CODES drifted at {name}"
            );
        }
        let resid_lookup = |name: &str| {
            RESIDUAL_CODES
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, c)| *c)
        };
        for (name, code) in [
            ("normal", 0),
            ("right_skewed", 2),
            ("left_skewed", 3),
            ("high_kurtosis", 4),
            ("uniform", 5),
        ] {
            assert_eq!(
                resid_lookup(name),
                Some(code),
                "RESIDUAL_CODES drifted at {name}"
            );
        }
        // The v1 aliases are gone — the table must NOT resolve them.
        for alias in ["t", "heavy_tailed", "skewed"] {
            assert_eq!(resid_lookup(alias), None, "alias {alias} must be dropped");
        }
        // RE vocabulary is its own table (normal/heavy_tailed/right_skewed).
        let re_lookup = |name: &str| {
            RE_DIST_CODES
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, c)| *c)
        };
        assert_eq!(re_lookup("normal"), Some(0));
        assert_eq!(re_lookup("heavy_tailed"), Some(1));
        assert_eq!(re_lookup("right_skewed"), Some(2));
        // Uploaded sentinels have no decode arm but must survive in the table.
        assert_eq!(dist_lookup("uploaded_data"), Some(99));
        // JSON bridge round-trips every entry. The residual / RE-dist serializers
        // are exercised only by the Python/R bridges (invisible to Rust tests), so
        // pin them here too: each must emit its full table, not "" / a wrong string.
        let parse =
            |s: &str| serde_json::from_str::<std::collections::BTreeMap<String, i32>>(s).unwrap();
        assert_eq!(parse(&dist_codes_json()).len(), DIST_CODES.len());
        assert_eq!(parse(&residual_codes_json()), {
            RESIDUAL_CODES
                .iter()
                .map(|(n, c)| (n.to_string(), *c))
                .collect::<std::collections::BTreeMap<_, _>>()
        });
        assert_eq!(parse(&re_dist_codes_json()), {
            RE_DIST_CODES
                .iter()
                .map(|(n, c)| (n.to_string(), *c))
                .collect::<std::collections::BTreeMap<_, _>>()
        });
    }

    /// `binary` (code 1) is in DIST_CODES but has no residual meaning, so
    /// `residual_dist_from_code(1)` must reject it by name — not fall through to the
    /// opaque "residual_code=1" catch-all.
    #[test]
    fn residual_dist_from_code_rejects_binary_by_name() {
        match residual_dist_from_code(1) {
            Err(SpecError::UnknownResidualDist { name }) => assert_eq!(name, "binary"),
            other => panic!("expected UnknownResidualDist(\"binary\"), got {other:?}"),
        }
    }

    #[test]
    fn build_columns_carries_sampled_proportions() {
        use crate::formula::parse_formula;
        use crate::variables::build_predictor_table;

        let f = parse_formula("y = group").unwrap();
        let preds = vec![PredictorSpec {
            name: "group".into(),
            pinned: false,
            kind: VarKind::Factor {
                levels: vec!["A".into(), "B".into(), "C".into()],
                proportions: vec![0.4, 0.3, 0.3],
                reference: "A".into(),
                sampled_proportions: Some(false),
            },
        }];
        let table = build_predictor_table(&f, &preds).unwrap();
        let cols = build_columns(&table);
        let f_col = cols
            .iter()
            .find(|c| matches!(c, ColumnSpec::FactorSynthetic { .. }))
            .unwrap();
        assert!(matches!(
            f_col,
            ColumnSpec::FactorSynthetic {
                sampled_proportions: Some(false),
                ..
            }
        ));
    }

    #[test]
    fn lme_scenario_knob_accepted_through_build_contract_with_skeleton() {
        // An LME spec whose scenario has random_effect_dist != 0 must succeed
        // through build_contract_with_skeleton (the outer pass sets Mle and
        // re-validates). Before this fix, the inner Ols template pre-validate
        // fires invariant_13 (LmeScenarioRequiresMle) and erroneously rejects it.
        use engine_contract::{ClusterSizing, ClusterSpec, OutcomeKind};

        let mut spec = simple_spec();
        spec.scenarios = vec![ScenarioInput {
            name: "lme_scenario".into(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: vec![],
            residual_change_prob: 0.0,
            residual_dists: vec![],
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            random_effect_dist: 1, // t — triggers lme = Some(...)
            random_effect_df: 5.0,
            icc_noise_sd: 0.0,
        }];
        let cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 10 }, 0.25);
        let result = build_contract_with_skeleton(
            &spec,
            OutcomeKind::Continuous,
            None,
            None, // default coupling → Mle (because clusters non-empty)
            0.0,
            vec![cluster],
        );
        assert!(
            result.is_ok(),
            "LME scenario knob must succeed through build_contract_with_skeleton; got: {result:?}"
        );
    }

    /// A 4-level factor in a factor-only formula (`y = group`) produces 3 non-reference
    /// dummies (k−1 = 3) plus the intercept = 5 design terms. The factor is the only
    /// predictor, so it lands at generation column id 0; the dummies carry level_index
    /// 1, 2, 3 (reference level 0 is suppressed). Pinning this prevents a dropped or
    /// duplicated dummy from silently passing downstream validation.
    #[test]
    fn four_level_factor_emits_4_design_terms() {
        let spec = LinearSpec {
            formula: "y = group".into(),
            predictors: vec![PredictorSpec {
                name: "group".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["A".into(), "B".into(), "C".into(), "D".into()],
                    proportions: vec![0.25, 0.25, 0.25, 0.25],
                    reference: "A".into(),
                    sampled_proportions: None,
                },
            }],
            effects: vec![
                EffectAssignment {
                    name: "group[B]".into(),
                    size: 0.2,
                },
                EffectAssignment {
                    name: "group[C]".into(),
                    size: 0.3,
                },
                EffectAssignment {
                    name: "group[D]".into(),
                    size: 0.4,
                },
            ],
            correlations: vec![],
            alpha: 0.05,
            correction: Correction::None,
            targets: vec!["overall".into()],
            heteroskedasticity: HeteroskedasticityInput::default(),
            residual: ResidualSpec::default(),
            max_failed_fraction: 0.1,
            scenarios: vec![],
            test_formula: None,
            report_overall: false,
            contrast_pairs: vec![],
            posthoc_requests: vec![],
            upload: None,
            cluster_level_vars: vec![],
            wald_se: Default::default(),
            nagq: 1,
        };
        let contracts = build_linear_contract(&spec).unwrap();
        let c = &contracts[0];
        // Factor-only: the factor is the sole generation column → ColumnId(0).
        let factor_col = c.generation.columns[0].clone();
        assert!(
            matches!(factor_col, ColumnSpec::FactorSynthetic { n_levels: 4, .. }),
            "expected FactorSynthetic{{n_levels:4}}, got {factor_col:?}"
        );
        // Const + (k−1) dummies = 1 + 3 = 4 terms.
        let terms = &c.design_generation.terms;
        assert_eq!(terms.len(), 4, "terms={terms:?}");
        // Read the actual column id from the contract rather than hardcoding.
        let factor_col_id = match &c.generation.columns[0] {
            ColumnSpec::FactorSynthetic { .. } => engine_contract::ids::ColumnId(0),
            other => panic!("unexpected column kind: {other:?}"),
        };
        // Collect the three non-reference DummyOf terms and check level indices.
        let dummies: Vec<_> = terms
            .iter()
            .filter_map(|t| match t {
                DesignTerm::DummyOf {
                    column,
                    level_index,
                } => Some((column, level_index)),
                _ => None,
            })
            .collect();
        assert_eq!(dummies.len(), 3, "expected 3 dummies, terms={terms:?}");
        for (col, _) in &dummies {
            assert_eq!(**col, factor_col_id, "dummy column id mismatch");
        }
        let mut indices: Vec<u32> = dummies.iter().map(|(_, &li)| li).collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![1, 2, 3], "dummy level_indices={indices:?}");
    }
}
