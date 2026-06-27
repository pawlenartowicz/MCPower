//! `run_find_power` / `run_find_sample_size` — binds `assemble_spec` to the orchestrator entry points; the single-core WASM twins (`run_single_core_find_power` / `run_single_core_find_sample_size`) sit beside them. Also houses `get_effects_from_data` — the effect-recovery path that fits a provided design (OLS/GLM/MLE) to retrieve standardized slopes. `set_n_threads` re-exports the engine-core thread-pool initialiser for Tauri callers.
//!
//! Invariants a maintainer must not break:
//!  * `run_single_core_*` are the WASM twins of `run_find_*` — any change to one
//!    (assemble path, parameter handling, output shape) lands in both.
//!  * `get_effects_from_data` duplicates the `assemble.rs` var_types→`VarKind`
//!    projection; the two must stay in sync.

use engine_orchestrator::{
    find_power, find_sample_size, single_core_find_power, single_core_find_sample_size,
    CancellationToken, PowerResult, SampleSizeMethod, SampleSizeResult, ScenarioResult,
};
use engine_spec_builder::upload::standardize_continuous;

use crate::{
    app_spec::{ClusterDim, EffectSize, MixedOutcome, MixedSpec},
    assemble::assemble_spec,
    error::AdapterError,
    progress::{EmitterSink, ProgressEmitter},
    AppSpec,
};

/// Translate `spec` → contracts via `assemble_spec`, then call the orchestrator's
/// `find_power` entry point.
///
/// Dispatch twin: `run_single_core_find_power` (WASM worker pool) — change together.
///
/// # Errors
/// - Spec validation failures propagate as `AdapterError::SpecBuilder` variants.
/// - Orchestrator failures propagate as `AdapterError::Orchestrator`.
pub fn run_find_power(
    spec: &AppSpec,
    sample_size: usize,
    emitter: &dyn ProgressEmitter,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<PowerResult>, AdapterError> {
    let contracts = assemble_spec(spec)?;
    let RunInputs {
        base_seed, n_sims, ..
    } = run_inputs_of(spec);
    let mut sink = EmitterSink::new(emitter);
    let result = find_power(
        &contracts,
        sample_size,
        n_sims,
        base_seed,
        Some(&mut sink),
        cancel,
    )?;
    Ok(result)
}

/// Translate `spec` → contracts via `assemble_spec`, then call the orchestrator's
/// `find_sample_size` entry point. `target_power` is read from `spec`.
///
/// Dispatch twin: `run_single_core_find_sample_size` (WASM worker pool, Grid only) — change together.
///
/// # Errors
/// - Spec validation failures propagate as `AdapterError::SpecBuilder` variants.
/// - Orchestrator failures propagate as `AdapterError::Orchestrator`.
pub fn run_find_sample_size(
    spec: &AppSpec,
    bounds: (usize, usize),
    method: SampleSizeMethod,
    emitter: &dyn ProgressEmitter,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<SampleSizeResult>, AdapterError> {
    let contracts = assemble_spec(spec)?;
    let RunInputs {
        base_seed,
        n_sims,
        target_power,
    } = run_inputs_of(spec);
    let mut sink = EmitterSink::new(emitter);
    let result = find_sample_size(
        &contracts,
        target_power,
        bounds,
        n_sims,
        method,
        base_seed,
        Some(&mut sink),
        cancel,
    )?;
    Ok(result)
}

/// Single-core twin of `run_find_power` (WASM worker pool); change together.
/// `n_sims` and `base_seed` are caller-supplied per worker (not read from the spec).
pub fn run_single_core_find_power(
    spec: &AppSpec,
    sample_size: usize,
    n_sims: usize,
    base_seed: u64,
    emitter: &dyn ProgressEmitter,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<PowerResult>, AdapterError> {
    let contracts = assemble_spec(spec)?;
    let mut sink = EmitterSink::new(emitter);
    let result = single_core_find_power(
        &contracts,
        sample_size,
        n_sims,
        base_seed,
        Some(&mut sink),
        cancel,
    )?;
    Ok(result)
}

/// Single-core twin of `run_find_sample_size` (Grid only, WASM worker pool);
/// change together. `n_sims` and `base_seed` are per-worker; `target_power` is
/// read from the spec (run-level, shared).
pub fn run_single_core_find_sample_size(
    spec: &AppSpec,
    bounds: (usize, usize),
    method: SampleSizeMethod,
    n_sims: usize,
    base_seed: u64,
    emitter: &dyn ProgressEmitter,
    cancel: &CancellationToken,
) -> Result<ScenarioResult<SampleSizeResult>, AdapterError> {
    let contracts = assemble_spec(spec)?;
    let target_power = run_inputs_of(spec).target_power;
    let mut sink = EmitterSink::new(emitter);
    let result = single_core_find_sample_size(
        &contracts,
        target_power,
        bounds,
        n_sims,
        method,
        base_seed,
        Some(&mut sink),
        cancel,
    )?;
    Ok(result)
}

struct RunInputs {
    base_seed: u64,
    n_sims: usize,
    target_power: f64,
}

fn run_inputs_of(spec: &AppSpec) -> RunInputs {
    match spec {
        AppSpec::Linear(l) => RunInputs {
            base_seed: l.seed,
            n_sims: l.n_sims as usize,
            target_power: l.target_power,
        },
        AppSpec::Logit(l) => RunInputs {
            base_seed: l.seed,
            n_sims: l.n_sims as usize,
            target_power: l.target_power,
        },
        AppSpec::Mixed(m) => RunInputs {
            base_seed: m.seed,
            n_sims: m.n_sims as usize,
            target_power: m.target_power,
        },
    }
}

/// Configure the rayon thread pool used by `run_find_power` / `run_find_sample_size`.
/// Must be called before the first engine invocation; a second call returns an error
/// because the pool is already initialised. Mirrors `mcpower._engine.set_n_threads(n)`.
///
/// # Errors
/// Returns a `String` error when `n < 1` or the pool is already initialised.
pub fn set_n_threads(n: usize) -> Result<(), String> {
    engine_core::set_n_threads(n).map_err(|e| format!("{e}"))
}

/// Effect-recovery preview returned by [`get_effects_from_data`]: the fitted
/// fixed effects plus the two scalars the same fit can recover but that are not
/// effects — the estimated cluster ICC and the binary-outcome baseline
/// probability. Hosts present these and apply them on demand; the engine never
/// writes them back into a spec.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EffectsFromData {
    pub effects: Vec<EffectSize>,
    /// Estimated intraclass correlation from the random-intercept fit's variance
    /// components. `Some` only for the `Mixed` arm when the fit converged to a
    /// non-empty `variance_components`; `None` otherwise (regression families and
    /// non-converged mixed fits).
    #[serde(default)]
    pub cluster_icc: Option<f64>,
    /// Baseline event probability recovered from the fitted intercept
    /// `1/(1+exp(−betas[0]))`. `Some` only for binary outcomes (`Logit`, or
    /// `Mixed` + `MixedOutcome::Binary`); `None` for Gaussian/linear arms, where
    /// `betas[0]` is a raw mean offset rather than a probability.
    #[serde(default)]
    pub baseline_probability: Option<f64>,
}

/// Recover standardized effect sizes from uploaded data by fitting the model to
/// the provided columns. The fitted estimator follows the spec variant: OLS for
/// `Linear`, GLM for `Logit`, MLE for `Mixed`.
///
/// The canonical column order mirrors Python and R:
/// `[Intercept, non-factors (z-scored or centred), factor dummies (0/1),
///   interactions (elementwise products)]`.
/// Outcome scaling is estimator-specific: OLS z-scores the outcome (population
/// SD, ddof=0), recovering the standardized β; GLM and MLE fit it on its native
/// scale — recovery is the inverse of the generation convention. MLE recovery is
/// fixed-effects-only and threads the uploaded grouping column (named by the
/// `(1|group)` term) as cluster IDs. Betas are mapped to semantic names and the
/// intercept is dropped from the returned `effects`.
///
/// Alongside the effects, the same fit yields two optional scalars (see
/// [`EffectsFromData`]): the estimated cluster ICC (mixed arm, converged fit
/// only) and the binary-outcome baseline probability (the inverse-logit of the
/// fitted intercept).
pub fn get_effects_from_data(spec: &AppSpec) -> Result<EffectsFromData, AdapterError> {
    use engine_spec_builder::variables::build_predictor_table;

    // Extract the fields common to every estimator spec plus the outcome-scaling
    // mode and (for MLE) the grouping-column name. OLS z-scores the outcome; GLM
    // and MLE fit it on its native scale. MLE additionally threads the uploaded
    // grouping column as cluster_ids (handled below).
    let (parsed_formula, var_types, csv_opt, base_seed, native_outcome, cluster_name) = match spec {
        AppSpec::Linear(l) => (&l.parsed_formula, &l.var_types, &l.csv, l.seed, false, None),
        AppSpec::Logit(l) => (&l.parsed_formula, &l.var_types, &l.csv, l.seed, true, None),
        AppSpec::Mixed(m) => (
            &m.parsed_formula,
            &m.var_types,
            &m.csv,
            m.seed,
            true,
            Some(m.cluster_name.as_str()),
        ),
    };

    let csv = csv_opt.as_ref().ok_or(AdapterError::GetEffectsNoCsv)?;

    let outcome_name = &parsed_formula.outcome;
    let y_col = csv
        .columns
        .iter()
        .find(|c| &c.name == outcome_name)
        .ok_or_else(|| AdapterError::GetEffectsOutcomeMissing {
            outcome: outcome_name.clone(),
        })?;

    let nrow = csv.n_rows as usize;

    let col_by_name: std::collections::HashMap<&str, &engine_spec_builder::input::UploadColumn> =
        csv.columns.iter().map(|c| (c.name.as_str(), c)).collect();

    // Guard up-front — every column we will actually read must have
    // exactly nrow values.  A short column would silently corrupt the design
    // (mismatched vector lengths, wrong means) if we only used .take(nrow).
    if y_col.values.len() != nrow {
        return Err(AdapterError::GetEffectsColumnLength {
            name: outcome_name.clone(),
            expected: nrow,
            got: y_col.values.len(),
        });
    }

    // Reconstruct the formula string and predictor specs exactly as assemble.rs does.
    let formula_str = {
        let outcome = &parsed_formula.outcome;
        let mut terms: Vec<String> = parsed_formula.predictors.clone();
        for iterm in &parsed_formula.interaction_terms {
            terms.push(iterm.join(":"));
        }
        if terms.is_empty() {
            format!("{outcome} ~ 1")
        } else {
            format!("{outcome} ~ {}", terms.join(" + "))
        }
    };

    let parsed = engine_spec_builder::parse_formula(&formula_str)?;

    // Build PredictorSpec list from the spec's var_types, mirroring assemble.rs logic.
    let predictors: Result<Vec<engine_spec_builder::PredictorSpec>, AdapterError> = parsed_formula
        .predictors
        .iter()
        .map(|pred_name| {
            use crate::app_spec::VarType;
            use engine_spec_builder::VarKind;

            let var_type = var_types.iter().find(|vt| match vt {
                VarType::Numeric { name, .. }
                | VarType::Binary { name, .. }
                | VarType::Factor { name, .. } => name == pred_name,
            });

            let kind = match var_type {
                None | Some(VarType::Numeric { .. }) => VarKind::Normal,
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
                    ..
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
                    // Level names drive the recovered effect names, so they must
                    // match what the host displays: user labels first, then the
                    // factor column's labels, then "1".."k" (matching assemble.rs).
                    let levels: Vec<String> = if factor_labels.len() == n {
                        factor_labels.clone()
                    } else if let Some(col) = col_by_name.get(name.as_str()) {
                        if col.labels.len() == n {
                            col.labels.clone()
                        } else {
                            (1..=*factor_n_levels).map(|i| format!("{i}")).collect()
                        }
                    } else {
                        (1..=*factor_n_levels).map(|i| format!("{i}")).collect()
                    };
                    // Honor the chosen reference level (mirrors assemble.rs); the
                    // recovery design must use the same reference as the power run
                    // or the recovered β labels are off by the reference shift.
                    let reference = levels[*factor_reference as usize].clone();
                    VarKind::Factor {
                        levels,
                        proportions: factor_proportions.clone(),
                        reference,
                        sampled_proportions: None,
                    }
                }
            };

            Ok(engine_spec_builder::PredictorSpec {
                name: pred_name.clone(),
                pinned: false,
                kind,
            })
        })
        .collect();
    let predictors = predictors?;

    let table = build_predictor_table(&parsed, &predictors)?;

    // Recovery design, single-sourced in engine-spec-builder: canonical order
    // [Intercept, non-factors, factor dummies, interactions]; py/R bind to the
    // same `build_recovery_design` via their bridges.
    let engine_spec_builder::RecoveryDesign {
        design_flat,
        semantic_names,
        ncol,
    } = engine_spec_builder::build_recovery_design(&table, &csv.columns, nrow)?;

    // Outcome scaling depends on the estimator: z-score for OLS (recovers the
    // standardized β); native scale for GLM/MLE (the inverse of generation — the
    // logit fitter expects raw 0/1, the mixed fitter the raw response). Length
    // already guarded above.
    let outcome: Vec<f64> = if native_outcome {
        y_col.values.clone()
    } else {
        standardize_continuous(&y_col.values)
    };

    // MLE (mixed) recovery is fixed-effects-only: the uploaded data must carry
    // the grouping column named by the formula's random-intercept term. Map its
    // distinct values to contiguous 0-based cluster IDs (first-appearance order)
    // so the LME fitter's per-cluster suff-stats line up; `n_clusters` then sizes
    // the workspace cluster buffers via the zeroed spec's cluster_dim override.
    let (cluster_ids, n_clusters): (Option<Vec<u32>>, usize) = if let Some(cname) = cluster_name {
        let gcol =
            col_by_name
                .get(cname)
                .ok_or_else(|| AdapterError::GetEffectsPredictorMissing {
                    name: cname.to_string(),
                })?;
        if gcol.values.len() != nrow {
            return Err(AdapterError::GetEffectsColumnLength {
                name: cname.to_string(),
                expected: nrow,
                got: gcol.values.len(),
            });
        }
        let mut id_of: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        let mut ids: Vec<u32> = Vec::with_capacity(nrow);
        for &v in &gcol.values {
            let next = id_of.len() as u32;
            let id = *id_of.entry(v.to_bits()).or_insert(next);
            ids.push(id);
        }
        let k = id_of.len();
        (Some(ids), k)
    } else {
        (None, 0)
    };

    // Build a copy of the spec with all effect sizes zeroed (the fit ignores
    // them — same pattern as Python's `get_effects_from_data`). The variant is
    // preserved so the contract carries the right estimator (Ols/Glm/Mle). For
    // Mixed, pin the cluster count to the uploaded data's distinct-group count so
    // the fitter's cluster buffers match the cluster_ids built above.
    // Effect recovery fits a single clean design to the uploaded data: zero the
    // effects and drop any scenario perturbations so `assemble_spec` yields one
    // baseline contract (the `.next()` below then takes it).
    let zeroed_spec = match spec {
        AppSpec::Linear(l) => AppSpec::Linear({
            let mut s = l.clone();
            for e in &mut s.effects {
                e.value = 0.0;
            }
            s.scenarios.clear();
            s
        }),
        AppSpec::Logit(l) => AppSpec::Logit({
            let mut s = l.clone();
            for e in &mut s.effects {
                e.value = 0.0;
            }
            s.scenarios.clear();
            s
        }),
        AppSpec::Mixed(m) => AppSpec::Mixed({
            let mut s = m.clone();
            for e in &mut s.effects {
                e.value = 0.0;
            }
            s.scenarios.clear();
            s.cluster_dim = ClusterDim::NClusters {
                value: n_clusters as u32,
            };
            s
        }),
    };

    // `assemble_spec` builds the contract's factor levels as "1".."k"
    // while `design_flat` above uses real label-based dummies.  This divergence
    // is safe: `debug_load_data` fits the CALLER-PROVIDED `design_flat` directly
    // and never rebuilds the design from the contract's dummy declarations — only
    // `n_levels` / the dummy COUNT matters for the fit, and that agrees.
    let contracts = assemble_spec(&zeroed_spec)?;
    let contract = contracts
        .into_iter()
        .next()
        .ok_or(AdapterError::EmptyContracts)?;

    let result = engine_orchestrator::debug::debug_load_data(
        &contract,
        base_seed,
        &design_flat,
        nrow,
        ncol,
        &outcome,
        cluster_ids.as_deref(),
        None,
    )?;

    if result.betas.len() != semantic_names.len() {
        return Err(AdapterError::GetEffectsBetaColumnMismatch {
            betas: result.betas.len(),
            cols: semantic_names.len(),
        });
    }

    let effects: Vec<EffectSize> = result.betas[1..]
        .iter()
        .zip(semantic_names[1..].iter())
        .map(|(&beta, name)| EffectSize {
            name: name.clone(),
            value: beta,
        })
        .collect();

    // Estimated ICC from the random-intercept fit's variance components — only
    // meaningful for the Mixed arm, and only when the fit converged to a
    // non-empty variance estimate (an empty `variance_components` is the
    // non-converged case the Py/R hosts report as "unavailable"). τ̂² is the
    // primary grouping's intercept variance; the residual scale depends on the
    // outcome: π²/3 (latent log-odds) for a binary GLMM — `sigma_sq_hat` is a
    // 1.0 placeholder there and must not be used — and σ̂² for a Gaussian LME.
    // Mirrors the Python/R `get_effects_from_data` ICC report.
    let cluster_icc: Option<f64> = match spec {
        AppSpec::Mixed(m) => result.variance_components.first().map(|&tau_sq| {
            let residual = match m.outcome {
                MixedOutcome::Binary { .. } => std::f64::consts::PI.powi(2) / 3.0,
                MixedOutcome::Gaussian => result.sigma_sq_hat,
            };
            tau_sq / (tau_sq + residual)
        }),
        _ => None,
    };

    // Baseline probability from the fitted intercept — the inverse-logit of
    // `betas[0]`, undoing the forward `logit(p)` the assembler applies. Only a
    // binary outcome has a baseline-probability knob (Logit, or Mixed+Binary);
    // for the Gaussian/linear arms `betas[0]` is a raw mean offset, not a
    // probability, so there is nothing to recover.
    let is_binary = matches!(
        spec,
        AppSpec::Logit(_)
            | AppSpec::Mixed(MixedSpec {
                outcome: MixedOutcome::Binary { .. },
                ..
            })
    );
    let baseline_probability = is_binary.then(|| crate::assemble::logistic(result.betas[0]));

    Ok(EffectsFromData {
        effects,
        cluster_icc,
        baseline_probability,
    })
}
