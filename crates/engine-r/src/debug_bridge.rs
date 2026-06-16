//! Binding for `engine_orchestrator::debug::debug_report`.
//! Decodes the msgpack `Vec<SimulationContract>` blob, selects the scenario at
//! `scenario_index`, builds a `StageMask` from five booleans, calls the
//! orchestrator's `debug_report`, and returns the `DebugReport` as a nested R
//! list.
//!
//! Matrix layout note: design matrices in `DebugReport` are column-major (same
//! as R's `matrix()` default), so the flat `Vec<f64>` + nrow + ncol can be
//! passed directly to `matrix(data, nrow, ncol)` in R with zero transpose.

use engine_orchestrator::debug::{
    debug_load_data as orch_load_data, DebugCrit, DebugData, DebugDispatch, DebugReport,
    DebugStats, DispatchRoute, Estimator, LoadDataResult, LoadDataTarget, Matrix, OutcomeKind,
    ResolvedInput, StageMask, TargetCrit, TargetStats,
};
use engine_orchestrator::ScenarioResult;
use extendr_api::error::Result;
use extendr_api::prelude::*;

use crate::orchestrator_bridge::{decode_contracts, power_result_to_list};

// ── top-level binding ─────────────────────────────────────────────────────────

/// Run the debug introspection pipeline for a single scenario.
///
/// * `contracts`       — msgpack `raw` vector produced by `build_contract_from_spec`.
/// * `scenario_index`  — 0-based index into the `Vec<SimulationContract>`.
/// * `seed`            — RNG base seed (f64 in R; cast to u64).
/// * `n`               — sample size for this debug run.
/// * `n_sims`          — number of simulations to run.
/// * `stage_*`         — booleans controlling which stages to compute.
///
/// Returns a named list with fields `input`, `data`, `dispatch`, `stats`,
/// `crit`, `power` (each NULL when the stage was not requested or not available).
#[extendr]
pub fn debug_report(
    contracts: &[u8],
    scenario_index: i32,
    seed: f64,
    n: i32,
    n_sims: i32,
    stage_input: bool,
    stage_data: bool,
    stage_dispatch: bool,
    stage_stats: bool,
    stage_crit: bool,
) -> Result<List> {
    let all_contracts = decode_contracts(contracts)?;
    let idx = scenario_index as usize;
    let contract = all_contracts
        .get(idx)
        .ok_or_else(|| Error::Other(format!("scenario_index {idx} out of range")))?;

    let mask = StageMask {
        input: stage_input,
        data: stage_data,
        dispatch: stage_dispatch,
        stats: stage_stats,
        crit: stage_crit,
    };

    let report = engine_orchestrator::debug::debug_report(
        contract,
        seed as u64,
        n as usize,
        n_sims as usize,
        mask,
    )
    .map_err(|e| Error::Other(format!("{e}")))?;

    Ok(debug_report_to_list(&report))
}

// ── DebugReport → R list ──────────────────────────────────────────────────────

fn debug_report_to_list(r: &DebugReport) -> List {
    let input_robj: Robj = match &r.input {
        Some(ri) => resolved_input_to_list(ri).into_robj(),
        None => r!(NULL),
    };
    let data_robj: Robj = match &r.data {
        Some(d) => debug_data_to_list(d).into_robj(),
        None => r!(NULL),
    };
    let dispatch_robj: Robj = match &r.dispatch {
        Some(d) => debug_dispatch_to_list(d).into_robj(),
        None => r!(NULL),
    };
    let stats_robj: Robj = match &r.stats {
        Some(s) => debug_stats_to_list(s).into_robj(),
        None => r!(NULL),
    };
    let crit_robj: Robj = match &r.crit {
        Some(c) => debug_crit_to_list(c).into_robj(),
        None => r!(NULL),
    };
    let power_robj: Robj = match &r.power {
        Some(pr) => {
            // Wrap the single PowerResult into a ScenarioResult so we can
            // reuse power_result_to_list (which expects the multi-scenario envelope).
            let sr = ScenarioResult {
                scenarios: vec![("optimistic".to_string(), pr.clone())],
            };
            power_result_to_list(&sr).into_robj()
        }
        None => r!(NULL),
    };

    list!(
        input = input_robj,
        data = data_robj,
        dispatch = dispatch_robj,
        stats = stats_robj,
        crit = crit_robj,
        power = power_robj
    )
}

// ── Matrix helper ─────────────────────────────────────────────────────────────

/// Decode a `Matrix` as `list(data = <f64 vec>, nrow = <i32>, ncol = <i32>)`.
/// Column-major layout matches R's `matrix()` default — zero transpose needed.
fn matrix_to_list(m: &Matrix) -> List {
    list!(
        data = m.data.clone(),
        nrow = m.nrow as i32,
        ncol = m.ncol as i32
    )
}

// ── Stage decoders ────────────────────────────────────────────────────────────

fn resolved_input_to_list(ri: &ResolvedInput) -> List {
    let eff_corr_robj: Robj = match &ri.effective_correlation {
        Some(m) => matrix_to_list(m).into_robj(),
        None => r!(NULL),
    };
    // effective_effects: named numeric vector (names = predictor labels, values = effect sizes).
    let eff_names: Vec<String> = ri
        .effective_effects
        .iter()
        .map(|(n, _)| n.clone())
        .collect();
    let eff_vals: Vec<f64> = ri.effective_effects.iter().map(|(_, v)| *v).collect();
    let eff_named =
        List::from_names_and_values(eff_names, eff_vals.iter().map(|v| r!(*v))).unwrap();

    list!(
        effective_correlation = eff_corr_robj,
        effective_effects = eff_named,
        resolved_alpha = ri.resolved_alpha
    )
}

fn debug_data_to_list(d: &DebugData) -> List {
    let design_robj = matrix_to_list(&d.design).into_robj();
    let cluster_robj: Robj = match &d.cluster_ids {
        Some(ids) => {
            let v: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
            r!(v)
        }
        None => r!(NULL),
    };
    // One integer vector per extra grouping (declaration order); an empty R
    // list when the design has no extra groupings.
    let extra_ids_robj: Robj = List::from_values(
        d.extra_grouping_ids
            .iter()
            .map(|ids| ids.iter().map(|&x| x as i32).collect::<Vec<i32>>()),
    )
    .into();
    list!(
        design = design_robj,
        design_columns = d.design_columns.clone(),
        outcome = d.outcome.clone(),
        cluster_ids = cluster_robj,
        extra_grouping_ids = extra_ids_robj,
        sim0_seed = d.sim0_seed as f64
    )
}

fn dispatch_route_string(route: &DispatchRoute) -> String {
    match route {
        DispatchRoute::Simulated => "simulated".to_string(),
    }
}

fn outcome_kind_string(ok: &OutcomeKind) -> &'static str {
    match ok {
        OutcomeKind::Continuous => "continuous",
        OutcomeKind::Binary => "binary",
    }
}

fn estimator_string(e: &Estimator) -> &'static str {
    match e {
        Estimator::Ols => "ols",
        Estimator::Glm => "glm",
        Estimator::Mle => "mle",
    }
}

fn debug_dispatch_to_list(d: &DebugDispatch) -> List {
    list!(
        formula = d.formula.clone(),
        route = dispatch_route_string(&d.route),
        outcome_kind = outcome_kind_string(&d.outcome_kind),
        estimator = estimator_string(&d.estimator)
    )
}

fn target_stats_to_list(ts: &TargetStats) -> List {
    list!(
        target_index = ts.target_index as i32,
        target_label = ts.target_label.clone(),
        statistic = ts.statistic.clone(),
        statistic_kind = ts.statistic_kind.as_str()
    )
}

fn debug_stats_to_list(s: &DebugStats) -> List {
    let targets = List::from_values(
        s.targets
            .iter()
            .map(|t| target_stats_to_list(t).into_robj()),
    );
    let converged: Vec<bool> = s.converged.clone();
    list!(targets = targets, converged = converged)
}

fn target_crit_to_list(tc: &TargetCrit) -> List {
    list!(
        target_index = tc.target_index as i32,
        target_label = tc.target_label.clone(),
        critical_value = tc.critical_value,
        alpha = tc.alpha,
        distribution = tc.distribution.as_str(),
        df = tc.df.clone(),
        two_sided = tc.two_sided
    )
}

fn debug_crit_to_list(c: &DebugCrit) -> List {
    let targets = List::from_values(c.targets.iter().map(|t| target_crit_to_list(t).into_robj()));
    list!(targets = targets)
}

/// Fit a provided dataset with the configured solver (the `data → results`
/// path). `design` is column-major (R `as.numeric(matrix)` order). Pass an
/// empty `cluster_ids` (`integer(0)`) for non-clustered designs.
///
/// Returns a named list: `betas` (numeric, length ncol), `design_columns`
/// (character), `converged` (logical), `targets` (list of per-target lists with
/// `target_index`, `target_label`, `beta`, `se`, `statistic`, `statistic_kind`,
/// `critical_value`, `alpha`, `df`, `two_sided`).
#[extendr]
pub fn debug_load_data(
    contracts: &[u8],
    scenario_index: i32,
    seed: f64,
    design: &[f64],
    nrow: i32,
    ncol: i32,
    outcome: &[f64],
    cluster_ids: &[i32],
) -> Result<List> {
    let all_contracts = decode_contracts(contracts)?;
    let idx = scenario_index as usize;
    let contract = all_contracts
        .get(idx)
        .ok_or_else(|| Error::Other(format!("scenario_index {idx} out of range")))?;

    let cluster_u32: Option<Vec<u32>> = if cluster_ids.is_empty() {
        None
    } else {
        Some(cluster_ids.iter().map(|&x| x as u32).collect())
    };

    let result = orch_load_data(
        contract,
        seed as u64,
        design,
        nrow as usize,
        ncol as usize,
        outcome,
        cluster_u32.as_deref(),
    )
    .map_err(|e| Error::Other(format!("{e}")))?;

    Ok(load_data_result_to_list(&result))
}

fn load_data_target_to_list(t: &LoadDataTarget) -> List {
    list!(
        target_index = t.target_index as i32,
        target_label = t.target_label.clone(),
        beta = t.beta,
        se = t.se,
        statistic = t.statistic,
        statistic_kind = t.statistic_kind.as_str(),
        critical_value = t.critical_value,
        alpha = t.alpha,
        df = t.df.clone(),
        two_sided = t.two_sided
    )
}

fn load_data_result_to_list(r: &LoadDataResult) -> List {
    let targets = List::from_values(
        r.targets
            .iter()
            .map(|t| load_data_target_to_list(t).into_robj()),
    );
    list!(
        betas = r.betas.clone(),
        design_columns = r.design_columns.clone(),
        converged = r.converged,
        targets = targets,
        variance_components = r.variance_components.clone(),
        sigma_sq_hat = r.sigma_sq_hat,
        re_corr = r.re_corr.clone()
    )
}

// ── extendr module registration ───────────────────────────────────────────────

extendr_module! {
    mod debug_bridge;
    fn debug_report;
    fn debug_load_data;
}
