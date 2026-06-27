//! Debug / introspection FFI contract — the frozen `DebugReport` report shape.
//!
//! Both the types and the `debug_report` capability are always compiled (the
//! types are serde derives only, no runtime cost; `debug_report` runs the
//! single-threaded engine-core observer driver, never the hot path). There is
//! no cargo feature gate. Shape frozen at 2.0.0.
//!
//! Stage labels `D-A`…`D-F` cross-reference derivation sites within this file:
//! D-A synthesized design-column labels / formula, D-B the natural (unsquared)
//! statistic + threshold convention, D-C the estimator → statistic-family map,
//! D-D the effective-correlation surfacing rule, D-E effective effect sizes
//! under synthesized names, D-F power derived iff both `stats` and `crit`.

use serde::{Deserialize, Serialize};

use crate::result::PowerResult;
use engine_contract::SimulationContract;

/// Which stages to compute. A `false` flag ⇒ that `Option` in `DebugReport`
/// is `None` and the engine skips its work. There is no `power` flag:
/// `DebugReport.power` is derived, computed iff both `stats` and `crit` are set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageMask {
    pub input: bool,
    pub data: bool,
    pub dispatch: bool,
    pub stats: bool,
    pub crit: bool,
}

impl StageMask {
    pub fn all() -> Self {
        Self {
            input: true,
            data: true,
            dispatch: true,
            stats: true,
            crit: true,
        }
    }
    pub fn none() -> Self {
        Self {
            input: false,
            data: false,
            dispatch: false,
            stats: false,
            crit: false,
        }
    }
}

/// Column-major dense matrix — R-native AND faer-native, maps to an R matrix
/// with zero transpose. Shared by `ResolvedInput` + `DebugData`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub nrow: usize,
    pub ncol: usize,
}

/// Generation kind. Serialises to `"continuous"` | `"binary"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Continuous,
    Binary,
}

/// Estimator family. Serialises to `"ols"` | `"glm"` | `"mle"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Estimator {
    Ols,
    Glm,
    Mle,
}

/// Distribution family a decision statistic is compared against. Shared by
/// `TargetStats.statistic_kind` and `TargetCrit.distribution` so the two line
/// up per target by construction. Serialises to `"t"|"f"|"wald_chi2"|"z"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticKind {
    T,
    F,
    WaldChi2,
    Z,
}

impl StatisticKind {
    /// Stable lowercase code every host surfaces (`"t"`, `"f"`, `"wald_chi2"`,
    /// `"z"`) — cross-port contract; do not re-spell in adapters.
    pub fn as_str(&self) -> &'static str {
        match self {
            StatisticKind::T => "t",
            StatisticKind::F => "f",
            StatisticKind::WaldChi2 => "wald_chi2",
            StatisticKind::Z => "z",
        }
    }
}

/// The internal route the orchestrator chose for a scenario. Self-describing
/// JSON envelope (`tag = "route"`), mirroring `EstimatorExtras`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "route", rename_all = "snake_case")]
pub enum DispatchRoute {
    /// Monte-Carlo simulator — the only path; all designs route here.
    Simulated,
}

/// The bundled report. Flat `Option`s: a `None` stage was not selected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugReport {
    pub input: Option<ResolvedInput>,
    pub data: Option<DebugData>,
    pub dispatch: Option<DebugDispatch>,
    pub stats: Option<DebugStats>,
    pub crit: Option<DebugCrit>,
    /// The `PowerResult` this debug run produced. `Some` iff both `stats` and
    /// `crit` stages ran. Lets a host show power next to the chain and makes
    /// the stats→power self-consistency check a direct compare.
    pub power: Option<PowerResult>,
}

/// Stage 0 — validates "the engine understood my inputs."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedInput {
    /// The validated contract with `Option` defaults filled — what the engine parsed.
    pub contract: SimulationContract,
    /// Correlation matrix actually used (col-major, n_non_factor²). `None` when
    /// no (non-identity) correlation is present. See D-D.
    pub effective_correlation: Option<Matrix>,
    /// Effect sizes (scenario base) by synthesized predictor name. See D-E.
    pub effective_effects: Vec<(String, f64)>,
    pub resolved_alpha: f64,
}

/// Stage 1 — sim-0 draw, for DGP checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugData {
    /// Design matrix for sim 0, column-major. nrow = n, ncol = n_terms.
    pub design: Matrix,
    /// Synthesized column labels (intercept, predictors, dummy levels). See D-A.
    pub design_columns: Vec<String>,
    /// Generated outcome for sim 0, length n.
    pub outcome: Vec<f64>,
    /// Cluster/group id per row for LME designs, length n. `None` if not clustered.
    pub cluster_ids: Option<Vec<u32>>,
    /// Per-extra-grouping level ids for LME designs (declaration order, each
    /// length n; layout-derived). Empty when the design has no extra groupings.
    #[serde(default)]
    pub extra_grouping_ids: Vec<Vec<u32>>,
    /// The per-sim seed engine-core used for sim 0: `pcg_mix64(scenario_base, 0)`.
    pub sim0_seed: u64,
}

/// Stage 2 — validates the router ("right number the right way").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugDispatch {
    /// Synthesized formula string (see D-A).
    pub formula: String,
    /// Which path the orchestrator chose for THIS scenario.
    pub route: DispatchRoute,
    pub outcome_kind: OutcomeKind,
    pub estimator: Estimator,
}

/// Stage 3 — the full empirical sampling distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugStats {
    pub targets: Vec<TargetStats>,
    /// Per-sim convergence flag, length N.
    pub converged: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetStats {
    pub target_index: usize,
    pub target_label: String,
    /// The DECISION statistic per sim, length N — the empirical sampling
    /// distribution. Natural (unsquared) statistic; see D-B.
    pub statistic: Vec<f64>,
    pub statistic_kind: StatisticKind,
}

/// Stage 4 — the decision threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugCrit {
    pub targets: Vec<TargetCrit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetCrit {
    pub target_index: usize,
    pub target_label: String,
    /// Natural (unsquared) threshold; see D-B.
    pub critical_value: f64,
    pub alpha: f64,
    pub distribution: StatisticKind,
    /// df slots: `[]` z/normal, `[df]` t/χ², `[df1, df2]` F.
    pub df: Vec<f64>,
    pub two_sided: bool,
}

/// Result of the `data → results` debug path: one fit of a provided dataset.
/// `betas` align to `design_columns` (length `ncol`); `targets` carry the
/// per-target se/statistic/critical-value for the tested coefficients.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadDataResult {
    pub design_columns: Vec<String>,
    pub betas: Vec<f64>,
    pub converged: bool,
    pub targets: Vec<LoadDataTarget>,
    /// Estimated variance components τ̂²_g = θ̂_g²·σ̂² (order [primary,
    /// extras…]); surfaced by the general multi-grouping Mle path only at this
    /// milestone, empty elsewhere.
    #[serde(default)]
    pub variance_components: Vec<f64>,
    /// σ̂² from the same fit; NaN when not surfaced.
    #[serde(default = "nan_default")]
    pub sigma_sq_hat: f64,
    /// Off-diagonal correlations of the q_p×q_p RE correlation matrix,
    /// vech column-major (q_p(q_p−1)/2 entries; empty when the primary has
    /// no slope). Component order [intercept, slope_0, …]; entry order
    /// corr(slope_0,int), corr(slope_1,int), corr(slope_1,slope_0), …
    /// The q_p diagonal variances ride in variance_components[0..q_p].
    #[serde(default)]
    pub re_corr: Vec<f64>,
}

fn nan_default() -> f64 {
    f64::NAN
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadDataTarget {
    /// Kernel design-column index this target tests (0 = intercept).
    pub target_index: usize,
    /// Human label for that column (e.g. `"col_1"`).
    pub target_label: String,
    pub beta: f64,
    pub se: f64,
    /// Natural (unsquared) decision statistic.
    pub statistic: f64,
    pub statistic_kind: StatisticKind,
    /// Natural (unsquared) threshold.
    pub critical_value: f64,
    pub alpha: f64,
    /// df slots: `[df_resid]` for OLS t.
    pub df: Vec<f64>,
    pub two_sided: bool,
}

/// Synthesized design-column labels (D-A): intercept, continuous, factor
/// dummies, then interaction columns (appended last, mirroring the kernel
/// design-matrix layout `1 + n_non_factor + n_factor_dummies + interactions`).
fn synth_design_columns(
    n_non_factor: u32,
    n_factor_dummies: u32,
    n_interactions: u32,
) -> Vec<String> {
    let mut v = Vec::with_capacity(
        1 + n_non_factor as usize + n_factor_dummies as usize + n_interactions as usize,
    );
    v.push("intercept".to_string());
    for i in 1..=n_non_factor {
        v.push(format!("col_{i}"));
    }
    for k in 0..n_factor_dummies {
        v.push(format!("factor_dummy_{k}"));
    }
    for k in 0..n_interactions {
        v.push(format!("interaction_{k}"));
    }
    v
}

fn synth_target_labels(spec: &engine_core::SimulationSpec) -> Vec<String> {
    let mut labels = Vec::with_capacity(spec.target_indices.len() + spec.contrast_pairs.len());
    for &ti in &spec.target_indices {
        labels.push(format!("col_{ti}"));
    }
    for &(p, n) in &spec.contrast_pairs {
        labels.push(format!("col_{p}_vs_col_{n}"));
    }
    labels
}

fn synth_formula(spec: &engine_core::SimulationSpec) -> String {
    let p = 1 + spec.n_non_factor + spec.n_factor_dummies;
    if p <= 1 {
        "outcome ~ 1".to_string()
    } else {
        let terms: Vec<String> = (1..p).map(|i| format!("col_{i}")).collect();
        format!("outcome ~ {}", terms.join(" + "))
    }
}

mod entry {
    use super::*;
    use crate::find_power::lower_contracts;
    use crate::result::OrchestratorError;
    use engine_contract::EstimatorSpec;

    /// Map kernel estimator → wire `Estimator`.
    fn wire_estimator(e: EstimatorSpec) -> Estimator {
        match e {
            EstimatorSpec::Ols => Estimator::Ols,
            EstimatorSpec::Glm => Estimator::Glm,
            EstimatorSpec::Mle => Estimator::Mle,
        }
    }

    /// Map kernel estimator → per-target statistic family (D-C). The OLS-vs-z
    /// partition is owned by `EstimatorSpec`; this only names the wire variants.
    fn wire_statistic_kind(e: EstimatorSpec) -> StatisticKind {
        if e.uses_student_t() {
            StatisticKind::T
        } else {
            StatisticKind::Z
        }
    }

    /// Run the staged debug pipeline for one contract: lowers it through
    /// `find_power`'s exact path (identical scenario base seed), then
    /// assembles the report stages requested in `stages`.
    ///
    /// # Errors
    /// Contract-lowering failures and engine errors from the single-threaded
    /// observer run.
    pub fn debug_report(
        c: &SimulationContract,
        seed: u64,
        n: usize,
        n_sims: usize,
        stages: StageMask,
    ) -> Result<DebugReport, OrchestratorError> {
        // Reuse find_power's EXACT lowering → identical scenario base seed.
        let scenarios = lower_contracts(std::slice::from_ref(c), seed)?;
        let scenario = scenarios
            .into_iter()
            .next()
            .ok_or_else(|| OrchestratorError::InvalidScenarios("empty".into()))?;
        let spec = &scenario.spec;
        let base_seed = scenario.base_seed;

        // Synthesized design-column labels (D-A). Computed once: the input
        // stage borrows it (effective_effects names) and the later data stage
        // moves it into its `design_columns` field — both run after this bind.
        let design_columns = synth_design_columns(
            spec.n_non_factor,
            spec.n_factor_dummies,
            spec.interactions.len() as u32,
        );

        let dispatch = if stages.dispatch {
            Some(DebugDispatch {
                formula: synth_formula(spec),
                route: DispatchRoute::Simulated,
                outcome_kind: match spec.outcome_kind {
                    engine_contract::OutcomeKind::Continuous => OutcomeKind::Continuous,
                    engine_contract::OutcomeKind::Binary => OutcomeKind::Binary,
                },
                estimator: wire_estimator(spec.estimator),
            })
        } else {
            None
        };

        let input = if stages.input {
            // effective_correlation (D-D): surface only when there is genuine
            // off-diagonal structure — ≥2 non-factors, a full matrix, and at
            // least one entry departing from the identity. Both guards
            // short-circuit before the flat index scan, so it never reads
            // out of bounds.
            let nnf = spec.n_non_factor as usize;
            let has_offdiagonal = nnf >= 2
                && spec.correlation.len() == nnf * nnf
                && !(0..nnf * nnf).all(|idx| {
                    let v = spec.correlation[idx];
                    if idx % nnf == idx / nnf {
                        (v - 1.0).abs() < 1e-12
                    } else {
                        v.abs() < 1e-12
                    }
                });
            let effective_correlation = has_offdiagonal.then(|| Matrix {
                data: spec.correlation.clone(),
                nrow: nnf,
                ncol: nnf,
            });

            // effective_effects (D-E): spec.effect_sizes with synthesized names.
            // effect_sizes[0] is the intercept slot; 1.. map to col_1.. then dummies.
            let effective_effects: Vec<(String, f64)> = spec
                .effect_sizes
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    let name = design_columns
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("col_{i}"));
                    (name, b)
                })
                .collect();

            // contract: echo the validated input with design_test default filled.
            let mut contract = c.clone();
            if contract.design_test.is_none() {
                contract.design_test = Some(contract.design_generation.clone());
            }

            Some(ResolvedInput {
                contract,
                effective_correlation,
                effective_effects,
                resolved_alpha: spec.crit_values.alpha,
            })
        } else {
            None
        };
        use engine_core::introspect::{run_introspect, IntrospectMask};

        let need_power = stages.stats && stages.crit; // D-F
        let icall = if stages.data || stages.stats || stages.crit || need_power {
            Some(run_introspect(
                spec,
                n as u32,
                n_sims as u32,
                base_seed,
                IntrospectMask {
                    stats: stages.stats,
                    data: stages.data,
                    crit: stages.crit || need_power,
                    power: need_power,
                },
            )?)
        } else {
            None
        };

        let data = if stages.data {
            let id = icall
                .as_ref()
                .and_then(|o| o.data.as_ref())
                .expect("data captured when stages.data");
            Some(DebugData {
                design: Matrix {
                    data: id.design.clone(),
                    nrow: id.nrow,
                    ncol: id.ncol,
                },
                design_columns,
                outcome: id.outcome.clone(),
                cluster_ids: id.cluster_ids.clone(),
                extra_grouping_ids: spec
                    .cluster
                    .as_ref()
                    .map(|cl| {
                        (0..cl.extra_groupings.len())
                            .map(|g| {
                                (0..id.nrow)
                                    .map(|i| cl.extra_level_of_row(g, i) as u32)
                                    .collect()
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                sim0_seed: id.sim0_seed,
            })
        } else {
            None
        };
        let labels = synth_target_labels(spec);
        let stat_kind = wire_statistic_kind(spec.estimator);

        let stats = if stages.stats {
            let o = icall.as_ref().expect("introspect call");
            let stat_sq = o.stat_sq.as_ref().expect("stat_sq when stages.stats");
            let converged_u = o.converged.as_ref().expect("converged when stages.stats");
            let nt = labels.len();
            let targets = (0..nt)
                .map(|t| {
                    // Natural statistic (D-B): sqrt of squared statistic per sim.
                    let statistic: Vec<f64> = (0..n_sims)
                        .map(|s| {
                            let v = stat_sq[s * nt + t];
                            if v.is_nan() {
                                f64::NAN
                            } else {
                                v.sqrt()
                            }
                        })
                        .collect();
                    TargetStats {
                        target_index: t,
                        target_label: labels[t].clone(),
                        statistic,
                        statistic_kind: stat_kind,
                    }
                })
                .collect();
            Some(DebugStats {
                targets,
                converged: converged_u.clone(),
            })
        } else {
            None
        };

        let crit = if stages.crit {
            let o = icall.as_ref().expect("introspect call");
            let ic = o.crit.as_ref().expect("crit when stages.crit");
            // Natural threshold (D-B). df slots per D-C.
            let crit_value = ic.crit_sq_uncorrected.sqrt();
            let df = spec.estimator.df_slots(ic.df_resid);
            let targets = labels
                .iter()
                .enumerate()
                .map(|(t, label)| TargetCrit {
                    target_index: t,
                    target_label: label.clone(),
                    critical_value: crit_value,
                    alpha: spec.crit_values.alpha,
                    distribution: stat_kind,
                    df: df.clone(),
                    two_sided: true,
                })
                .collect();
            Some(DebugCrit { targets })
        } else {
            None
        };

        let power = if need_power {
            let o = icall.as_ref().expect("introspect call");
            let batch = o.batch.as_ref().expect("batch when need_power");
            let results = crate::aggregation::aggregate_batch(
                batch,
                &spec.target_indices,
                &spec.contrast_pairs,
                &spec.estimator,
            );
            // aggregate_batch ships n:0 (filled by caller in find_power.rs); replicate.
            results.into_iter().next().map(|mut pr| {
                pr.n = n;
                pr
            })
        } else {
            None
        };

        Ok(DebugReport {
            input,
            data,
            dispatch,
            stats,
            crit,
            power,
        })
    }

    /// Fit a provided dataset with the configured solver (the `data → results`
    /// path). Lowers the contract exactly as `debug_report` does so the spec /
    /// test definition is identical, then calls `fit_provided_data`. Wired for
    /// all three estimators (OLS t, GLM/MLE Wald z).
    #[allow(clippy::too_many_arguments)] // raw data buffers for one validation call site
    pub fn debug_load_data(
        c: &SimulationContract,
        seed: u64,
        design: &[f64],
        nrow: usize,
        ncol: usize,
        outcome: &[f64],
        cluster_ids: Option<&[u32]>,
        // Validation-only override of the contract's `wald_se` SE mode. `None`
        // keeps the contract's configured mode; `Some(_)` forces the kernel SE
        // (used by the R Oracle harness to read per-fit `rx` vs `hessian` SE on
        // one dataset). Bypasses the port-level `rx` rejection by design — no
        // production path passes a `Some`.
        wald_se: Option<engine_contract::WaldSe>,
    ) -> Result<LoadDataResult, OrchestratorError> {
        let scenarios = lower_contracts(std::slice::from_ref(c), seed)?;
        let mut scenario = scenarios
            .into_iter()
            .next()
            .ok_or_else(|| OrchestratorError::InvalidScenarios("empty".into()))?;
        if let Some(w) = wald_se {
            scenario.spec.wald_se = w;
        }
        let spec = &scenario.spec;

        let r = engine_core::introspect::fit_provided_data(
            spec,
            design,
            nrow,
            ncol,
            outcome,
            cluster_ids,
        )?;

        let design_columns = synth_design_columns(
            spec.n_non_factor,
            spec.n_factor_dummies,
            spec.interactions.len() as u32,
        );
        let stat_kind = wire_statistic_kind(spec.estimator);
        let df = spec.estimator.df_slots(r.df_resid);
        let crit_value = r.crit_sq_uncorrected.sqrt();

        let targets = r
            .target_cols
            .iter()
            .enumerate()
            .map(|(t, &col)| {
                let ci = col as usize;
                LoadDataTarget {
                    target_index: ci,
                    target_label: design_columns
                        .get(ci)
                        .cloned()
                        .unwrap_or_else(|| format!("col_{ci}")),
                    beta: r.betas[ci],
                    se: r.se[t],
                    statistic: r.statistic[t],
                    statistic_kind: stat_kind,
                    critical_value: crit_value,
                    alpha: spec.crit_values.alpha,
                    df: df.clone(),
                    two_sided: true,
                }
            })
            .collect();

        Ok(LoadDataResult {
            design_columns,
            betas: r.betas,
            converged: r.converged,
            targets,
            variance_components: r.variance_components,
            sigma_sq_hat: r.sigma_sq_hat,
            re_corr: r.re_corr,
        })
    }
}

pub use entry::{debug_load_data, debug_report};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enums_serialize_as_snake_case_plain_strings() {
        assert_eq!(
            serde_json::to_string(&OutcomeKind::Continuous).unwrap(),
            "\"continuous\""
        );
        assert_eq!(
            serde_json::to_string(&OutcomeKind::Binary).unwrap(),
            "\"binary\""
        );
        assert_eq!(serde_json::to_string(&Estimator::Ols).unwrap(), "\"ols\"");
        assert_eq!(serde_json::to_string(&Estimator::Glm).unwrap(), "\"glm\"");
        assert_eq!(serde_json::to_string(&Estimator::Mle).unwrap(), "\"mle\"");
        assert_eq!(serde_json::to_string(&StatisticKind::T).unwrap(), "\"t\"");
        assert_eq!(serde_json::to_string(&StatisticKind::F).unwrap(), "\"f\"");
        assert_eq!(
            serde_json::to_string(&StatisticKind::WaldChi2).unwrap(),
            "\"wald_chi2\""
        );
        assert_eq!(serde_json::to_string(&StatisticKind::Z).unwrap(), "\"z\"");
    }

    #[test]
    fn dispatch_route_is_self_describing_tagged() {
        let s = serde_json::to_string(&DispatchRoute::Simulated).unwrap();
        assert_eq!(s, "{\"route\":\"simulated\"}");
        let back: DispatchRoute = serde_json::from_str(&s).unwrap();
        assert_eq!(back, DispatchRoute::Simulated);
    }

    #[test]
    fn stage_mask_constructors() {
        let all = StageMask::all();
        assert!(all.input && all.data && all.dispatch && all.stats && all.crit);
        let none = StageMask::none();
        assert!(!none.input && !none.data && !none.dispatch && !none.stats && !none.crit);
    }

    /// DBG-04: the declared stage set is exactly five flags
    /// (`input`, `data`, `dispatch`, `stats`, `crit`), frozen at 2.0.0, and
    /// there is no `power` flag (power is derived from stats && crit). The
    /// exhaustive destructuring is a compile-time forward-trap: adding or
    /// removing a flag breaks this test, signalling a contract change.
    #[test]
    fn stage_mask_has_exactly_five_frozen_flags() {
        let StageMask {
            input,
            data,
            dispatch,
            stats,
            crit,
        } = StageMask::all();
        // Count the flags by binding each exactly once; all are `true` here.
        let flags = [input, data, dispatch, stats, crit];
        assert_eq!(flags.len(), 5, "StageMask must declare exactly five stages");
        assert!(
            flags.iter().all(|&f| f),
            "StageMask::all() must set every flag"
        );

        // No `power` flag exists: DebugReport.power is derived, gated on
        // stats && crit, never a StageMask field. Round-trip the mask through
        // JSON and confirm the key set is exactly the five frozen names.
        let json = serde_json::to_value(StageMask::none()).unwrap();
        let obj = json
            .as_object()
            .expect("StageMask serialises to a JSON object");
        let mut keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["crit", "data", "dispatch", "input", "stats"]);
        assert!(
            !obj.contains_key("power"),
            "no `power` flag is allowed on StageMask"
        );
    }

    #[test]
    fn matrix_roundtrips_msgpack() {
        let m = Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0],
            nrow: 2,
            ncol: 2,
        };
        let bytes = rmp_serde::to_vec(&m).unwrap();
        let back: Matrix = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn debug_report_full_roundtrips_msgpack_and_json() {
        let report = DebugReport {
            input: Some(ResolvedInput {
                contract: engine_contract::fixtures::example1_simple_ols(),
                effective_correlation: None,
                effective_effects: vec![("col_1".into(), 0.5)],
                resolved_alpha: 0.05,
            }),
            data: Some(DebugData {
                design: Matrix {
                    data: vec![1.0, 1.0, 0.1, 0.9],
                    nrow: 2,
                    ncol: 2,
                },
                design_columns: vec!["intercept".into(), "col_1".into()],
                outcome: vec![1.2, 3.4],
                cluster_ids: None,
                extra_grouping_ids: vec![],
                sim0_seed: 0xABCD,
            }),
            dispatch: Some(DebugDispatch {
                formula: "outcome ~ col_1".into(),
                route: DispatchRoute::Simulated,
                outcome_kind: OutcomeKind::Continuous,
                estimator: Estimator::Ols,
            }),
            stats: Some(DebugStats {
                targets: vec![TargetStats {
                    target_index: 0,
                    target_label: "col_1".into(),
                    statistic: vec![2.1, 1.9],
                    statistic_kind: StatisticKind::T,
                }],
                converged: vec![true, true],
            }),
            crit: Some(DebugCrit {
                targets: vec![TargetCrit {
                    target_index: 0,
                    target_label: "col_1".into(),
                    critical_value: 1.98,
                    alpha: 0.05,
                    distribution: StatisticKind::T,
                    df: vec![98.0],
                    two_sided: true,
                }],
            }),
            power: None,
        };

        let bytes = rmp_serde::to_vec(&report).unwrap();
        let back: DebugReport = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(report, back);

        let json = serde_json::to_string(&report).unwrap();
        let back2: DebugReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, back2);
    }

    #[test]
    fn debug_report_all_none_roundtrips() {
        let empty = DebugReport {
            input: None,
            data: None,
            dispatch: None,
            stats: None,
            crit: None,
            power: None,
        };
        let back: DebugReport = rmp_serde::from_slice(&rmp_serde::to_vec(&empty).unwrap()).unwrap();
        assert_eq!(empty, back);
    }
}
