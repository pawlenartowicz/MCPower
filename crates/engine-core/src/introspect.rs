//! Debug introspection driver. (Always compiled; no cargo feature.)
//!
//! Leaves the hot-path `run_one_sim` untouched. For the authoritative answer it
//! calls the real `run_batch_st`; for observation it re-walks the same per-sim
//! seed chain. Single-threaded, fully deterministic.

use crate::batch::run_batch_st;
use crate::batch::run_batch_st_capture;
use crate::critvals::CritValueTable;
use crate::data_gen::generate_sim_data;
use crate::rng::pcg_mix64;
use crate::spec::{BatchResult, EstimatorSpec, SimulationSpec};
use crate::workspace::SimWorkspace;
use crate::EngineError;
use glmm::loop_advanced::{fit_suff_stats_t_sq, OlsScratch, OlsSuffStats};
use glmm::loop_advanced::{glm_irls_fit, GlmScratch};

/// Which observations to capture. Engine-core-local (orchestrator maps its
/// `StageMask` onto this — engine-core never depends on orchestrator types).
#[derive(Debug, Clone, Copy)]
pub struct IntrospectMask {
    pub stats: bool,
    pub data: bool,
    pub crit: bool,
    pub power: bool,
}

/// Sim-0 raw draw.
#[derive(Debug, Clone)]
pub struct IntrospectData {
    /// Column-major, `nrow * ncol`.
    pub design: Vec<f64>,
    pub nrow: usize,
    pub ncol: usize,
    pub outcome: Vec<f64>,
    pub cluster_ids: Option<Vec<u32>>,
    pub sim0_seed: u64,
}

/// Per-target uncorrected threshold + df, in the engine's squared space.
#[derive(Debug, Clone)]
pub struct IntrospectCrit {
    /// First target's (squared) uncorrected threshold. Every target shares it
    /// except under the between-within denominator df, which this single slot
    /// cannot express; NaN when the spec has no targets.
    pub crit_sq_uncorrected: f64,
    /// `N` minus the FITTED predictor count as f64 — the same entry width the
    /// per-target rows below are built at (OLS df_resid; ignored by z-families).
    pub df_resid: f64,
    /// Per-target (squared) uncorrected threshold, aligned to `target_indices`
    /// then `contrast_pairs` — the full row `CritValueTable::t_crit_sq_uncorrected`
    /// carries for this N. Differs across targets only under the between-within
    /// denominator df; otherwise every entry equals `crit_sq_uncorrected`.
    pub crit_sq_uncorrected_per_target: Vec<f64>,
    /// Per-target denominator df, mirroring `CritValueTable::df_per_target`'s row
    /// for this N: empty when the threshold is standard-normal (Wald-z) for
    /// every target, one entry per target when it is Student-t/F-referenced
    /// (OLS always; a mixed target under the between-within denominator df).
    pub df_per_target: Vec<f64>,
}

/// Result of fitting a *provided* dataset (the `data → results` debug path).
/// Mirrors what the sim loop computes per sim, but for one externally-supplied
/// dataset instead of a generated draw. `betas`/`se` are aligned to the design
/// columns / targets respectively; `statistic` is the natural (unsquared)
/// per-target decision statistic. Estimator-agnostic in shape; all three
/// estimator producers (OLS, GLM, MLE) are wired.
#[derive(Debug, Clone)]
pub struct IntrospectResults {
    /// Fitted coefficient per design column of the design that was HANDED IN,
    /// length `ncol`. A column a `test_formula` reduction drops carries NaN —
    /// the reduced fit never estimated it.
    pub betas: Vec<f64>,
    /// Standard error per *target*, length `n_targets` = `sqrt(var_diag[t])`.
    pub se: Vec<f64>,
    /// Natural (unsquared) decision statistic per target, length `n_targets`.
    pub statistic: Vec<f64>,
    /// Kernel column index each target refers to, length `n_targets`
    /// (= `spec.target_indices`). Lets the host align targets to design columns.
    pub target_cols: Vec<u32>,
    pub converged: bool,
    /// First target's (squared) uncorrected threshold — see
    /// [`IntrospectCrit::crit_sq_uncorrected`].
    pub crit_sq_uncorrected: f64,
    /// `N` minus the FITTED predictor count (the full design width unless a
    /// `test_formula` reduces it). Only meaningful for OLS (Student-t df_resid);
    /// the z-based GLM/MLE estimators are df-independent and ignore it.
    pub df_resid: f64,
    /// Per-target (squared) uncorrected threshold — see
    /// [`IntrospectCrit::crit_sq_uncorrected_per_target`].
    pub crit_sq_uncorrected_per_target: Vec<f64>,
    /// Per-target denominator df — see [`IntrospectCrit::df_per_target`].
    pub df_per_target: Vec<f64>,
    /// Estimated variance components: the diagonals of each grouping's RE
    /// covariance D = σ̂²·ΛΛ′. Order [primary (q_p entries), extras in
    /// declaration order (q_g entries each)]. Filled by the general (multi-
    /// grouping) Mle path and the Glm+cluster path; empty elsewhere.
    pub variance_components: Vec<f64>,
    /// σ̂² from the same fit; NaN when not surfaced.
    pub sigma_sq_hat: f64,
    /// Off-diagonal correlations of the q_p×q_p RE correlation matrix,
    /// vech column-major (q_p(q_p−1)/2 entries; empty when the primary has
    /// no slope). Component order [intercept, slope_0, …]; entry order
    /// corr(slope_0,int), corr(slope_1,int), corr(slope_1,slope_0), …
    /// The q_p diagonal variances ride in variance_components[0..q_p].
    pub re_corr: Vec<f64>,
}

/// Everything `run_introspect` can surface, one `Option` per stage mask flag.
#[derive(Debug, Clone)]
pub struct IntrospectOut {
    /// Authoritative batch (run_batch_st). `Some` iff `mask.power`.
    pub batch: Option<BatchResult>,
    /// Squared decision statistic, shape (n_sims, n_targets) row-major.
    /// `Some` iff `mask.stats`.
    pub stat_sq: Option<Vec<f64>>,
    /// Per-sim convergence, length n_sims. `Some` iff `mask.stats`.
    pub converged: Option<Vec<bool>>,
    /// Sim-0 draw. `Some` iff `mask.data`.
    pub data: Option<IntrospectData>,
    /// Threshold + df. `Some` iff `mask.crit`.
    pub crit: Option<IntrospectCrit>,
}

fn capture_sim0_data(
    spec: &SimulationSpec,
    n: u32,
    base_seed: u64,
) -> Result<IntrospectData, EngineError> {
    // Debug-only spec preparation — mirrors run_batch_impl's `spec_with_tables`
    // block (t3_table + het_coeffs); change together. Without it the sim-0 draw
    // runs a DIFFERENT DGP than the stats stage fits: default het_coeffs read
    // driver-std 0 ⇒ heteroskedasticity silently inert, and a HighKurtosis
    // marginal panics on the missing t(3) table — breaking debug invariant 4
    // (data → stat coherence).
    let mut prepared = spec.clone();
    prepared.t3_table = Some(crate::marginals::T3PpfTable::build_default());
    prepared.het_coeffs = prepared.compute_het_coeffs();
    let spec = &prepared;

    let n_usize = n as usize;
    let p = n_predictors_total(spec) as usize;
    let n_non_factor = spec.n_non_factor as usize;
    let n_factors = spec.factor_n_levels.len();
    let n_clusters_ws = spec
        .cluster
        .as_ref()
        .map(|c| c.sizing.n_clusters_at(n_usize))
        .unwrap_or(0);

    let mut ws = SimWorkspace::new(n_usize, p, n_non_factor, n_factors, spec.cluster.as_ref());
    generate_sim_data(spec, 0, base_seed, &mut ws)?;

    // Column-major extraction of the first `n` rows of `x_full` (max_n == n here).
    let mut design = Vec::with_capacity(n_usize * p);
    for j in 0..p {
        for i in 0..n_usize {
            design.push(ws.x_full[(i, j)] as f64);
        }
    }
    let outcome: Vec<f64> = ws.y_full[..n_usize].iter().map(|&v| v as f64).collect();
    let cluster_ids = if n_clusters_ws > 0 {
        Some(ws.cluster_ids[..n_usize].to_vec())
    } else {
        None
    };

    Ok(IntrospectData {
        design,
        nrow: n_usize,
        ncol: p,
        outcome,
        cluster_ids,
        sim0_seed: pcg_mix64(base_seed, 0),
    })
}

/// Fit a provided `(design, outcome[, cluster_ids])` with the shipped solver
/// for `spec.estimator` and return betas/se/statistic + the crit threshold.
/// The data is injected into the same `SimWorkspace` buffers the hot loop fills
/// by generation, then dispatched through the *identical* kernel each
/// `run_one_sim` arm uses — OLS (`OlsSuffStats::add_rows` →
/// `fit_suff_stats_t_sq`), unclustered GLM (`glm_irls_fit`), or the mixed arms
/// (clustered GLMM / LMM) via `build_workspace` + `fit_on` — so this is the
/// shipped solver, not a parallel impl (proven by the orchestrator
/// bit-equivalence tests).
///
/// `design` is column-major `nrow * ncol` (the shape `capture_sim0_data`
/// emits). All three estimator arms are wired; `df_resid` is OLS-only-meaningful
/// (z-families are df-independent).
///
/// This function deliberately does NOT mirror the sparse-factor exclusion branch
/// of `run_one_sim` — debug fits provided bytes as-is so the L3 same-bytes
/// contract holds. It DOES honour a `test_formula` reduction (`spec.fit_columns`),
/// because that reduction is spec-constant: every sim of the corresponding
/// `find_power` run fits the same reduced design, so a debug fit of the full
/// design would answer a different question. `design`/`ncol` stay the full
/// generation design; the reduction happens on the way into the fit.
pub fn fit_provided_data(
    spec: &SimulationSpec,
    design: &[f64],
    nrow: usize,
    ncol: usize,
    outcome: &[f64],
    cluster_ids: Option<&[u32]>,
) -> Result<IntrospectResults, EngineError> {
    // `test_formula` reduction: fit only the columns the test design keeps, in
    // `build_exclusion_remap` order — the same reduced design every sim of the
    // hot path fits (batch.rs `test_reduces`; change together). `kept` maps a
    // fitted position back to the caller's design column. An empty
    // `fit_columns` makes `kept` the identity over the full design, so every
    // branch below collapses to the former full-width behaviour bit-for-bit.
    let (p_fit, _, kept) = entry_width(spec);
    let p_fit = p_fit as usize;
    let reduces = p_fit < ncol;
    let n_non_factor = spec.n_non_factor as usize;
    let n_factors = spec.factor_n_levels.len();
    let n_clusters_ws = spec
        .cluster
        .as_ref()
        .map(|c| c.sizing.n_clusters_at(nrow))
        .unwrap_or(0);

    let mut ws = SimWorkspace::new(nrow, p_fit, n_non_factor, n_factors, spec.cluster.as_ref());

    // Inject the provided data into the workspace buffers (column-major in),
    // gathering only the fitted columns. Captured design/outcome are f64;
    // narrow into the f32 data plane.
    for (j, &cj) in kept.iter().enumerate() {
        let src = cj as usize * nrow;
        for i in 0..nrow {
            ws.x_full[(i, j)] = design[src + i] as f32;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..nrow {
        ws.y_full[i] = outcome[i] as f32;
    }
    // f64 ingress widen — mirrors batch.rs (the fit kernels read x_full_f64/
    // y_full_f64; data plane stays f32). introspect injects directly rather than
    // via generate_sim_data, so widen here too. Per-column contiguous slices so
    // the cast autovectorizes (see batch.rs).
    for j in 0..p_fit {
        let src = ws.x_full.col(j).try_as_col_major().unwrap().as_slice();
        let dst = ws
            .x_full_f64
            .col_mut(j)
            .try_as_col_major_mut()
            .unwrap()
            .as_slice_mut();
        for (d, &s) in dst[..nrow].iter_mut().zip(&src[..nrow]) {
            *d = s as f64;
        }
    }
    for (d, &s) in ws.y_full_f64[..nrow].iter_mut().zip(&ws.y_full[..nrow]) {
        *d = s as f64;
    }
    if let Some(ids) = cluster_ids {
        if n_clusters_ws > 0 {
            ws.cluster_ids[..nrow].copy_from_slice(&ids[..nrow]);
        }
    }

    // Targets and the truth-β start move into the fitted design's index space
    // (mirrors the reduced arms of batch.rs). Every test target is in the test
    // design by construction, so no target is dropped. Both are exact clones of
    // the spec's own vectors when nothing is reduced.
    let fit_targets: Vec<u32> = if reduces {
        spec.target_indices
            .iter()
            .map(|&t| {
                kept.iter()
                    .position(|&c| c == t)
                    .expect("a test target is always in the test design") as u32
            })
            .collect()
    } else {
        spec.target_indices.clone()
    };
    let fit_beta: Vec<f64> = kept
        .iter()
        .map(|&cj| spec.effect_sizes[cj as usize])
        .collect();
    let n_test_targets = fit_targets.len();

    // Dispatch on the estimator — each arm mirrors the corresponding
    // `run_one_sim` arm exactly (same kernel, same injected bytes).
    #[allow(clippy::type_complexity)]
    // `pinned`: the mixed fit's pinned-component bitmask (0 for OLS / unclustered
    // GLM) — decides whether the collapsed crit table applies below.
    let (betas, se, statistic, converged, variance_components, sigma_sq_hat, re_corr, pinned): (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        bool,
        Vec<f64>,
        f64,
        Vec<f64>,
        u64,
    ) = match spec.estimator {
        EstimatorSpec::Ols => {
            // OLS — suff-stats then closed-form fit. var_diag / t_sq are
            // TARGET-indexed.
            ws.reset_suff_stats();
            {
                let x_seg = ws.x_full_f64.as_ref().subrows(0, nrow);
                let y_seg = &ws.y_full_f64[..nrow];
                let mut suff = OlsSuffStats {
                    xtx: ws.suff_xtx.as_mut(),
                    xty: &mut ws.suff_xty,
                    yty: &mut ws.suff_yty,
                    sum_y: &mut ws.suff_sum_y,
                    n_rows: &mut ws.suff_n_rows,
                    panel_x: &mut ws.panel_x,
                    panel_y: &mut ws.panel_y,
                };
                suff.add_rows(x_seg, y_seg);
            }
            let scratch = OlsScratch {
                fit_betas: &mut ws.fit_betas,
                fit_var_diag: &mut ws.fit_var_diag,
                fit_t_sq: &mut ws.fit_t_sq,
                fit_u_scratch: &mut ws.fit_u_scratch,
                fit_factor: ws.fit_factor.as_mut(),
                fit_rhs: ws.fit_rhs.as_mut(),
            };
            let fit = fit_suff_stats_t_sq(
                ws.suff_xtx.as_ref(),
                &ws.suff_xty,
                ws.suff_yty,
                ws.suff_sum_y,
                ws.suff_n_rows,
                &fit_targets,
                ws.suff_xtx_work.as_mut(),
                scratch,
            );
            let betas = fit.betas[..p_fit].to_vec();
            let se = (0..n_test_targets)
                .map(|t| fit.var_diag[t].sqrt())
                .collect();
            let statistic = (0..n_test_targets).map(|t| fit.t_sq[t].sqrt()).collect();
            (
                betas,
                se,
                statistic,
                fit.converged,
                vec![],
                f64::NAN,
                vec![],
                0,
            )
        }
        EstimatorSpec::Glm if spec.cluster.is_some() => {
            // Clustered-GLMM path — one-shot `build_workspace` + `fit_on`, mirroring
            // batch.rs's Glm+cluster arm (same kernel, same start). `into_fit` then
            // surfaces the RE (co)variance: D̂ = σ̂²·Λ̂Λ̂' with σ̂² ≡ 1 for the
            // fixed-dispersion families (binomial logit/probit, Poisson).
            let cluster = spec.cluster.as_ref().expect("guard ⇒ cluster");
            // Derive the family from the contract's outcome + link — the same helper
            // build_mixed_workspaces uses, so an uploaded clustered probit/Poisson
            // GLMM fits with the right family instead of falling back to logit.
            let family = crate::mixed_workspace::glmm_family(spec.outcome_kind, spec.link);
            let model = crate::mixed_workspace::cluster_to_model_spec(
                cluster,
                family,
                &spec.cluster_slope_design_cols,
                &spec.extra_slope_cols,
            );
            // One `fit_on` workspace at this single N (introspect fits one dataset,
            // not a grid). build_mixed_workspaces builds the sized spec, per-point
            // ids, and the frozen opts (target_indices/wald_se/nagq) exactly as the
            // production find_power path does.
            // Sized at the FITTED width: `p_fit` already equals
            // `fit_columns.len()` under a `test_formula`, so
            // build_mixed_workspaces' own reduction guard is inert and the
            // β-dimension matches the reduced x injected above. The RE design Z
            // stays full — only fixed columns shrink.
            let (mut mixed, ids, opts) = crate::mixed_workspace::build_mixed_workspaces(
                spec,
                &[nrow as u32],
                p_fit,
                &ws.cluster_ids,
                &ws.extra_grouping_ids,
            )
            .expect("Glm+cluster ⇒ build_mixed_workspaces is Some");
            // The batch opts carry the FULL-design target set; a reduced fit
            // needs its own (target_indices is not frozen by `fit_on`).
            let opts_red;
            let opts = if reduces {
                opts_red = glmm::FitOptions {
                    target_indices: fit_targets.clone(),
                    wald_se: crate::mixed_workspace::wald_se_to_glmm(spec.wald_se),
                    nagq: spec.nagq,
                    ..Default::default()
                };
                &opts_red
            } else {
                &opts
            };
            // `fit_on` takes row-major x[i·p + j]; x_full_f64 is column-major faer.
            for i in 0..nrow {
                for j in 0..p_fit {
                    ws.x_rowmajor[i * p_fit + j] = ws.x_full_f64[(i, j)];
                }
            }
            // Truth-start under the scenario's `truth_start` assumption: seed the
            // optimizer at the DGP-truth [β | θ] (well-behaved estimation) or start
            // blind. β is warm unconditionally (mirrors the retired `fit_glmm`, which
            // took β separately); when `truth_start` is off, θ must equal glmm's
            // blind None-θ seed (every entry THETA0 = 1.0) — passing it explicitly
            // reproduces that bit-for-bit. NOT a speed knob — the MLE is
            // start-sensitive. MIRRORS batch.rs's Glm+cluster arm — change together.
            let theta = crate::mixed_workspace::truth_theta(cluster, true);
            let start_theta = if spec.scenario.truth_start {
                theta
            } else {
                vec![1.0; theta.len()]
            };
            let start = glmm::StartValues {
                beta: fit_beta.clone(),
                theta: start_theta,
            };
            let view = glmm::loop_advanced::fit_on(
                &mut mixed[0],
                &ws.x_rowmajor[..nrow * p_fit],
                &ws.y_full_f64[..nrow],
                &ids[0],
                Some(&start),
                opts,
            );
            let converged = view.converged();
            let pinned = view.diagnostics().pinned_components;
            let betas = view.betas()[..p_fit].to_vec();
            // var_diag / t_sq are PREDICTOR-indexed for the mixed arms.
            let se = fit_targets
                .iter()
                .map(|&t| view.var_diag()[t as usize].sqrt())
                .collect();
            let statistic = fit_targets
                .iter()
                .map(|&t| view.t_sq()[t as usize].sqrt())
                .collect();
            // `into_fit` consumes the view; read the accessors above first.
            let fit = view.into_fit(
                &ws.x_rowmajor[..nrow * p_fit],
                &ws.y_full_f64[..nrow],
                &ids[0],
                nrow,
                p_fit,
                &model,
                opts,
            );
            let (variance_components, re_corr) = unpack_varcorr(&fit);
            (
                betas,
                se,
                statistic,
                converged,
                variance_components,
                fit.dispersion, // ≡ 1.0 for binomial/Poisson (dispersion fixed)
                re_corr,
                pinned,
            )
        }
        EstimatorSpec::Glm => {
            // Mirrors run_one_sim's GLM arm, including the spec-derived
            // truth start (β₀ = effect_sizes) — required for the
            // load_data ≡ hot-loop bit-equivalence invariant. No
            // suff-stats. var_diag / t_sq are TARGET-indexed (like OLS).
            let x_slice = ws.x_full_f64.as_ref().subrows(0, nrow);
            let y_slice = &ws.y_full_f64[..nrow];
            let scratch = GlmScratch {
                irls_eta: &mut ws.irls_eta[..nrow],
                irls_p: &mut ws.irls_p[..nrow],
                irls_w: &mut ws.irls_w[..nrow],
                irls_z: &mut ws.irls_z[..nrow],
                irls_betas: &mut ws.irls_betas,
                irls_betas_new: &mut ws.irls_betas_new,
                irls_var_diag: &mut ws.irls_var_diag,
                irls_t_sq: &mut ws.irls_t_sq,
                irls_u_scratch: &mut ws.irls_u_scratch,
                irls_xtwx: ws.irls_xtwx.as_mut(),
                irls_xtwz: &mut ws.irls_xtwz,
                irls_l: ws.irls_l.as_mut(),
                irls_wx: &mut ws.irls_wx,
            };
            let fit = glm_irls_fit(
                // Unclustered GLM family: Binary→logit/probit, Count→Poisson log
                // (mirrors the GLMM site — see mixed_workspace::glmm_family).
                crate::mixed_workspace::glmm_family(spec.outcome_kind, spec.link),
                f64::NAN, // nb_theta: ignored outside NB family (glm.rs doc comment)
                x_slice,
                y_slice,
                &fit_targets,
                Some(&fit_beta),
                None, // prior_w: MCPower has no prior-observation-weights feature
                None, // offset: no offset feature — byte-identical to the offset-free fit
                scratch,
            );
            let betas = fit.betas[..p_fit].to_vec();
            let se = (0..n_test_targets)
                .map(|t| fit.var_diag[t].sqrt())
                .collect();
            let statistic = (0..n_test_targets).map(|t| fit.t_sq[t].sqrt()).collect();
            (
                betas,
                se,
                statistic,
                fit.converged,
                vec![],
                f64::NAN,
                vec![],
                0,
            )
        }
        EstimatorSpec::Mle => {
            // Clustered LMM — one-shot `build_workspace` + `fit_on`, mirroring
            // batch.rs's single Mle arm. Both the intercept-only (former Brent
            // scalar) and the general (slopes / extra groupings) cases route
            // through the same `fit_on` (BOBYQA) now — the scalar Brent path is
            // retired (a deliberate Brent → BOBYQA numeric move on the
            // intercept-only case). Extra-grouping / slope layout is carried by
            // ws.cluster_ids + ws.extra_grouping_ids (laid out by the ctor from the
            // same layout functions as the hot path); build_mixed_workspaces slices
            // them by N. `into_fit` surfaces D̂ = σ̂²·Λ̂Λ̂' per grouping.
            let cluster = spec.cluster.as_ref().expect("Mle requires ClusterSpec");
            let mut model = crate::mixed_workspace::cluster_to_model_spec(
                cluster,
                glmm::Family::Gaussian,
                &spec.cluster_slope_design_cols,
                &spec.extra_slope_cols,
            );
            // A `test_formula` reduction strips every random slope (primary and
            // extras) so q_g = 1 everywhere: a slope may reference a fixed column
            // the reduced design no longer carries, and the kept columns don't map
            // to the full-design slope cols. MIRRORS batch.rs's reduced LMM fit —
            // change together.
            if reduces {
                if let Some(re) = model.re.as_mut() {
                    re.slopes.clear();
                    for g in &mut re.extra_groupings {
                        g.slopes.clear();
                    }
                }
            }
            // The reduced fit needs its own workspace: `build_mixed_workspaces`
            // rebuilds the model from `spec` (slopes intact) and sizes the LMM
            // β-dimension at the full width, so it cannot express the stripped
            // reduced model. Same construction as batch.rs's cold reduced fit.
            let (mut mixed, ids, opts) = if reduces {
                let opts = glmm::FitOptions {
                    target_indices: fit_targets.clone(),
                    wald_se: crate::mixed_workspace::wald_se_to_glmm(spec.wald_se),
                    nagq: spec.nagq,
                    ..Default::default()
                };
                let raw = glmm::GroupIds {
                    primary: ws.cluster_ids[..nrow].to_vec(),
                    extra: ws
                        .extra_grouping_ids
                        .iter()
                        .map(|v| v[..nrow].to_vec())
                        .collect(),
                };
                let (sized, sized_ids, perm) =
                    glmm::loop_advanced::spec_sized_from_ids_pub(&model, &raw);
                let w =
                    glmm::loop_advanced::build_workspace(&sized, perm, nrow, p_fit, &opts);
                (vec![w], vec![sized_ids.into_owned()], opts)
            } else {
                crate::mixed_workspace::build_mixed_workspaces(
                    spec,
                    &[nrow as u32],
                    p_fit,
                    &ws.cluster_ids,
                    &ws.extra_grouping_ids,
                )
                .expect("Mle ⇒ build_mixed_workspaces is Some")
            };
            // `fit_on` takes row-major x[i·p + j]; x_full_f64 is column-major faer.
            for i in 0..nrow {
                for j in 0..p_fit {
                    ws.x_rowmajor[i * p_fit + j] = ws.x_full_f64[(i, j)];
                }
            }
            // Truth-start under the scenario's `truth_start` assumption: seed the
            // optimizer at the DGP-truth θ (well-behaved estimation) or start blind
            // (None ⇒ glmm's blind θ). LMM is start-independent in β; it threads only
            // θ. NOT a speed knob — the MLE is start-sensitive. MIRRORS batch.rs's
            // Mle arm; the hot loop and this debug path must derive the same hint —
            // change together.
            // The stripped reduced model has one intercept scalar τ per grouping,
            // so its θ warm start must drop the slopes too (`include_slopes`).
            let theta = crate::mixed_workspace::truth_theta(cluster, !reduces);
            let start = spec.scenario.truth_start.then(|| glmm::StartValues {
                beta: fit_beta.clone(),
                theta,
            });
            let view = glmm::loop_advanced::fit_on(
                &mut mixed[0],
                &ws.x_rowmajor[..nrow * p_fit],
                &ws.y_full_f64[..nrow],
                &ids[0],
                start.as_ref(),
                &opts,
            );
            let converged = view.converged();
            let pinned = view.diagnostics().pinned_components;
            let betas = view.betas()[..p_fit].to_vec();
            // var_diag / t_sq are PREDICTOR-indexed for the mixed arms: extract by
            // the fitted target columns, NOT by target rank.
            let se = fit_targets
                .iter()
                .map(|&t| view.var_diag()[t as usize].sqrt())
                .collect();
            let statistic = fit_targets
                .iter()
                .map(|&t| view.t_sq()[t as usize].sqrt())
                .collect();
            // `into_fit` consumes the view; read the accessors above first. It also
            // covers the sparse route (Prebuilt holds a full Fit — varcorr/dispersion
            // read fine; only joint_t_sq/theta degrade, which introspect never reads).
            let fit = view.into_fit(
                &ws.x_rowmajor[..nrow * p_fit],
                &ws.y_full_f64[..nrow],
                &ids[0],
                nrow,
                p_fit,
                &model,
                &opts,
            );
            let (variance_components, re_corr) = unpack_varcorr(&fit);
            (
                betas,
                se,
                statistic,
                converged,
                variance_components,
                fit.dispersion, // σ̂² (NaN on a degenerate LMM fit — varcorr is then empty too)
                re_corr,
                pinned,
            )
        }
    };

    // Build the crit table with the SAME args the pipeline uses (`run_introspect`):
    // `n_targets(spec)` = target_indices + contrast_pairs. The uncorrected
    // threshold read below is target-count-independent, but matching the
    // pipeline's build keeps the two paths provably identical (the basis of the
    // bit-equivalence test). `n_targets` is the module-private helper already in
    // introspect.rs — distinct from `n_test_targets` above. A collapsed mixed
    // fit swaps in the collapsed table, as the hot path does.
    let table = build_crit_table_for_fit(spec, nrow as u32, pinned)?;

    // Report betas in the CALLER's design-column space: hosts index this vector
    // by the same column ids they passed in (and that `target_cols` carries).
    // A `test_formula`-dropped column has no estimate, so its slot is NaN.
    let betas = if reduces {
        let mut full = vec![f64::NAN; ncol];
        for (rj, &cj) in kept.iter().enumerate() {
            full[cj as usize] = betas[rj];
        }
        full
    } else {
        betas
    };

    Ok(IntrospectResults {
        betas,
        se,
        statistic,
        target_cols: spec.target_indices.clone(),
        converged,
        crit_sq_uncorrected: table.t_crit_sq_uncorrected[0]
            .first()
            .copied()
            .unwrap_or(f64::NAN),
        df_resid: (nrow as i64 - p_fit as i64) as f64,
        crit_sq_uncorrected_per_target: table.t_crit_sq_uncorrected[0].clone(),
        df_per_target: table.df_per_target[0].clone(),
        variance_components,
        sigma_sq_hat,
        re_corr,
    })
}

/// Unpack a mixed-model `Fit`'s RE (co)variance into introspect's
/// `(variance_components, re_corr)`. `fit.varcorr` holds one vech-packed
/// COLUMN-MAJOR LOWER-TRIANGULAR block D̂ = σ̂²·Λ̂Λ̂' per grouping (primary, then
/// extras in declaration order). For a q×q block the diagonal (j,j) sits at vech
/// index `j·q − j(j−1)/2` and off-diagonal (r,c) with c < r at
/// `c·q − c(c−1)/2 + (r−c)`.
///
/// `variance_components` = every block's diagonal in declaration order (order
/// [primary diag, extras…] mirrors generation.rs `n_variance_components`);
/// `re_corr` = the PRIMARY block's off-diagonals as correlations, vech
/// column-major (empty when q_p == 1). Both are empty on a degenerate fit
/// (`fit.varcorr` is empty when the LMM/GLMM fit has no finite endpoint).
fn unpack_varcorr(fit: &glmm::Fit) -> (Vec<f64>, Vec<f64>) {
    // Block length L = q(q+1)/2 ⇒ q = (√(8L+1) − 1)/2.
    let vech_q = |len: usize| -> usize { (((8 * len + 1) as f64).sqrt() as usize - 1) / 2 };
    // vech index of diagonal (j,j) in a column-major lower-tri q×q block. The
    // triangular offset is written `(j·j − j)/2` (equal to j(j−1)/2 but computed
    // so `j == 0` never underflows the usize `j − 1`).
    let diag_idx = |j: usize, q: usize| j * q - (j * j - j) / 2;
    let mut variance_components = Vec::new();
    for block in &fit.varcorr {
        let q = vech_q(block.len());
        for j in 0..q {
            variance_components.push(block[diag_idx(j, q)]);
        }
    }
    let mut re_corr = Vec::new();
    if let Some(block) = fit.varcorr.first() {
        let q = vech_q(block.len());
        let diag = |j: usize| block[diag_idx(j, q)];
        for c in 0..q {
            for r in (c + 1)..q {
                let den = (diag(r) * diag(c)).sqrt();
                let off = block[diag_idx(c, q) + (r - c)];
                re_corr.push(if den > 0.0 { off / den } else { 0.0 });
            }
        }
    }
    (variance_components, re_corr)
}

/// n_targets = marginals + contrasts (matches `BatchResult` target ordering).
fn n_targets(spec: &SimulationSpec) -> usize {
    spec.target_indices.len() + spec.contrast_pairs.len()
}

fn n_predictors_total(spec: &SimulationSpec) -> u32 {
    1 + spec.n_non_factor + spec.n_factor_dummies + spec.interactions.len() as u32
}

/// The column set the run's entry crit table is built over (batch.rs
/// `p_entry` / `p_between_entry` — change together): a `test_formula`
/// reduction makes every sim decide against the reduced design, so the debug
/// table follows it rather than the full generation width. Returns
/// `(p, p_between, kept_cols)`, where `kept_cols[j]` is the full-design column
/// sitting at fitted position `j`.
fn entry_width(spec: &SimulationSpec) -> (u32, u32, Vec<u32>) {
    let flags = vec![0u8; spec.factor_n_levels.len()];
    let mut kept: Vec<u32> = Vec::new();
    let mut remap = vec![-1i32; n_predictors_total(spec) as usize];
    let p = crate::batch::build_exclusion_remap(spec, &flags, &mut kept, &mut remap) as u32;
    let p_between = crate::critvals::between_column_count(spec, Some(&kept));
    (p, p_between, kept)
}

/// Build the crit table `run_introspect` reports, resolving the between-within
/// denominator df exactly as the hot path's `build_crit_tables` does (same
/// entry width and `p_between` input, same `BetweenWithinDf::resolve` gate) so
/// a debug run and a real `find_power` run see the same per-target reference.
/// Tukey `k` is not threaded here — no debug caller reports a Tukey-adjusted
/// threshold, matching this module's pre-existing behaviour.
fn build_crit_table_for_debug(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
) -> Result<CritValueTable, EngineError> {
    let (p, p_between, _) = entry_width(spec);
    let bw = crate::critvals::BetweenWithinDf::resolve(spec, sample_sizes, p_between);
    build_debug_table(spec, sample_sizes, p, bw.as_ref())
}

/// The table a single provided-data fit is decided against: the collapsed
/// twin when every variance component pinned at zero, else the live one —
/// the same rule as the hot path's `fit_collapsed` routing in batch.rs.
/// `pinned` is the fit's component bitmask. Built at the FITTED width (see
/// `entry_width`), because `fit_provided_data` fits the reduced design whenever
/// a `test_formula` narrows it.
fn build_crit_table_for_fit(
    spec: &SimulationSpec,
    sample_size: u32,
    pinned: u64,
) -> Result<CritValueTable, EngineError> {
    let (p, p_between, _) = entry_width(spec);
    let bw = crate::critvals::BetweenWithinDf::resolve(spec, &[sample_size], p_between);
    let collapsed = spec
        .cluster
        .as_ref()
        .is_some_and(|c| crate::batch::fit_collapsed(c.n_variance_components(), pinned));
    let bw = match (bw, collapsed) {
        (Some(b), true) => b.collapsed(spec.estimator),
        (b, _) => b,
    };
    build_debug_table(spec, &[sample_size], p, bw.as_ref())
}

fn build_debug_table(
    spec: &SimulationSpec,
    sample_sizes: &[u32],
    n_predictors: u32,
    bw: Option<&crate::critvals::BetweenWithinDf>,
) -> Result<CritValueTable, EngineError> {
    CritValueTable::build_with_tukey_k(
        &spec.crit_values,
        sample_sizes,
        n_predictors,
        n_targets(spec) as u32,
        spec.correction_method,
        spec.estimator,
        &[],
        bw,
    )
}

pub fn run_introspect(
    spec: &SimulationSpec,
    n: u32,
    n_sims: u32,
    base_seed: u64,
    mask: IntrospectMask,
) -> Result<IntrospectOut, EngineError> {
    let crit = if mask.crit {
        let table = build_crit_table_for_debug(spec, &[n])?;
        // Fitted width, matching `df_per_target` below — both come from the
        // entry crit table, which is built over the fitted design.
        let (p_entry, _, _) = entry_width(spec);
        Some(IntrospectCrit {
            crit_sq_uncorrected: table.t_crit_sq_uncorrected[0]
                .first()
                .copied()
                .unwrap_or(f64::NAN),
            df_resid: (n as i64 - p_entry as i64) as f64,
            crit_sq_uncorrected_per_target: table.t_crit_sq_uncorrected[0].clone(),
            df_per_target: table.df_per_target[0].clone(),
        })
    } else {
        None
    };

    // stats + power share one authoritative run. converged is read from the
    // BatchResult, never recomputed.
    let (batch, stat_sq, converged) = if mask.stats {
        let (b, s) = run_batch_st_capture(spec, &[n], n_sims, base_seed)?;
        let conv: Vec<bool> = b.converged.iter().map(|&c| c != 0).collect();
        let batch = if mask.power { Some(b) } else { None };
        (batch, Some(s), Some(conv))
    } else if mask.power {
        let b = run_batch_st(spec, &[n], n_sims, base_seed, None)?;
        (Some(b), None, None)
    } else {
        (None, None, None)
    };

    let data = if mask.data {
        Some(capture_sim0_data(spec, n, base_seed)?)
    } else {
        None
    };

    Ok(IntrospectOut {
        batch,
        stat_sq,
        converged,
        data,
        crit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // `engine_contract::fixtures` exposes `example1_simple_ols` (2 marginal targets,
    // OLS, no clusters) — there is NO `minimal_ols_contract` there. These engine-core
    // tests are target-count-agnostic, so alias it.
    use crate::contract_adapter::contract_to_simulation_spec;
    use crate::marginals::T3PpfTable;
    use crate::spec::Distribution;
    use engine_contract::fixtures::example1_simple_ols as minimal_ols_contract;

    /// Tiny deterministic LCG (mirrors glmm.rs tests) — reproducible without RNG caveats.
    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Build a (spec, design, nrow, ncol, outcome, cluster_ids) tuple for a
    /// Glm+cluster intercept-only GLMM scenario:
    ///   - n=80, p=2 (intercept+x1), 8 clusters, FixedClusters, target=[1].
    ///   - outcome is a balanced Bernoulli mix (LCG-generated from a logit with a
    ///     modest random intercept per cluster) so the fit converges reliably.
    ///   - design column-major, f64.
    fn glmm_intercept_provided_fixture() -> (
        crate::spec::SimulationSpec,
        Vec<f64>,
        usize,
        usize,
        Vec<f64>,
        Vec<u32>,
    ) {
        use crate::spec::{
            ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, EstimatorSpec,
            HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations,
        };
        let spec = crate::spec::SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![crate::spec::Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.5],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Binary,
            link: None,
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            df_method: Default::default(),
            nagq: 1,
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: Some(ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 8 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            }),
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            extra_slope_cols: Vec::new(),
            fit_columns: Vec::new(),
        };

        let (n, nc, p) = (80usize, 8usize, 2usize);
        let mut st = 7u64;
        // One random intercept per cluster (modest magnitude for convergence).
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        // Column-major design: col 0 = intercept (1.0), col 1 = x1.
        let mut design = vec![0.0_f64; n * p];
        let mut outcome = vec![0.0_f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % nc;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            design[i] = 1.0; // col 0 = intercept
            design[n + i] = x1; // col 1 = x1
            let eta = 0.2 + 0.5 * x1 + u0[c];
            let pr = 1.0 / (1.0 + (-eta).exp());
            outcome[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
        }
        (spec, design, n, p, outcome, ids)
    }

    #[test]
    fn introspect_glm_cluster_emits_variance_components() {
        let (spec, design, nrow, ncol, outcome, ids) = glmm_intercept_provided_fixture();
        let res = fit_provided_data(&spec, &design, nrow, ncol, &outcome, Some(&ids)).unwrap();
        assert!(res.converged);
        assert_eq!(res.variance_components.len(), 1); // [τ̂²]
        assert!(res.variance_components[0] >= 0.0 && res.variance_components[0].is_finite());
        assert!(res.re_corr.is_empty()); // no slope
        assert!(res.betas.iter().all(|b| b.is_finite()));
    }

    #[test]
    fn introspect_data_captures_sim0_draw() {
        let c = minimal_ols_contract();
        let spec = contract_to_simulation_spec(&c).unwrap();
        let n = 80usize;
        let out = run_introspect(
            &spec,
            n as u32,
            30,
            7,
            IntrospectMask {
                stats: false,
                data: true,
                crit: false,
                power: false,
            },
        )
        .unwrap();
        let d = out.data.expect("data present");
        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        assert_eq!(d.nrow, n);
        assert_eq!(d.ncol, p);
        assert_eq!(d.design.len(), n * p);
        assert_eq!(d.outcome.len(), n);
        // Column 0 is the intercept → all 1.0 (col-major: first `nrow` entries).
        assert!(d.design[..n].iter().all(|&v| v == 1.0));
        // sim0_seed is the documented per-sim seed for sim 0.
        assert_eq!(d.sim0_seed, pcg_mix64(7, 0));
        // Non-clustered OLS fixture ⇒ no cluster ids.
        assert!(d.cluster_ids.is_none());
    }

    #[test]
    fn introspect_data_regenerates_identically() {
        // Determinism: same (spec, n, base_seed) ⇒ byte-identical sim-0 draw.
        let c = minimal_ols_contract();
        let spec = contract_to_simulation_spec(&c).unwrap();
        let mask = IntrospectMask {
            stats: false,
            data: true,
            crit: false,
            power: false,
        };
        let a = run_introspect(&spec, 64, 10, 99, mask)
            .unwrap()
            .data
            .unwrap();
        let b = run_introspect(&spec, 64, 10, 99, mask)
            .unwrap()
            .data
            .unwrap();
        assert_eq!(a.design, b.design);
        assert_eq!(a.outcome, b.outcome);
    }

    /// CRN row-prefix nesting at the generated-data level: for a fixed (spec,
    /// base_seed) the sim-0 draw at a small N must be the exact prefix of the draw at
    /// a larger N — per design column (column-major) and for the outcome. Guards the
    /// batch generation loop against a per-N reseed or a `factor_prefix_counts`
    /// mis-reset that silently breaks common-random-numbers nesting across grid points
    /// (a historical bug class). `rng_rows_stable_across_max_n` guards the stream
    /// source; this guards that the batch consumes it nested at the data level.
    #[test]
    fn introspect_data_is_row_prefix_nested_across_n() {
        let c = minimal_ols_contract();
        let spec = contract_to_simulation_spec(&c).unwrap();
        let mask = IntrospectMask {
            stats: false,
            data: true,
            crit: false,
            power: false,
        };
        let (n_small, n_large, seed) = (40u32, 100u32, 4242u64);
        let small = run_introspect(&spec, n_small, 8, seed, mask)
            .unwrap()
            .data
            .unwrap();
        let large = run_introspect(&spec, n_large, 8, seed, mask)
            .unwrap()
            .data
            .unwrap();
        let (ns, nl, p) = (n_small as usize, n_large as usize, small.ncol);
        assert!(
            p >= 2,
            "fixture must carry a non-intercept predictor for the test to bite"
        );
        assert_eq!(large.ncol, p);
        // Outcome is row-indexed: small is the exact prefix of large.
        assert_eq!(small.outcome, large.outcome[..ns]);
        // Design is column-major (nrow*ncol): the first `ns` rows of each column nest.
        for col in 0..p {
            assert_eq!(
                &small.design[col * ns..col * ns + ns],
                &large.design[col * nl..col * nl + ns],
                "design column {col} not row-prefix-nested across N"
            );
        }
    }

    /// The data stage must generate from the SAME prepared spec the stats stage
    /// fits (t3_table + het_coeffs — debug invariant 4, data → stat coherence).
    /// Red before the fix: a HighKurtosis marginal panicked on the missing t(3)
    /// table. Oracle: generate_sim_data on a manually prepared spec, bit-equal.
    #[test]
    fn introspect_data_runs_prepared_spec() {
        let c = minimal_ols_contract();
        let mut spec = contract_to_simulation_spec(&c).unwrap();
        spec.var_types[0] = Distribution::HighKurtosis;
        spec.scenario.heteroskedasticity_ratio = 4.0;

        let n = 200u32;
        let mask = IntrospectMask {
            stats: false,
            data: true,
            crit: false,
            power: false,
        };
        let d = run_introspect(&spec, n, 1, 7, mask)
            .expect("data stage must not panic on HighKurtosis")
            .data
            .unwrap();

        // Oracle: run_batch_impl's preparation, applied by hand.
        let mut prepared = spec.clone();
        prepared.t3_table = Some(T3PpfTable::build_default());
        prepared.het_coeffs = prepared.compute_het_coeffs();
        let p = (1 + prepared.n_non_factor + prepared.n_factor_dummies) as usize
            + prepared.interactions.len();
        let mut ws = SimWorkspace::new(
            n as usize,
            p,
            prepared.n_non_factor as usize,
            prepared.factor_n_levels.len(),
            None,
        );
        generate_sim_data(&prepared, 0, 7, &mut ws).unwrap();
        for i in 0..n as usize {
            // Capture is f64; the regenerated buffer is f32. Compare in f32
            // (re-derivation determinism — holds exactly in the f32 plane).
            assert_eq!(
                (d.outcome[i] as f32).to_bits(),
                ws.y_full[i].to_bits(),
                "outcome row {i} diverges from the prepared-spec generation"
            );
        }
    }

    /// Heteroskedasticity must be LIVE in the data stage: with λ = 4 pinned,
    /// the sim-0 outcomes must differ from a raw-spec generation (default
    /// het_coeffs ⇒ driver-std 0 ⇒ hsk silently inert — the pre-fix behavior).
    #[test]
    fn introspect_data_hsk_is_live() {
        let c = minimal_ols_contract();
        let mut spec = contract_to_simulation_spec(&c).unwrap();
        spec.scenario.heteroskedasticity_ratio = 4.0;
        let n = 200u32;
        let mask = IntrospectMask {
            stats: false,
            data: true,
            crit: false,
            power: false,
        };
        let d = run_introspect(&spec, n, 1, 7, mask).unwrap().data.unwrap();

        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize + spec.interactions.len();
        let mut ws = SimWorkspace::new(
            n as usize,
            p,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            None,
        );
        generate_sim_data(&spec, 0, 7, &mut ws).unwrap(); // raw spec: hsk inert
        let identical =
            (0..n as usize).all(|i| (d.outcome[i] as f32).to_bits() == ws.y_full[i].to_bits());
        assert!(
            !identical,
            "λ=4 must change sim-0 outcomes vs the inert-hsk (raw spec) generation"
        );
    }

    #[test]
    fn introspect_crit_only_returns_threshold_and_df() {
        let c = minimal_ols_contract();
        let spec = contract_to_simulation_spec(&c).unwrap();
        let out = run_introspect(
            &spec,
            100,
            50,
            42,
            IntrospectMask {
                stats: false,
                data: false,
                crit: true,
                power: false,
            },
        )
        .unwrap();
        assert!(out.stat_sq.is_none());
        assert!(out.data.is_none());
        assert!(out.batch.is_none());
        let crit = out.crit.expect("crit present");
        // OLS df = N - n_predictors_total (1 + n_non_factor + n_factor_dummies).
        let p = 1 + spec.n_non_factor + spec.n_factor_dummies;
        assert_eq!(crit.df_resid, (100 - p) as f64);
        assert!(crit.crit_sq_uncorrected.is_finite() && crit.crit_sq_uncorrected > 0.0);
    }

    // Engine-level self-consistency: recomputing rejects from the captured
    // squared statistic vs the squared threshold reproduces batch.uncorrected.
    fn assert_stats_reproduce_uncorrected(estimator_contract: engine_contract::SimulationContract) {
        let spec = contract_to_simulation_spec(&estimator_contract).unwrap();
        let n = 120u32;
        let n_sims = 200u32;
        let seed = 12345u64;
        let out = run_introspect(
            &spec,
            n,
            n_sims,
            seed,
            IntrospectMask {
                stats: true,
                data: false,
                crit: true,
                power: true,
            },
        )
        .unwrap();

        let batch = out.batch.expect("batch");
        let stat_sq = out.stat_sq.expect("stat_sq");
        let crit_sq = out.crit.expect("crit").crit_sq_uncorrected;
        let nt = spec.target_indices.len() + spec.contrast_pairs.len();
        let ns = n_sims as usize;
        assert_eq!(stat_sq.len(), ns * nt);
        // BatchResult.uncorrected shape: (n_sims, n_sample_sizes=1, n_targets) row-major.
        for t in 0..nt {
            let mut from_stats = 0u64;
            let mut from_batch = 0u64;
            for s in 0..ns {
                if stat_sq[s * nt + t] > crit_sq {
                    from_stats += 1;
                }
                from_batch += batch.uncorrected[s * nt + t] as u64;
            }
            assert_eq!(
                from_stats, from_batch,
                "target {t}: stats vs batch reject mismatch"
            );
        }
    }

    #[test]
    fn ols_stats_reproduce_uncorrected_counts() {
        assert_stats_reproduce_uncorrected(minimal_ols_contract());
    }

    /// `run_introspect` with `mask.stats=true` returns `stat_sq` of
    /// shape (n_sims × n_targets) row-major, where n_targets =
    /// |target_indices| + |contrast_pairs|.
    #[test]
    fn introspect_stats_shape() {
        let c = minimal_ols_contract();
        let spec = contract_to_simulation_spec(&c).unwrap();
        let n_sims = 64u32;
        let out = run_introspect(
            &spec,
            100,
            n_sims,
            42,
            IntrospectMask {
                stats: true,
                data: false,
                crit: false,
                power: false,
            },
        )
        .unwrap();
        let stat_sq = out.stat_sq.expect("stat_sq present when mask.stats=true");
        let n_targets = spec.target_indices.len() + spec.contrast_pairs.len();
        assert_eq!(
            stat_sq.len(),
            n_sims as usize * n_targets,
            "stat_sq must be n_sims × n_targets row-major"
        );
    }

    // -----------------------------------------------------------------
    // H1 — interaction contract: ncol==4 and df_resid==n-4
    //      Guards the OOB-panic regression (n_predictors_total must include
    //      interactions.len()); if it regresses, generate_sim_data writes to
    //      column index 3 in a 3-column workspace → panic before any assert.
    // -----------------------------------------------------------------

    /// Build the interaction OLS contract inline (y ~ x1 + x2 + x1:x2).
    /// Same structure as `contract_adapter::tests::continuous_interaction_maps_to_appended_kernel_column`.
    fn interaction_ols_contract_for_introspect() -> engine_contract::SimulationContract {
        use engine_contract::{
            ColumnId, CorrectionMethod, DesignSpec, DesignTerm, EstimatorSpec, GenerationSpec,
            OutcomeKind, OutcomeSpec, ResidualDist, ResidualSpec, ScenarioPerturbations, TestSpec,
            TestTarget,
        };
        engine_contract::SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    engine_contract::ColumnSpec::Synthetic {
                        kind: engine_contract::SyntheticKind::Normal,
                        pinned: false,
                    },
                    engine_contract::ColumnSpec::Synthetic {
                        kind: engine_contract::SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: engine_contract::Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::Interaction {
                        components: vec![
                            DesignTerm::Direct {
                                column: ColumnId(0),
                            },
                            DesignTerm::Direct {
                                column: ColumnId(1),
                            },
                        ],
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5, 0.3, 0.2],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: None,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            df_method: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 3 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        }
    }

    #[test]
    fn introspect_interaction_contract_correct_ncol_and_df() {
        // H1 regression guard: if n_predictors_total reverts to omitting
        // interactions.len(), SimWorkspace is sized for 3 cols but generate_sim_data
        // writes to column index 3 → OOB panic. The test panics before reaching
        // the ncol/df_resid asserts, so any reversion is caught.
        let c = interaction_ols_contract_for_introspect();
        let spec = contract_to_simulation_spec(&c).unwrap();

        // Spec must have 1 interaction (x1:x2 → [kernel_col 1, kernel_col 2]).
        assert_eq!(
            spec.interactions.len(),
            1,
            "interaction spec must have 1 interaction"
        );
        // Total predictors: 1 (intercept) + 2 (x1,x2) + 0 (factors) + 1 (x1:x2) = 4.
        let expected_p = 4usize;

        let n = 100u32;
        let out = run_introspect(
            &spec,
            n,
            20,
            7,
            IntrospectMask {
                stats: false,
                data: true,
                crit: true,
                power: false,
            },
        )
        .unwrap();

        // Data stage: 4-column design matrix (guards OOB panic regression).
        let d = out.data.expect("data present");
        assert_eq!(
            d.ncol, expected_p,
            "interaction contract must produce 4-column design, got ncol={}",
            d.ncol
        );
        assert_eq!(
            d.design.len(),
            n as usize * expected_p,
            "design data length must be n*p"
        );
        // Column 0 is intercept (all 1.0), column-major layout.
        assert!(
            d.design[..n as usize].iter().all(|&v| v == 1.0),
            "column 0 of design must be all 1.0 (intercept)"
        );

        // Crit stage: df_resid = n - 4 (all 4 predictors accounted for).
        let crit = out.crit.expect("crit present");
        assert_eq!(
            crit.df_resid,
            (n as i64 - expected_p as i64) as f64,
            "df_resid must account for interaction column: got {}, expected {}",
            crit.df_resid,
            n as i64 - expected_p as i64
        );
        assert!(
            crit.crit_sq_uncorrected.is_finite() && crit.crit_sq_uncorrected > 0.0,
            "crit_sq_uncorrected must be positive finite, got {}",
            crit.crit_sq_uncorrected
        );
    }

    /// Non-degenerate Mle specs route through the general path and surface
    /// per-grouping variance components (τ̂²_g = θ̂_g²·σ̂², [primary, extras…]).
    /// Generate a crossed-grouping dataset with the engine itself, then refit it.
    #[test]
    fn fit_provided_data_general_path_surfaces_variance_components() {
        use engine_contract::{
            ClusterSizing, ClusterSpec, EstimatorSpec, GroupingRelation, GroupingSpec,
        };
        let mut c = minimal_ols_contract();
        c.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, 0.20);
        cluster.extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.15,
            slopes: vec![],
        }];
        c.generation.cluster = Some(cluster);
        let spec = contract_to_simulation_spec(&c).unwrap();
        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        let n = 96usize; // atom 6·4 = 24 → multiple
        let mut ws = SimWorkspace::new(
            n,
            p,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            spec.cluster.as_ref(),
        );
        crate::data_gen::generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        // Column-major design copy of the first n rows.
        let mut design = vec![0.0; n * p];
        for j in 0..p {
            for i in 0..n {
                design[j * n + i] = ws.x_full[(i, j)] as f64;
            }
        }
        let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
        let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();
        let r = fit_provided_data(&spec, &design, n, p, &outcome, Some(&cids)).unwrap();
        assert!(r.converged);
        assert_eq!(r.variance_components.len(), 2); // primary + 1 crossed
        assert!(r
            .variance_components
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
        assert!(r.sigma_sq_hat.is_finite() && r.sigma_sq_hat > 0.0);
    }

    /// Extra-grouping random SLOPE: the crossed factor carries intercept + 1
    /// slope (q_g = 2), so its RE covariance D_g is 2×2 and contributes TWO
    /// diagonal variance components, not one. Primary intercept-only (q_p = 1) ⇒
    /// total length 1 (primary) + 2 (extra) = 3. Before the cursor-walk fix the
    /// extra pushed a single scalar (length 2) and read the wrong θ slot
    /// (`theta[base+e]`, which assumed q_g==1). Clone of
    /// `fit_provided_data_general_path_surfaces_variance_components` with one
    /// slope added to the crossed extra.
    #[test]
    fn introspect_surfaces_extra_grouping_slope_variance() {
        use engine_contract::{
            ClusterSizing, ClusterSpec, ColumnId, EstimatorSpec, GroupingRelation, GroupingSpec,
            SlopeTerm,
        };
        let mut c = minimal_ols_contract();
        c.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, 0.20);
        cluster.extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.15,
            slopes: vec![SlopeTerm {
                column: ColumnId(0),
                variance: 0.12,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
        }];
        c.generation.cluster = Some(cluster);
        let spec = contract_to_simulation_spec(&c).unwrap();
        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        let n = 96usize; // atom 6·4 = 24 → multiple
        let mut ws = SimWorkspace::new(
            n,
            p,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            spec.cluster.as_ref(),
        );
        crate::data_gen::generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let mut design = vec![0.0; n * p];
        for j in 0..p {
            for i in 0..n {
                design[j * n + i] = ws.x_full[(i, j)] as f64;
            }
        }
        let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
        let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();
        let r = fit_provided_data(&spec, &design, n, p, &outcome, Some(&cids)).unwrap();
        assert!(r.converged);
        // q_p=1 (primary) + q_g=2 (extra intercept+slope) = 3 components.
        assert_eq!(
            r.variance_components.len(),
            3,
            "primary intercept + extra (intercept + slope) = 3 components"
        );
        assert!(
            r.variance_components
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "all RE variances finite and non-negative: {:?}",
            r.variance_components,
        );
        // The extra's slope variance — previously dropped (length-2 bug).
        assert!(
            r.variance_components[2] > 0.0,
            "extra slope variance must be positive, got {}",
            r.variance_components[2],
        );
    }

    /// Random-intercept-only Mle (no slopes, no extra groupings) routes through
    /// the scalar `EstimatorSpec::Mle` branch — which historically returned an
    /// empty `variance_components`. It must now surface [τ̂²] (= θ̂²·σ̂²) and a
    /// finite σ̂², matching the general-path and GLMM branches, so hosts can
    /// recover the ICC. Generate a clustered dataset with the engine, then refit.
    #[test]
    fn fit_provided_data_intercept_only_surfaces_variance_component() {
        use engine_contract::{ClusterSizing, ClusterSpec, EstimatorSpec};
        let mut c = minimal_ols_contract();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters { n_clusters: 8 },
            0.30,
        ));
        let spec = contract_to_simulation_spec(&c).unwrap();
        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        let n = 160usize; // multiple of 8 clusters
        let mut ws = SimWorkspace::new(
            n,
            p,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            spec.cluster.as_ref(),
        );
        crate::data_gen::generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let mut design = vec![0.0; n * p];
        for j in 0..p {
            for i in 0..n {
                design[j * n + i] = ws.x_full[(i, j)] as f64;
            }
        }
        let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
        let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();
        let r = fit_provided_data(&spec, &design, n, p, &outcome, Some(&cids)).unwrap();
        assert!(r.converged);
        assert_eq!(r.variance_components.len(), 1); // [τ̂²], scalar random intercept
        assert!(r.variance_components[0].is_finite() && r.variance_components[0] >= 0.0);
        assert!(r.sigma_sq_hat.is_finite() && r.sigma_sq_hat > 0.0);
        assert!(r.re_corr.is_empty()); // no slope ⇒ no off-diagonal
    }

    /// A collapsed between-within LMM fit (τ̂² pinned at 0) reports the
    /// residual df `N − P` on every target; a live fit keeps the within df
    /// `N − P − (g − 1)`. True τ² = 0 so seeds split between the two; the
    /// loop must see both. Mirrors the hot path's `lme_collapsed_fit_uses_residual_df`.
    #[test]
    fn fit_provided_data_collapsed_lmm_reports_residual_df() {
        use engine_contract::{ClusterSizing, ClusterSpec, DfMethod, EstimatorSpec};
        let mut c = minimal_ols_contract();
        c.estimator = EstimatorSpec::Mle;
        c.df_method = DfMethod::BetweenWithin;
        c.generation.cluster = Some(ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters { n_clusters: 8 },
            0.0,
        ));
        let spec = contract_to_simulation_spec(&c).unwrap();
        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        let n = 160usize;
        let df_resid = (n - p) as f64;
        let df_within = df_resid - 7.0; // g − p_between = 8 − 1
        let (mut seen_live, mut seen_collapsed) = (false, false);
        for seed in 0..12u64 {
            let mut ws = SimWorkspace::new(
                n,
                p,
                spec.n_non_factor as usize,
                spec.factor_n_levels.len(),
                spec.cluster.as_ref(),
            );
            crate::data_gen::generate_sim_data(&spec, 0, 2137 + seed, &mut ws).unwrap();
            let mut design = vec![0.0; n * p];
            for j in 0..p {
                for i in 0..n {
                    design[j * n + i] = ws.x_full[(i, j)] as f64;
                }
            }
            let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
            let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();
            let r = fit_provided_data(&spec, &design, n, p, &outcome, Some(&cids)).unwrap();
            if !r.converged {
                continue;
            }
            let expected = if r.variance_components[0] == 0.0 {
                seen_collapsed = true;
                df_resid
            } else {
                seen_live = true;
                df_within
            };
            assert!(
                r.df_per_target.iter().all(|&d| d == expected),
                "seed {seed}: τ̂² = {} ⇒ df {expected}, got {:?}",
                r.variance_components[0],
                r.df_per_target
            );
            let t = crate::critvals::t_ppf(0.975, expected);
            assert!(r
                .crit_sq_uncorrected_per_target
                .iter()
                .all(|&c| (c - t * t).abs() < 1e-12));
        }
        assert!(
            seen_collapsed && seen_live,
            "need both a pinned and a live fit"
        );
    }

    /// A `test_formula` reduction must reach the provided-data fit too: the
    /// FULL design here has no between-cluster df left (4 clusters, 4
    /// cluster-level predictors ⇒ p_between = 5, g − p_between = −1, which the
    /// between-within crit table refuses), while the fitted design
    /// `{intercept, x1}` has p_between = 2 ⇒ df_between = 4 − 2 = 2. Same shape
    /// as the hot path's `between_within_full_design_df_gate_does_not_block_reduced_fit`.
    /// Before honouring `fit_columns` this fit errored `InvalidSpec` on the
    /// full-width df gate.
    #[test]
    fn fit_provided_data_test_formula_uses_reduced_between_within_df() {
        use crate::spec::{
            ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, DfMethod, EstimatorSpec,
            HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations,
        };
        let spec = SimulationSpec {
            n_non_factor: 4,
            n_factor_dummies: 0,
            correlation: vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            var_types: vec![Distribution::Normal; 4],
            var_pinned: vec![],
            var_params: vec![0.0; 4],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![1, 2, 3, 4],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 1.0, 0.0, 0.0, 0.0],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            link: None,
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
            df_method: DfMethod::BetweenWithin,
            nagq: 1,
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: Some(ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 4 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            }),
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            extra_slope_cols: Vec::new(),
            fit_columns: vec![0, 1],
        };

        // Data is generated at the FULL design width (5 columns) — the caller
        // hands in the whole design; the reduction happens inside the fit.
        let n = 40usize;
        let p_full = 5usize;
        let mut ws = SimWorkspace::new(
            n,
            p_full,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            spec.cluster.as_ref(),
        );
        // Seed 8 lands a comfortably live τ̂² (≈ 0.40); a τ̂² pinned at 0 would be
        // decided against the residual df instead (see
        // `fit_provided_data_collapsed_lmm_reports_residual_df`).
        crate::data_gen::generate_sim_data(&spec, 0, 8, &mut ws).unwrap();
        let mut design = vec![0.0_f64; n * p_full];
        for j in 0..p_full {
            for i in 0..n {
                design[j * n + i] = ws.x_full[(i, j)] as f64;
            }
        }
        let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
        let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();

        let r = fit_provided_data(&spec, &design, n, p_full, &outcome, Some(&cids))
            .expect("the reduced fit has between-within df to spare and must build a crit table");
        assert!(r.converged);
        assert!(
            r.variance_components[0] > 0.0,
            "a collapsed fit would be decided against the residual df instead; got τ̂² = {}",
            r.variance_components[0]
        );
        // g − p_between = 4 − 2 = 2, the REDUCED design's between df.
        assert_eq!(
            r.df_per_target,
            vec![2.0],
            "cluster-level target must take the reduced design's between df"
        );
        let t = crate::critvals::t_ppf(0.975, 2.0);
        assert!((r.crit_sq_uncorrected - t * t).abs() < 1e-12);
        // df_resid follows the fitted width too: 40 − 2.
        assert_eq!(r.df_resid, 38.0);
        // Targets stay in the CALLER's column space; dropped columns have no
        // estimate.
        assert_eq!(r.target_cols, vec![1]);
        assert_eq!(r.betas.len(), p_full);
        assert!(r.betas[0].is_finite() && r.betas[1].is_finite());
        assert!(r.betas[2..].iter().all(|b| b.is_nan()));
    }

    /// `fit_provided_data` on a (1+x1|g) spec surfaces [τ̂₀², τ̂₁²] in
    /// `variance_components` and one finite correlation in `re_corr` (q_p=2
    /// primary block → q_p(q_p−1)/2 = 1 off-diagonal entry, |·| ≤ 1).
    #[test]
    fn slope_introspect_surfaces_variance_and_corr() {
        use crate::spec::{
            ClusterSizing, ClusterSpec, CorrectionMethod, CritValues, EstimatorSpec,
            HeteroskedasticityCoeffs, OutcomeKind, ResidualDist, ScenarioPerturbations,
        };
        use engine_contract::SlopeTerm;

        // Replicate batch.rs::tests::slope_power_spec / minimal_lme_spec inline,
        // using enough clusters and observations that the LMM can converge.
        let spec = SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![crate::spec::Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.0, 0.5],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            link: None,
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
            df_method: Default::default(),
            nagq: 1,
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: Some(ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
                tau_squared: 0.25,
                slopes: vec![SlopeTerm {
                    column: engine_contract::ColumnId(0),
                    variance: 0.10,
                    corr_with_intercept: 0.0,
                    corr_with: vec![],
                }],
                extra_groupings: vec![],
            }),
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            // x_full col 1 = x1 (col 0 = intercept, col 1 = first continuous).
            cluster_slope_design_cols: vec![1],
            extra_slope_cols: Vec::new(),
            fit_columns: Vec::new(),
        };

        let p = (1 + spec.n_non_factor + spec.n_factor_dummies) as usize;
        let n = 200usize; // 20 clusters × 10 obs each
        let mut ws = SimWorkspace::new(
            n,
            p,
            spec.n_non_factor as usize,
            spec.factor_n_levels.len(),
            spec.cluster.as_ref(),
        );
        crate::data_gen::generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();

        // Column-major design copy.
        let mut design = vec![0.0_f64; n * p];
        for j in 0..p {
            for i in 0..n {
                design[j * n + i] = ws.x_full[(i, j)] as f64;
            }
        }
        let outcome: Vec<f64> = ws.y_full[..n].iter().map(|&v| v as f64).collect();
        let cids: Vec<u32> = ws.cluster_ids[..n].to_vec();

        let r = fit_provided_data(&spec, &design, n, p, &outcome, Some(&cids)).unwrap();
        // q_p = 2 → 2 primary RE variances; no extras.
        assert_eq!(r.variance_components.len(), 2, "expect [τ̂₀², τ̂₁²]");
        assert!(
            r.variance_components
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "RE variances must be finite and non-negative: {:?}",
            r.variance_components,
        );
        // q_p(q_p−1)/2 = 1 off-diagonal correlation.
        assert_eq!(r.re_corr.len(), 1, "expect one intercept↔slope correlation");
        assert!(
            r.re_corr[0].is_finite() && r.re_corr[0].abs() <= 1.0,
            "re_corr[0] must be finite and in [-1,1], got {}",
            r.re_corr[0],
        );
    }
}
