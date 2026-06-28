//! Debug introspection driver. (Always compiled; no cargo feature.)
//!
//! Leaves the hot-path `run_one_sim` untouched. For the authoritative answer it
//! calls the real `run_batch_st`; for observation it re-walks the same per-sim
//! seed chain. Single-threaded, fully deterministic.

use crate::batch::run_batch_st;
use crate::batch::run_batch_st_capture;
use crate::batch::EPS_RANK;
use crate::critvals::CritValueTable;
use crate::data_gen::generate_sim_data;
use crate::rng::pcg_mix64;
use crate::spec::{BatchResult, EstimatorSpec, SimulationSpec};
use crate::workspace::SimWorkspace;
use crate::EngineError;
use glmm::mcpower::{fit_suff_stats_t_sq, OlsScratch, OlsSuffStats};
use glmm::mcpower::{glm_irls_fit, GlmScratch};
use glmm::mcpower::{lme_fit, LmeScratch, LmeSuffStats};

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
    /// `t_crit_sq_uncorrected[0]` — the shared (squared) uncorrected threshold.
    pub crit_sq_uncorrected: f64,
    /// `N - n_predictors_total` as f64 (OLS df_resid; ignored by z-families).
    pub df_resid: f64,
}

/// Result of fitting a *provided* dataset (the `data → results` debug path).
/// Mirrors what the sim loop computes per sim, but for one externally-supplied
/// dataset instead of a generated draw. `betas`/`se` are aligned to the design
/// columns / targets respectively; `statistic` is the natural (unsquared)
/// per-target decision statistic. Estimator-agnostic in shape; all three
/// estimator producers (OLS, GLM, MLE) are wired.
#[derive(Debug, Clone)]
pub struct IntrospectResults {
    /// Fitted coefficient per design column, length `ncol`.
    pub betas: Vec<f64>,
    /// Standard error per *target*, length `n_targets` = `sqrt(var_diag[t])`.
    pub se: Vec<f64>,
    /// Natural (unsquared) decision statistic per target, length `n_targets`.
    pub statistic: Vec<f64>,
    /// Kernel column index each target refers to, length `n_targets`
    /// (= `spec.target_indices`). Lets the host align targets to design columns.
    pub target_cols: Vec<u32>,
    pub converged: bool,
    /// Shared (squared) uncorrected threshold, `t_crit_sq_uncorrected[0]`.
    pub crit_sq_uncorrected: f64,
    /// `N - n_predictors_total`. Only meaningful for OLS (Student-t df_resid);
    /// the z-based GLM/MLE estimators are df-independent and ignore it.
    pub df_resid: f64,
    /// Estimated variance components τ̂²_g = θ̂_g²·σ̂², order [primary,
    /// extras in declaration order]. Filled by the general (multi-grouping)
    /// Mle path only at this milestone; empty elsewhere.
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
/// `fit_suff_stats_t_sq`), GLM (`glm_irls_fit`), or MLE
/// (`LmeSuffStats::add_rows` → `lme_fit`) — so this is the shipped solver, not a
/// parallel impl (proven by the orchestrator bit-equivalence tests).
///
/// `design` is column-major `nrow * ncol` (the shape `capture_sim0_data`
/// emits). All three estimator arms are wired; `df_resid` is OLS-only-meaningful
/// (z-families are df-independent).
///
/// This function deliberately does NOT mirror the sparse-factor exclusion branch
/// of `run_one_sim` — debug fits provided bytes as-is so the L3 same-bytes
/// contract holds.
pub fn fit_provided_data(
    spec: &SimulationSpec,
    design: &[f64],
    nrow: usize,
    ncol: usize,
    outcome: &[f64],
    cluster_ids: Option<&[u32]>,
) -> Result<IntrospectResults, EngineError> {
    let p = n_predictors_total(spec) as usize;
    let n_non_factor = spec.n_non_factor as usize;
    let n_factors = spec.factor_n_levels.len();
    let n_clusters_ws = spec
        .cluster
        .as_ref()
        .map(|c| c.sizing.n_clusters_at(nrow))
        .unwrap_or(0);

    let mut ws = SimWorkspace::new(nrow, p, n_non_factor, n_factors, spec.cluster.as_ref());

    // Inject the provided data into the workspace buffers (column-major in).
    // Captured design/outcome are f64; narrow into the f32 data plane.
    for j in 0..ncol {
        for i in 0..nrow {
            ws.x_full[(i, j)] = design[j * nrow + i] as f32;
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
    for j in 0..ncol {
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

    let n_test_targets = spec.target_indices.len();

    // Dispatch on the estimator — each arm mirrors the corresponding
    // `run_one_sim` arm exactly (same kernel, same injected bytes).
    #[allow(clippy::type_complexity)]
    let (betas, se, statistic, converged, variance_components, sigma_sq_hat, re_corr): (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        bool,
        Vec<f64>,
        f64,
        Vec<f64>,
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
                &spec.target_indices,
                EPS_RANK,
                ws.suff_xtx_work.as_mut(),
                scratch,
            );
            let betas = fit.betas[..ncol].to_vec();
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
            )
        }
        EstimatorSpec::Glm if spec.cluster.is_some() => {
            // Clustered-logistic (GLMM) path — mirrors the Mle general branch's
            // structure (build workspace, build Z, fit, extract D̂). No σ² for
            // the binomial family: D̂ = Λ̂Λ̂' directly; sigma_sq_hat = 1.0.
            let cluster = spec.cluster.as_ref().expect("guard ⇒ cluster");
            let slope_cols: Vec<usize> = spec
                .cluster_slope_design_cols
                .iter()
                .map(|&c| c as usize)
                .collect();
            let model = crate::mixed_workspace::cluster_to_model_spec(
                cluster,
                spec.estimator,
                spec.wald_se,
            );
            let mut glmm =
                glmm::mcpower::GlmmWorkspace::for_cluster_spec(ncol, &model, nrow, &slope_cols);
            glmm::mcpower::build_z(
                &mut glmm,
                ws.x_full_f64.as_ref().subrows(0, nrow),
                &ws.cluster_ids[..nrow],
                &ws.extra_grouping_ids,
                nrow,
            );
            // Truth-start mirrors batch.rs Glm+cluster arm — change together.
            let nt = glmm.theta_truth.len();
            let mut tbuf = [0.0_f64; glmm::consts::MAX_THETA];
            tbuf[..nt].copy_from_slice(&glmm.theta_truth);
            let f = glmm::mcpower::fit_glmm(
                &mut glmm,
                ws.x_full_f64.as_ref().subrows(0, nrow),
                &ws.y_full_f64[..nrow],
                &ws.cluster_ids[..nrow],
                &spec.target_indices,
                Some(&tbuf[..nt]),
                &spec.effect_sizes,
                nrow,
                crate::mixed_workspace::wald_se_to_glmm(spec.wald_se),
            );
            // D̂ = Λ̂Λ̂' (no σ² — binomial dispersion fixed at 1). Diagonal
            // entries are RE variances; off-diagonals yield re_corr. Mirrors
            // the Mle general branch's q_p > 1 block, with σ=1. primary_lambda
            // refreshes glmm.lam; for q_p == 1 re_corr is empty (inner loop vacuous).
            let q = glmm.groupings.primary_q;
            glmm::mcpower::primary_lambda(&glmm.params[..glmm.n_theta], q, &mut glmm.lam);
            let mut d = vec![0.0_f64; q * q];
            for i in 0..q {
                for j in 0..q {
                    let mut acc = 0.0;
                    for kk in 0..q {
                        acc += glmm.lam[i * q + kk] * glmm.lam[j * q + kk];
                    }
                    d[i * q + j] = acc;
                }
            }
            // Primary diagonal variances, then extra-grouping scalars (θ_e²).
            // Order mirrors lmm.rs diagonal_theta()/variance_components — change together.
            let mut variance_components: Vec<f64> = (0..q).map(|dd| d[dd * q + dd]).collect();
            let base = q * (q + 1) / 2;
            for e in 0..glmm.groupings.extra_offsets.len() {
                let te = glmm.params[base + e];
                variance_components.push(te * te);
            }
            // re_corr: primary off-diagonals vech column-major (empty when q_p == 1).
            let mut re_corr = Vec::with_capacity(q * (q - 1) / 2);
            for c in 0..q {
                for r in (c + 1)..q {
                    let den = (d[r * q + r] * d[c * q + c]).sqrt();
                    re_corr.push(if den > 0.0 { d[r * q + c] / den } else { 0.0 });
                }
            }
            let betas = glmm.betas[..ncol].to_vec();
            let se = spec
                .target_indices
                .iter()
                .map(|&t| glmm.var_diag[t as usize].sqrt())
                .collect();
            let statistic = spec
                .target_indices
                .iter()
                .map(|&t| glmm.t_sq[t as usize].sqrt())
                .collect();
            // sigma_sq_hat = 1.0 (binomial dispersion is fixed; no scale parameter).
            (
                betas,
                se,
                statistic,
                f.converged,
                variance_components,
                1.0,
                re_corr,
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
                x_slice,
                y_slice,
                &spec.target_indices,
                Some(&spec.effect_sizes),
                scratch,
            );
            let betas = fit.betas[..ncol].to_vec();
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
            )
        }
        EstimatorSpec::Mle
            if spec.cluster.as_ref().is_some_and(|c| {
                !c.extra_groupings.is_empty() || !spec.cluster_slope_design_cols.is_empty()
            }) =>
        {
            // This guard mirrors build_lmm_workspace's dispatch (lmm.rs): slopes
            // OR extra groupings ⇒ general lmm path — change together.
            // General (multi-grouping) path. Extra-grouping membership is
            // layout-derived: rows are a layout prefix (true for
            // engine-generated data by construction; arbitrary uploaded
            // bytes inherit the documented assumption). ws.extra_grouping_ids
            // is already laid out (length nrow) by the ctor (Task 4) from
            // the same extra_level_of_row functions — reuse it (mirrors the
            // hot path) rather than rebuilding; add_rows_multi slices each
            // by nrow. Provided cluster_ids drive the primary as on the
            // Brent path.
            let cluster = spec.cluster.as_ref().expect("guard ⇒ cluster");
            let slope_cols: Vec<usize> = spec
                .cluster_slope_design_cols
                .iter()
                .map(|&c| c as usize)
                .collect();
            let model = crate::mixed_workspace::cluster_to_model_spec(
                cluster,
                spec.estimator,
                spec.wald_se,
            );
            let mut lmm =
                glmm::mcpower::LmmWorkspace::for_cluster_spec(ncol, &model, nrow, &slope_cols);
            let cid_slice = &ws.cluster_ids[..nrow];
            lmm.suff.add_rows_multi(
                ws.x_full_f64.as_ref().subrows(0, nrow),
                &ws.y_full_f64[..nrow],
                cid_slice,
                &ws.extra_grouping_ids,
            );
            // Truth-start (P1) — MIRRORS batch.rs's general branch; hot loop
            // and debug path must derive the same hint; change together.
            let mut tbuf = [0.0_f64; 1 + glmm::consts::MAX_EXTRA_GROUPINGS];
            let nt = lmm.theta_truth.len();
            tbuf[..nt].copy_from_slice(&lmm.theta_truth);
            let f = glmm::mcpower::fit_lmm(&mut lmm, &spec.target_indices, Some(&tbuf[..nt]));
            let sig = f.sigma_sq;
            let q = lmm.suff.groupings.primary_q;
            let (variance_components, re_corr) = if q > 1 {
                // Slope spec: D̂ = σ̂²·Λ̂Λ̂′ where Λ̂ is the row-major
                // lower-triangular primary Cholesky factor. Diagonals of D̂
                // are the RE variances [τ̂₀², τ̂₁², …]; off-diagonals yield
                // the correlation vector (vech column-major).
                let mut lam = vec![0.0_f64; q * q];
                glmm::mcpower::primary_lambda(&lmm.theta, q, &mut lam);
                let mut d = vec![0.0_f64; q * q];
                for i in 0..q {
                    for j in 0..q {
                        let mut acc = 0.0;
                        for k in 0..q {
                            acc += lam[i * q + k] * lam[j * q + k];
                        }
                        d[i * q + j] = sig * acc;
                    }
                }
                // variance_components order [intercept, slope_0, …, extras] mirrors
                // diagonal_theta()/boundary_rate_per_component (lmm.rs / result.rs) —
                // change together. Primary diagonals first, then extra scalars at
                // lmm.theta[base + e] (base = q(q+1)/2 = primary vech length).
                let mut vc: Vec<f64> = (0..q).map(|dd| d[dd * q + dd]).collect();
                let base = q * (q + 1) / 2;
                for e in 0..lmm.suff.groupings.extra_offsets.len() {
                    let te = lmm.theta[base + e];
                    vc.push(sig * te * te);
                }
                // re_corr = primary off-diagonals vech column-major:
                // corr(r, c) for c < r.
                let mut rc = Vec::with_capacity(q * (q - 1) / 2);
                for c in 0..q {
                    for r in (c + 1)..q {
                        let den = (d[r * q + r] * d[c * q + c]).sqrt();
                        rc.push(if den > 0.0 { d[r * q + c] / den } else { 0.0 });
                    }
                }
                (vc, rc)
            } else {
                // q_p == 1 ⇒ every θ entry is a diagonal variance (1×1 primary vech
                // + extra scalars), so θ²·σ̂² is the per-component variance; no off-diagonals.
                let vc: Vec<f64> = lmm.theta.iter().map(|t| t * t * sig).collect();
                (vc, vec![])
            };
            let betas = lmm.fit.betas[..ncol].to_vec();
            let se = spec
                .target_indices
                .iter()
                .map(|&t| lmm.fit.var_diag[t as usize].sqrt())
                .collect();
            let statistic = spec
                .target_indices
                .iter()
                .map(|&t| lmm.fit.t_sq[t as usize].sqrt())
                .collect();
            (
                betas,
                se,
                statistic,
                f.converged,
                variance_components,
                sig,
                re_corr,
            )
        }
        EstimatorSpec::Mle => {
            // Mirrors run_one_sim's MLE arm. Build per-cluster suff-stats,
            // then lme_fit. var_diag / t_sq are PREDICTOR-indexed: extract by
            // spec.target_indices, NOT by target rank.
            use faer::reborrow::IntoConst;
            ws.reset_lme_suff_stats();
            {
                let mut suff = LmeSuffStats {
                    xtx: ws.lme_xtx.as_mut(),
                    xty: &mut ws.lme_xty,
                    yty: &mut ws.lme_yty,
                    sum_xc: ws.lme_sum_xc.as_mut(),
                    sum_yc: &mut ws.lme_sum_yc,
                    cluster_sizes: &mut ws.lme_cluster_sizes,
                    n_clusters_seen: &mut ws.lme_n_clusters_seen,
                    panel_x: &mut ws.panel_x,
                    panel_y: &mut ws.panel_y,
                };
                let x_slice = ws.x_full_f64.as_ref().subrows(0, nrow);
                let y_slice = &ws.y_full_f64[..nrow];
                let cid_slice = &ws.cluster_ids[..nrow];
                suff.add_rows(x_slice, y_slice, cid_slice);
            }
            let n_clusters = ws.lme_n_clusters_seen;
            let scratch = LmeScratch {
                xtx: ws.lme_xtx.as_ref(),
                xty: &ws.lme_xty,
                yty: ws.lme_yty,
                ols_scratch: OlsScratch {
                    fit_betas: &mut ws.fit_betas,
                    fit_var_diag: &mut ws.fit_var_diag,
                    fit_t_sq: &mut ws.fit_t_sq,
                    fit_u_scratch: &mut ws.fit_u_scratch,
                    fit_factor: ws.fit_factor.as_mut(),
                    fit_rhs: ws.fit_rhs.as_mut(),
                },
                sum_xc: ws.lme_sum_xc.as_mut().into_const(),
                sum_yc: &ws.lme_sum_yc,
                cluster_sizes: &ws.lme_cluster_sizes,
                n_clusters,
                n_rows: nrow as u32,
                xtvix: ws.lme_xtvix.as_mut(),
                xtviy: &mut ws.lme_xtviy,
                xtvix_factor: ws.lme_xtvix_factor.as_mut(),
                v_diag_inv: &mut ws.lme_v_diag_inv,
                betas: &mut ws.lme_betas,
                var_diag: &mut ws.lme_var_diag,
                t_sq: &mut ws.lme_t_sq,
                u_scratch: &mut ws.lme_u_scratch,
                sigma_sq: 0.0,
                brent_log_a: &mut ws.lme_brent_log_a,
                brent_log_b: &mut ws.lme_brent_log_b,
                brent_log_c: &mut ws.lme_brent_log_c,
                brent_fa: &mut ws.lme_brent_fa,
                brent_fb: &mut ws.lme_brent_fb,
                brent_fc: &mut ws.lme_brent_fc,
                joint_sigma_t_chol: ws.lme_joint_sigma_t_chol.as_mut(),
                joint_rhs: &mut ws.lme_joint_rhs,
                joint_k_inv: ws.lme_joint_k_inv.as_mut(),
            };
            let x_slice = ws.x_full_f64.as_ref().subrows(0, nrow);
            let y_slice = &ws.y_full_f64[..nrow];
            let cid_slice = &ws.cluster_ids[..nrow];
            // Truth start from the spec's BASE τ² (mirrors the hot loop,
            // which starts at the per-block √τ²_block). Provided bytes
            // carry no block identity, so the L5 ICC-jitter perturbation
            // cannot be reproduced here: under that knob the refit reaches
            // the same minimum within Brent tolerance rather than
            // bit-identically; for every non-jittered scenario the two
            // starts are equal and load_data ≡ hot-loop stays bit-exact.
            let theta_start = Some(
                spec.cluster
                    .as_ref()
                    .map(|c| c.tau_squared)
                    .unwrap_or(0.0)
                    .max(0.0)
                    .sqrt(),
            );
            let fit = lme_fit(
                x_slice,
                y_slice,
                cid_slice,
                &spec.target_indices,
                theta_start,
                scratch,
            );
            let betas = fit.betas[..ncol].to_vec();
            let se = spec
                .target_indices
                .iter()
                .map(|&ti| fit.var_diag[ti as usize].sqrt())
                .collect();
            let statistic = spec
                .target_indices
                .iter()
                .map(|&ti| fit.t_sq[ti as usize].sqrt())
                .collect();
            // Surface τ̂² (primary random-intercept variance, index 0) and σ̂²
            // for the random-intercept-only path, matching the general-MLE and
            // GLMM branches. Empty / NaN on a non-finite (failed) fit so hosts
            // detect the absence the same way they do for OLS. No off-diagonals
            // exist for a scalar random intercept ⇒ re_corr stays empty.
            let (variance_components, sigma_sq_hat) =
                if fit.tau_sq_hat.is_finite() && fit.sigma_sq.is_finite() {
                    (vec![fit.tau_sq_hat], fit.sigma_sq)
                } else {
                    (vec![], f64::NAN)
                };
            (
                betas,
                se,
                statistic,
                fit.converged,
                variance_components,
                sigma_sq_hat,
                vec![],
            )
        }
    };

    // Build the crit table with the SAME args the pipeline uses (`run_introspect`):
    // `n_targets(spec)` = target_indices + contrast_pairs. The uncorrected
    // threshold read below is target-count-independent, but matching the
    // pipeline's build keeps the two paths provably identical (the basis of the
    // bit-equivalence test). `n_targets` is the module-private helper already in
    // introspect.rs — distinct from `n_test_targets` above.
    let table = CritValueTable::build(
        &spec.crit_values,
        &[nrow as u32],
        n_predictors_total(spec),
        n_targets(spec) as u32,
        spec.correction_method,
        spec.estimator,
    )?;

    Ok(IntrospectResults {
        betas,
        se,
        statistic,
        target_cols: spec.target_indices.clone(),
        converged,
        crit_sq_uncorrected: table.t_crit_sq_uncorrected[0],
        df_resid: (nrow as i64 - n_predictors_total(spec) as i64) as f64,
        variance_components,
        sigma_sq_hat,
        re_corr,
    })
}

/// n_targets = marginals + contrasts (matches `BatchResult` target ordering).
fn n_targets(spec: &SimulationSpec) -> usize {
    spec.target_indices.len() + spec.contrast_pairs.len()
}

fn n_predictors_total(spec: &SimulationSpec) -> u32 {
    1 + spec.n_non_factor + spec.n_factor_dummies + spec.interactions.len() as u32
}

pub fn run_introspect(
    spec: &SimulationSpec,
    n: u32,
    n_sims: u32,
    base_seed: u64,
    mask: IntrospectMask,
) -> Result<IntrospectOut, EngineError> {
    let crit = if mask.crit {
        let table = CritValueTable::build(
            &spec.crit_values,
            &[n],
            n_predictors_total(spec),
            n_targets(spec) as u32,
            spec.correction_method,
            spec.estimator,
        )?;
        Some(IntrospectCrit {
            crit_sq_uncorrected: table.t_crit_sq_uncorrected[0],
            df_resid: (n as i64 - n_predictors_total(spec) as i64) as f64,
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
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
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
            },
            design_test: None,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
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
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
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
