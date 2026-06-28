//! Per-thread workspace skeleton.
//!
//! Holds preallocated buffers reused across sims assigned to a single thread,
//! including the per-sim scenario design slots (perturbed Σ/L, var types,
//! residual choice, τ²) that `data_gen::populate_design` overwrites every sim.

use crate::correlation::EvdScratch;
use crate::spec::{Distribution, ResidualDist};
use faer::Mat;

/// One memoized reduced-model crit entry — see
/// [`SimWorkspace::reduced_crit_cache`].
pub struct ReducedCritEntry {
    /// Sample size the entry was built for.
    pub n: u32,
    /// Reduced predictor-column count (intercept included).
    pub p_red: u32,
    /// Single-N `CritValueTable` (built with a one-element sample-size slice;
    /// all per-N vectors have exactly one row, read at index 0).
    pub crit: crate::critvals::CritValueTable,
    /// Per-contrast post-hoc correction crits for that single N (flat across
    /// post-hoc blocks); empty when the spec has no post-hoc blocks.
    pub posthoc_corr_row: Vec<f64>,
}

/// Per-thread simulation workspace. Allocated once via `map_init` and reused
/// across all sims assigned to that thread.
pub struct SimWorkspace {
    /// Full design matrix at `max_n`, columns = predictors. f32 data plane —
    /// fit kernels widen `as f64` at read sites (f32 generation, f64 fit).
    pub x_full: Mat<f32>,
    /// Outcome vector at `max_n`.
    pub y_full: Vec<f32>,
    /// Residual buffer (Xβ noise contribution) at `max_n`.
    pub residuals: Vec<f32>,
    /// Planar standard-normal draws for the continuous block, column-major
    /// `[j·max_n + i]`, `n_non_factor` columns. Filled per sim by
    /// `rng::fill_normal_column` (CLASS_XNORM); consumed by the lower-tri
    /// mix column sweep.
    pub z_planar: Vec<f32>,
    /// f64 accumulator column for the L·z mix (`max_n`).
    pub mix_acc: Vec<f64>,
    /// f64 staging column for the outcome-assembly passes: η = lp + u_re for the
    /// binary sigmoid pass (and lp + u_re for the continuous y when the hsk exp
    /// pass needs mix_acc). Sized max_n, allocated once — warm path stays zero-alloc.
    pub eta_acc: Vec<f64>,
    /// Word scratch for the batched Philox fills (`max_n`).
    pub gen_words: Vec<u32>,
    /// Tail-lane index scratch for `fill_normal_column` (~4.85% of a column;
    /// capacity `max_n` so the warm path never reallocates).
    pub gen_tail_idx: Vec<u32>,
    /// f32 scratch column for the batched t/χ² residual accumulator draws
    /// (`max_n`). Stage-2 residual pass only.
    pub resid_g: Vec<f32>,
    /// Fixed-allocation running counts, one slot per factor level
    /// (`factor_proportions` layout). Reset per sim by `generate_sim_data`;
    /// sized lazily there so the constructor signature stays unchanged.
    pub factor_alloc_counts: Vec<u32>,
    /// Running per-level counts for the current row prefix, factor_proportions
    /// layout. Reset per sim in run_one_sim; grown incrementally with the grid.
    pub factor_prefix_counts: Vec<u32>,
    /// Per-factor exclusion code for the current (sim, n_idx):
    /// 0 = included, 1 = sparse-excluded, 2 = separation-dropped (GLM fallback).
    pub factor_excluded_flags: Vec<u8>,
    /// Memoized reduced-model crit tables for sparse-exclusion rounds, filled
    /// lazily by `batch::reduced_crit_entry`. Keyed by `(n, p_red)` — every
    /// df-dependent threshold depends only on the kept column count, never on
    /// *which* factors were dropped (the per-target Tukey k and the planned-m
    /// correction sequence are spec constants), so distinct exclusion patterns
    /// with equal width share an entry. Cold path only; bounded by
    /// (#distinct reduced widths × grid points).
    pub reduced_crit_cache: Vec<ReducedCritEntry>,

    // ------------------------------------------------------------------
    // Per-sim scenario design slots. Overwritten every sim by
    // `data_gen::populate_design` before the data draws read them.
    // ------------------------------------------------------------------
    /// Cholesky factor of the (possibly perturbed) correlation matrix for the
    /// current sim. `n_non_factor × n_non_factor`.
    pub l_design: Mat<f64>,
    /// Perturbed correlation scratch (scenario path).
    /// `n_non_factor × n_non_factor`.
    pub sigma_design: Mat<f64>,
    /// Base-correlation scratch (`n_non_factor × n_non_factor`) — refilled
    /// from `spec.correlation` on every scenario draw, so the scenario path
    /// never heap-allocates a per-sim copy.
    pub corr_base: Mat<f64>,
    /// Noise-snapshot scratch for `perturb_correlation` (`n_non_factor²`,
    /// fully overwritten before read on every use).
    pub perturb_noise_buf: Vec<f64>,
    /// Current sim's variable types (length `n_non_factor`).
    pub var_types_design: Vec<Distribution>,
    /// Current sim's residual distribution.
    pub residual_dist_design: ResidualDist,
    /// Current sim's residual df.
    pub residual_df_design: f64,
    /// Current sim's effective cluster variance τ²_eff: `cluster.tau_squared`
    /// plus an optional per-sim ICC jitter (scenario path), clamped ≥ 0. Equals
    /// `cluster.tau_squared` on the optimistic path / no-jitter scenarios. 0.0
    /// when there is no cluster. Read by the per-sim cluster-effect draw.
    pub tau_squared_design: f64,

    /// EVD workspace used by `psd_repair_and_factor` when scenario
    /// perturbations push the correlation matrix off the PSD cone.
    pub evd_scratch: EvdScratch,

    // ------------------------------------------------------------------
    // OLS fit scratch — written by `fit_suff_stats_t_sq` and consumed by posthoc.
    // Sized for the largest possible fit (p predictors).
    // ------------------------------------------------------------------
    pub fit_betas: Vec<f64>,
    pub fit_var_diag: Vec<f64>,
    pub fit_t_sq: Vec<f64>,
    pub fit_u_scratch: Vec<f64>,
    /// `P × P` factor: the lower-triangular Cholesky factor `L` of `X'X`
    /// (so `L · L' = X'X`), written by `fit_suff_stats_t_sq`; posthoc
    /// gates on `OlsFitView::converged`.
    pub fit_factor: Mat<f64>,
    /// `max_n × 1` rhs buffer for the lstsq solver (also reused as a length-`p`
    /// rhs for the Cholesky solve in the suff-stats path; only rows `0..p` are
    /// touched there).
    pub fit_rhs: Mat<f64>,

    // ------------------------------------------------------------------
    // Batch correction / posthoc staging scratch (reused per (sim, N)).
    // ------------------------------------------------------------------
    /// Length-`P` staging buffer for the compact (target-rank indexed) `t²`
    /// array fed to `apply_correction` in the MLE batch arm, where `fit.t_sq`
    /// is indexed by predictor (length `P`) and must be gathered down to the
    /// `n_targets` test ranks. Reused per (sim, N); only the first `n_targets`
    /// slots are touched. Disjoint from every field the LME fit view borrows.
    pub compact_t_sq_scratch: Vec<f64>,
    /// Per-contrast `t²` staging buffer for `evaluate_posthoc`. Its length is
    /// the posthoc contrast count (`Σ k(k−1)/2` over factor blocks), which is
    /// **not** bounded by `P` and is unknown at workspace construction — so it
    /// starts empty and is grown once on the first posthoc call, then reused
    /// (zero warm-path allocations thereafter).
    pub posthoc_t_sq_scratch: Vec<f64>,

    // ------------------------------------------------------------------
    // OLS sufficient statistics — accumulated row-by-row across the sample-
    // size grid by `OlsSuffStats`. Reset at the top of every sim.
    // ------------------------------------------------------------------
    /// `P × P` accumulator for `X'X`. Only the lower triangle is written
    /// (Cholesky `Side::Lower` ignores the upper triangle, so storing it would
    /// waste flops).
    pub suff_xtx: Mat<f64>,
    /// Length `P` accumulator for `X'y`.
    pub suff_xty: Vec<f64>,
    /// Scalar `y'y`.
    pub suff_yty: f64,
    /// Σ yᵢ accumulator parallel to `suff_yty`. Used by the OLS overall F-test
    /// to derive `SST = Σyᵢ² − (Σyᵢ)²/n` at the batch site.
    pub suff_sum_y: f64,
    /// Number of rows accumulated so far.
    pub suff_n_rows: usize,
    /// `P × P` working buffer — `suff_xtx` is copied here before factoring
    /// because `Cholesky` consumes the matrix it reads from.
    pub suff_xtx_work: Mat<f64>,
    /// Column-major f64 panel (`PANEL_ROWS × P`, leading dim = the current
    /// panel's rows) — `OlsSuffStats`/`LmeSuffStats::add_rows` widen f32 design
    /// rows here panel-by-panel before the GEMM accumulate. Allocated once;
    /// warm path stays zero-alloc.
    pub panel_x: Vec<f64>,
    /// y twin of `panel_x`, length `PANEL_ROWS`.
    pub panel_y: Vec<f64>,

    // ------------------------------------------------------------------
    // IRLS scratch (Logit). All per-fit buffers live here per
    // the fix1 pattern; `glm::glm_irls_fit` takes a `GlmScratch<'w>` built
    // inline at the call site.
    //
    // Marginal per-thread footprint at max_n=10^4, P=10:
    //   per-row buffers (4 × 8 × max_n) ≈ 320 KB
    //   f64 X copy (8 × max_n × P) ≈ 800 KB
    //   P×P faer Mats (2 × 8 × P²) ≈ 1.6 KB
    //   per-P slots (6 × 8 × P) ≈ 0.48 KB
    // ≈ 1.1 MB per thread.
    // ------------------------------------------------------------------
    /// length `max_n` — per-row linear predictor η = Xβ.
    pub irls_eta: Vec<f64>,
    /// length `max_n` — per-row p_i = σ(η_i).
    pub irls_p: Vec<f64>,
    /// length `max_n` — per-row W_i = max(p_i(1-p_i), WEIGHT_CLAMP).
    pub irls_w: Vec<f64>,
    /// length `max_n` — per-row working response z_i = η_i + (y_i - p_i)/W_i.
    pub irls_z: Vec<f64>,
    /// length `P` — current β estimate.
    pub irls_betas: Vec<f64>,
    /// length `P` — candidate β before step-halving / accept check.
    pub irls_betas_new: Vec<f64>,
    /// length `P` — per-target Var(β̂_j) = ((X'WX)⁻¹)_jj (prefix written).
    pub irls_var_diag: Vec<f64>,
    /// length `P` — per-target z² values (named `t_sq` for uniformity with
    /// `OlsFitView`); prefix written.
    pub irls_t_sq: Vec<f64>,
    /// length `P` — forward-solve scratch for the per-target var_diag.
    pub irls_u_scratch: Vec<f64>,
    /// `P × P` — accumulator for X'WX (lower triangle).
    pub irls_xtwx: Mat<f64>,
    /// length `P` — RHS X'Wz for the normal-eq solve.
    pub irls_xtwz: Vec<f64>,
    /// `P × P` — cached lower-triangular Cholesky factor L of the last
    /// accepted X'WX. Valid only when `GlmFitView::converged == true`.
    pub irls_l: Mat<f64>,
    /// `max_n × P` capacity — per-fit f64 copy of X, column-major with
    /// stride = the fit's n (only the `[..n*p]` prefix is live per fit).
    /// Filled once per `glm_irls_fit` call so the IRLS hot loops read
    /// contiguous f64 instead of widening bounds-checked f32 MatRef loads
    /// every iteration.
    pub irls_x_f64: Vec<f64>,
    /// Per-iteration W∘X scratch (column-major, mirrors `irls_x_f64`); needs `len ≥ n·p`.
    pub irls_wx: Vec<f64>,

    // ------------------------------------------------------------------
    // LME suff-stats — rebuilt per (sim, N) by `lme_add_rows`.
    // The first three slots are numerically equal to the OLS X'X / X'y / y'y
    // and are aliased into a nested OlsScratch on the `boundary_hit = 1` path.
    // ------------------------------------------------------------------
    /// `P × P` accumulator for X'X (lower triangle only — fix2 invariant).
    pub lme_xtx: Mat<f64>,
    /// Length-P X'y accumulator.
    pub lme_xty: Vec<f64>,
    /// Scalar y'y.
    pub lme_yty: f64,
    /// `P × max_n_clusters` per-cluster sum of X rows.
    pub lme_sum_xc: Mat<f64>,
    /// Length-`max_n_clusters` per-cluster sum of y rows.
    pub lme_sum_yc: Vec<f64>,
    /// Length-`max_n_clusters` per-cluster row counts (u32 — capped at `u32::MAX`).
    pub lme_cluster_sizes: Vec<u32>,
    /// Current cluster count seen (≤ max_n_clusters at all times).
    pub lme_n_clusters_seen: u32,
    /// `P × P` working buffer for the Cholesky factor of `X' V(θ)⁻¹ X`.
    pub lme_xtvix: Mat<f64>,
    /// Length-P RHS `X' V(θ)⁻¹ y`.
    pub lme_xtviy: Vec<f64>,
    /// Cached Cholesky L of `lme_xtvix` at θ̂ (for var_diag forward-solve).
    pub lme_xtvix_factor: Mat<f64>,
    /// Length-`max_n_clusters` per-cluster `(1 + θ²·n_c)⁻¹`.
    pub lme_v_diag_inv: Vec<f64>,
    /// Length-P β̂(θ̂).
    pub lme_betas: Vec<f64>,
    /// Length-P per-target Var(β̂_j).
    pub lme_var_diag: Vec<f64>,
    /// Length-P per-target β̂_j² / Var(β̂_j).
    pub lme_t_sq: Vec<f64>,
    /// Length-P forward-solve scratch (same role as `fit_u_scratch`).
    /// Also borrowed by the OLS contrast path as the forward-solve scratch
    /// for `ols_contrast_t_sq` (L · u = c, length P ≥ n_predictors).
    pub lme_u_scratch: Vec<f64>,
    /// Brent scalar state (one per call; lives in workspace so no per-call alloc).
    pub lme_brent_log_a: f64,
    pub lme_brent_log_b: f64,
    pub lme_brent_log_c: f64,
    pub lme_brent_fa: f64,
    pub lme_brent_fb: f64,
    pub lme_brent_fc: f64,
    // ------------------------------------------------------------------
    // Joint Wald-χ² scratch. Sized at construction to worst-case P; reused per sim.
    // ------------------------------------------------------------------
    /// `P × P` scratch; top-left k×k holds the Cholesky factor L_T of Σ_T
    /// where Σ_T is the k×k principal submatrix of K⁻¹ = (X'V⁻¹X)⁻¹ gathered
    /// by `target_indices`. Overwritten per call.
    pub lme_joint_sigma_t_chol: Mat<f64>,
    /// Length-`P` RHS for the joint solve. On entry: β̂_T copied in. On exit:
    /// x = Σ_T⁻¹ β̂_T (only the first k slots are touched).
    /// Also borrowed by the OLS path as a staging scratch for the combined
    /// marginal + contrast t² array, which can exceed P (full pairwise
    /// contrast sets) — `run_batch_impl` grows it to max(P, n_targets) right
    /// after construction; LME consumers only take `[..P]` views.
    pub lme_joint_rhs: Vec<f64>,
    /// `P × P` scratch. Starts as `I_p` (refilled by `reset_lme_suff_stats`
    /// per sim) and is overwritten in-place to `K⁻¹` by `chol.solve_in_place`.
    pub lme_joint_k_inv: Mat<f64>,
    /// Length-`max_n_clusters` per-cluster random-effect draws (`τ · N(0,1)`)
    /// — written by `data_gen.rs` step 1, read in step 4.
    pub cluster_u_draws: Vec<f32>,
    /// Per-cluster random-SLOPE draws, row-major `[cluster · #slopes + k]` =
    /// `u_{k+1}` for slope k (the intercept draw `u_0` stays in
    /// `cluster_u_draws`). Written by data-gen's `q_p×q_p` Cholesky draw, read in
    /// the assembly loop as `Σ_k u_{k+1}·x_slope_k`. Zero-length when the primary
    /// has no slope.
    pub cluster_slope_u_draws: Vec<f32>,
    /// Length-`max_n` cluster assignments, interleaved (`i % n_clusters`).
    /// Invariant in N — populated once at construction.
    pub cluster_ids: Vec<u32>,
    /// Per-extra-grouping level ids per row, declaration order — one
    /// `Vec<u32>` of length max_n per extra grouping, built once at
    /// construction from `ClusterSpec::extra_level_of_row` (invariant in N,
    /// like `cluster_ids`). Empty when the spec has no extra groupings.
    pub extra_grouping_ids: Vec<Vec<u32>>,
    /// Per-extra-grouping random-effect draws (`τ_g · unit draw` per level) —
    /// written by `data_gen.rs` step 2b, read in the assembly loop. Sized by
    /// `extra_n_levels_at(g, max_n)`.
    pub extra_u_draws: Vec<Vec<f32>>,
    /// Per-extra-grouping effective τ² for the current sim — base
    /// `tau_squared` plus the scenario ICC jitter (independent draw per
    /// grouping). Mirrors `tau_squared_design`; both are set together by
    /// `populate_design` — change together.
    pub extra_tau_sq_design: Vec<f64>,
    /// General-path solver workspace — Some iff the spec dispatches to the
    /// lmm path (estimator Mle + non-degenerate ClusterSpec). Built by the
    /// batch/introspect construction sites via `lmm::build_lmm_workspace`
    /// (the ctor doesn't know the estimator). Boxed: the tail/zx buffers are
    /// large relative to the rest of the workspace.
    pub lmm: Option<Box<glmm::mcpower::LmmWorkspace>>,
    /// GLMM solver workspace — `Some` for Glm + cluster specs (built at the batch
    /// entry, like `lmm`). `None` for OLS / plain logistic / LME paths.
    pub glmm: Option<Box<glmm::mcpower::GlmmWorkspace>>,
}

impl SimWorkspace {
    /// Allocate a fresh workspace sized for the given configuration.
    ///
    /// - `max_n`: largest sample size that will be drawn from this workspace.
    /// - `n_predictors`: total columns of `X_full` (intercept + non-factor +
    ///   factor dummies).
    /// - `n_non_factor`: number of correlated continuous predictors.
    /// - `_n_factors`: number of factor columns — unused (factor categorical
    ///   draws write straight into `x_full`); retained to keep the many
    ///   existing call sites stable.
    /// - `cluster`: the full cluster spec (sizing regime plus extra
    ///   groupings); pass `None` for OLS / Logit specs (no clusters). Used to
    ///   derive cluster count and build `cluster_ids` with the correct layout
    ///   for the regime (round-robin for FixedClusters, contiguous blocks for
    ///   FixedSize), plus the per-extra-grouping id/draw buffers from the
    ///   `ClusterSpec` layout helpers.
    pub fn new(
        max_n: usize,
        n_predictors: usize,
        n_non_factor: usize,
        _n_factors: usize,
        cluster: Option<&engine_contract::ClusterSpec>,
    ) -> Self {
        let max_n_clusters = cluster.map(|c| c.sizing.n_clusters_at(max_n)).unwrap_or(0);
        let n_extras = cluster.map(|c| c.extra_groupings.len()).unwrap_or(0);
        Self {
            x_full: Mat::<f32>::zeros(max_n, n_predictors),
            y_full: vec![0.0f32; max_n],
            residuals: vec![0.0f32; max_n],
            z_planar: vec![0.0f32; max_n * n_non_factor],
            mix_acc: vec![0.0f64; max_n],
            eta_acc: vec![0.0f64; max_n],
            gen_words: vec![0u32; max_n],
            gen_tail_idx: Vec::with_capacity(max_n),
            resid_g: vec![0.0f32; max_n],
            factor_alloc_counts: Vec::new(),
            factor_prefix_counts: Vec::new(),
            factor_excluded_flags: Vec::new(),
            reduced_crit_cache: Vec::new(),

            l_design: Mat::<f64>::zeros(n_non_factor, n_non_factor),
            sigma_design: Mat::<f64>::zeros(n_non_factor, n_non_factor),
            corr_base: Mat::<f64>::zeros(n_non_factor, n_non_factor),
            perturb_noise_buf: vec![0.0; n_non_factor * n_non_factor],
            var_types_design: vec![Distribution::Normal; n_non_factor],
            residual_dist_design: ResidualDist::Normal,
            residual_df_design: 0.0,
            tau_squared_design: 0.0,

            evd_scratch: EvdScratch::new(),

            fit_betas: vec![0.0; n_predictors],
            fit_var_diag: vec![0.0; n_predictors],
            fit_t_sq: vec![0.0; n_predictors],
            fit_u_scratch: vec![0.0; n_predictors],
            fit_factor: Mat::<f64>::zeros(n_predictors, n_predictors),
            fit_rhs: Mat::<f64>::zeros(max_n.max(n_predictors), 1),

            compact_t_sq_scratch: vec![0.0; n_predictors],
            posthoc_t_sq_scratch: Vec::new(),

            suff_xtx: Mat::<f64>::zeros(n_predictors, n_predictors),
            suff_xty: vec![0.0; n_predictors],
            suff_yty: 0.0,
            suff_sum_y: 0.0,
            suff_n_rows: 0,
            suff_xtx_work: Mat::<f64>::zeros(n_predictors, n_predictors),
            panel_x: vec![0.0f64; glmm::mcpower::PANEL_ROWS * n_predictors],
            panel_y: vec![0.0f64; glmm::mcpower::PANEL_ROWS],

            irls_eta: vec![0.0; max_n],
            irls_p: vec![0.0; max_n],
            irls_w: vec![0.0; max_n],
            irls_z: vec![0.0; max_n],
            irls_betas: vec![0.0; n_predictors],
            irls_betas_new: vec![0.0; n_predictors],
            irls_var_diag: vec![0.0; n_predictors],
            irls_t_sq: vec![0.0; n_predictors],
            irls_u_scratch: vec![0.0; n_predictors],
            irls_xtwx: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_xtwz: vec![0.0; n_predictors],
            irls_l: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_x_f64: vec![0.0; max_n * n_predictors],
            irls_wx: vec![0.0; max_n * n_predictors],

            lme_xtx: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_xty: vec![0.0; n_predictors],
            lme_yty: 0.0,
            lme_sum_xc: Mat::<f64>::zeros(n_predictors, max_n_clusters.max(1)),
            lme_sum_yc: vec![0.0; max_n_clusters.max(1)],
            lme_cluster_sizes: vec![0u32; max_n_clusters.max(1)],
            lme_n_clusters_seen: 0,
            lme_xtvix: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_xtviy: vec![0.0; n_predictors],
            lme_xtvix_factor: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_v_diag_inv: vec![0.0; max_n_clusters.max(1)],
            lme_betas: vec![0.0; n_predictors],
            lme_var_diag: vec![0.0; n_predictors],
            lme_t_sq: vec![0.0; n_predictors],
            lme_u_scratch: vec![0.0; n_predictors],
            lme_brent_log_a: 0.0,
            lme_brent_log_b: 0.0,
            lme_brent_log_c: 0.0,
            lme_brent_fa: 0.0,
            lme_brent_fb: 0.0,
            lme_brent_fc: 0.0,
            lme_joint_sigma_t_chol: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_joint_rhs: vec![0.0; n_predictors],
            lme_joint_k_inv: {
                let mut m = Mat::<f64>::zeros(n_predictors, n_predictors);
                for i in 0..n_predictors {
                    m[(i, i)] = 1.0;
                }
                m
            },
            cluster_u_draws: vec![0.0f32; max_n_clusters],
            cluster_slope_u_draws: match cluster {
                Some(c) if !c.slopes.is_empty() => vec![0.0f32; max_n_clusters * c.slopes.len()],
                _ => Vec::new(),
            },
            cluster_ids: {
                let mut ids = vec![0u32; max_n];
                if let Some(c) = cluster {
                    for (i, id) in ids.iter_mut().enumerate().take(max_n) {
                        *id = c.sizing.cluster_of_row(i) as u32;
                    }
                }
                ids
            },
            extra_grouping_ids: (0..n_extras)
                .map(|g| {
                    let c = cluster.unwrap();
                    (0..max_n)
                        .map(|i| c.extra_level_of_row(g, i) as u32)
                        .collect()
                })
                .collect(),
            extra_u_draws: (0..n_extras)
                .map(|g| vec![0.0f32; cluster.unwrap().extra_n_levels_at(g, max_n)])
                .collect(),
            extra_tau_sq_design: vec![0.0; n_extras],
            lmm: None,
            glmm: None,
        }
    }

    /// Reset the OLS sufficient-statistics accumulator to "no rows seen".
    /// Reuses storage — zeros only the lower triangle of `suff_xtx` (the upper
    /// triangle is never read or written by `add_rows_suff` / Cholesky).
    pub fn reset_suff_stats(&mut self) {
        let p = self.suff_xty.len();
        for j in 0..p {
            for i in j..p {
                self.suff_xtx[(i, j)] = 0.0;
            }
            self.suff_xty[j] = 0.0;
        }
        self.suff_yty = 0.0;
        self.suff_sum_y = 0.0;
        self.suff_n_rows = 0;
    }

    /// Reset the LME sufficient-statistics accumulator to "no rows seen".
    /// Zeros the lower triangle of `lme_xtx`, all of `lme_xty`, `lme_yty`,
    /// `lme_sum_xc` (entire buffer), `lme_sum_yc`, `lme_cluster_sizes`, and
    /// resets `lme_n_clusters_seen = 0`.
    /// Brent state and per-θ scratch are **not** zeroed here — they are
    /// overwritten on every `profiled_deviance` call.
    pub fn reset_lme_suff_stats(&mut self) {
        let p = self.lme_xty.len();
        let k = self.lme_sum_yc.len();
        for j in 0..p {
            for i in j..p {
                self.lme_xtx[(i, j)] = 0.0;
            }
            self.lme_xty[j] = 0.0;
            for c in 0..k {
                self.lme_sum_xc[(j, c)] = 0.0;
            }
        }
        for v in self.lme_sum_yc.iter_mut() {
            *v = 0.0;
        }
        for v in self.lme_cluster_sizes.iter_mut() {
            *v = 0;
        }
        self.lme_yty = 0.0;
        self.lme_n_clusters_seen = 0;

        // Refill identity into the joint K⁻¹ scratch (consumed in-place by
        // chol.solve_in_place — see `lme.rs` joint Wald-χ² block).
        // `lme_joint_sigma_t_chol` and `lme_joint_rhs` are fully overwritten
        // per call so they need no reset.
        for j in 0..p {
            for i in 0..p {
                self.lme_joint_k_inv[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_contract::ClusterSizing;

    #[test]
    fn workspace_allocates_with_correct_shapes() {
        let ws = SimWorkspace::new(
            100,
            5,
            3,
            2,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters: 20 },
                0.25,
            )),
        );
        assert_eq!(ws.x_full.nrows(), 100);
        assert_eq!(ws.x_full.ncols(), 5);
        assert_eq!(ws.y_full.len(), 100);
        assert_eq!(ws.residuals.len(), 100);
        assert_eq!(ws.z_planar.len(), 100 * 3);
        assert_eq!(ws.mix_acc.len(), 100);
        assert_eq!(ws.eta_acc.len(), 100);
        assert_eq!(ws.gen_words.len(), 100);
        assert!(ws.gen_tail_idx.is_empty() && ws.gen_tail_idx.capacity() >= 100);
        assert_eq!(ws.resid_g.len(), 100);
        assert_eq!(ws.l_design.nrows(), 3);
        assert_eq!(ws.l_design.ncols(), 3);
        assert_eq!(ws.sigma_design.nrows(), 3);
        assert_eq!(ws.sigma_design.ncols(), 3);
        assert_eq!(ws.corr_base.nrows(), 3);
        assert_eq!(ws.corr_base.ncols(), 3);
        assert_eq!(ws.perturb_noise_buf.len(), 9);
        assert_eq!(ws.var_types_design.len(), 3);

        assert_eq!(ws.fit_betas.len(), 5);
        assert_eq!(ws.fit_var_diag.len(), 5);
        assert_eq!(ws.fit_t_sq.len(), 5);
        assert_eq!(ws.fit_u_scratch.len(), 5);
        assert_eq!(ws.fit_factor.nrows(), 5);
        assert_eq!(ws.fit_factor.ncols(), 5);
        assert_eq!(ws.fit_rhs.nrows(), 100);
        assert_eq!(ws.fit_rhs.ncols(), 1);

        assert_eq!(ws.suff_xtx.nrows(), 5);
        assert_eq!(ws.suff_xtx.ncols(), 5);
        assert_eq!(ws.suff_xty.len(), 5);
        assert_eq!(ws.suff_yty, 0.0);
        assert_eq!(ws.suff_n_rows, 0);
        assert_eq!(ws.suff_xtx_work.nrows(), 5);
        assert_eq!(ws.suff_xtx_work.ncols(), 5);
        assert_eq!(ws.panel_x.len(), glmm::mcpower::PANEL_ROWS * 5);
        assert_eq!(ws.panel_y.len(), glmm::mcpower::PANEL_ROWS);

        // IRLS scratch (Logit).
        assert_eq!(ws.irls_eta.len(), 100);
        assert_eq!(ws.irls_p.len(), 100);
        assert_eq!(ws.irls_w.len(), 100);
        assert_eq!(ws.irls_z.len(), 100);
        assert_eq!(ws.irls_betas.len(), 5);
        assert_eq!(ws.irls_betas_new.len(), 5);
        assert_eq!(ws.irls_var_diag.len(), 5);
        assert_eq!(ws.irls_t_sq.len(), 5);
        assert_eq!(ws.irls_u_scratch.len(), 5);
        assert_eq!(ws.irls_xtwx.nrows(), 5);
        assert_eq!(ws.irls_xtwx.ncols(), 5);
        assert_eq!(ws.irls_xtwz.len(), 5);
        assert_eq!(ws.irls_l.nrows(), 5);
        assert_eq!(ws.irls_l.ncols(), 5);
        assert_eq!(ws.irls_x_f64.len(), 100 * 5);
        assert_eq!(ws.irls_wx.len(), 100 * 5);

        // LME scratch.
        assert_eq!(ws.lme_xtx.nrows(), 5);
        assert_eq!(ws.lme_xtx.ncols(), 5);
        assert_eq!(ws.lme_xty.len(), 5);
        assert_eq!(ws.lme_yty, 0.0);
        assert_eq!(ws.lme_sum_xc.nrows(), 5);
        assert_eq!(ws.lme_sum_xc.ncols(), 20);
        assert_eq!(ws.lme_sum_yc.len(), 20);
        assert_eq!(ws.lme_cluster_sizes.len(), 20);
        assert_eq!(ws.lme_n_clusters_seen, 0);
        assert_eq!(ws.lme_xtvix.nrows(), 5);
        assert_eq!(ws.lme_xtvix.ncols(), 5);
        assert_eq!(ws.lme_xtviy.len(), 5);
        assert_eq!(ws.lme_xtvix_factor.nrows(), 5);
        assert_eq!(ws.lme_xtvix_factor.ncols(), 5);
        assert_eq!(ws.lme_v_diag_inv.len(), 20);
        assert_eq!(ws.lme_betas.len(), 5);
        assert_eq!(ws.lme_var_diag.len(), 5);
        assert_eq!(ws.lme_t_sq.len(), 5);
        assert_eq!(ws.lme_u_scratch.len(), 5);
        assert_eq!(ws.lme_joint_sigma_t_chol.nrows(), 5);
        assert_eq!(ws.lme_joint_sigma_t_chol.ncols(), 5);
        assert_eq!(ws.lme_joint_rhs.len(), 5);
        assert_eq!(ws.lme_joint_k_inv.nrows(), 5);
        assert_eq!(ws.lme_joint_k_inv.ncols(), 5);
        // joint_k_inv starts as identity.
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(ws.lme_joint_k_inv[(i, j)], expected);
            }
        }
        assert_eq!(ws.cluster_u_draws.len(), 20);
        assert_eq!(ws.cluster_ids.len(), 100);
        // Interleaved layout: cluster_ids[i] == i % 20.
        for i in 0..100 {
            assert_eq!(ws.cluster_ids[i], (i % 20) as u32);
        }

        // FixedSize block layout: cluster_ids[i] == i / cluster_size.
        let ws_block = SimWorkspace::new(
            100,
            5,
            3,
            2,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedSize { cluster_size: 5 },
                0.25,
            )),
        );
        assert_eq!(ws_block.cluster_ids[0], 0);
        assert_eq!(ws_block.cluster_ids[4], 0);
        assert_eq!(ws_block.cluster_ids[5], 1); // block boundary
        assert_eq!(ws_block.cluster_ids[99], 19); // 99 / 5
                                                  // Buffer is sized to max_n / cluster_size = 100 / 5 = 20.
        assert_eq!(ws_block.cluster_u_draws.len(), 20);
    }

    #[test]
    fn reset_lme_suff_stats_zeros_state() {
        let mut ws = SimWorkspace::new(
            100,
            5,
            3,
            2,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters: 20 },
                0.25,
            )),
        );
        // Poke non-zero values into every field that reset should clear.
        for j in 0..5 {
            for i in j..5 {
                ws.lme_xtx[(i, j)] = 1.0;
            }
            ws.lme_xty[j] = 2.0;
            for c in 0..20 {
                ws.lme_sum_xc[(j, c)] = 3.0;
            }
        }
        for v in ws.lme_sum_yc.iter_mut() {
            *v = 4.0;
        }
        for v in ws.lme_cluster_sizes.iter_mut() {
            *v = 5;
        }
        ws.lme_yty = 6.0;
        ws.lme_n_clusters_seen = 7;
        // Poke non-identity into joint K⁻¹ to confirm reset restores identity.
        for j in 0..5 {
            for i in 0..5 {
                ws.lme_joint_k_inv[(i, j)] = 99.0;
            }
        }

        ws.reset_lme_suff_stats();

        // Assert all are zero / reset.
        for j in 0..5 {
            for i in j..5 {
                assert_eq!(ws.lme_xtx[(i, j)], 0.0);
            }
            assert_eq!(ws.lme_xty[j], 0.0);
            for c in 0..20 {
                assert_eq!(ws.lme_sum_xc[(j, c)], 0.0);
            }
        }
        for v in &ws.lme_sum_yc {
            assert_eq!(*v, 0.0);
        }
        for v in &ws.lme_cluster_sizes {
            assert_eq!(*v, 0);
        }
        assert_eq!(ws.lme_yty, 0.0);
        assert_eq!(ws.lme_n_clusters_seen, 0);
        // joint_k_inv must be refilled to identity by the reset.
        for j in 0..5 {
            for i in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(ws.lme_joint_k_inv[(i, j)], expected);
            }
        }
    }

    /// DGEN-12: `reset_suff_stats` zeroes the OLS sufficient-statistics
    /// accumulators (lower triangle of suff_xtx, all of suff_xty/yty/sum_y,
    /// suff_n_rows) without touching any LME or IRLS field.
    #[test]
    fn reset_suff_stats_leaves_lme_irls() {
        let mut ws = SimWorkspace::new(
            100,
            5,
            3,
            2,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters: 20 },
                0.25,
            )),
        );

        // Poke OLS suff-stats.
        for j in 0..5 {
            for i in j..5 {
                ws.suff_xtx[(i, j)] = 7.0;
            }
            ws.suff_xty[j] = 8.0;
        }
        ws.suff_yty = 9.0;
        ws.suff_sum_y = 10.0;
        ws.suff_n_rows = 11;

        // Poke LME + IRLS fields that reset_suff_stats must NOT touch.
        ws.lme_yty = 42.0;
        ws.lme_n_clusters_seen = 5;
        ws.irls_betas[0] = 99.0;
        ws.irls_t_sq[2] = 88.0;

        ws.reset_suff_stats();

        // OLS suff-stats zeroed.
        for j in 0..5 {
            for i in j..5 {
                assert_eq!(ws.suff_xtx[(i, j)], 0.0, "suff_xtx[{i},{j}] not zeroed");
            }
            assert_eq!(ws.suff_xty[j], 0.0, "suff_xty[{j}] not zeroed");
        }
        assert_eq!(ws.suff_yty, 0.0);
        assert_eq!(ws.suff_sum_y, 0.0);
        assert_eq!(ws.suff_n_rows, 0);

        // LME + IRLS untouched.
        assert_eq!(
            ws.lme_yty, 42.0,
            "reset_suff_stats must not touch LME fields"
        );
        assert_eq!(
            ws.lme_n_clusters_seen, 5,
            "reset_suff_stats must not touch LME fields"
        );
        assert_eq!(
            ws.irls_betas[0], 99.0,
            "reset_suff_stats must not touch IRLS fields"
        );
        assert_eq!(
            ws.irls_t_sq[2], 88.0,
            "reset_suff_stats must not touch IRLS fields"
        );
    }

    #[test]
    fn extra_grouping_buffers_sized_and_laid_out() {
        let mut spec = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 4 },
            0.25,
        );
        spec.extra_groupings.push(engine_contract::GroupingSpec {
            relation: engine_contract::GroupingRelation::Crossed { n_clusters: 3 },
            tau_squared: 0.1,
            slopes: vec![],
        });
        spec.extra_groupings.push(engine_contract::GroupingSpec {
            relation: engine_contract::GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.05,
            slopes: vec![],
        });
        let ws = SimWorkspace::new(24, 3, 2, 0, Some(&spec));
        assert_eq!(ws.extra_grouping_ids.len(), 2);
        assert_eq!(ws.extra_u_draws.len(), 2);
        assert_eq!(ws.extra_tau_sq_design, vec![0.0, 0.0]);
        assert_eq!(ws.extra_u_draws[0].len(), 3); // crossed: fixed count
        assert_eq!(ws.extra_u_draws[1].len(), 8); // nested: 4 parents × 2
        for i in 0..24 {
            assert_eq!(
                ws.extra_grouping_ids[0][i],
                spec.extra_level_of_row(0, i) as u32
            );
            assert_eq!(
                ws.extra_grouping_ids[1][i],
                spec.extra_level_of_row(1, i) as u32
            );
        }
        // Degenerate spec ⇒ all extras empty (today's shape).
        let plain = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 4 },
            0.25,
        );
        let ws2 = SimWorkspace::new(24, 3, 2, 0, Some(&plain));
        assert!(ws2.extra_grouping_ids.is_empty());
        assert!(ws2.extra_u_draws.is_empty());
        assert!(ws2.extra_tau_sq_design.is_empty());
    }
}
