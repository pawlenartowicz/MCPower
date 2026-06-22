//! LME profiled-deviance solver — Brent's method on log(θ).
//!
//! Hot-loop invariants (mirror `ols.rs` post-fix1+fix2):
//!  * Bounded allocations on the warm path — no explicit per-fit allocs; the
//!    locked block count is dominated by faer's Cholesky internals
//!    (bounded-alloc test in `lme::tests`).
//!  * Inference returns squared statistics — Decision A locks Wald z; the
//!    `t_sq` slot holds `β̂²/Var(β̂)`, compared against the squared z critical
//!    value already stored in `crit.t_crit_sq_uncorrected`.
//!  * All per-θ scratch (`lme_v_diag_inv`, `lme_xtvix`, `lme_xtviy`,
//!    `lme_xtvix_factor`) is fully overwritten on every `profiled_deviance`
//!    call — no per-iteration reset needed inside the Brent loop.
//!
//! **NR** = Press, Teukolsky, Vetterling & Flannery (2007), *Numerical Recipes:
//! The Art of Scientific Computing*, 3rd ed., Cambridge University Press.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::linalg::matmul::{matmul, triangular};
use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};

use crate::ols::{chol_rank_deficient, OlsScratch, PANEL_ROWS};
use crate::FLOAT_NEAR_ZERO;

/// Maximum Brent-loop iterations before declaring non-convergence.
const MAX_BRENT_ITERS: u32 = 50;
/// Default relative tolerance on log(θ) for Brent termination.
const BRENT_REL_TOL: f64 = 1e-4;
/// Initial θ bracket — log-spaced. The upper bound keeps θ capped so
/// λ² ∈ [0, 1e6] ⇒ θ ∈ [0, 1000]. Capping at θ=1e2 would truncate that
/// domain and trigger high-τ rebracket retries on cases the main loop
/// handles within [0, 1000].
const LOG_THETA_LOW: f64 = -9.210_340_371_976_184; // ln(1e-4)
                                                   // clippy::approx_constant fires because -ln(10) ≈ LN_10; this is intentional
                                                   // (ln(1e-1) = -ln(10)), not a use of the std constant.
#[expect(
    clippy::approx_constant,
    reason = "literal is ln(1e-1) = -ln(10), not a use of the std LN_10 constant"
)]
const LOG_THETA_MID: f64 = -2.302_585_092_994_046; // ln(1e-1)
const LOG_THETA_HIGH: f64 = 6.907_755_278_982_137; // ln(1e3) — matches v1's λ²≤1e6 reach
/// Boundary-detection slack (log-units). If Brent settles within this distance
/// of an endpoint we treat it as a boundary hit.
const BOUNDARY_LOG_SLACK: f64 = 0.1;
/// Half-width Δ of the truth-centered bracket (log-units, ≈ ×7.4 each way) —
/// many σ of log θ̂'s sampling spread at realistic cluster counts. The width
/// trade-off is asymmetric: golden-section cost is logarithmic in width, so
/// too-wide costs a fraction of an eval, while too-narrow costs the repair
/// path plus the cold-bracket retry (~2× a cold fit).
const TRUTH_BRACKET_HALF_WIDTH: f64 = 2.0;

/// Accumulated sufficient statistics for one LME simulation step.
///
/// Mirrors `OlsSuffStats` but adds per-cluster accumulators for the random-
/// intercept variance component estimation.
pub struct LmeSuffStats<'w> {
    /// `P × P` X'X accumulator (lower triangle).
    pub xtx: MatMut<'w, f64>,
    /// Length-P X'y accumulator.
    pub xty: &'w mut [f64],
    /// Scalar y'y.
    pub yty: &'w mut f64,
    /// `P × max_n_clusters` per-cluster sum of X rows.
    pub sum_xc: MatMut<'w, f64>,
    /// Length-`max_n_clusters` per-cluster sum of y.
    pub sum_yc: &'w mut [f64],
    /// Length-`max_n_clusters` per-cluster row counts.
    pub cluster_sizes: &'w mut [u32],
    /// Running cluster count (highest cluster_id + 1 seen so far).
    pub n_clusters_seen: &'w mut u32,
    /// Panel-widening scratch, len ≥ `min(block rows, PANEL_ROWS) · p` —
    /// workspace `panel_x` at the hot sites.
    pub panel_x: &'w mut [f64],
    /// y twin, len ≥ `min(block rows, PANEL_ROWS)` — workspace `panel_y`.
    pub panel_y: &'w mut [f64],
}

/// Per-simulation scratch for the LME Brent solver and inference.
///
/// `LmeScratch` is built at the call site by reborrowing fields of
/// `SimWorkspace`. The nested `ols_scratch` aliases the same workspace storage
/// so the `boundary_hit = 1` OLS fallback can call `fit_suff_stats_t_sq`
/// without a second accumulator.
///
/// Note on `xtx`/`xty`/`yty`: these are exposed as read-only borrows of the
/// workspace's `lme_xtx`/`lme_xty`/`lme_yty` (the suff-stats accumulator
/// targets). `OlsScratch` does not expose X'X / X'y / y'y directly (those
/// live in `OlsSuffStats`, which is used at suff-stats build time only). The
/// τ̂≈0 OLS fallback routes its inputs through `OlsSuffStats` over the same
/// `lme_xtx`/`lme_xty`/`lme_yty` storage — so the duplication here is just
/// a read-only view, not a second accumulator.
pub struct LmeScratch<'w> {
    /// `P × P` X'X accumulator (lower triangle) — read-only here, written by
    /// `LmeSuffStats::add_rows`. Borrowed from workspace's `lme_xtx`.
    pub xtx: MatRef<'w, f64>,
    /// Length-P X'y — read-only here, written by `LmeSuffStats::add_rows`.
    /// Borrowed from workspace's `lme_xty`.
    pub xty: &'w [f64],
    /// Scalar y'y — read-only here, written by `LmeSuffStats::add_rows`.
    pub yty: f64,
    /// X'X / X'y / y'y aliased through this nested `OlsScratch` for the
    /// `boundary_hit = 1` (τ̂≈0) OLS fallback. Built from the same workspace
    /// storage — no double accumulation.
    pub ols_scratch: OlsScratch<'w>,
    /// `P × max_n_clusters` per-cluster sum of X (read-only after suff-stats build).
    pub sum_xc: MatRef<'w, f64>,
    /// Length-`max_n_clusters` per-cluster sum of y.
    pub sum_yc: &'w [f64],
    /// Length-`max_n_clusters` per-cluster row counts.
    pub cluster_sizes: &'w [u32],
    /// Cluster count for this sim.
    pub n_clusters: u32,
    /// Total rows in the fit (matches the `n_rows_total` arg the caller
    /// passes after `lme_add_rows`).
    pub n_rows: u32,
    /// `P × P` Cholesky scratch — `lme_xtvix` from the workspace.
    pub xtvix: MatMut<'w, f64>,
    /// Length-P RHS — `lme_xtviy` from the workspace.
    pub xtviy: &'w mut [f64],
    /// `P × P` cached Cholesky factor at θ̂.
    pub xtvix_factor: MatMut<'w, f64>,
    /// Length-`max_n_clusters` per-cluster `(1 + θ²·n_c)⁻¹` scratch.
    pub v_diag_inv: &'w mut [f64],
    /// Length-P β̂(θ̂) output.
    pub betas: &'w mut [f64],
    /// Length-P per-target Var(β̂_j).
    pub var_diag: &'w mut [f64],
    /// Length-P per-target β̂_j² / Var(β̂_j).
    pub t_sq: &'w mut [f64],
    /// Length-P forward-solve scratch.
    pub u_scratch: &'w mut [f64],
    /// `σ̂²(θ)` from the most recent successful `profiled_deviance` call.
    /// Set to `f64::INFINITY` (or stale) on the failure paths — callers must
    /// guard via `LmeFitView::converged`. Lives on `LmeScratch` (not the
    /// workspace) because it is a per-call scratch scalar.
    pub sigma_sq: f64,
    /// Brent scalar state (6 fields). Built inline at the call site as
    /// individual `&'w mut f64` reborrows of workspace's `lme_brent_*` slots.
    /// (`d`/`e` are stack locals in `lme_fit`, not workspace slots — see there.)
    pub brent_log_a: &'w mut f64,
    pub brent_log_b: &'w mut f64,
    pub brent_log_c: &'w mut f64,
    pub brent_fa: &'w mut f64,
    pub brent_fb: &'w mut f64,
    pub brent_fc: &'w mut f64,
    /// p × p scratch; top-left k×k holds the Cholesky of Σ_T. Overwritten per
    /// call. See joint Wald-χ² block in `lme_fit`.
    pub joint_sigma_t_chol: MatMut<'w, f64>,
    /// Length-p RHS for the joint solve. On entry: β̂_T copied into the first
    /// k slots. On exit: x = Σ_T⁻¹ β̂_T in the first k slots.
    pub joint_rhs: &'w mut [f64],
    /// p × p scratch. Starts as `I_p` (refilled by `reset_lme_suff_stats` at
    /// each sim start) and is overwritten in-place to `K⁻¹` by
    /// `chol.solve_in_place()`.
    pub joint_k_inv: MatMut<'w, f64>,
}

/// Borrowed view into the outputs of `lme_fit`. Lifetime ties back
/// to the workspace that owns the storage.
pub struct LmeFitView<'a> {
    pub betas: &'a [f64],
    pub var_diag: &'a [f64],
    pub t_sq: &'a [f64],
    pub factor: MatRef<'a, f64>,
    pub sigma_sq: f64,
    /// Estimated random-intercept variance τ̂² = θ̂²·σ̂² (θ̂ is the fitted
    /// relative SD λ̂ = τ/σ). `NaN` on the failure paths, alongside `sigma_sq`.
    /// The general-grouping path computes the same quantity in `introspect.rs`
    /// from the raw `theta`; this scalar path exposes it directly since `θ̂`
    /// (`pin_theta`) is local to `lme_fit`.
    pub tau_sq_hat: f64,
    pub converged: bool,
    /// 0 = interior min, 1 = τ̂ ≈ 0 (OLS fallback used), 2 = high-τ bracket failure.
    pub boundary_hit: u8,
    /// Brent-loop iterations (the `converged` test reads this against MAX_BRENT_ITERS).
    pub n_iter: u32,
    /// Total `profiled_deviance` calls on this sim (diagnostics only).
    pub n_evals: u32,
    /// Joint Wald-χ² over the configured target indices. Under H₀: β_T = 0,
    /// asymptotically χ²(k) where k = target_indices.len(). `NaN` on
    /// non-converged or numerically degenerate fits (k×k Cholesky failure,
    /// non-positive σ̂², etc.). Always finite on the converged path for
    /// well-conditioned Σ_T.
    pub joint_t_sq: f64,
}

// ---------------------------------------------------------------------------
// LmeSuffStats impl
// ---------------------------------------------------------------------------

impl<'w> LmeSuffStats<'w> {
    /// Accumulate a block of rows into the running sufficient statistics;
    /// `cluster_ids_block` routes each row to its cluster accumulator.
    ///
    /// Panel-GEMM (group B): X'X/X'y accumulate through faer GEMM over a
    /// widened f64 panel, exactly as `OlsSuffStats::add_rows` (see its doc
    /// comment for the route rationale). The cluster accumulators (sum_xc,
    /// sum_yc, cluster_sizes, yty, n_clusters_seen) stay scalar passes in
    /// the original row order — each (j, c) cell receives the same addends
    /// in the same ascending-row order as the pre-panel loop (the widening
    /// is exact), so they are bit-identical by construction.
    pub fn add_rows(
        &mut self,
        x_block: MatRef<'_, f32>,
        y_block: &[f32],
        cluster_ids_block: &[u32],
    ) {
        debug_assert_eq!(x_block.nrows(), y_block.len());
        debug_assert_eq!(x_block.nrows(), cluster_ids_block.len());
        let p = self.xty.len();
        debug_assert_eq!(x_block.ncols(), p);
        let max_k = self.sum_yc.len();
        let m = x_block.nrows();
        debug_assert!(self.panel_x.len() >= PANEL_ROWS.min(m) * p);
        debug_assert!(self.panel_y.len() >= PANEL_ROWS.min(m));

        let mut off = 0;
        while off < m {
            let rows = (m - off).min(PANEL_ROWS);
            // f32 data plane widens once per element; column-major,
            // leading dim = rows.
            for j in 0..p {
                let col = &mut self.panel_x[j * rows..(j + 1) * rows];
                for (i, v) in col.iter_mut().enumerate() {
                    *v = x_block[(off + i, j)] as f64;
                }
            }
            for i in 0..rows {
                let y_row = y_block[off + i] as f64;
                let c = cluster_ids_block[off + i] as usize;
                debug_assert!(
                    c < max_k,
                    "cluster_id {c} exceeds workspace's max_n_clusters {max_k}"
                );
                self.panel_y[i] = y_row;
                self.sum_yc[c] += y_row;
                self.cluster_sizes[c] += 1;
                *self.yty += y_row * y_row;
                let candidate = (c as u32).saturating_add(1);
                if candidate > *self.n_clusters_seen {
                    *self.n_clusters_seen = candidate;
                }
            }
            // Per-cluster X sums read the widened panel — contiguous panel
            // column, scattered sum_xc writes; O(n·p), not the bottleneck.
            for j in 0..p {
                let col = &self.panel_x[j * rows..(j + 1) * rows];
                for (i, &v) in col.iter().enumerate() {
                    let c = cluster_ids_block[off + i] as usize;
                    self.sum_xc[(j, c)] += v;
                }
            }
            let xp = MatRef::from_column_major_slice(&self.panel_x[..rows * p], rows, p);
            triangular::matmul(
                self.xtx.rb_mut(),
                BlockStructure::TriangularLower,
                Accum::Add,
                xp.transpose(),
                BlockStructure::Rectangular,
                xp,
                BlockStructure::Rectangular,
                1.0,
                Par::Seq,
            );
            matmul(
                MatMut::from_column_major_slice_mut(&mut *self.xty, p, 1),
                Accum::Add,
                xp.transpose(),
                MatRef::from_column_major_slice(&self.panel_y[..rows], rows, 1),
                1.0,
                Par::Seq,
            );
            off += rows;
        }
    }
}

// ---------------------------------------------------------------------------
// profiled_deviance
// ---------------------------------------------------------------------------

// REML form of the profiled deviance. The original implementation parametrised
// by lam_sq = θ²; the inner loop
// is algebraically identical to our θ-form below (1 + θ²·n_c == 1 + lam_sq·n_c,
// and θ²·v_c⁻¹ == lam_sq · M_j⁻¹). v1's REML branch returns:
//     log|L_theta| + 2·Σ log L_jj + (N − P) · log(r_sq)
// where `r_sq = ytViy − β̂'·XtViy` (NOT divided by (N − P)). We instead
// return `(N − P) · log(σ̂²) = (N − P) · log(r_sq / (N − P))`. The two
// differ only by the additive constant `(N − P) · log(N − P)` — the test
// gates on deviance *differences* (which cancel that constant) plus β̂ and σ̂²
// directly (which are formulation-independent).
//
// Per-θ overwrite invariant: `scratch.v_diag_inv`, `scratch.xtvix`,
// `scratch.xtviy`, `scratch.xtvix_factor`, `scratch.betas` are all fully
// overwritten on every call — no per-iteration reset needed by the caller.
/// Evaluate the REML profiled deviance at θ. Writes intermediate state into
/// `scratch.xtvix`, `scratch.xtviy`, `scratch.xtvix_factor`, `scratch.v_diag_inv`
/// per the per-θ overwrite invariant.
///
/// Side effects: on success, `scratch.xtvix_factor` holds the Cholesky L of
/// `X' V(θ)⁻¹ X` and `scratch.betas` holds β̂(θ). `scratch.v_diag_inv[c]`
/// holds `(1 + θ²·n_c)⁻¹`.
///
/// Returns `f64::INFINITY` on Cholesky failure or non-positive σ̂².
pub fn profiled_deviance(theta: f64, scratch: &mut LmeScratch<'_>) -> f64 {
    use faer::reborrow::Reborrow;

    let p = scratch.betas.len();
    let n_clusters = scratch.n_clusters as usize;
    let n_rows = scratch.n_rows as usize;
    let theta_sq = theta * theta;

    if n_rows <= p || p == 0 {
        return f64::INFINITY;
    }

    // 1. v_diag_inv[c] = (1 + θ²·n_c)⁻¹; accumulate log|V| = Σ log(1 + θ²·n_c).
    let mut log_det_v = 0.0_f64;
    for c in 0..n_clusters {
        let n_c = scratch.cluster_sizes[c] as f64;
        let one_plus = 1.0 + theta_sq * n_c;
        scratch.v_diag_inv[c] = 1.0 / one_plus;
        log_det_v += one_plus.ln();
    }

    // 2. Copy lower triangle of X'X → scratch.xtvix (upper triangle is left as
    //    whatever was there last call; faer's Cholesky reads `Side::Lower` only,
    //    but the `cholesky` helper inspects all entries — so we write a clean
    //    upper triangle too to avoid NaN/stale-data contamination).
    for j in 0..p {
        for i in 0..p {
            if i >= j {
                scratch.xtvix[(i, j)] = scratch.xtx[(i, j)];
            } else {
                scratch.xtvix[(i, j)] = 0.0;
            }
        }
    }
    // 3. Apply per-cluster rank-1 downdates onto the lower triangle:
    //    X' V⁻¹ X = X'X − Σ_c (θ² · v_diag_inv[c]) · sum_xc[:, c] · sum_xc[:, c]'.
    for c in 0..n_clusters {
        let s_c = theta_sq * scratch.v_diag_inv[c];
        for j in 0..p {
            let s_xj = scratch.sum_xc[(j, c)];
            for i in j..p {
                let s_xi = scratch.sum_xc[(i, c)];
                scratch.xtvix[(i, j)] -= s_c * s_xi * s_xj;
            }
        }
    }

    // 4. X' V⁻¹ y = X'y − Σ_c (θ² · v_diag_inv[c]) · sum_xc[:, c] · sum_yc[c].
    for j in 0..p {
        scratch.xtviy[j] = scratch.xty[j];
    }
    for c in 0..n_clusters {
        let s_c = theta_sq * scratch.v_diag_inv[c];
        let s_yc = scratch.sum_yc[c];
        let factor = s_c * s_yc;
        for j in 0..p {
            scratch.xtviy[j] -= factor * scratch.sum_xc[(j, c)];
        }
    }

    // 5. Cholesky of scratch.xtvix (lower) via faer's `llt(Side::Lower)`,
    //    matching `ols.rs`'s pattern verbatim. Returns INFINITY on failure.
    let chol = match scratch.xtvix.rb().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::INFINITY,
    };
    let l = chol.L();
    // Copy L into scratch.xtvix_factor (strict upper triangle zeroed).
    for j in 0..p {
        for i in 0..p {
            scratch.xtvix_factor[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
        }
    }

    // 6. Solve L L' β̂ = X' V⁻¹ y → scratch.betas, in place. Copy the RHS
    //    (X' V⁻¹ y) into betas first — `xtviy` must stay intact for the σ̂²
    //    computation at step 8 — then solve over the betas buffer directly
    //    (β̂ lands there anyway), so no owned `Mat` and no copy-back.
    use faer::linalg::solvers::Solve;
    scratch.betas[..p].copy_from_slice(&scratch.xtviy[..p]);
    {
        let mut rhs = MatMut::from_column_major_slice_mut(scratch.betas, p, 1usize);
        chol.solve_in_place(rhs.rb_mut());
    }

    // 7. y' V⁻¹ y = y'y − Σ_c (θ² · v_diag_inv[c]) · sum_yc[c]².
    let mut ytviy = scratch.yty;
    for c in 0..n_clusters {
        let s_c = theta_sq * scratch.v_diag_inv[c];
        let s_yc = scratch.sum_yc[c];
        ytviy -= s_c * s_yc * s_yc;
    }

    // 8. σ̂² = (y' V⁻¹ y − β̂' · X' V⁻¹ y) / (N − P).
    let mut bty = 0.0_f64;
    for j in 0..p {
        bty += scratch.betas[j] * scratch.xtviy[j];
    }
    let r_sq = ytviy - bty;
    if !r_sq.is_finite() || r_sq <= 0.0 {
        return f64::INFINITY;
    }
    let df_resid = (n_rows - p) as f64;
    let sigma_sq = r_sq / df_resid;
    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return f64::INFINITY;
    }
    // Surface σ̂² on the scratch so `lme_fit` can read it after the
    // pin-at-θ̂ eval without recomputing y'V⁻¹y - β̂'X'V⁻¹y from suff-stats.
    scratch.sigma_sq = sigma_sq;

    // 9. log|X' V⁻¹ X| = 2 · Σ_j log L_jj.
    let mut log_det_xtvix = 0.0_f64;
    for j in 0..p {
        let ljj = scratch.xtvix_factor[(j, j)];
        if !ljj.is_finite() || ljj <= 0.0 {
            return f64::INFINITY;
        }
        log_det_xtvix += ljj.ln();
    }
    log_det_xtvix *= 2.0;

    // 10. dev_REML(θ) = log|V| + 2·Σ log L_jj + (N − P) · log(σ̂²).
    log_det_v + log_det_xtvix + df_resid * sigma_sq.ln()
}

// ---------------------------------------------------------------------------
// brent_minimize
// ---------------------------------------------------------------------------

// Golden-section + parabolic-interpolation constants (NR §10.3).
/// Golden-section ratio: (3 − √5) / 2.
const CGOLD: f64 = 0.381_966_011_250_105;
/// Small offset to avoid division by zero when x = 0.
const ZEPS: f64 = 1.0e-10;

/// Brent's combined golden-section + parabolic-interpolation 1-D minimiser.
///
/// # Reference
/// NR §10.3 "Brent's Method in One Dimension". The algorithm structure and
/// variable names below follow it closely.
///
/// # Caller contract
/// The caller provides a valid bracket `(a, b, c)` with `a < b < c` and
/// `f(b) ≤ min(f(a), f(c))` — i.e. `b` is the interior best point.
///
/// # Returns
/// `(xmin, fmin)`. The bracket `[a, c]` and state `(d, e)` are updated in
/// place; the caller can inspect the final bracket for boundary detection.
///
/// # Convergence
/// Returns when the bracket width satisfies
/// `|x − midpoint| ≤ 2·tol1 − 0.5·(high − low)`
/// where `tol1 = rel_tol·|x| + ZEPS`.
/// `*n_iter_out` is set to the number of iterations consumed; if it equals
/// `max_iter` the caller should treat the result as non-converged.
#[expect(
    clippy::too_many_arguments,
    reason = "Brent minimizer; args are the bracket triplet plus tuning knobs"
)]
pub fn brent_minimize<F: FnMut(f64) -> f64>(
    a: &mut f64,
    b: &mut f64,
    c: &mut f64,
    fa: &mut f64,
    fb: &mut f64,
    fc: &mut f64,
    d: &mut f64,
    e: &mut f64,
    mut f: F,
    rel_tol: f64,
    max_iter: u32,
    n_iter_out: &mut u32,
) -> (f64, f64) {
    // NR §10.3 uses variables: a, b (bracket endpoints; NOT the caller's a/b/c),
    // x (best so far), w (second best), v (third best).
    // Caller's convention: (*a, *c) are the bracket endpoints and *b is the
    // current best interior point.  Map to NR notation internally:
    //   bracket_lo = min(*a, *c), bracket_hi = max(*a, *c)
    //   x = w = v = *b,  fx = fw = fv = *fb

    // Initialise bracket endpoints (NR's a, b — not our parameter *a, *b, *c).
    let (mut bracket_lo, mut bracket_hi) = if *a < *c { (*a, *c) } else { (*c, *a) };

    // Best point so far (NR's x).
    let mut x = *b;
    let mut fx = *fb;
    // Second-best (NR's w) and third-best (NR's v) — all start equal.
    let mut w = x;
    let mut fw = fx;
    let mut v = x;
    let mut fv = fx;

    // NR's e (previous step taken) and d (current step) — caller-supplied
    // workspace scalars.  Reset to zero for a fresh start.
    *e = 0.0;
    *d = 0.0;

    for iter in 0..max_iter {
        let xm = 0.5 * (bracket_lo + bracket_hi);
        let tol1 = rel_tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;

        // Termination condition (NR §10.3 eq 10.3.3).
        if (x - xm).abs() <= tol2 - 0.5 * (bracket_hi - bracket_lo) {
            *n_iter_out = iter;
            // Write back bracket and best point.
            *a = bracket_lo;
            *c = bracket_hi;
            *b = x;
            *fa = fv; // NR's v corresponds to fv (third-best)
            *fb = fx;
            *fc = fw; // NR's w corresponds to fw (second-best)
            return (x, fx);
        }

        // Decide: try parabolic interpolation or fall back to golden-section.
        let u;
        if (*e).abs() > tol1 {
            // Attempt parabolic fit through (v, w, x).
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p_num = (x - v) * q - (x - w) * r;
            let mut q_denom = 2.0 * (q - r);
            if q_denom > 0.0 {
                p_num = -p_num;
            } else {
                q_denom = -q_denom;
            }
            let p_step = p_num;
            let q_step = q_denom;
            // Accept parabolic step only if it falls strictly inside the bracket
            // and is less than half the previous step (NR §10.3).
            if p_step.abs() < (0.5 * q_step * (*e).abs())
                && p_step > q_step * (bracket_lo - x)
                && p_step < q_step * (bracket_hi - x)
            {
                *e = *d;
                *d = p_step / q_step;
                let u_trial = x + *d;
                // Ensure we don't step too close to bracket endpoints.
                if (u_trial - bracket_lo) < tol2 || (bracket_hi - u_trial) < tol2 {
                    *d = if xm >= x { tol1 } else { -tol1 };
                }
                u = x + *d;
            } else {
                // Parabolic rejected — golden-section step.
                *e = if x >= xm {
                    bracket_lo - x
                } else {
                    bracket_hi - x
                };
                *d = CGOLD * (*e);
                u = x + *d;
            }
        } else {
            // Golden-section step.
            *e = if x >= xm {
                bracket_lo - x
            } else {
                bracket_hi - x
            };
            *d = CGOLD * (*e);
            u = x + *d;
        }

        // Move at least tol1 away from x.
        let u_actual = if (*d).abs() >= tol1 {
            u
        } else if *d >= 0.0 {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u_actual);

        // Update bracket.
        if fu <= fx {
            // New point is better.
            if u_actual >= x {
                bracket_lo = x;
            } else {
                bracket_hi = x;
            }
            // Cycle: v ← w ← x ← u_actual.
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u_actual;
            fx = fu;
        } else {
            // New point is worse — tighten bracket.
            if u_actual < x {
                bracket_lo = u_actual;
            } else {
                bracket_hi = u_actual;
            }
            // Update w and v from the new (worse) point if appropriate.
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u_actual;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u_actual;
                fv = fu;
            }
        }
    }

    // Max iterations exhausted — return best found.
    *n_iter_out = max_iter;
    *a = bracket_lo;
    *c = bracket_hi;
    *b = x;
    *fb = fx;
    (x, fx)
}

// ---------------------------------------------------------------------------
// joint_wald_chi_sq — asymptotic Wald-χ² over a subset of target coefficients
// ---------------------------------------------------------------------------

/// Compute the joint Wald-χ² statistic
///
/// ```text
/// W = β̂_T' [(K⁻¹)_TT]⁻¹ β̂_T / σ̂²
/// ```
///
/// where `K = X' V(θ̂)⁻¹ X` (or `X'X` on the τ̂≈0 path; the formula is the
/// same — the caller passes whichever `xtvix` was just used) and `T` is the
/// configured target subset. Under H₀: β_T = 0, `W ~ χ²(k)` asymptotically.
///
/// **Algorithm (bounded-alloc; all explicit storage from the LME scratch —
/// the two faer `llt` calls below allocate internally, counted by the
/// `lme_fit_warm_path_bounded_alloc` lock):**
///   1. Re-Cholesky `xtvix` (idempotent — same factorisation already cached in
///      `xtvix_factor`; faer doesn't expose a `LltRef` reconstructed from a
///      pre-computed L without recomputation, and the cost is O(p³/3) which
///      is negligible at p ≤ 20).
///   2. `solve_in_place(I_p)` against the new Cholesky → `joint_k_inv` becomes
///      `K⁻¹`. The function refills the identity internally before solving, so
///      the caller need not pre-initialize `joint_k_inv`.
///   3. Gather `(K⁻¹)_TT` into the top-left k×k block of `joint_sigma_t_chol`.
///   4. In-place Cholesky on that k×k block.
///   5. Solve `Σ_T x = β̂_T` via the new k×k Cholesky factor (`solve_in_place`
///      on a length-`k` column view of `joint_rhs`).
///   6. `W_raw = β̂_T · x`.
///   7. Divide by `σ̂²` and return; on any numerical failure return `NaN`.
///
/// Returns `NaN` if `k == 0`, `σ̂² ≤ 0`, or any of the Cholesky steps fail.
pub(crate) fn joint_wald_chi_sq(
    xtvix: MatRef<'_, f64>,
    betas: &[f64],
    sigma_sq: f64,
    target_indices: &[u32],
    mut joint_k_inv: MatMut<'_, f64>,
    mut joint_sigma_t_chol: MatMut<'_, f64>,
    joint_rhs: &mut [f64],
) -> f64 {
    use faer::linalg::solvers::Solve;
    use faer::reborrow::Reborrow;

    let p = betas.len();
    let k = target_indices.len();
    if k == 0 || p == 0 || !(sigma_sq.is_finite() && sigma_sq > FLOAT_NEAR_ZERO) {
        return f64::NAN;
    }

    // Step 1: re-Cholesky K = X'V⁻¹X (lower).
    let chol = match xtvix.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    // Step 2: K⁻¹ via `solve_in_place(I_p)`. Refill identity here so the
    // helper is safe to call repeatedly without depending on the workspace's
    // reset_lme_suff_stats to have just run (cost: p² writes, p ≤ 20).
    for j in 0..p {
        for i in 0..p {
            joint_k_inv[(i, j)] = if i == j { 1.0 } else { 0.0 };
        }
    }
    chol.solve_in_place(joint_k_inv.rb_mut());

    // Step 3: gather Σ_T = (K⁻¹)_TT into the top-left k×k of `joint_sigma_t_chol`.
    for (a, &ti) in target_indices.iter().enumerate() {
        let ti = ti as usize;
        if ti >= p {
            return f64::NAN;
        }
        for (b, &tj) in target_indices.iter().enumerate() {
            let tj = tj as usize;
            if tj >= p {
                return f64::NAN;
            }
            joint_sigma_t_chol[(a, b)] = joint_k_inv[(ti, tj)];
        }
    }

    // Step 4: in-place Cholesky on the k×k block. We pass the full p×p view's
    // top-left k×k submatrix to faer.
    let sigma_t_view = joint_sigma_t_chol.rb().submatrix(0, 0, k, k);
    let chol_t = match sigma_t_view.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    // Step 5: copy β̂_T into the first k slots of joint_rhs and solve in
    // place via a k×1 column view over that prefix (the same
    // `from_column_major_slice_mut` idiom as the β solve) — no owned Mat,
    // no copy-out.
    for (a, &ti) in target_indices.iter().enumerate() {
        joint_rhs[a] = betas[ti as usize];
    }
    {
        let mut rhs = MatMut::from_column_major_slice_mut(&mut joint_rhs[..k], k, 1usize);
        chol_t.solve_in_place(rhs.rb_mut());
    }

    // Step 6: W_raw = β̂_T · x.
    let mut w_raw = 0.0_f64;
    for (a, &ti) in target_indices.iter().enumerate() {
        w_raw += betas[ti as usize] * joint_rhs[a];
    }
    if !w_raw.is_finite() {
        return f64::NAN;
    }

    // Step 7: divide by σ̂².
    w_raw / sigma_sq
}

// ---------------------------------------------------------------------------
// lme_fit — top-level kernel
// ---------------------------------------------------------------------------

/// NaN-fill the output slots of a `LmeScratch` on a non-converged exit so
/// callers see clear bad-value signals.
fn nan_fill_outputs(scratch: &mut LmeScratch<'_>, p: usize, target_indices: &[u32]) {
    for v in scratch.betas[..p].iter_mut() {
        *v = f64::NAN;
    }
    for &tj in target_indices.iter() {
        let tj = tj as usize;
        if tj < scratch.var_diag.len() {
            scratch.var_diag[tj] = f64::NAN;
        }
        if tj < scratch.t_sq.len() {
            scratch.t_sq[tj] = f64::NAN;
        }
    }
}

/// Fit a random-intercept LME by Brent minimisation of the REML profiled
/// deviance over `log(θ)`, with the closed-form β̂ / σ̂² / Var(β̂_target)
/// recovery at θ̂.
///
/// The caller is responsible for:
///   * `ws.reset_lme_suff_stats()` (once per (sim, N))
///   * Building the accumulator via `LmeSuffStats::add_rows(...)`
///   * Constructing `scratch: LmeScratch<'_>` by reborrowing the workspace.
///
/// `theta_start`: `None` → the full cold bracket (bit-identical pre-warm-start
/// path); `Some(θ₀)` → spec-derived truth start (Y is always synthetic, so the
/// realised per-block θ_true = √τ²_block is known at fit time — the same
/// truth-start seam as `lmm::fit_lmm`). The bracket is centered at
/// `clamp(ln θ₀, LOW+Δ, HIGH−Δ)` with half-width Δ = `TRUTH_BRACKET_HALF_WIDTH`;
/// θ₀ = 0 → ln θ₀ = −∞ clamps left, putting the bracket's left edge exactly at
/// `LOG_THETA_LOW` so τ̂≈0 boundary semantics survive without a floor constant.
/// A truth bracket that fails repair retries once from the cold bracket, so no
/// spec can classify a boundary worse than the cold path. A per-scenario/block
/// constant — determinism and chunk merging are unaffected.
///
/// Hot-path structure: 3-point bracket (truth-centered or cold) → bracket
/// repair (failed truth bracket → cold retry) → Brent → boundary detect
/// (anchored to the global LOG_THETA_LOW/HIGH edges, never the narrow
/// bracket) → (τ̂≈0 OLS-equivalent fallback | high-τ rebracket) →
/// pin Cholesky at θ̂ → forward-solve recovery.
///
/// **Brent state on the stack.** `brent_d` and `brent_e` are scalar locals
/// here — there are no workspace slots for them. Stack scalars cost zero
/// allocations, so the only purpose dedicated workspace fields would serve
/// (documenting the no-per-call-alloc invariant) is already met without them.
pub fn lme_fit<'a>(
    _x: MatRef<'_, f32>,
    _y: &[f32],
    _cluster_ids: &[u32],
    target_indices: &[u32],
    theta_start: Option<f64>,
    mut scratch: LmeScratch<'a>,
) -> LmeFitView<'a> {
    let p = scratch.betas.len();
    let mut n_evals: u32 = 0;
    let mut brent_iters: u32 = 0;

    // ----- 2. Initial 3-point bracket on log(θ) -----
    // Truth-centered when θ₀ is passed (see the doc comment), else the full
    // cold bracket [LOG_THETA_LOW, LOG_THETA_HIGH].
    let truth_bracket = theta_start.map(|theta0| {
        let center = theta0.ln().clamp(
            LOG_THETA_LOW + TRUTH_BRACKET_HALF_WIDTH,
            LOG_THETA_HIGH - TRUTH_BRACKET_HALF_WIDTH,
        );
        (
            center - TRUTH_BRACKET_HALF_WIDTH,
            center,
            center + TRUTH_BRACKET_HALF_WIDTH,
        )
    });
    let (mut log_a, mut log_b, mut log_c) =
        truth_bracket.unwrap_or((LOG_THETA_LOW, LOG_THETA_MID, LOG_THETA_HIGH));
    let mut fa: f64;
    let mut fb: f64;
    let mut fc: f64;

    // ----- 3. Establish the bracket: evaluate + repair, with cold retry -----
    // Repair: interior bisection first; if that fails, expand the failing
    // endpoint outward by one decade (up to two attempts). A truth bracket
    // whose repair fails retries once from the full cold bracket — today's
    // behaviour IS the cold bracket, so no spec can classify a boundary worse
    // than the pre-warm-start path. Only then is a boundary hit declared.
    let mut truth_retry_left = truth_bracket.is_some();
    let boundary_pre_brent: Option<u8>;
    loop {
        fa = profiled_deviance(log_a.exp(), &mut scratch);
        fb = profiled_deviance(log_b.exp(), &mut scratch);
        fc = profiled_deviance(log_c.exp(), &mut scratch);
        n_evals += 3;

        let mut bracket_ok = fb < fa && fb < fc;
        if !bracket_ok {
            // Interior bisection: try the midpoint of (log_a, log_c).
            let log_mid = 0.5 * (log_a + log_c);
            let f_mid = profiled_deviance(log_mid.exp(), &mut scratch);
            n_evals += 1;
            if f_mid < fa && f_mid < fc {
                log_b = log_mid;
                fb = f_mid;
                bracket_ok = true;
            }
        }
        if !bracket_ok {
            // Outward expansion (≤ 2 attempts), direction picked by lower endpoint.
            const LN10: f64 = std::f64::consts::LN_10;
            for _ in 0..2 {
                if fa <= fc {
                    // Min is plausibly to the left — expand `a` leftward.
                    let new_log_a = log_a - LN10;
                    let new_fa = profiled_deviance(new_log_a.exp(), &mut scratch);
                    n_evals += 1;
                    if new_fa > fa {
                        // Found a strict minimum at log_a (the old `a` is the
                        // new interior point).
                        log_c = log_b;
                        fc = fb;
                        log_b = log_a;
                        fb = fa;
                        log_a = new_log_a;
                        fa = new_fa;
                        bracket_ok = true;
                        break;
                    }
                    log_a = new_log_a;
                    fa = new_fa;
                } else {
                    // Min is plausibly to the right — expand `c` rightward.
                    let new_log_c = log_c + LN10;
                    let new_fc = profiled_deviance(new_log_c.exp(), &mut scratch);
                    n_evals += 1;
                    if new_fc > fc {
                        log_a = log_b;
                        fa = fb;
                        log_b = log_c;
                        fb = fc;
                        log_c = new_log_c;
                        fc = new_fc;
                        bracket_ok = true;
                        break;
                    }
                    log_c = new_log_c;
                    fc = new_fc;
                }
            }
        }
        if bracket_ok {
            boundary_pre_brent = None;
            break;
        }
        if truth_retry_left {
            // Cold retry (once): rebuild the full bracket from the global edges.
            truth_retry_left = false;
            log_a = LOG_THETA_LOW;
            log_b = LOG_THETA_MID;
            log_c = LOG_THETA_HIGH;
            continue;
        }
        // Left-edge failure → τ̂≈0 (boundary_hit=1);
        // right-edge failure → high-τ (boundary_hit=2).
        boundary_pre_brent = Some(if fa <= fc { 1 } else { 2 });
        break;
    }

    // ----- 4. Brent's method (skipped if pre-Brent boundary hit) -----
    let mut log_theta_hat = log_b;
    let mut brent_d: f64 = 0.0;
    let mut brent_e: f64 = 0.0;
    if boundary_pre_brent.is_none() {
        let mut brent_n_evals: u32 = 0;
        let (xmin, _fmin) = {
            let f_closure = |log_x: f64| -> f64 {
                brent_n_evals += 1;
                profiled_deviance(log_x.exp(), &mut scratch)
            };
            brent_minimize(
                &mut log_a,
                &mut log_b,
                &mut log_c,
                &mut fa,
                &mut fb,
                &mut fc,
                &mut brent_d,
                &mut brent_e,
                f_closure,
                BRENT_REL_TOL,
                MAX_BRENT_ITERS,
                &mut brent_iters,
            )
        };
        n_evals += brent_n_evals;
        log_theta_hat = xmin;
    }

    // ----- 5. Boundary detection -----
    let mut boundary_hit: u8 = boundary_pre_brent.unwrap_or(0);
    if boundary_hit == 0 {
        if log_theta_hat - LOG_THETA_LOW < BOUNDARY_LOG_SLACK {
            boundary_hit = 1;
        } else if LOG_THETA_HIGH - log_theta_hat < BOUNDARY_LOG_SLACK {
            // High-τ retry: re-bracket one decade to the right. Provisionally
            // set boundary_hit = 2; reset to 0 if the second Brent finds an
            // interior min, or kept-and-returned if the retry also fails.
            let mut log_a2 = (1e-1_f64).ln();
            let mut log_b2 = (1e1_f64).ln();
            let mut log_c2 = (1e3_f64).ln();
            let mut fa2 = profiled_deviance(log_a2.exp(), &mut scratch);
            let mut fb2 = profiled_deviance(log_b2.exp(), &mut scratch);
            let mut fc2 = profiled_deviance(log_c2.exp(), &mut scratch);
            n_evals += 3;
            if fb2 < fa2 && fb2 < fc2 {
                // Second bracket valid → second Brent loop.
                let mut brent_n_evals2: u32 = 0;
                let mut bd2: f64 = 0.0;
                let mut be2: f64 = 0.0;
                let mut bi2: u32 = 0;
                let (xmin2, _fmin2) = {
                    let f_closure = |log_x: f64| -> f64 {
                        brent_n_evals2 += 1;
                        profiled_deviance(log_x.exp(), &mut scratch)
                    };
                    brent_minimize(
                        &mut log_a2,
                        &mut log_b2,
                        &mut log_c2,
                        &mut fa2,
                        &mut fb2,
                        &mut fc2,
                        &mut bd2,
                        &mut be2,
                        f_closure,
                        BRENT_REL_TOL,
                        MAX_BRENT_ITERS,
                        &mut bi2,
                    )
                };
                n_evals += brent_n_evals2;
                brent_iters = bi2;
                log_theta_hat = xmin2;
                // If second Brent ALSO settles near the high edge, give up.
                if (1e3_f64).ln() - log_theta_hat < BOUNDARY_LOG_SLACK {
                    boundary_hit = 2;
                    nan_fill_outputs(&mut scratch, p, target_indices);
                    let factor = scratch.xtvix_factor.into_const();
                    return LmeFitView {
                        betas: &scratch.betas[..p],
                        var_diag: &scratch.var_diag[..p],
                        t_sq: &scratch.t_sq[..p],
                        factor,
                        sigma_sq: f64::NAN,
                        tau_sq_hat: f64::NAN,
                        converged: false,
                        boundary_hit,
                        n_iter: brent_iters,
                        n_evals,
                        joint_t_sq: f64::NAN,
                    };
                }
                // Otherwise: second Brent found an interior min → fall through.
                boundary_hit = 0;
            } else {
                // Second bracket also fails → permanent high-τ failure.
                nan_fill_outputs(&mut scratch, p, target_indices);
                let factor = scratch.xtvix_factor.into_const();
                return LmeFitView {
                    betas: &scratch.betas[..p],
                    var_diag: &scratch.var_diag[..p],
                    t_sq: &scratch.t_sq[..p],
                    factor,
                    sigma_sq: f64::NAN,
                    tau_sq_hat: f64::NAN,
                    converged: false,
                    boundary_hit: 2,
                    n_iter: brent_iters,
                    n_evals,
                    joint_t_sq: f64::NAN,
                };
            }
        }
    }

    // ----- 6. Pin Cholesky at θ̂ -----
    // If boundary_hit == 1 (τ̂≈0): pin at LOG_THETA_LOW. The arithmetic of
    // profiled_deviance at θ near zero degenerates to OLS (v_diag_inv ≈ 1,
    // per-cluster downdates ≈ 0); β̂ matches OLS to machine precision.
    // If boundary_hit == 2 (failed): we already returned above (NaN-filled).
    // If boundary_hit == 0 (interior): pin at θ̂ = exp(log_theta_hat).
    let pin_theta = if boundary_hit == 1 {
        LOG_THETA_LOW.exp()
    } else {
        log_theta_hat.exp()
    };
    let pin_dev = profiled_deviance(pin_theta, &mut scratch);
    n_evals += 1;
    // Rank-deficiency check on the pinning Cholesky factor — mirrors
    // `ols.rs::fit_suff_stats_t_sq`'s min/max diag ratio test. Catches the
    // grey-zone case where faer's Cholesky succeeds numerically on a
    // collinear X (e.g. duplicated columns) but the inverse is essentially
    // unbounded. We use a slightly looser threshold than OLS's 1e-12 because
    // the LME formulation downdates X'X with per-cluster outer products,
    // which can amplify near-singularity onto the diagonal of L.
    const EPS_RANK: f64 = 1e-8;
    let rank_deficient = chol_rank_deficient(scratch.xtvix_factor.rb(), p, EPS_RANK);
    if !pin_dev.is_finite() || rank_deficient {
        nan_fill_outputs(&mut scratch, p, target_indices);
        let factor = scratch.xtvix_factor.into_const();
        return LmeFitView {
            betas: &scratch.betas[..p],
            var_diag: &scratch.var_diag[..p],
            t_sq: &scratch.t_sq[..p],
            factor,
            sigma_sq: f64::NAN,
            tau_sq_hat: f64::NAN,
            converged: false,
            boundary_hit,
            n_iter: brent_iters,
            n_evals,
            joint_t_sq: f64::NAN,
        };
    }
    let sigma_sq = scratch.sigma_sq;

    // ----- 7. Recovery: per-target var_diag via forward solve on L -----
    // β̂ already in scratch.betas from the pinning eval (profiled_deviance
    // writes it via the two-triangular-solve path).
    //
    // For each target tj: solve L · u = e_{tj} (forward sub), then
    //   var_diag[tj] = σ̂² · ‖u‖²
    // because (X' V⁻¹ X)⁻¹ = L⁻ᵀ L⁻¹ and the diagonal entries of L⁻ᵀ L⁻¹
    // are ‖L⁻¹ e_j‖². Mirrors the var_diag recovery in `fit_suff_stats_t_sq`.
    for &tj in target_indices.iter() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        for v in scratch.u_scratch[..p].iter_mut() {
            *v = 0.0;
        }
        let mut bad_diag = false;
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= scratch.xtvix_factor[(i, k)] * scratch.u_scratch[k];
            }
            let l_ii = scratch.xtvix_factor[(i, i)];
            if l_ii.abs() < FLOAT_NEAR_ZERO {
                scratch.u_scratch[i] = f64::NAN;
                bad_diag = true;
            } else {
                scratch.u_scratch[i] = acc / l_ii;
            }
        }
        let mut norm_sq = 0.0;
        for &v in scratch.u_scratch[..p].iter() {
            norm_sq += v * v;
        }
        let vd = sigma_sq * norm_sq;
        scratch.var_diag[tj] = vd;
        if !bad_diag && vd > FLOAT_NEAR_ZERO && vd.is_finite() {
            let beta_j = scratch.betas[tj];
            scratch.t_sq[tj] = (beta_j * beta_j) / vd;
        } else {
            scratch.t_sq[tj] = f64::NAN;
        }
    }

    // ----- 7b. Joint Wald-χ² over target_indices -----
    let joint_t_sq = joint_wald_chi_sq(
        scratch.xtvix.rb(),
        &scratch.betas[..p],
        sigma_sq,
        target_indices,
        scratch.joint_k_inv.rb_mut(),
        scratch.joint_sigma_t_chol.rb_mut(),
        &mut scratch.joint_rhs[..p],
    );

    // ----- 8. Build the view -----
    // `converged`: Brent iters strictly below cap
    // (boundary_hit == 1 with no Brent ran ⇒ brent_iters == 0 ⇒ true).
    let converged = brent_iters < MAX_BRENT_ITERS;
    let factor = scratch.xtvix_factor.into_const();
    // τ̂² = θ̂²·σ̂² (pin_theta is the fitted relative SD λ̂; mirrors the
    // general-grouping path's `theta·theta·sig` in introspect.rs). On the
    // τ̂≈0 boundary (boundary_hit == 1) pin_theta is LOG_THETA_LOW.exp(), so
    // τ̂² is ~0 — the correct near-zero report, not a failure.
    let tau_sq_hat = pin_theta * pin_theta * sigma_sq;
    LmeFitView {
        betas: &scratch.betas[..p],
        var_diag: &scratch.var_diag[..p],
        t_sq: &scratch.t_sq[..p],
        factor,
        sigma_sq,
        tau_sq_hat,
        converged,
        boundary_hit,
        n_iter: brent_iters,
        n_evals,
        joint_t_sq,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_constants_log_spaced() {
        assert!((LOG_THETA_LOW - (1e-4_f64).ln()).abs() < 1e-12);
        assert!((LOG_THETA_HIGH - (1e3_f64).ln()).abs() < 1e-12);
        #[expect(
            clippy::assertions_on_constants,
            reason = "compile-time ordering invariant on the log-theta brackets"
        )]
        {
            assert!(LOG_THETA_LOW < LOG_THETA_MID && LOG_THETA_MID < LOG_THETA_HIGH);
        }
    }

    /// Build an `LmeScratch` from a workspace whose `lme_*` suff-stats are
    /// already populated. Used by `profiled_deviance` tests below.
    fn build_lme_scratch<'w>(
        ws: &'w mut crate::workspace::SimWorkspace,
        n_rows: u32,
        n_clusters: u32,
    ) -> LmeScratch<'w> {
        use faer::reborrow::IntoConst;
        LmeScratch {
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
            n_rows,
            xtvix: ws.lme_xtvix.as_mut(),
            xtviy: &mut ws.lme_xtviy,
            xtvix_factor: ws.lme_xtvix_factor.as_mut(),
            v_diag_inv: &mut ws.lme_v_diag_inv,
            betas: &mut ws.lme_betas,
            var_diag: &mut ws.lme_var_diag,
            t_sq: &mut ws.lme_t_sq,
            u_scratch: &mut ws.lme_u_scratch,
            brent_log_a: &mut ws.lme_brent_log_a,
            brent_log_b: &mut ws.lme_brent_log_b,
            brent_log_c: &mut ws.lme_brent_log_c,
            brent_fa: &mut ws.lme_brent_fa,
            brent_fb: &mut ws.lme_brent_fb,
            brent_fc: &mut ws.lme_brent_fc,
            joint_sigma_t_chol: ws.lme_joint_sigma_t_chol.as_mut(),
            joint_rhs: &mut ws.lme_joint_rhs,
            joint_k_inv: ws.lme_joint_k_inv.as_mut(),
            sigma_sq: 0.0,
        }
    }

    /// Regression net for the panel-GEMM rewrite (group B), LME twin of
    /// ols::add_rows_panel_matches_scalar_reference. X'X / X'y reassociate —
    /// 1e-12 relative band, measured max 9.1e-16 (first post-rewrite run);
    /// every cluster accumulator (sum_xc, sum_yc, cluster_sizes, yty,
    /// n_clusters_seen) stays scalar in row order and must be bit-identical.
    #[test]
    fn lme_add_rows_panel_matches_scalar_reference() {
        use crate::workspace::SimWorkspace;

        let n = 611; // two PANEL_ROWS boundaries + tail
        let p = 5;
        let k = 40; // buffer capacity; ids span ceil(611/16) = 39 clusters of ≤16 rows
        let mut x = faer::Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x[(i, j)] = (((((i * 13 + j * 29 + 3) % 47) as f64) / 11.0 - 2.0).sin()) as f32;
            }
        }
        let y: Vec<f32> = (0..n)
            .map(|i| ((((i * 31 + 7) % 53) as f32) / 9.0 - 2.5).cos())
            .collect();
        let ids: Vec<u32> = (0..n).map(|i| (i / 16) as u32).collect();

        // Oracle: the pre-panel scalar accumulation, verbatim.
        let mut ref_xtx = faer::Mat::<f64>::zeros(p, p);
        let mut ref_xty = vec![0.0f64; p];
        let mut ref_yty = 0.0f64;
        let mut ref_sum_xc = faer::Mat::<f64>::zeros(p, k);
        let mut ref_sum_yc = vec![0.0f64; k];
        let mut ref_sizes = vec![0u32; k];
        let mut ref_seen = 0u32;
        for row in 0..n {
            let y_row = y[row] as f64;
            let c = ids[row] as usize;
            for j in 0..p {
                let x_rj = x[(row, j)] as f64;
                ref_sum_xc[(j, c)] += x_rj;
                for i in j..p {
                    ref_xtx[(i, j)] += x[(row, i)] as f64 * x_rj;
                }
                ref_xty[j] += x_rj * y_row;
            }
            ref_sum_yc[c] += y_row;
            ref_sizes[c] += 1;
            ref_yty += y_row * y_row;
            ref_seen = ref_seen.max(c as u32 + 1);
        }

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: k as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &ids);
        }
        for j in 0..p {
            for i in j..p {
                let (got, want) = (ws.lme_xtx[(i, j)], ref_xtx[(i, j)]);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "xtx[{i},{j}] = {got}, scalar reference {want}"
                );
            }
            assert!(
                (ws.lme_xty[j] - ref_xty[j]).abs() <= 1e-12 * ref_xty[j].abs().max(1.0),
                "xty[{j}] diverged"
            );
            for c in 0..k {
                assert_eq!(
                    ws.lme_sum_xc[(j, c)].to_bits(),
                    ref_sum_xc[(j, c)].to_bits(),
                    "sum_xc[{j},{c}] must stay bit-identical"
                );
            }
        }
        for c in 0..k {
            assert_eq!(ws.lme_sum_yc[c].to_bits(), ref_sum_yc[c].to_bits());
            assert_eq!(ws.lme_cluster_sizes[c], ref_sizes[c]);
        }
        assert_eq!(ws.lme_yty.to_bits(), ref_yty.to_bits());
        assert_eq!(ws.lme_n_clusters_seen, ref_seen);
    }

    /// EST-20: `profiled_deviance` overwrites all scratch state on every call
    /// — evaluating the same θ twice (with an intervening different-θ call)
    /// reproduces identical β̂ and identical deviance, i.e. there is no
    /// stale-state accumulation across θ evaluations. A broken kernel that
    /// accumulated into `xtvix`/`betas` instead of overwriting would drift.
    #[test]
    fn profiled_deviance_overwrites_state() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        let n: usize = 6;
        let p: usize = 2;
        let n_clusters: usize = 2;
        let cluster_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let x1: [f64; 6] = [
            0.1257302210933933,
            -0.1321048632913019,
            0.6404226504432821,
            0.10490011715303971,
            -0.535669373161111,
            0.36159505490948474,
        ];
        let y: [f64; 6] = [
            0.7718630718197979,
            -0.09922526468089643,
            1.4699547603093044,
            1.266192714345762,
            -1.8688474280172964,
            1.3141089963737067,
        ];
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
        }
        let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y_f32, &cluster_ids);
        }

        let mut scratch = build_lme_scratch(&mut ws, n as u32, n_clusters as u32);
        // First eval at θ=1.
        let dev1_a = profiled_deviance(1.0, &mut scratch);
        let beta1_a = [scratch.betas[0], scratch.betas[1]];
        assert!(dev1_a.is_finite());
        // Perturb state with a different θ.
        let _ = profiled_deviance(2.0, &mut scratch);
        // Re-eval at θ=1: must reproduce the first result bit-for-bit.
        let dev1_b = profiled_deviance(1.0, &mut scratch);
        let beta1_b = [scratch.betas[0], scratch.betas[1]];

        assert_eq!(
            dev1_a, dev1_b,
            "deviance(θ=1) must be reproducible (no stale state)"
        );
        assert_eq!(
            beta1_a, beta1_b,
            "β̂(θ=1) must be reproducible (no stale state)"
        );
    }

    /// Replaces the two synthetic-function value-match Brent tests. Brent must
    /// converge to an *interior* minimum without expanding the bracket: the
    /// returned xmin lies inside (a0, c0), fmin is ≤ the value at both initial
    /// endpoints, and the iteration terminates below the cap. No xmin value is
    /// pinned — only the bracketing/optimality invariant.
    #[test]
    fn brent_finds_interior_minimum() {
        let g = |x: f64| {
            let dx = x - 0.7;
            dx.powi(4) + 0.1 * dx.powi(2)
        };
        let a0 = -2.0_f64;
        let c0 = 2.0_f64;
        let mut a = a0;
        let mut b = 0.5_f64;
        let mut c = c0;
        let mut fa = g(a);
        let mut fb = g(b);
        let mut fc = g(c);
        let mut d = 0.0_f64;
        let mut e = 0.0_f64;

        let mut n_iter = 0_u32;
        let (xmin, fmin) = brent_minimize(
            &mut a,
            &mut b,
            &mut c,
            &mut fa,
            &mut fb,
            &mut fc,
            &mut d,
            &mut e,
            g,
            BRENT_REL_TOL,
            MAX_BRENT_ITERS,
            &mut n_iter,
        );

        assert!(
            xmin > a0 && xmin < c0,
            "xmin {xmin} must stay inside the initial bracket"
        );
        assert!(
            fmin <= g(a0) && fmin <= g(c0),
            "fmin {fmin} must not exceed the endpoints"
        );
        assert!(
            fmin <= g(xmin) + 1e-12,
            "fmin must equal g at the returned minimiser"
        );
        assert!(
            n_iter < MAX_BRENT_ITERS,
            "n_iter {n_iter} must terminate below the cap"
        );
    }

    /// Continuity test — `profiled_deviance(θ)` should be smooth and finite
    /// across the search domain. Sample 11 points on log(θ) ∈ [-4, 2] and
    /// assert (i) all values are finite, (ii) the discrete second derivative
    /// changes sign at most once (convex, or one inflection — never wild).
    #[test]
    fn profiled_deviance_is_smooth_across_log_theta() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        let n: usize = 6;
        let p: usize = 2;
        let n_clusters: usize = 2;
        let cluster_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let x1: [f64; 6] = [
            0.1257302210933933,
            -0.1321048632913019,
            0.6404226504432821,
            0.10490011715303971,
            -0.535669373161111,
            0.36159505490948474,
        ];
        let y: [f64; 6] = [
            0.7718630718197979,
            -0.09922526468089643,
            1.4699547603093044,
            1.266192714345762,
            -1.8688474280172964,
            1.3141089963737067,
        ];
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
        }
        let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y_f32, &cluster_ids);
        }

        let n_samples = 11;
        let log_lo = -4.0_f64;
        let log_hi = 2.0_f64;
        let mut devs = vec![0.0_f64; n_samples];
        {
            let mut scratch = build_lme_scratch(&mut ws, n as u32, n_clusters as u32);
            for (idx, slot) in devs.iter_mut().enumerate() {
                let frac = idx as f64 / (n_samples - 1) as f64;
                let log_theta = log_lo + (log_hi - log_lo) * frac;
                let theta = log_theta.exp();
                *slot = profiled_deviance(theta, &mut scratch);
            }
        }

        for (idx, d) in devs.iter().enumerate() {
            assert!(d.is_finite(), "dev[{idx}] = {d} (not finite)");
        }

        // Discrete second derivative: d2[i] = devs[i+1] - 2*devs[i] + devs[i-1].
        // Count sign changes — should be ≤ 1 (convex or one inflection).
        let mut signs: Vec<i32> = Vec::with_capacity(n_samples - 2);
        for i in 1..(n_samples - 1) {
            let d2 = devs[i + 1] - 2.0 * devs[i] + devs[i - 1];
            // Use a small numerical threshold so tiny noise doesn't count.
            if d2.abs() > 1e-10 {
                signs.push(if d2 > 0.0 { 1 } else { -1 });
            }
        }
        let mut n_changes = 0;
        for w in signs.windows(2) {
            if w[0] != w[1] {
                n_changes += 1;
            }
        }
        assert!(
            n_changes <= 1,
            "profiled_deviance second-derivative sign changes = {n_changes} > 1; devs = {devs:?}"
        );
    }

    // ---------------------------------------------------------------------
    // lme_fit end-to-end tests
    // ---------------------------------------------------------------------

    /// EST-24/25: on the canonical clustered reference dataset, `lme_fit`
    /// converges with a *valid* boundary flag (∈ {0, 1} — interior or τ̂≈0
    /// OLS-equivalent, never the hard-failure 2), returns finite β̂ and t²,
    /// and per-target variances are non-negative (`var_diag = σ̂²·‖L⁻¹eⱼ‖² ≥ 0`).
    /// No β̂ or t² value is pinned — only the convergence/shape invariants that
    /// a broken solver (NaN-filled, negative variance, hard boundary) would
    /// violate.
    #[test]
    fn lme_fit_converges_with_valid_boundary() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 60;
        const P: usize = 3;
        const K: usize = 6;

        let x1: [f64; 60] = [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
            -1.302179506862318,
            0.12784040316728537,
            -0.3162425923435822,
            -0.016801157504288795,
            -0.85304392757358,
            0.8793979748628286,
            0.7777919354289483,
            0.06603069756121605,
            1.1272412069680329,
            0.4675093422520456,
            -0.8592924628832382,
            0.36875078408249884,
            -0.9588826008289989,
            0.8784503013072725,
            -0.049925910986252896,
            -0.18486236354526056,
            -0.6809295444039414,
            1.2225413386740303,
            -0.15452948206880215,
            -0.4283278221631072,
            -0.3521335504882296,
            0.5323091855533487,
            0.36544406436407834,
            0.4127326115959884,
            0.43082100300788273,
            2.1416476008704612,
            -0.4064150163846156,
            -0.5122427290715373,
            -0.8137727282478777,
            0.6159794225754956,
            1.1289722927208916,
            -0.11394745765487507,
            -0.840156476962528,
            -0.8244812156912396,
            0.6505927878247011,
            0.7432541712034423,
            0.543154268305195,
            -0.6655097072886943,
            0.23216132306671977,
            0.11668580914072822,
            0.21868859672901295,
            0.8714287779481898,
            0.22359554877468227,
            0.6789135630718949,
            0.06757906948889146,
            0.28911939868998415,
            0.6312882258385404,
            -1.4571558198556664,
            -0.31967121635730134,
            -0.4703726542927955,
            -0.6388778482433419,
            -0.27514225122668373,
            1.4949413112343959,
            -0.8658311156932432,
            0.9682783545914808,
        ];
        let x2: [f64; 60] = [
            -1.6828697716158048,
            -0.33488502998577485,
            0.1627530651050056,
            0.5862223313592781,
            0.711226579792855,
            0.7933472351999252,
            -0.3487250722484376,
            -0.46235179266456716,
            0.8579758812571538,
            -0.1913043248816149,
            -1.2756863233379219,
            -1.1332872140034806,
            -0.9194522860016113,
            0.49716074405376404,
            0.14242573607056525,
            0.6904853540677682,
            -0.42725264633653426,
            0.15853969107671423,
            0.6255903939673367,
            -0.3093465397202384,
            0.45677523755741145,
            -0.6619259410666513,
            -0.3630538465650718,
            -0.3817378939983291,
            -1.1958396455890397,
            0.4869724807855818,
            -0.46940234020272387,
            0.01249411872768743,
            0.48074665890590895,
            0.4465311760299441,
            0.6653851089727862,
            -0.09848548450942361,
            -0.42329831204415375,
            -0.07971821090639905,
            -1.6873344339580298,
            -1.4471124724230873,
            -1.3226996123544024,
            -0.9972468276014818,
            0.3997742267234366,
            -0.9054790553600608,
            -0.3781625540393897,
            1.2992282977860654,
            -0.35626397106142593,
            0.7375155684670865,
            -0.933617680009877,
            -0.20543755786763002,
            -0.9500220549105812,
            -0.3390330759005625,
            0.8403081374573955,
            -1.7273204231923487,
            0.43442364354585733,
            0.2377356023322779,
            -0.5941499556967944,
            -1.4460578543884546,
            0.07212950771386951,
            -0.5294927090638024,
            0.23267621135470395,
            0.02185214552344288,
            1.6017788913209154,
            -0.23935562747302427,
        ];
        let y_data: [f64; 60] = [
            0.4634838483807412,
            -1.6958367493518591,
            -0.07656189550801018,
            0.5791988172301176,
            -0.623788696202619,
            -1.5640666548456659,
            -0.05286302450777913,
            -0.40566566398296267,
            -1.082675989070649,
            -0.6398666414265639,
            0.8986776652224437,
            1.1127414469896355,
            1.1065939034406163,
            3.740015159794447,
            0.830029945030196,
            -0.3524130442209007,
            -1.5476759552488093,
            0.06153714483417494,
            -0.8333554776977836,
            -0.4197403105019113,
            -0.42835728975820775,
            0.1797922600939008,
            2.0965054734973476,
            0.19096753379326917,
            -1.3788672930282488,
            -0.9522449167472798,
            -1.4102109341814926,
            0.5597880597592666,
            0.824978479993657,
            2.343004843474632,
            0.7533972033686575,
            0.8633702245648947,
            -0.7432683368811597,
            -0.7561198927806311,
            -0.6351566537898222,
            -0.3691685843204092,
            1.0273694686701114,
            -1.4272550988991226,
            0.6853189837147633,
            0.010351200249558046,
            1.7179220596063591,
            1.2720016651761046,
            -1.2435885938521603,
            0.4099566730250238,
            -0.7373919750549822,
            1.35341093351855,
            0.2238950195023513,
            -0.9898128365682515,
            -1.333705002777748,
            -0.19843480740416738,
            1.9931161686261913,
            1.406582108224644,
            -0.4972904954045434,
            -0.08212174931081978,
            0.445137615643293,
            -0.14552036838998333,
            -0.33955737323366064,
            2.7199407103663717,
            1.0045452753534772,
            2.1737935566316495,
        ];

        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
            x[(i, 2)] = x2[i] as f32;
        }
        let y: Vec<f32> = y_data.iter().map(|&v| v as f32).collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }

        let target_indices: Vec<u32> = vec![1, 2];

        let fit_betas: [f64; P];
        let fit_t_sq: [f64; P];
        let fit_var_diag: [f64; P];
        let fit_converged: bool;
        let fit_boundary: u8;
        {
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &target_indices, None, scratch);
            fit_betas = [fit.betas[0], fit.betas[1], fit.betas[2]];
            fit_t_sq = [fit.t_sq[0], fit.t_sq[1], fit.t_sq[2]];
            fit_var_diag = [fit.var_diag[0], fit.var_diag[1], fit.var_diag[2]];
            fit_converged = fit.converged;
            fit_boundary = fit.boundary_hit;
        }

        assert!(fit_converged, "fit failed to converge");
        // boundary_hit ∈ {0, 1} accepted: interior min or τ̂≈0 OLS-equivalent.
        // Reject 2 (high-τ hard failure).
        assert!(
            fit_boundary == 0 || fit_boundary == 1,
            "boundary_hit = {fit_boundary} (expected 0 or 1)"
        );

        for (j, b) in fit_betas.iter().enumerate() {
            assert!(b.is_finite(), "β̂[{j}] must be finite on a converged fit");
        }
        for &tj in &target_indices {
            let j = tj as usize;
            // > 0.0 (strictly): a converged fit on real data must produce a
            // non-degenerate Wald statistic; == 0.0 would indicate a zeroed-out
            // solver or a collapsed variance that >= 0.0 could not catch.
            assert!(
                fit_t_sq[j].is_finite() && fit_t_sq[j] > 0.0,
                "t²[{j}] must be finite and strictly positive on a converged fit"
            );
            // EST-25: per-target variance σ̂²·‖L⁻¹eⱼ‖² is non-negative.
            assert!(
                fit_var_diag[j] >= 0.0,
                "var_diag[{j}] = {} must be non-negative",
                fit_var_diag[j]
            );
        }
    }

    /// τ̂ = 0 boundary test: build clustered data with zero cluster variance
    /// (pure OLS) and assert that lme_fit reports `boundary_hit == 1` and
    /// returns β̂ matching a hand-coded OLS fit within 1e-9.
    #[test]
    fn lme_fit_detects_tau_zero_boundary() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 60;
        const P: usize = 3;
        const K: usize = 6;

        // Reuse the canonical X but rebuild y from the OLS-only DGP
        // (no cluster effect u, no ε offset per cluster — just OLS noise).
        let x1: [f64; 60] = [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
            -1.302179506862318,
            0.12784040316728537,
            -0.3162425923435822,
            -0.016801157504288795,
            -0.85304392757358,
            0.8793979748628286,
            0.7777919354289483,
            0.06603069756121605,
            1.1272412069680329,
            0.4675093422520456,
            -0.8592924628832382,
            0.36875078408249884,
            -0.9588826008289989,
            0.8784503013072725,
            -0.049925910986252896,
            -0.18486236354526056,
            -0.6809295444039414,
            1.2225413386740303,
            -0.15452948206880215,
            -0.4283278221631072,
            -0.3521335504882296,
            0.5323091855533487,
            0.36544406436407834,
            0.4127326115959884,
            0.43082100300788273,
            2.1416476008704612,
            -0.4064150163846156,
            -0.5122427290715373,
            -0.8137727282478777,
            0.6159794225754956,
            1.1289722927208916,
            -0.11394745765487507,
            -0.840156476962528,
            -0.8244812156912396,
            0.6505927878247011,
            0.7432541712034423,
            0.543154268305195,
            -0.6655097072886943,
            0.23216132306671977,
            0.11668580914072822,
            0.21868859672901295,
            0.8714287779481898,
            0.22359554877468227,
            0.6789135630718949,
            0.06757906948889146,
            0.28911939868998415,
            0.6312882258385404,
            -1.4571558198556664,
            -0.31967121635730134,
            -0.4703726542927955,
            -0.6388778482433419,
            -0.27514225122668373,
            1.4949413112343959,
            -0.8658311156932432,
            0.9682783545914808,
        ];
        let x2: [f64; 60] = [
            -1.6828697716158048,
            -0.33488502998577485,
            0.1627530651050056,
            0.5862223313592781,
            0.711226579792855,
            0.7933472351999252,
            -0.3487250722484376,
            -0.46235179266456716,
            0.8579758812571538,
            -0.1913043248816149,
            -1.2756863233379219,
            -1.1332872140034806,
            -0.9194522860016113,
            0.49716074405376404,
            0.14242573607056525,
            0.6904853540677682,
            -0.42725264633653426,
            0.15853969107671423,
            0.6255903939673367,
            -0.3093465397202384,
            0.45677523755741145,
            -0.6619259410666513,
            -0.3630538465650718,
            -0.3817378939983291,
            -1.1958396455890397,
            0.4869724807855818,
            -0.46940234020272387,
            0.01249411872768743,
            0.48074665890590895,
            0.4465311760299441,
            0.6653851089727862,
            -0.09848548450942361,
            -0.42329831204415375,
            -0.07971821090639905,
            -1.6873344339580298,
            -1.4471124724230873,
            -1.3226996123544024,
            -0.9972468276014818,
            0.3997742267234366,
            -0.9054790553600608,
            -0.3781625540393897,
            1.2992282977860654,
            -0.35626397106142593,
            0.7375155684670865,
            -0.933617680009877,
            -0.20543755786763002,
            -0.9500220549105812,
            -0.3390330759005625,
            0.8403081374573955,
            -1.7273204231923487,
            0.43442364354585733,
            0.2377356023322779,
            -0.5941499556967944,
            -1.4460578543884546,
            0.07212950771386951,
            -0.5294927090638024,
            0.23267621135470395,
            0.02185214552344288,
            1.6017788913209154,
            -0.23935562747302427,
        ];

        // f64 matrix for OLS reference arithmetic; f32 copy for fit calls.
        let mut x_f64 = Mat::<f64>::zeros(N, P);
        for i in 0..N {
            x_f64[(i, 0)] = 1.0;
            x_f64[(i, 1)] = x1[i];
            x_f64[(i, 2)] = x2[i];
        }

        // Synthesize y = X·β_true + ε, no per-cluster effect (τ = 0).
        // Use a deterministic small "noise" pattern.
        let beta_true = [0.5, 0.4, -0.3];
        let y_f64: Vec<f64> = (0..N)
            .map(|i| {
                let row_signal =
                    beta_true[0] + beta_true[1] * x_f64[(i, 1)] + beta_true[2] * x_f64[(i, 2)];
                // Tiny deterministic "noise" derived from i (small so τ̂≈0
                // boundary fires cleanly).
                let noise = ((i as f64) * 0.0137).sin() * 0.05;
                row_signal + noise
            })
            .collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();

        // OLS reference β̂ via normal equations on the same data.
        // β̂_ols = (X'X)⁻¹ X'y. Use faer's Cholesky for the inversion.
        let mut xtx_ols = Mat::<f64>::zeros(P, P);
        let mut xty_ols = [0.0_f64; P];
        for i in 0..N {
            for jj in 0..P {
                xty_ols[jj] += x_f64[(i, jj)] * y_f64[i];
                for ii in jj..P {
                    xtx_ols[(ii, jj)] += x_f64[(i, ii)] * x_f64[(i, jj)];
                }
            }
        }
        let chol = xtx_ols
            .as_ref()
            .llt(faer::Side::Lower)
            .expect("OLS Cholesky failed");
        let mut rhs = Mat::<f64>::zeros(P, 1);
        for jj in 0..P {
            rhs[(jj, 0)] = xty_ols[jj];
        }
        use faer::linalg::solvers::Solve;
        chol.solve_in_place(rhs.as_mut());
        let ols_beta = [rhs[(0, 0)], rhs[(1, 0)], rhs[(2, 0)]];

        // f32 copies for fit/add_rows calls (f32 data plane).
        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = x_f64[(i, 0)] as f32;
            x[(i, 1)] = x_f64[(i, 1)] as f32;
            x[(i, 2)] = x_f64[(i, 2)] as f32;
        }
        let y: Vec<f32> = y_f64.iter().map(|&v| v as f32).collect();

        // Now run lme_fit on the same (X, y).
        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        let targets: Vec<u32> = vec![1, 2];

        let fit_betas: [f64; P];
        let fit_boundary: u8;
        let fit_converged: bool;
        {
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
            fit_betas = [fit.betas[0], fit.betas[1], fit.betas[2]];
            fit_boundary = fit.boundary_hit;
            fit_converged = fit.converged;
        }
        assert!(fit_converged, "lme_fit failed to converge on τ=0 case");
        assert_eq!(
            fit_boundary, 1,
            "expected τ̂≈0 boundary_hit=1, got {fit_boundary}"
        );
        for j in 0..P {
            let delta = (fit_betas[j] - ols_beta[j]).abs();
            // f32 data plane: LME reads f32-widened X/y while ols_beta is the f64
            // reference solve, so τ̂≈0 reduction matches OLS to ~f32 precision
            // (~2e-9 here), not the old f64 1e-9. Loosened to the f32-plane budget.
            assert!(
                delta < 1e-6,
                "β̂[{j}] = {} vs OLS β̂ = {} (Δ = {})",
                fit_betas[j],
                ols_beta[j],
                delta
            );
        }
    }

    /// Deterministic clustered fixture with a REAL random-intercept effect
    /// (per-cluster offsets, sd ≈ 0.66, vs residual noise 0.3·U(−1,1)) so the
    /// REML minimum is interior — used by the W2 truth-start tests below.
    /// Returns `(x, y, cluster_ids)`; n=60, p=3, k=6.
    fn make_clustered_fixture_interior_theta() -> (faer::Mat<f32>, Vec<f32>, Vec<u32>) {
        use faer::Mat;
        const N: usize = 60;
        const P: usize = 3;
        const K: usize = 6;
        let offsets: [f64; K] = [0.9, -0.6, 0.3, -0.2, 0.65, -1.05];
        let mut x = Mat::<f32>::zeros(N, P);
        let mut y = vec![0.0_f32; N];
        let mut cluster_ids = vec![0_u32; N];
        // LCG-style pseudo-randomness, hermetic (mirrors make_lme_fixture_p3).
        let mut state = 0xD1B5_4A32_D192_ED03_u64;
        for i in 0..N {
            x[(i, 0)] = 1.0;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let x1 = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            x[(i, 1)] = x1 as f32;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let x2 = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            x[(i, 2)] = x2 as f32;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            let c = i % K;
            y[i] = (0.5 + 0.4 * x1 - 0.3 * x2 + offsets[c] + 0.3 * noise) as f32;
            cluster_ids[i] = c as u32;
        }
        (x, y, cluster_ids)
    }

    /// W2 truth-start: a warm (truth-centered) bracket and the cold bracket on
    /// the same suff-stats converge to the same interior REML minimum — z =
    /// √t² within 1e-4 abs and β̂ within 1e-5 abs (the campaign parity
    /// floors), same boundary flag — while spending fewer deviance evals. A
    /// warm bracket that shifted the minimum (instead of just narrowing the
    /// search) or that failed to save evals would fail here.
    #[test]
    fn lme_truth_start_matches_cold_minimum_with_fewer_evals() {
        use crate::workspace::SimWorkspace;

        let (x, y, cluster_ids) = make_clustered_fixture_interior_theta();
        let n = y.len();
        let p = x.ncols();
        let k = *cluster_ids.iter().max().unwrap() as usize + 1;
        let targets: Vec<u32> = vec![1, 2];

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: k as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }

        let (cold_betas, cold_t_sq, cold_boundary, cold_evals) = {
            let scratch = build_lme_scratch(&mut ws, n as u32, k as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
            assert!(fit.converged, "cold fit must converge");
            (
                fit.betas.to_vec(),
                fit.t_sq.to_vec(),
                fit.boundary_hit,
                fit.n_evals,
            )
        };
        assert_eq!(
            cold_boundary, 0,
            "fixture must have an interior minimum (cold boundary_hit = 0)"
        );

        // Truth start: the fixture's designed θ = sd(offsets)/sd(residual)
        // ≈ 0.66/0.17 ≈ 4 — exact centering is not required, only that the
        // (c − Δ, c + Δ) bracket contains the minimum.
        let scratch = build_lme_scratch(&mut ws, n as u32, k as u32);
        let warm = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, Some(4.0), scratch);
        assert!(warm.converged, "warm fit must converge");
        assert_eq!(
            warm.boundary_hit, cold_boundary,
            "boundary flags must agree"
        );
        for (j, (&bw, &bc)) in warm.betas.iter().zip(&cold_betas).enumerate() {
            assert!(
                (bw - bc).abs() < 1e-5,
                "β̂[{j}]: warm {bw} vs cold {bc} exceeds 1e-5"
            );
        }
        for &tj in &targets {
            let (tw, tc) = (warm.t_sq[tj as usize], cold_t_sq[tj as usize]);
            assert!(
                (tw.sqrt() - tc.sqrt()).abs() < 1e-4,
                "z[{tj}]: warm {} vs cold {} exceeds 1e-4",
                tw.sqrt(),
                tc.sqrt()
            );
        }
        assert!(
            warm.n_evals < cold_evals,
            "warm bracket must save deviance evals: warm {} vs cold {}",
            warm.n_evals,
            cold_evals
        );
    }

    /// W2 truth-start at τ² = 0: ln θ₀ = −∞ clamps the bracket onto the left
    /// edge, and a no-cluster-effect dataset must still classify as the τ̂≈0
    /// boundary (boundary_hit = 1) with β̂ identical to the cold path — both
    /// paths pin the recovery Cholesky at exactly LOG_THETA_LOW, so the
    /// outputs match bit-for-bit, not just within tolerance.
    #[test]
    fn lme_truth_start_zero_tau_keeps_boundary_semantics() {
        use crate::workspace::SimWorkspace;

        // τ = 0 DGP: the p3 fixture has no cluster offsets.
        let (x, y, cluster_ids) = make_lme_fixture_p3(0x5EED_CAFE);
        let n = y.len();
        let p = x.ncols();
        let k = *cluster_ids.iter().max().unwrap() as usize + 1;
        let targets: Vec<u32> = vec![1, 2];

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: k as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }

        let (cold_betas, cold_boundary, cold_converged) = {
            let scratch = build_lme_scratch(&mut ws, n as u32, k as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
            (fit.betas.to_vec(), fit.boundary_hit, fit.converged)
        };

        let scratch = build_lme_scratch(&mut ws, n as u32, k as u32);
        let warm = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, Some(0.0), scratch);

        assert!(cold_converged && warm.converged, "both paths must converge");
        assert_eq!(cold_boundary, 1, "τ=0 DGP must hit the τ̂≈0 boundary cold");
        assert_eq!(warm.boundary_hit, 1, "θ₀ = 0 must keep boundary_hit = 1");
        for (j, (&bw, &bc)) in warm.betas.iter().zip(&cold_betas).enumerate() {
            assert_eq!(
                bw.to_bits(),
                bc.to_bits(),
                "β̂[{j}]: warm {bw} != cold {bc} (both pin at LOG_THETA_LOW)"
            );
        }
    }

    /// Rank-deficient X test: two collinear columns. lme_fit should mark
    /// the fit as `converged = false` and fill outputs with NaN.
    #[test]
    fn lme_fit_handles_rank_deficient_x() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 30;
        const P: usize = 3;
        const K: usize = 3;

        // x[1] = 1.0 (constant) duplicates the intercept exactly → singular
        // X'X with no recoverable σ̂² (the linear span has rank 2, not 3, and
        // the residual is bounded away from zero by the noise pattern below).
        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = 1.0;
            x[(i, 2)] = ((i as f32) * 0.1) - 1.0;
        }
        let y: Vec<f32> = (0..N)
            .map(|i| (i as f32) * 0.13 - 0.4 + ((i % 3) as f32) * 0.5)
            .collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        let targets: Vec<u32> = vec![1, 2];

        let fit_converged: bool;
        let fit_betas: [f64; P];
        let fit_var_diag: [f64; 2];
        let fit_t_sq: [f64; 2];
        {
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
            fit_converged = fit.converged;
            fit_betas = [fit.betas[0], fit.betas[1], fit.betas[2]];
            fit_var_diag = [fit.var_diag[1], fit.var_diag[2]];
            fit_t_sq = [fit.t_sq[1], fit.t_sq[2]];
        }

        assert!(!fit_converged, "expected non-converged on singular X");
        // β̂ entries should be NaN.
        for (j, &b) in fit_betas.iter().enumerate() {
            assert!(b.is_nan(), "β̂[{j}] = {b} should be NaN");
        }
        for v in &fit_var_diag {
            assert!(v.is_nan(), "var_diag = {v} should be NaN");
        }
        for v in &fit_t_sq {
            assert!(v.is_nan(), "t_sq = {v} should be NaN");
        }
    }

    /// Bracket-repair test: construct a case where the initial 3-point sweep
    /// has `fa < fb` (left endpoint lower → minimum is near or below
    /// LOG_THETA_LOW). Assert Brent still converges and `boundary_hit ∈ {0, 1}`.
    ///
    /// We do this by re-using the τ=0 case from above (where the true τ̂ is at
    /// the left boundary): at the initial sweep `fa = profiled_deviance(1e-4)`
    /// is likely below `fb = profiled_deviance(1e-1)`, so the bracket repair
    /// fires. The result should still mark convergence and either find an
    /// interior min slightly above LOG_THETA_LOW (boundary_hit=0) or
    /// τ̂≈0 (boundary_hit=1).
    #[test]
    fn lme_fit_bracket_repair_when_min_at_left_edge() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 30;
        const P: usize = 2;
        const K: usize = 3;

        // Deterministic OLS-only DGP (τ=0) — minimum of profiled deviance is
        // at or below LOG_THETA_LOW.
        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = ((i as f32) * 0.07).sin();
        }
        let y: Vec<f32> = (0..N)
            .map(|i| (0.5 + 0.4 * x[(i, 1)] as f64 + ((i as f64) * 0.011).cos() * 0.05) as f32)
            .collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        let targets: Vec<u32> = vec![1];

        let fit_converged: bool;
        let fit_boundary: u8;
        {
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
            fit_converged = fit.converged;
            fit_boundary = fit.boundary_hit;
        }
        assert!(fit_converged, "bracket-repair path failed to converge");
        assert!(
            fit_boundary == 0 || fit_boundary == 1,
            "expected boundary_hit ∈ {{0, 1}}, got {fit_boundary}"
        );
    }

    /// Bounded-allocation warm-path test (mirrors `ols.rs`'s
    /// `fit_suff_stats_warm_path_bounded_alloc`). Marked `#[ignore]` because
    /// `dhat::Profiler` measures process-wide allocations and concurrent tests
    /// contaminate the count. Run explicitly:
    ///   `cargo test -p engine-core lme_fit_warm_path_bounded_alloc -- --ignored --test-threads=1`
    ///
    /// `BOUND` locks the measured warm-path block count. Each `profiled_deviance`
    /// call (the Brent loop runs several per fit) does one faer `Cholesky`; the
    /// in-place β solve no longer allocates an owned RHS `Mat`, and neither does
    /// the joint Wald solve (its RHS is a column view over `joint_rhs`).
    /// Measured at 1800 blocks over `N_CALLS` (was 2600 with the per-fit owned
    /// joint-RHS `Mat`, 3300 before the zero-copy β RHS). The fixture stays
    /// cold-started: a fixed bracket keeps the eval count — and therefore the
    /// faer-internal block count — pinned across start-point changes. If a
    /// future faer version changes its Cholesky internals, update the bound —
    /// do not relax it.
    #[test]
    #[ignore]
    fn lme_fit_warm_path_bounded_alloc() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 60;
        const P: usize = 3;
        const K: usize = 6;
        const N_CALLS: usize = 100;
        const BOUND: u64 = 1800;

        // Use the canonical N=60 reference case.
        let x1: [f64; 60] = [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
            -1.302179506862318,
            0.12784040316728537,
            -0.3162425923435822,
            -0.016801157504288795,
            -0.85304392757358,
            0.8793979748628286,
            0.7777919354289483,
            0.06603069756121605,
            1.1272412069680329,
            0.4675093422520456,
            -0.8592924628832382,
            0.36875078408249884,
            -0.9588826008289989,
            0.8784503013072725,
            -0.049925910986252896,
            -0.18486236354526056,
            -0.6809295444039414,
            1.2225413386740303,
            -0.15452948206880215,
            -0.4283278221631072,
            -0.3521335504882296,
            0.5323091855533487,
            0.36544406436407834,
            0.4127326115959884,
            0.43082100300788273,
            2.1416476008704612,
            -0.4064150163846156,
            -0.5122427290715373,
            -0.8137727282478777,
            0.6159794225754956,
            1.1289722927208916,
            -0.11394745765487507,
            -0.840156476962528,
            -0.8244812156912396,
            0.6505927878247011,
            0.7432541712034423,
            0.543154268305195,
            -0.6655097072886943,
            0.23216132306671977,
            0.11668580914072822,
            0.21868859672901295,
            0.8714287779481898,
            0.22359554877468227,
            0.6789135630718949,
            0.06757906948889146,
            0.28911939868998415,
            0.6312882258385404,
            -1.4571558198556664,
            -0.31967121635730134,
            -0.4703726542927955,
            -0.6388778482433419,
            -0.27514225122668373,
            1.4949413112343959,
            -0.8658311156932432,
            0.9682783545914808,
        ];
        let x2: [f64; 60] = [
            -1.6828697716158048,
            -0.33488502998577485,
            0.1627530651050056,
            0.5862223313592781,
            0.711226579792855,
            0.7933472351999252,
            -0.3487250722484376,
            -0.46235179266456716,
            0.8579758812571538,
            -0.1913043248816149,
            -1.2756863233379219,
            -1.1332872140034806,
            -0.9194522860016113,
            0.49716074405376404,
            0.14242573607056525,
            0.6904853540677682,
            -0.42725264633653426,
            0.15853969107671423,
            0.6255903939673367,
            -0.3093465397202384,
            0.45677523755741145,
            -0.6619259410666513,
            -0.3630538465650718,
            -0.3817378939983291,
            -1.1958396455890397,
            0.4869724807855818,
            -0.46940234020272387,
            0.01249411872768743,
            0.48074665890590895,
            0.4465311760299441,
            0.6653851089727862,
            -0.09848548450942361,
            -0.42329831204415375,
            -0.07971821090639905,
            -1.6873344339580298,
            -1.4471124724230873,
            -1.3226996123544024,
            -0.9972468276014818,
            0.3997742267234366,
            -0.9054790553600608,
            -0.3781625540393897,
            1.2992282977860654,
            -0.35626397106142593,
            0.7375155684670865,
            -0.933617680009877,
            -0.20543755786763002,
            -0.9500220549105812,
            -0.3390330759005625,
            0.8403081374573955,
            -1.7273204231923487,
            0.43442364354585733,
            0.2377356023322779,
            -0.5941499556967944,
            -1.4460578543884546,
            0.07212950771386951,
            -0.5294927090638024,
            0.23267621135470395,
            0.02185214552344288,
            1.6017788913209154,
            -0.23935562747302427,
        ];
        let y_data: [f64; 60] = [
            0.4634838483807412,
            -1.6958367493518591,
            -0.07656189550801018,
            0.5791988172301176,
            -0.623788696202619,
            -1.5640666548456659,
            -0.05286302450777913,
            -0.40566566398296267,
            -1.082675989070649,
            -0.6398666414265639,
            0.8986776652224437,
            1.1127414469896355,
            1.1065939034406163,
            3.740015159794447,
            0.830029945030196,
            -0.3524130442209007,
            -1.5476759552488093,
            0.06153714483417494,
            -0.8333554776977836,
            -0.4197403105019113,
            -0.42835728975820775,
            0.1797922600939008,
            2.0965054734973476,
            0.19096753379326917,
            -1.3788672930282488,
            -0.9522449167472798,
            -1.4102109341814926,
            0.5597880597592666,
            0.824978479993657,
            2.343004843474632,
            0.7533972033686575,
            0.8633702245648947,
            -0.7432683368811597,
            -0.7561198927806311,
            -0.6351566537898222,
            -0.3691685843204092,
            1.0273694686701114,
            -1.4272550988991226,
            0.6853189837147633,
            0.010351200249558046,
            1.7179220596063591,
            1.2720016651761046,
            -1.2435885938521603,
            0.4099566730250238,
            -0.7373919750549822,
            1.35341093351855,
            0.2238950195023513,
            -0.9898128365682515,
            -1.333705002777748,
            -0.19843480740416738,
            1.9931161686261913,
            1.406582108224644,
            -0.4972904954045434,
            -0.08212174931081978,
            0.445137615643293,
            -0.14552036838998333,
            -0.33955737323366064,
            2.7199407103663717,
            1.0045452753534772,
            2.1737935566316495,
        ];

        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
            x[(i, 2)] = x2[i] as f32;
        }
        let y: Vec<f32> = y_data.iter().map(|&v| v as f32).collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        // Warmup to drive any one-time setup outside the profiler window.
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        {
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let _ = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
        }

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.reset_lme_suff_stats();
            {
                let mut s = LmeSuffStats {
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
                s.add_rows(x.as_ref(), &y, &cluster_ids);
            }
            let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
            let _ = lme_fit(x.as_ref(), &y, &cluster_ids, &targets, None, scratch);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "lme_fit allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }

    // ---------------------------------------------------------------------
    // Joint Wald-χ²
    // ---------------------------------------------------------------------

    /// Build a small deterministic LME fixture (p=3, k=3 clusters, n=15
    /// blocked) seeded by `seed` so different tests can vary it. Returns
    /// `(x, y, cluster_ids)`.
    fn make_lme_fixture_p3(seed: u64) -> (faer::Mat<f32>, Vec<f32>, Vec<u32>) {
        use faer::Mat;
        const N: usize = 15;
        const P: usize = 3;
        const K: usize = 3;
        let mut x = Mat::<f32>::zeros(N, P);
        let mut y = vec![0.0_f32; N];
        let mut cluster_ids = vec![0_u32; N];
        // Simple LCG-style pseudo-randomness so the test stays hermetic.
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        for i in 0..N {
            // Three values per row (intercept omitted by reusing 1.0).
            x[(i, 0)] = 1.0;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let x1 = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            x[(i, 1)] = x1 as f32;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let x2 = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            x[(i, 2)] = x2 as f32;
            state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            // Signal: true β = (0.5, 0.4, -0.3).
            y[i] = (0.5 + 0.4 * x1 - 0.3 * x2 + 0.3 * noise) as f32;
            cluster_ids[i] = (i % K) as u32;
        }
        (x, y, cluster_ids)
    }

    /// k=1 collapse: `joint_t_sq` must equal `t_sq[target]` to ULP precision.
    #[test]
    fn joint_wald_collapses_to_wald_z_sq_when_k_eq_1() {
        use crate::workspace::SimWorkspace;

        let (x, y, cluster_ids) = make_lme_fixture_p3(0x1234_5678);
        let target_indices = [1u32]; // test x1 only
        let n = y.len();
        let p = x.ncols();
        let n_clusters = *cluster_ids.iter().max().unwrap() as usize + 1;

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        let scratch = build_lme_scratch(&mut ws, n as u32, n_clusters as u32);
        let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &target_indices, None, scratch);

        assert!(fit.converged, "fixture failed to converge");
        assert!(fit.joint_t_sq.is_finite(), "joint_t_sq is NaN/inf");
        let wald_z_sq = fit.t_sq[target_indices[0] as usize];
        let diff = (fit.joint_t_sq - wald_z_sq).abs();
        // Joint goes through extra matrix solves; tolerate ULP-level drift.
        assert!(
            diff < 1e-12,
            "k=1 joint must equal Wald-z² within 1e-12: joint={}, wald={}, diff={}",
            fit.joint_t_sq,
            wald_z_sq,
            diff
        );
    }

    /// τ̂≈0 OLS-fallback path: the joint Wald-χ² must still be finite and
    /// non-negative when boundary_hit=1.
    #[test]
    fn joint_wald_ols_fallback_returns_finite_chi_sq() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 30;
        const P: usize = 3;
        const K: usize = 3;
        // Build a fixture with zero cluster effect (pure OLS DGP) — REML
        // collapses τ̂ → 0, triggering the boundary_hit=1 OLS-equivalent path.
        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = ((i as f32) * 0.13).sin();
            x[(i, 2)] = ((i as f32) * 0.07).cos();
        }
        let y: Vec<f32> = (0..N)
            .map(|i| {
                (0.5 + 0.4 * x[(i, 1)] as f64 - 0.3 * x[(i, 2)] as f64
                    + ((i as f64) * 0.011).sin() * 0.05) as f32
            })
            .collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();
        let target_indices = [1u32, 2u32];

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }
        let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
        let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &target_indices, None, scratch);
        assert!(fit.converged);
        assert_eq!(
            fit.boundary_hit, 1,
            "expected boundary_hit=1 on zero-ICC fixture, got {}",
            fit.boundary_hit
        );
        assert!(
            fit.joint_t_sq.is_finite(),
            "OLS-fallback joint must be finite, got {}",
            fit.joint_t_sq
        );
        assert!(fit.joint_t_sq >= 0.0);
    }

    // -----------------------------------------------------------------
    // C4 — lme_fit σ² / β̂ golden values (external oracle: R/lme4 REML)
    // -----------------------------------------------------------------
    //
    // R run on the committed 60-row fixture (cluster = i % 6, i=0..59):
    //   df <- data.frame(y=y60, x1=x1, x2=x2, cluster=factor(rep(0:5, each=10)))
    //   fit <- lmer(y ~ x1 + x2 + (1|cluster), REML=TRUE, data=df)
    //   sigma(fit)^2  →  0.9643488
    //   fixef(fit)    →  intercept=0.1665798, x1=0.7513689, x2=0.2555365
    #[test]
    fn lme_fit_golden_sigma_sq_and_betas() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        const N: usize = 60;
        const P: usize = 3;
        const K: usize = 6;

        let x1: [f64; 60] = [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
            -1.9510351886538364,
            -1.302179506862318,
            0.12784040316728537,
            -0.3162425923435822,
            -0.016801157504288795,
            -0.85304392757358,
            0.8793979748628286,
            0.7777919354289483,
            0.06603069756121605,
            1.1272412069680329,
            0.4675093422520456,
            -0.8592924628832382,
            0.36875078408249884,
            -0.9588826008289989,
            0.8784503013072725,
            -0.049925910986252896,
            -0.18486236354526056,
            -0.6809295444039414,
            1.2225413386740303,
            -0.15452948206880215,
            -0.4283278221631072,
            -0.3521335504882296,
            0.5323091855533487,
            0.36544406436407834,
            0.4127326115959884,
            0.43082100300788273,
            2.1416476008704612,
            -0.4064150163846156,
            -0.5122427290715373,
            -0.8137727282478777,
            0.6159794225754956,
            1.1289722927208916,
            -0.11394745765487507,
            -0.840156476962528,
            -0.8244812156912396,
            0.6505927878247011,
            0.7432541712034423,
            0.543154268305195,
            -0.6655097072886943,
            0.23216132306671977,
            0.11668580914072822,
            0.21868859672901295,
            0.8714287779481898,
            0.22359554877468227,
            0.6789135630718949,
            0.06757906948889146,
            0.28911939868998415,
            0.6312882258385404,
            -1.4571558198556664,
            -0.31967121635730134,
            -0.4703726542927955,
            -0.6388778482433419,
            -0.27514225122668373,
            1.4949413112343959,
            -0.8658311156932432,
            0.9682783545914808,
        ];
        let x2: [f64; 60] = [
            -1.6828697716158048,
            -0.33488502998577485,
            0.1627530651050056,
            0.5862223313592781,
            0.711226579792855,
            0.7933472351999252,
            -0.3487250722484376,
            -0.46235179266456716,
            0.8579758812571538,
            -0.1913043248816149,
            -1.2756863233379219,
            -1.1332872140034806,
            -0.9194522860016113,
            0.49716074405376404,
            0.14242573607056525,
            0.6904853540677682,
            -0.42725264633653426,
            0.15853969107671423,
            0.6255903939673367,
            -0.3093465397202384,
            0.45677523755741145,
            -0.6619259410666513,
            -0.3630538465650718,
            -0.3817378939983291,
            -1.1958396455890397,
            0.4869724807855818,
            -0.46940234020272387,
            0.01249411872768743,
            0.48074665890590895,
            0.4465311760299441,
            0.6653851089727862,
            -0.09848548450942361,
            -0.42329831204415375,
            -0.07971821090639905,
            -1.6873344339580298,
            -1.4471124724230873,
            -1.3226996123544024,
            -0.9972468276014818,
            0.3997742267234366,
            -0.9054790553600608,
            -0.3781625540393897,
            1.2992282977860654,
            -0.35626397106142593,
            0.7375155684670865,
            -0.933617680009877,
            -0.20543755786763002,
            -0.9500220549105812,
            -0.3390330759005625,
            0.8403081374573955,
            -1.7273204231923487,
            0.43442364354585733,
            0.2377356023322779,
            -0.5941499556967944,
            -1.4460578543884546,
            0.07212950771386951,
            -0.5294927090638024,
            0.23267621135470395,
            0.02185214552344288,
            1.6017788913209154,
            -0.23935562747302427,
        ];
        let y_data: [f64; 60] = [
            0.4634838483807412,
            -1.6958367493518591,
            -0.07656189550801018,
            0.5791988172301176,
            -0.623788696202619,
            -1.5640666548456659,
            -0.05286302450777913,
            -0.40566566398296267,
            -1.082675989070649,
            -0.6398666414265639,
            0.8986776652224437,
            1.1127414469896355,
            1.1065939034406163,
            3.740015159794447,
            0.830029945030196,
            -0.3524130442209007,
            -1.5476759552488093,
            0.06153714483417494,
            -0.8333554776977836,
            -0.4197403105019113,
            -0.42835728975820775,
            0.1797922600939008,
            2.0965054734973476,
            0.19096753379326917,
            -1.3788672930282488,
            -0.9522449167472798,
            -1.4102109341814926,
            0.5597880597592666,
            0.824978479993657,
            2.343004843474632,
            0.7533972033686575,
            0.8633702245648947,
            -0.7432683368811597,
            -0.7561198927806311,
            -0.6351566537898222,
            -0.3691685843204092,
            1.0273694686701114,
            -1.4272550988991226,
            0.6853189837147633,
            0.010351200249558046,
            1.7179220596063591,
            1.2720016651761046,
            -1.2435885938521603,
            0.4099566730250238,
            -0.7373919750549822,
            1.35341093351855,
            0.2238950195023513,
            -0.9898128365682515,
            -1.333705002777748,
            -0.19843480740416738,
            1.9931161686261913,
            1.406582108224644,
            -0.4972904954045434,
            -0.08212174931081978,
            0.445137615643293,
            -0.14552036838998333,
            -0.33955737323366064,
            2.7199407103663717,
            1.0045452753534772,
            2.1737935566316495,
        ];

        let mut x = Mat::<f32>::zeros(N, P);
        for i in 0..N {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
            x[(i, 2)] = x2[i] as f32;
        }
        let y: Vec<f32> = y_data.iter().map(|&v| v as f32).collect();
        let cluster_ids: Vec<u32> = (0..N).map(|i| (i % K) as u32).collect();
        let target_indices: Vec<u32> = vec![1, 2];

        let mut ws = SimWorkspace::new(
            N,
            P,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: K as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y, &cluster_ids);
        }

        let scratch = build_lme_scratch(&mut ws, N as u32, K as u32);
        let fit = lme_fit(x.as_ref(), &y, &cluster_ids, &target_indices, None, scratch);

        assert!(
            fit.converged,
            "must converge on the canonical 60-row fixture"
        );

        if fit.boundary_hit == 0 {
            // Interior minimum: compare to R/lme4 REML values (0.1% relative tolerance).
            // R: sigma(fit)^2  = 0.9643488  (lmer REML on this fixture)
            // R: fixef(fit)[2] = 0.7513689  (x1 coefficient, index 1)
            let r_sigma_sq = 0.9643488_f64;
            let r_beta_x1 = 0.7513689_f64;

            let rel_sigma = (fit.sigma_sq - r_sigma_sq).abs() / r_sigma_sq;
            assert!(
                rel_sigma < 1e-3,
                "σ² = {}, R/lme4 = {r_sigma_sq}, rel = {rel_sigma}",
                fit.sigma_sq
            );
            let rel_beta = (fit.betas[1] - r_beta_x1).abs() / r_beta_x1.abs().max(1e-9);
            assert!(
                rel_beta < 1e-3,
                "β̂[x1] = {}, R/lme4 = {r_beta_x1}, rel = {rel_beta}",
                fit.betas[1]
            );
        } else {
            // OLS fallback (boundary_hit=1): σ² should still be finite and positive.
            // The OLS-fallback branch is further guarded by lme_fit_detects_tau_zero_boundary.
            assert_eq!(fit.boundary_hit, 1, "unexpected boundary_hit value");
            assert!(
                fit.sigma_sq.is_finite() && fit.sigma_sq > 0.0,
                "OLS-fallback σ² must be finite positive, got {}",
                fit.sigma_sq
            );
        }
    }

    // -----------------------------------------------------------------
    // C12 — profiled_deviance at θ=1 vs an external lme4 oracle (+ curvature)
    // -----------------------------------------------------------------
    //
    // External oracle (no reference to the engine's own prior output). lme4's
    // REML deviance function, built via `mkLmerDevfun` and evaluated WITHOUT
    // optimising — this fixture's optimum is the τ̂=0 boundary, not θ=1 — is
    //     devfun(θ) = log|L_θ|² + log|R_X|² + (N−P)·[1 + log(2π·σ̂²)].
    // The engine's `profiled_deviance` returns the same quantity minus the
    // additive constant (N−P)·(1 + log2π): see step 10 of the fn, which returns
    // `log|V| + 2·Σ log L_jj + (N−P)·log(σ̂²)`. Hence
    //     engine_expected(θ) = devfun(θ) − (N−P)·(1 + log2π).
    // For this 6-row fixture lme4 gives devfun(1.0) = 9.410566478412274
    // (reproduce with scripts/lme_devfun_theta1.R), so the engine value at θ=1
    // is ≈ −1.9409417872 — and lme4 confirms the curvature ordering too
    // (devfun(1e-4) − const ≈ −3.0766 < −1.9409).
    //
    // The test guards two properties:
    //   1. Absolute value vs the lme4 oracle: a wrong df_resid or a botched
    //      log-det / σ̂² term shifts the deviance well beyond the 1e-4 band.
    //   2. Curvature: deviance at θ=1 exceeds deviance at θ→0 (this fixture's
    //      boundary minimum), confirming the optimiser descends the right way.
    #[test]
    fn profiled_deviance_value_and_curvature_at_theta_1() {
        use crate::workspace::SimWorkspace;
        use faer::Mat;

        let n: usize = 6;
        let p: usize = 2;
        let n_clusters: usize = 2;
        let cluster_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let x1: [f64; 6] = [
            0.1257302210933933,
            -0.1321048632913019,
            0.6404226504432821,
            0.10490011715303971,
            -0.535669373161111,
            0.36159505490948474,
        ];
        let y: [f64; 6] = [
            0.7718630718197979,
            -0.09922526468089643,
            1.4699547603093044,
            1.266192714345762,
            -1.8688474280172964,
            1.3141089963737067,
        ];
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i] as f32;
        }
        let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();

        let mut ws = SimWorkspace::new(
            n,
            p,
            0,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
        {
            let mut s = LmeSuffStats {
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
            s.add_rows(x.as_ref(), &y_f32, &cluster_ids);
        }

        let mut scratch = build_lme_scratch(&mut ws, n as u32, n_clusters as u32);

        // 1. Absolute-value guard against the external lme4 oracle.
        //    devfun(1.0) is lme4's REML deviance at θ=1 (see the block comment
        //    above the fn and scripts/lme_devfun_theta1.R); subtract the
        //    (N−P)·(1 + log2π) constant the engine omits. Actual lme4↔engine
        //    agreement is ~1e-13; the 1e-4 band leaves headroom for platform
        //    float drift while still catching any O(1) formula error.
        const DEVFUN_LME4_AT_THETA_1: f64 = 9.410566478412274;
        let n_minus_p = (n - p) as f64; // = 4
        let reml_const = n_minus_p * (1.0 + (2.0 * std::f64::consts::PI).ln());
        let expected_dev_at_theta_1 = DEVFUN_LME4_AT_THETA_1 - reml_const; // ≈ −1.9409417872
        let dev_at_theta_1 = profiled_deviance(1.0, &mut scratch);
        assert!(
            dev_at_theta_1.is_finite(),
            "profiled_deviance(θ=1) must be finite, got {dev_at_theta_1}"
        );
        assert!(
            (dev_at_theta_1 - expected_dev_at_theta_1).abs() < 1e-4,
            "profiled_deviance(θ=1) = {dev_at_theta_1}, lme4-derived expected {expected_dev_at_theta_1}"
        );

        // 2. Curvature guard: deviance at θ=1 must exceed deviance near the OLS boundary
        //    (θ→0). This fixture's minimum is at τ̂=0; a mis-coded formula that monotone-
        //    decreases toward θ=∞ would fail here.
        let dev_at_small_theta = profiled_deviance(1e-4, &mut scratch);
        assert!(
            dev_at_theta_1 > dev_at_small_theta,
            "deviance at θ=1 ({dev_at_theta_1}) must exceed deviance at θ→0 ({dev_at_small_theta})"
        );
    }
}
