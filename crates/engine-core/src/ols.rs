//! OLS sufficient-statistics fit + t² inference.
//!
//! Hot-loop invariant: never call sqrt on the SE side, never call abs on β,
//! never call Student-t CDF. All thresholds come from the precomputed
//! `CritValueTable` (squared). Per-coefficient inference outputs are
//! `t_sq_j = β̂_j² / var_diag_j`, where `var_diag_j = σ̂² · (X'X)⁻¹_jj`,
//! computed via `L u = e_j` → `‖u‖²` on the Cholesky factor of `X'X`.
//!
//! Scratch lives in `SimWorkspace` and is passed through `OlsScratch<'_>`.
//! `fit_suff_stats_t_sq` writes through the scratch and returns
//! `OlsFitView<'_>` borrowing the same storage — no per-fit heap allocations
//! on our side of the warm path apart from what faer's `llt` does internally.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::linalg::matmul::{matmul, triangular};
use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};

use crate::FLOAT_NEAR_ZERO;

/// Rows per widened f64 panel in the suff-stats GEMM accumulate
/// (`OlsSuffStats::add_rows` / `LmeSuffStats::add_rows`). Bounds the f64
/// working set to PANEL_ROWS·P so the GEMM stays cache-resident instead of
/// streaming a full n×p copy (group-B route (a) vs (b)).
/// Tuned 2026-06-12 over {128, 256, 512}, clock-locked, off-mode fits/s:
/// ols_large_n 3686 / 3802 / 3831, ols_wide 20792 / 20746 / 20657 — flat
/// within ~1–3% and opposite preferences, so 256 keeps the balanced middle.
pub(crate) const PANEL_ROWS: usize = 256;

/// Caller-owned scratch passed into `fit_suff_stats_t_sq`. Built at the call site
/// by reborrowing fields of `SimWorkspace` directly — this is the form that
/// composes with NLL split-borrowing (a `&mut self` helper method would
/// borrow the whole workspace and conflict with the shared `x_full`/`y_full`
/// reads in the same loop iteration).
pub struct OlsScratch<'w> {
    pub fit_betas: &'w mut [f64],
    pub fit_var_diag: &'w mut [f64],
    pub fit_t_sq: &'w mut [f64],
    pub fit_u_scratch: &'w mut [f64],
    pub fit_factor: MatMut<'w, f64>,
    pub fit_rhs: MatMut<'w, f64>,
}

/// Borrowed view into the scratch produced by `fit_suff_stats_t_sq`.
/// Lifetime ties back to the workspace that owns
/// the storage.
pub struct OlsFitView<'a> {
    pub betas: &'a [f64],
    /// Length `t` — only the first `n_targets` entries are populated.
    pub var_diag: &'a [f64],
    /// Length `t`.
    pub t_sq: &'a [f64],
    /// `P × P` factor: the lower-triangular Cholesky factor `L` of `X'X`
    /// (`L · L' = X'X`), which is what `fit_suff_stats_t_sq` writes and
    /// posthoc consumes. Contents are valid only when `converged == true`. On the
    /// `n <= p || p == 0` early-return path the factor is not written, so it
    /// holds either zeros (first call after workspace construction) or stale
    /// data from a previous fit. Posthoc gates on `converged` so this is safe
    /// in practice; new consumers must do the same.
    pub factor: MatRef<'a, f64>,
    pub sigma_sq: f64,
    pub df_resid: u32,
    pub converged: bool,
    /// `‖y − X β̂‖²`. `NaN` on every non-converged / rank-deficient return.
    pub rss: f64,
    /// `Σ (yᵢ − ȳ)²`. `NaN` on every non-converged / rank-deficient return.
    /// Computed from the `sum_y` / `yty` running sums.
    pub sst: f64,
}

/// Caller-owned scratch for the sufficient-statistics OLS path, parallel in
/// shape to `OlsScratch`. Built at the call site by reborrowing fields of
/// `SimWorkspace` so NLL split-borrowing keeps it composable with the design-
/// matrix reads inside the same loop iteration.
pub struct OlsSuffStats<'w> {
    /// `P × P` accumulator for `X'X` (only the lower triangle is meaningful).
    pub xtx: MatMut<'w, f64>,
    /// Length `P` accumulator for `X'y`.
    pub xty: &'w mut [f64],
    /// Scalar `y'y`.
    pub yty: &'w mut f64,
    /// Σ yᵢ — drives `SST = Σyᵢ² − (Σyᵢ)²/n` at the batch site.
    pub sum_y: &'w mut f64,
    /// Total rows added so far.
    pub n_rows: &'w mut usize,
    /// Panel-widening scratch, len ≥ `min(block rows, PANEL_ROWS) · p` —
    /// workspace `panel_x` at the hot sites.
    pub panel_x: &'w mut [f64],
    /// y twin, len ≥ `min(block rows, PANEL_ROWS)` — workspace `panel_y`.
    pub panel_y: &'w mut [f64],
}

impl<'w> OlsSuffStats<'w> {
    /// Accumulate a contiguous block of rows. Writes only into the lower
    /// triangle of `xtx` (i ≥ j in `xtx[(i, j)]`); the Cholesky path reads
    /// `Side::Lower` only, so storing the upper triangle would be wasted work.
    ///
    /// Panel-GEMM (group B): each ≤PANEL_ROWS slice of the block is widened
    /// f32→f64 into `panel_x`/`panel_y` once, then X'X/X'y accumulate through
    /// faer GEMM (`Accum::Add`, `Par::Seq` — per-fit parallelism is the outer
    /// rayon loop). GEMM accumulation order, deliberately NOT the old per-row
    /// rank-1 triangle: the serial FP-add chain was the latency floor
    /// (mirrors glm.rs's X'WX conversion). `yty`/`sum_y` stay scalar in row
    /// order — bit-identical to the pre-panel loop.
    ///
    /// Caller's responsibility: `x_block.nrows() == y_block.len()` and the
    /// column count matches the workspace's predictor count.
    pub fn add_rows(&mut self, x_block: MatRef<'_, f32>, y_block: &[f32]) {
        debug_assert_eq!(x_block.nrows(), y_block.len());
        let p = self.xty.len();
        debug_assert_eq!(x_block.ncols(), p);
        let m = x_block.nrows();
        debug_assert!(self.panel_x.len() >= PANEL_ROWS.min(m) * p);
        debug_assert!(self.panel_y.len() >= PANEL_ROWS.min(m));

        let mut off = 0;
        while off < m {
            let rows = (m - off).min(PANEL_ROWS);
            // f32 data plane widens once per element into the f64 panel;
            // the old triangle re-widened each x value up to p times.
            // Column-major, leading dim = rows (re-packed densely per panel).
            for j in 0..p {
                let col = &mut self.panel_x[j * rows..(j + 1) * rows];
                for (i, v) in col.iter_mut().enumerate() {
                    *v = x_block[(off + i, j)] as f64;
                }
            }
            for i in 0..rows {
                let y_row = y_block[off + i] as f64;
                self.panel_y[i] = y_row;
                *self.yty += y_row * y_row;
                *self.sum_y += y_row;
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
        *self.n_rows += m;
    }
}

// ---------------------------------------------------------------------------
// Shared triangular-solve helper
// ---------------------------------------------------------------------------

/// Solve a triangular system `T·u = b` into `scratch` and return `‖u‖²`.
///
/// - `upper=false`: lower-triangular factor; row access `factor[(i, k)]`.
/// - `upper=true`:  upper-triangular factor; column access `factor[(k, i)]`.
///
/// Returns `f64::NAN` immediately on any near-zero diagonal element.
/// The `b` closure supplies the right-hand side: `b(i)` is called once per row.
///
/// This de-duplicates the forward-substitution + norm-squared accumulation
/// shared by `fit_suff_stats_t_sq` and `ols_contrast_t_sq` (both Cholesky
/// path, lower).
#[inline]
fn triangular_solve_norm_sq(
    factor: MatRef<'_, f64>,
    b: impl Fn(usize) -> f64,
    scratch: &mut [f64],
    p: usize,
    upper: bool,
) -> f64 {
    for v in &mut scratch[..p] {
        *v = 0.0;
    }
    for i in 0..p {
        let mut acc = b(i);
        for k in 0..i {
            acc -= if upper {
                factor[(k, i)]
            } else {
                factor[(i, k)]
            } * scratch[k];
        }
        let diag = factor[(i, i)];
        if diag.abs() < FLOAT_NEAR_ZERO {
            return f64::NAN;
        }
        scratch[i] = acc / diag;
    }
    let mut norm_sq = 0.0;
    for &v in &scratch[..p] {
        norm_sq += v * v;
    }
    norm_sq
}

/// NaN-fill the three scratch output slices (`betas[..p]`, `var_diag[..t]`,
/// `t_sq[..t]`) on the rank-deficient / early-return paths so callers see the
/// v1-contract NaN signal. Successful paths overwrite every populated slot.
/// Shared by the OLS fit and the GLM IRLS preamble.
#[inline]
pub(crate) fn nan_fill_ols_scratch(
    betas: &mut [f64],
    var_diag: &mut [f64],
    t_sq: &mut [f64],
    p: usize,
    t: usize,
) {
    betas[..p].fill(f64::NAN);
    var_diag[..t].fill(f64::NAN);
    t_sq[..t].fill(f64::NAN);
}

/// Build the canonical non-converged `OlsFitView`: all-NaN numerics,
/// `converged: false`, slices/factor passed through as the NaN-filled scratch.
/// `df_resid` is taken as an argument because the `n <= p` early-return sites
/// compute it with a saturating sub while the post-rank-check sites use an
/// unchecked `n - p` (where `n > p` already holds) — each caller keeps owning
/// its own expression so no panic path is reintroduced.
#[inline]
fn nonconverged_view<'a>(
    betas: &'a [f64],
    var_diag: &'a [f64],
    t_sq: &'a [f64],
    factor: MatRef<'a, f64>,
    df_resid: u32,
) -> OlsFitView<'a> {
    OlsFitView {
        betas,
        var_diag,
        t_sq,
        factor,
        sigma_sq: f64::NAN,
        df_resid,
        converged: false,
        rss: f64::NAN,
        sst: f64::NAN,
    }
}

/// Min/max diagonal-ratio rank-deficiency test on a Cholesky factor `L`
/// (`p × p`, lower-triangular). Returns `true` when `L` is degenerate — either
/// a non-positive max diagonal or `min|L_ii| < eps · max|L_ii|`. Catches the
/// near-singular grey zone faer's `Llt` accepts numerically but whose inverse
/// is essentially unbounded (e.g. collinear / duplicated columns).
///
/// `eps` is passed by the caller, not fixed here, because the threshold is
/// deliberately different across fits: OLS uses `1e-12` (caller-supplied
/// `eps_rank`), while the mixed models use `1e-8` — looser, because their
/// per-cluster outer-product downdates of `X'X` amplify near-singularity onto
/// the diagonal of `L`. Keeping `eps` at the call site preserves that contrast.
pub(crate) fn chol_rank_deficient(factor: MatRef<'_, f64>, p: usize, eps: f64) -> bool {
    let mut max_diag: f64 = 0.0;
    let mut min_diag: f64 = f64::INFINITY;
    for i in 0..p {
        let d = factor[(i, i)].abs();
        if d > max_diag {
            max_diag = d;
        }
        if d < min_diag {
            min_diag = d;
        }
    }
    max_diag <= 0.0 || min_diag < eps * max_diag
}

/// The production OLS fit: Cholesky on accumulated sufficient statistics.
/// Starts from `(X'X, X'y, y'y, n_rows)` instead of a raw `(X, y)` pair —
/// this is the cross-N reuse path: each successive call sees a strictly larger
/// `n_rows` without re-doing the O(N·p²) reduction from scratch.
///
/// Inputs:
/// - `xtx_lower`: lower triangle of `X'X` (read-only). Upper triangle is
///   ignored — `Cholesky(Side::Lower)` only reads `i ≥ j` entries.
/// - `xty`: length-`P` vector of `X'y`.
/// - `yty`: scalar `y'y`.
/// - `n_rows`: number of rows accumulated into `xtx`/`xty`/`yty`.
/// - `target_indices`: per-coefficient indices into β̂ to test.
/// - `eps_rank`: rank-deficiency threshold on `min|L_diag| / max|L_diag|`.
/// - `xtx_work`: `P × P` scratch buffer — `xtx_lower` is copied here because
///   faer's Cholesky reads the input matrix.
/// - `scratch`: caller-owned scratch from `SimWorkspace`.
#[expect(
    clippy::too_many_arguments,
    reason = "sufficient-statistics kernel; each arg is a distinct precomputed input"
)]
pub fn fit_suff_stats_t_sq<'a>(
    xtx_lower: MatRef<'_, f64>,
    xty: &[f64],
    yty: f64,
    sum_y: f64,
    n_rows: usize,
    target_indices: &[u32],
    eps_rank: f64,
    mut xtx_work: MatMut<'_, f64>,
    scratch: OlsScratch<'a>,
) -> OlsFitView<'a> {
    let p = xty.len();
    let t = target_indices.len();
    let n = n_rows;

    debug_assert_eq!(xtx_lower.nrows(), p);
    debug_assert_eq!(xtx_lower.ncols(), p);
    debug_assert_eq!(xtx_work.nrows(), p);
    debug_assert_eq!(xtx_work.ncols(), p);

    let OlsScratch {
        fit_betas,
        fit_var_diag,
        fit_t_sq,
        fit_u_scratch,
        mut fit_factor,
        mut fit_rhs,
    } = scratch;

    debug_assert!(p <= fit_betas.len(), "scratch sized for fewer predictors");
    debug_assert!(t <= fit_var_diag.len());
    debug_assert!(p <= fit_rhs.nrows(), "fit_rhs must hold at least p rows");

    // NaN-fill on the rank-deficient / early-return paths so callers see the
    // v1-contract NaN signal. Successful paths overwrite every populated slot.
    nan_fill_ols_scratch(fit_betas, fit_var_diag, fit_t_sq, p, t);

    if n <= p || p == 0 {
        return nonconverged_view(
            &fit_betas[..p],
            &fit_var_diag[..t],
            &fit_t_sq[..t],
            fit_factor.into_const(),
            n.saturating_sub(p) as u32,
        );
    }

    // Copy the lower triangle of X'X into the working buffer (faer's Cholesky
    // only reads `Side::Lower`, but it inspects all entries it reads — write
    // them all). Upper triangle is irrelevant; we leave it untouched.
    for j in 0..p {
        for i in j..p {
            xtx_work[(i, j)] = xtx_lower[(i, j)];
        }
    }

    // Cholesky factorization on the lower triangle.
    let chol = match xtx_work.rb().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            return nonconverged_view(
                &fit_betas[..p],
                &fit_var_diag[..t],
                &fit_t_sq[..t],
                fit_factor.into_const(),
                (n - p) as u32,
            );
        }
    };

    // Materialize L (owned Mat; strict upper triangle zeroed) and run the
    // rank-deficiency check on its diagonal via the min/max diagonal-ratio
    // rule. (`Llt::new(...)` already rejects strictly non-PD inputs; this
    // extra band catches the near-singular grey zone.)
    let l = chol.L();
    if chol_rank_deficient(l, p, eps_rank) {
        return nonconverged_view(
            &fit_betas[..p],
            &fit_var_diag[..t],
            &fit_t_sq[..t],
            fit_factor.into_const(),
            (n - p) as u32,
        );
    }

    // Copy L into the caller-owned `fit_factor` so the returned view borrows
    // workspace storage rather than the locally-owned `l` Mat.
    for j in 0..p {
        for i in 0..p {
            fit_factor[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
        }
    }

    // β̂ via two triangular solves: L · z = X'y; L' · β̂ = z. Use the top
    // `p` rows of `fit_rhs` as the rhs buffer.
    for j in 0..p {
        fit_rhs[(j, 0)] = xty[j];
    }
    use faer::linalg::solvers::Solve;
    chol.solve_in_place(fit_rhs.rb_mut().subrows_mut(0, p));
    for j in 0..p {
        fit_betas[j] = fit_rhs[(j, 0)];
    }

    // RSS = y'y - β̂' · X'y. Closed form — no residual sweep needed.
    let mut bty = 0.0;
    for j in 0..p {
        bty += fit_betas[j] * xty[j];
    }
    let rss = yty - bty;
    let df_resid = (n - p) as u32;
    let sigma_sq = rss / df_resid as f64;
    let sst = yty - (sum_y * sum_y) / n as f64;

    // For each target: forward-solve L · u = e_{tj} (read L[i, k] directly —
    // no transposed access — since L is already lower-triangular). Then
    // var_diag = σ̂² · ‖u‖² because (X'X)⁻¹ = L⁻ᵀ L⁻¹ and the diagonal entries
    // are ‖L⁻¹ e_j‖².
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let norm_sq = triangular_solve_norm_sq(
            fit_factor.rb(),
            |i| if i == tj { 1.0 } else { 0.0 },
            fit_u_scratch,
            p,
            false, // lower-triangular L: row access factor[(i, k)]
        );
        let vd = sigma_sq * norm_sq;
        fit_var_diag[out_idx] = vd;
        if vd > FLOAT_NEAR_ZERO && vd.is_finite() {
            let beta_j = fit_betas[tj];
            fit_t_sq[out_idx] = (beta_j * beta_j) / vd;
        } else {
            fit_t_sq[out_idx] = f64::NAN;
        }
    }

    OlsFitView {
        betas: &fit_betas[..p],
        var_diag: &fit_var_diag[..t],
        t_sq: &fit_t_sq[..t],
        factor: fit_factor.into_const(),
        sigma_sq,
        df_resid,
        converged: true,
        rss,
        sst,
    }
}

// ---------------------------------------------------------------------------
// Contrast t² — pairwise contrast β_p − β_n via Cholesky forward solve
// ---------------------------------------------------------------------------

/// Compute the Wald t² statistic for the pairwise contrast `β_p − β_n`.
///
/// Uses the lower-triangular Cholesky factor `L` of `X'X` (stored in the
/// `factor` field of `OlsFitView`) to compute:
///
/// ```text
/// c = e_p − e_n          (contrast vector, length p)
/// L · u = c               (forward solve)
/// var(β_p − β_n) = σ̂² · ‖u‖²
/// t² = (β̂_p − β̂_n)² / var
/// ```
///
/// Returns `NaN` on any numerical failure (non-converged fit, out-of-range
/// indices, near-zero variance).
///
/// `scratch` must be length ≥ `p`; it is overwritten and not read on output.
pub fn ols_contrast_t_sq(fit: &OlsFitView<'_>, p_col: u32, n_col: u32, scratch: &mut [f64]) -> f64 {
    if !fit.converged {
        return f64::NAN;
    }
    let p = fit.betas.len();
    let pc = p_col as usize;
    let nc = n_col as usize;
    if pc >= p || nc >= p || scratch.len() < p {
        return f64::NAN;
    }

    // Forward solve L · u = c where c = e_pc − e_nc.
    // L is lower-triangular (fit.factor).
    let norm_sq = triangular_solve_norm_sq(
        fit.factor,
        |i| {
            if i == pc {
                1.0
            } else if i == nc {
                -1.0
            } else {
                0.0
            }
        },
        scratch,
        p,
        false, // lower-triangular L: row access factor[(i, k)]
    );
    let var = fit.sigma_sq * norm_sq;
    if var <= FLOAT_NEAR_ZERO || !var.is_finite() {
        return f64::NAN;
    }

    let beta_diff = fit.betas[pc] - fit.betas[nc];
    (beta_diff * beta_diff) / var
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::SimWorkspace;
    use faer::Mat;

    fn suff_stats(ws: &mut SimWorkspace) -> OlsSuffStats<'_> {
        OlsSuffStats {
            xtx: ws.suff_xtx.as_mut(),
            xty: &mut ws.suff_xty,
            yty: &mut ws.suff_yty,
            sum_y: &mut ws.suff_sum_y,
            n_rows: &mut ws.suff_n_rows,
            panel_x: &mut ws.panel_x,
            panel_y: &mut ws.panel_y,
        }
    }

    fn build_x(n: usize, p: usize, mut fill: impl FnMut(usize, usize) -> f64) -> Mat<f32> {
        let mut m = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                m[(i, j)] = fill(i, j) as f32;
            }
        }
        m
    }

    // ---------------------------------------------------------------------
    // Suff-stats path tests
    // ---------------------------------------------------------------------

    /// EST-07: `OlsSuffStats::add_rows` is batch-split invariant — adding the
    /// rows in two segments accumulates the identical X'X / X'y / y'y / Σy /
    /// n_rows as adding them in one full pass. No hand-computed reference: the
    /// invariant compares the kernel against itself under a different split.
    #[test]
    fn suff_stats_batch_split_invariance() {
        let n = 5;
        let p = 3;
        let x = build_x(n, p, |i, j| ((i + 1) as f64).powi(j as i32));
        let y = [1.0_f32, 2.0, 3.0, 4.0, 5.0];

        // Path A: two-segment accumulation.
        let mut ws_split = SimWorkspace::new(n, p, 0, 0, None);
        ws_split.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws_split);
            s.add_rows(x.as_ref().subrows(0, 2), &y[0..2]);
            s.add_rows(x.as_ref().subrows(2, 3), &y[2..5]);
        }

        // Path B: single full accumulation.
        let mut ws_full = SimWorkspace::new(n, p, 0, 0, None);
        ws_full.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws_full);
            s.add_rows(x.as_ref(), &y);
        }

        // Lower triangle (the only entries `add_rows` writes) must agree.
        for j in 0..p {
            for i in j..p {
                assert!(
                    (ws_split.suff_xtx[(i, j)] - ws_full.suff_xtx[(i, j)]).abs() < 1e-12,
                    "xtx[{i},{j}] split-add != full-add"
                );
            }
        }
        for k in 0..p {
            assert!(
                (ws_split.suff_xty[k] - ws_full.suff_xty[k]).abs() < 1e-12,
                "xty[{k}] split != full"
            );
        }
        assert!(
            (ws_split.suff_yty - ws_full.suff_yty).abs() < 1e-12,
            "yty split != full"
        );
        assert!(
            (ws_split.suff_sum_y - ws_full.suff_sum_y).abs() < 1e-12,
            "sum_y split != full"
        );
        assert_eq!(ws_split.suff_n_rows, ws_full.suff_n_rows);
        assert_eq!(ws_split.suff_n_rows, n);
    }

    /// Regression net for the panel-GEMM rewrite (group B): `add_rows` must match
    /// the pre-panel per-row rank-1 triangle (inlined verbatim below as the
    /// oracle). X'X / X'y reassociate under the GEMM — band 1e-12 relative,
    /// measured max 2.0e-15 (first post-rewrite run); yty / sum_y stay scalar in
    /// the original row order and must be bit-identical. n = 611 crosses two
    /// PANEL_ROWS=256 boundaries plus a 99-row tail.
    #[test]
    fn add_rows_panel_matches_scalar_reference() {
        let n = 611;
        let p = 7;
        let x = build_x(n, p, |i, j| {
            ((((i * 13 + j * 29 + 3) % 47) as f64) / 11.0 - 2.0).sin()
        });
        let y: Vec<f32> = (0..n)
            .map(|i| ((((i * 31 + 7) % 53) as f32) / 9.0 - 2.5).cos())
            .collect();

        // Oracle: the pre-panel scalar accumulation, verbatim.
        let mut ref_xtx = Mat::<f64>::zeros(p, p);
        let mut ref_xty = vec![0.0f64; p];
        let (mut ref_yty, mut ref_sum_y) = (0.0f64, 0.0f64);
        for row in 0..n {
            let y_row = y[row] as f64;
            for j in 0..p {
                let x_rj = x[(row, j)] as f64;
                for i in j..p {
                    ref_xtx[(i, j)] += x[(row, i)] as f64 * x_rj;
                }
                ref_xty[j] += x_rj * y_row;
            }
            ref_yty += y_row * y_row;
            ref_sum_y += y_row;
        }

        let mut ws = SimWorkspace::new(n, p, 0, 0, None);
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws);
            s.add_rows(x.as_ref(), &y);
        }
        for j in 0..p {
            for i in j..p {
                let (got, want) = (ws.suff_xtx[(i, j)], ref_xtx[(i, j)]);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "xtx[{i},{j}] = {got}, scalar reference {want}"
                );
            }
        }
        #[allow(clippy::needless_range_loop)]
        for k in 0..p {
            assert!(
                (ws.suff_xty[k] - ref_xty[k]).abs() <= 1e-12 * ref_xty[k].abs().max(1.0),
                "xty[{k}] = {}, scalar reference {}",
                ws.suff_xty[k],
                ref_xty[k]
            );
        }
        assert_eq!(
            ws.suff_yty.to_bits(),
            ref_yty.to_bits(),
            "yty must stay bit-identical (scalar row-order pass)"
        );
        assert_eq!(
            ws.suff_sum_y.to_bits(),
            ref_sum_y.to_bits(),
            "sum_y must stay bit-identical"
        );
        assert_eq!(ws.suff_n_rows, n);
    }

    /// Golden values for the production OLS fit (numeric kernels pin golden
    /// values). Oracle hand-computed exactly from the closed-form normal
    /// equations on x = [0, 1, 2, 3], y = [1, 3, 4, 8] with intercept:
    /// X'X = [[4, 6], [6, 14]], X'y = [16, 35], y'y = 90, Σy = 16 →
    /// β̂ = (0.7, 2.2); residuals (0.3, 0.1, −1.1, 0.7) → RSS = 1.8;
    /// SST = 90 − 16²/4 = 26; σ̂² = RSS/2 = 0.9;
    /// (X'X)⁻¹ = [[0.7, −0.3], [−0.3, 0.2]] → var_diag = (0.63, 0.18);
    /// t² = (0.49/0.63, 4.84/0.18) = (7/9, 242/9).
    #[test]
    fn fit_suff_stats_golden_values() {
        let n = 4;
        let p = 2;
        let x = build_x(n, p, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let y = [1.0_f32, 3.0, 4.0, 8.0];
        let targets: Vec<u32> = vec![0, 1];

        let mut ws = SimWorkspace::new(n, p, 0, 0, None);
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws);
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let res = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &targets,
            1e-12,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(res.converged);
        assert_eq!(res.df_resid, 2);

        let golden_betas = [0.7, 2.2];
        let golden_var = [0.63, 0.18];
        let golden_t_sq = [7.0 / 9.0, 242.0 / 9.0];
        for (j, (&got, &want)) in res.betas.iter().zip(golden_betas.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "β̂[{j}] = {got}, golden {want}");
        }
        for k in 0..targets.len() {
            assert!(
                (res.var_diag[k] - golden_var[k]).abs() < 1e-9,
                "var_diag[{k}] = {}, golden {}",
                res.var_diag[k],
                golden_var[k]
            );
            assert!(
                (res.t_sq[k] - golden_t_sq[k]).abs() < 1e-9,
                "t²[{k}] = {}, golden {}",
                res.t_sq[k],
                golden_t_sq[k]
            );
        }
        assert!(
            (res.rss - 1.8).abs() < 1e-9,
            "rss = {}, golden 1.8",
            res.rss
        );
        assert!(
            (res.sst - 26.0).abs() < 1e-9,
            "sst = {}, golden 26.0",
            res.sst
        );
        assert!(
            (res.sigma_sq - 0.9).abs() < 1e-9,
            "σ̂² = {}, golden 0.9",
            res.sigma_sq
        );
    }

    /// `n ≤ p` early-return on the production path: non-converged view with
    /// all-NaN outputs (v1 NaN contract), including `rss`/`sst`.
    #[test]
    fn suff_stats_non_converged_when_n_le_p() {
        let p = 4;
        let n = 3; // n < p → underdetermined
        let x = build_x(n, p, |i, j| ((i + j) as f64).sin() + 1.0);
        let y: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut ws = SimWorkspace::new(p, p, 0, 0, None); // alloc for p rows
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws);
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let res = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[1, 2, 3],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(!res.converged, "n ≤ p must not converge");
        for v in res.t_sq.iter() {
            assert!(v.is_nan(), "t² must be NaN when n ≤ p");
        }
        for v in res.betas.iter() {
            assert!(v.is_nan(), "β̂ must be NaN when n ≤ p");
        }
        assert!(res.rss.is_nan(), "rss must be NaN when n ≤ p");
        assert!(res.sst.is_nan(), "sst must be NaN when n ≤ p");
    }

    // -----------------------------------------------------------------------
    // ols_contrast_t_sq unit tests
    // -----------------------------------------------------------------------

    /// EST-10: `ols_contrast_t_sq` is symmetric under swapping `p_col` and
    /// `n_col` — the beta difference is negated but squared away, and ‖L⁻¹c‖²
    /// is identical for c and −c. A broken kernel that forgot to square the
    /// numerator (or used a one-sided statistic) would fail. The result must
    /// also be a positive finite value, not a pinned number.
    #[test]
    fn ols_contrast_t_sq_is_symmetric() {
        let p = 3;
        let mut factor = Mat::<f64>::zeros(p, p);
        factor[(0, 0)] = 2.0;
        factor[(1, 0)] = 1.0;
        factor[(1, 1)] = 3.0;
        factor[(2, 1)] = 1.0;
        factor[(2, 2)] = 4.0;

        let betas = [0.5_f64, 1.2, -0.7];
        let var_diag = [0.0_f64; 3];
        let t_sq_dummy = [0.0_f64; 3];

        let fit = OlsFitView {
            betas: &betas,
            var_diag: &var_diag,
            t_sq: &t_sq_dummy,
            factor: factor.as_ref(),
            sigma_sq: 0.4,
            df_resid: 10,
            converged: true,
            rss: 0.0,
            sst: 0.0,
        };

        let mut scratch = vec![0.0_f64; p];
        let forward = ols_contrast_t_sq(&fit, 1, 2, &mut scratch);
        let reversed = ols_contrast_t_sq(&fit, 2, 1, &mut scratch);

        assert!(
            forward.is_finite() && forward > 0.0,
            "contrast t² must be positive finite"
        );
        assert!(
            (forward - reversed).abs() / forward.abs().max(1.0) < 1e-12,
            "contrast t² must be symmetric under p/n swap: {forward} vs {reversed}"
        );
    }

    /// Golden contrast t² — the symmetry test above is blind to a `−`→`+` swap in
    /// the β-difference and to `*`↔`/` in `(β_p−β_n)²/var` (both survive because the
    /// p/n swap cancels the operator change). Pin an exact value to close that gap.
    ///
    /// Reuses the `fit_suff_stats_golden_values` fixture (x = [0,1,2,3] with
    /// intercept, y = [1,3,4,8]) → β̂ = (0.7, 2.2), σ̂² = 0.9,
    /// (X'X)⁻¹ = [[0.7,−0.3],[−0.3,0.2]]. Contrast slope−intercept: c = (−1, 1),
    /// c'(X'X)⁻¹c = 1.5 → var = 0.9·1.5 = 1.35; β̂₁−β̂₀ = 1.5 → t² = 2.25/1.35 = 5/3.
    #[test]
    fn ols_contrast_t_sq_golden_value() {
        let n = 4;
        let p = 2;
        let x = build_x(n, p, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let y = [1.0_f32, 3.0, 4.0, 8.0];

        let mut ws = SimWorkspace::new(n, p, 0, 0, None);
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws);
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let res = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &[0, 1],
            1e-12,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(res.converged);

        let mut cscratch = vec![0.0_f64; p];
        let t_sq = ols_contrast_t_sq(&res, 1, 0, &mut cscratch);
        assert!(
            (t_sq - 5.0 / 3.0).abs() < 1e-9,
            "contrast t² = {t_sq}, golden 5/3"
        );
        // Symmetric under swap — same golden, different code path through the sign.
        let rev = ols_contrast_t_sq(&res, 0, 1, &mut cscratch);
        assert!(
            (rev - 5.0 / 3.0).abs() < 1e-9,
            "reversed t² = {rev}, golden 5/3"
        );
    }

    /// Non-converged fit must return NaN.
    #[test]
    fn contrast_t_sq_returns_nan_on_non_converged() {
        let p = 2;
        let factor = Mat::<f64>::zeros(p, p);
        let betas = [0.0_f64, 1.0];
        let var_diag = [0.0_f64; 2];
        let t_sq_dummy = [0.0_f64; 2];
        let fit = OlsFitView {
            betas: &betas,
            var_diag: &var_diag,
            t_sq: &t_sq_dummy,
            factor: factor.as_ref(),
            sigma_sq: 1.0,
            df_resid: 10,
            converged: false, // non-converged
            rss: f64::NAN,
            sst: f64::NAN,
        };
        let mut scratch = vec![0.0_f64; p];
        let got = ols_contrast_t_sq(&fit, 0, 1, &mut scratch);
        assert!(got.is_nan(), "non-converged fit must return NaN, got={got}");
    }

    #[test]
    fn suff_stats_rank_deficiency_detected() {
        // Two collinear columns — the suff-stats fit must classify this as non-converged.
        let n = 50;
        let p = 3;
        let x = build_x(n, p, |i, j| match j {
            0 => 1.0,
            1 => (i as f64) * 0.1,
            // Zero column → exactly-zero Cholesky pivot, robustly rank-deficient
            // in f32. (An exact-duplicate column leaves a ~1e-7 roundoff pivot
            // that f32 generation tips above eps_rank=1e-12 — the f32 LLT grey
            // zone the f32-overhaul plan flagged; a zero column avoids it.)
            _ => 0.0,
        });
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws_su = SimWorkspace::new(n, p, 0, 0, None);
        ws_su.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws_su);
            s.add_rows(x.as_ref(), &y);
        }
        let xtx_ref = ws_su.suff_xtx.as_ref();
        let xty_ref = ws_su.suff_xty.clone();
        let yty_val = ws_su.suff_yty;
        let sum_y_val = ws_su.suff_sum_y;
        let n_rows_val = ws_su.suff_n_rows;
        let scratch = OlsScratch {
            fit_betas: &mut ws_su.fit_betas,
            fit_var_diag: &mut ws_su.fit_var_diag,
            fit_t_sq: &mut ws_su.fit_t_sq,
            fit_u_scratch: &mut ws_su.fit_u_scratch,
            fit_factor: ws_su.fit_factor.as_mut(),
            fit_rhs: ws_su.fit_rhs.as_mut(),
        };
        let res_su = fit_suff_stats_t_sq(
            xtx_ref,
            &xty_ref,
            yty_val,
            sum_y_val,
            n_rows_val,
            &targets,
            1e-12,
            ws_su.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(
            !res_su.converged,
            "Cholesky must reject near-collinear design"
        );
        assert!(res_su.rss.is_nan(), "rss must be NaN on rank-deficient");
        assert!(res_su.sst.is_nan(), "sst must be NaN on rank-deficient");
        for v in res_su.t_sq.iter() {
            assert!(v.is_nan(), "rank-deficient t_sq must be NaN");
        }
        for v in res_su.betas.iter() {
            assert!(v.is_nan(), "rank-deficient betas must be NaN");
        }
    }

    /// Warm-path allocation guard for the production suff-stats OLS fit
    /// (`reset_suff_stats` → `add_rows` → `fit_suff_stats_t_sq`). Our side is
    /// zero-alloc by construction; the bound below is faer's `llt` + `L()`
    /// internals per fit, pinned so a faer upgrade that regresses the warm
    /// path fails loudly here instead of surfacing as a benchmark mystery.
    ///
    /// `#[ignore]` because `dhat::Profiler` measures process-wide allocations
    /// and must run single-threaded:
    ///   `cargo test -p engine-core fit_suff_stats_warm_path_bounded_alloc -- --ignored --test-threads=1`
    #[test]
    #[ignore]
    fn fit_suff_stats_warm_path_bounded_alloc() {
        // 2 blocks/fit measured on faer 0.24 (`llt` factor + `L()` internals);
        // no one-time block — the GEMM-backend lazy init lands in the warmup
        // call outside the profiler window.
        const FAER_LLT_BLOCKS_PER_FIT: usize = 2;
        const N_CALLS: usize = 100;
        const ONE_TIME: usize = 0;
        const BOUND: usize = FAER_LLT_BLOCKS_PER_FIT * N_CALLS + ONE_TIME;

        let n = 200;
        let p = 6;
        let x = build_x(n, p, |i, j| {
            if j == 0 {
                1.0
            } else {
                ((i * 7 + j * 13 + 5) % 23) as f64 / 5.0 - 2.0
            }
        });
        let y: Vec<f32> = (0..n).map(|i| ((i * 11) % 13) as f32 / 10.0).collect();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws = SimWorkspace::new(n, p, 0, 0, None);

        let run_fit = |ws: &mut SimWorkspace| {
            ws.reset_suff_stats();
            {
                let mut s = suff_stats(ws);
                s.add_rows(x.as_ref(), &y);
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
                &targets,
                1e-12,
                ws.suff_xtx_work.as_mut(),
                scratch,
            );
            assert!(fit.converged);
        };

        // Warm everything once outside the measured window.
        run_fit(&mut ws);

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            run_fit(&mut ws);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks as usize <= BOUND,
            "fit_suff_stats_t_sq allocated {} blocks across {} warm-path calls (expected ≤ {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }
}
