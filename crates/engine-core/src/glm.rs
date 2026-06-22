//! Adaptive IRLS logistic regression kernel; outputs z² = β̂²/Var(β̂) for hot-loop comparison against precomputed squared critical values from `critvals.rs` — no CDF calls, no SE sqrt.
//!
//! Hot-loop invariant: no z-CDF calls, no SE sqrt, no per-coefficient
//! square roots. Inference outputs `z_sq_j = β̂_j² / Var(β̂_j)` against the
//! precomputed `z_crit_sq` table built in `critvals.rs`.
//!
//! All per-fit buffers live in `SimWorkspace` (the `irls_*` fields, see workspace.rs).
//! The kernel takes a `GlmScratch<'w>` built inline at the call site (NLL split-borrow)
//! and returns a borrowed `GlmFitView<'a>`. No owned result struct.
//!
//! Algorithm — guards and tolerances are v1 parity:
//!   - Adaptive convergence: `|Δdeviance| < DEVIANCE_TOL = 1e-8`
//!   - Safety cap: `MAX_IRLS_ITERS = 50`
//!   - BETA_CAP divergence guard: `iter ≥ 3 ∧ ‖β‖_∞ > 30 → non-converged`
//!   - All-0 / all-1 short circuit
//!   - Post-fit saturation guard (50% of weights < 1e-5 ⇒ non-converged)
//!   - No step-halving: β_new is accepted directly (v1 parity; step-halving
//!     drifts power ~3.5% at N=50 — see the accept step in the IRLS loop)
//!
//! The start point deviates from v1 deliberately: the shipped hot loop passes
//! `beta_start = Some(spec.effect_sizes)` (Y is synthetic, so the true β on
//! the logit scale is known — the spec-derived truth-start seam, mirroring
//! `lmm::fit_lmm`'s `theta_start`). `None` keeps v1's β = 0 cold start
//! bit-identically. Either way the accept rule and the |Δdev| < 1e-8 fixpoint
//! are unchanged — only the path to it shortens.
//!
//! Working-response IRLS form and the canonical-link weights follow McCullagh &
//! Nelder (1989), *Generalized Linear Models*, 2nd ed., Chapman & Hall — as MN89.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::linalg::matmul::{matmul, triangular};
use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};

use crate::ols::nan_fill_ols_scratch;
use crate::FLOAT_NEAR_ZERO;

/// IRLS safety cap; v1 parity.
pub const MAX_IRLS_ITERS: u32 = 50;
/// Adaptive convergence tolerance on `|Δdeviance|`; v1 parity.
pub const DEVIANCE_TOL: f64 = 1e-8;
/// Divergence guard: any |β_j| > BETA_CAP at iter ≥ 3 marks non-converged.
pub const BETA_CAP: f64 = 30.0;
/// Floor on per-row IRLS weight `W_i = p_i (1-p_i)` to avoid division by zero
/// in the working response. v1 parity.
pub const WEIGHT_CLAMP: f64 = 1e-6;
/// Saturation post-fit guard: rows with `p_i(1-p_i) < SATURATION_W` count as
/// saturated. If the fraction exceeds `SATURATION_FRAC`, the fit is marked
/// non-converged. v1 parity.
pub const SATURATION_W: f64 = 1e-5;
pub const SATURATION_FRAC: f64 = 0.5;

/// Borrowed view into the workspace `irls_*` scratch produced by `glm_irls_fit`.
/// Lifetime ties back to the workspace.
///
/// `t_sq` field name reuses the OLS slot for uniformity — values are z² under
/// Logit (Wald-z²); the threshold comparison `stat_sq > crit_sq` is family-
/// agnostic so writeback code in `batch.rs` stays uniform.
pub struct GlmFitView<'a> {
    /// length P — fitted coefficients (NaN-filled on non-converged paths).
    pub betas: &'a [f64],
    /// length T — `((X'WX)⁻¹)_jj` for each target.
    pub var_diag: &'a [f64],
    /// length T — Wald z² for each target. Compared against `z_crit²` from
    /// `CritValueTable`.
    pub t_sq: &'a [f64],
    /// Cached lower-triangular Cholesky factor L of the last accepted X'WX.
    /// Valid only when `converged == true` (stale-or-zero otherwise — same
    /// staleness contract as `OlsFitView::factor`).
    pub l: MatRef<'a, f64>,
    pub n_iter: u32,
    pub converged: bool,
    /// Final-iteration Bernoulli deviance −2·Σ[y log p̂ + (1−y) log(1−p̂)].
    /// `NaN` on every non-converged / short-circuit return.
    pub deviance: f64,
    /// Null-model deviance −2·(Σy · log ȳ + (n − Σy) · log(1 − ȳ)).
    /// `NaN` whenever `Σy ∈ {0, n}` (the short-circuit path) or any other
    /// non-converged path.
    pub deviance_null: f64,
}

/// Caller-owned scratch borrowed from `SimWorkspace` field-by-field. Built
/// inline at the call site (NLL split-borrow with simultaneous shared borrows
/// of `ws.x_full` / `ws.y_full`). **Do not** wrap in a helper method
/// `ws.glm_scratch()` — that re-introduces the whole-struct exclusive borrow
/// problem (NLL cannot split borrow a method receiver from its fields).
pub struct GlmScratch<'w> {
    pub irls_eta: &'w mut [f64],
    pub irls_p: &'w mut [f64],
    pub irls_w: &'w mut [f64],
    pub irls_z: &'w mut [f64],
    pub irls_betas: &'w mut [f64],
    pub irls_betas_new: &'w mut [f64],
    pub irls_var_diag: &'w mut [f64],
    pub irls_t_sq: &'w mut [f64],
    pub irls_u_scratch: &'w mut [f64],
    pub irls_xtwx: MatMut<'w, f64>,
    pub irls_xtwz: &'w mut [f64],
    pub irls_l: MatMut<'w, f64>,
    /// Per-fit f64 copy of X (column-major, stride n); needs `len ≥ n·p`.
    pub irls_x_f64: &'w mut [f64],
    /// Per-iteration W∘X scratch (column-major, mirrors `irls_x_f64`); needs `len ≥ n·p`.
    pub irls_wx: &'w mut [f64],
}

// ---------------------------------------------------------------------------
// Numerically stable helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid. Mirrors v1's data_generation.cpp::generate_y_binary
/// branch. Production paths now compute `p` in the vectorized `simd_transcendental`
/// kernels (fit: `pw_and_log1pexp_sum`; generation: `sigmoid_fill`, ≤2 ULP of this
/// form); this stays as the libm reference for tests.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn sigmoid_stable(eta: f64) -> f64 {
    if eta >= 0.0 {
        let z = (-eta).exp();
        1.0 / (1.0 + z)
    } else {
        let z = eta.exp();
        z / (1.0 + z)
    }
}

// ---------------------------------------------------------------------------
// IRLS kernel
// ---------------------------------------------------------------------------

/// Fit a logistic regression via adaptive IRLS. All buffers live in `scratch`;
/// the returned view borrows from the same storage.
///
/// - `x`: `n × p` design (column-major faer).
/// - `y`: length `n`, values in {0.0, 1.0} (produced by `data_gen::generate_sim_data`).
/// - `target_indices`: per-coefficient indices to compute `z²` for.
/// - `beta_start`: `None` → β = 0 cold start (bit-identical pre-warm-start
///   path); `Some(β₀)` (length `p`) → spec-derived truth start — seeds β and
///   computes η = X·β₀ once. A per-scenario constant, so determinism and
///   chunk merging are unaffected.
/// - `scratch`: borrowed mutable slots from `SimWorkspace.irls_*`.
pub fn glm_irls_fit<'a>(
    x: MatRef<'_, f32>,
    y: &[f32],
    target_indices: &[u32],
    beta_start: Option<&[f64]>,
    scratch: GlmScratch<'a>,
) -> GlmFitView<'a> {
    let n = x.nrows();
    let p = x.ncols();
    let t = target_indices.len();

    debug_assert_eq!(n, y.len(), "glm_irls_fit: y length must match X.nrows()");

    let GlmScratch {
        irls_eta,
        irls_p,
        irls_w,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_var_diag,
        irls_t_sq,
        irls_u_scratch,
        mut irls_xtwx,
        irls_xtwz,
        mut irls_l,
        irls_x_f64,
        irls_wx,
    } = scratch;

    debug_assert!(p <= irls_betas.len(), "scratch sized for fewer predictors");
    debug_assert!(t <= irls_var_diag.len());
    debug_assert!(n <= irls_eta.len());
    debug_assert!(n * p <= irls_x_f64.len());

    // NaN-fill on early-return / non-converged paths so callers see the
    // v1-contract NaN signal. Successful paths overwrite every populated slot.
    nan_fill_ols_scratch(irls_betas, irls_var_diag, irls_t_sq, p, t);

    // Short-circuit: n ≤ p, or no predictors.
    if n <= p || p == 0 {
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
        };
    }

    // All-0 / all-1 short circuit. Without this the working response divides
    // by zero on iter 1 (W collapses, p collapses).
    let mut y_sum = 0.0;
    for &yi in &y[..n] {
        y_sum += yi as f64;
    }
    if y_sum <= 0.0 || y_sum >= n as f64 {
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
        };
    }

    // Null-model deviance — computed once, reused at the batch site for the
    // LRT. Stored in a local so the per-iteration deviance tracker doesn't
    // clobber it. The all-0 / all-1 branch above guarantees `y_bar ∉ {0, 1}`
    // here, so no `ln(0)` risk.
    let y_bar = y_sum / n as f64;
    let deviance_null = -2.0 * (y_sum * y_bar.ln() + (n as f64 - y_sum) * (1.0 - y_bar).ln());

    // Per-fit f64 copy of X (column-major, stride n). The IRLS hot loops
    // re-read X elementwise ~2× per iteration; through the f32 MatRef each
    // read pays a load + widen + bounds-check chain (measured ~33% of
    // glm_rare runtime in the X'WX build alone, scalar SSE). One faithful
    // copy turns those into contiguous f64 slice reads. f32→f64 is
    // value-exact and every consumer reads the same values in the same
    // order, so the fit is bit-identical. Placed after the short circuits
    // so the cheap exits don't pay the fill.
    let x_f64 = &mut irls_x_f64[..n * p];
    for j in 0..p {
        for i in 0..n {
            x_f64[j * n + i] = x[(i, j)] as f64;
        }
    }
    let x_f64: &[f64] = x_f64;

    // Seed β and η here; the IRLS loop carries η forward from each
    // iteration's post-accept recompute (the β-accept step) instead of
    // recomputing X·β at the top of every iteration.
    match beta_start {
        // Truth start: β ← β₀, η = X·β₀ computed once (O(np)). The η loop
        // matches the β-accept step's summation order (column-sweep axpy) so a warm fit
        // walks the same arithmetic the accept path uses — change together.
        Some(b0) => {
            debug_assert_eq!(b0.len(), p, "beta_start length must match X.ncols()");
            irls_betas[..p].copy_from_slice(&b0[..p]);
            irls_eta[..n].fill(0.0);
            for j in 0..p {
                let xj = &x_f64[j * n..(j + 1) * n];
                let b_j = irls_betas[j];
                for i in 0..n {
                    irls_eta[i] += xj[i] * b_j;
                }
            }
        }
        // Cold start: β ← 0, so η = X·β = 0.
        None => {
            irls_betas[..p].fill(0.0);
            irls_eta[..n].fill(0.0);
        }
    }

    // Cholesky factor of the last accepted X'WX. The lower-triangular L is
    // materialised once after the loop (deferred from per-iteration so each
    // IRLS step skips an owned-Mat allocation). `Llt` owns its storage, so
    // keeping the last one past the next iter's irls_xtwx rebuild is safe.
    let mut last_chol = None;

    let mut deviance_prev = f64::INFINITY;
    let mut deviance_final = f64::NAN;
    let mut converged = false;
    let mut had_pd_failure = false;
    let mut n_iter: u32 = 0;

    // IRLS: each pass is a weighted least squares solve
    // β_new = (X'WX)⁻¹ X'Wz on the working response z = η + (y − μ)/W with
    // weights W = diag(μ(1−μ)), μ = σ(η). Iterated to the |Δdeviance| fixpoint
    // it is the maximum-likelihood β. MN89 §4.
    // `0..=`: pass k's top checks convergence of the deviance that pass k−1's
    // solve produced (off the carried η), so the final solve still gets its
    // check without the extra pass buying another factorization. The deviance
    // moved from a post-accept `bernoulli_deviance` call to the top-of-pass
    // fused kernel — same values one pass later, no standalone `log1pexp` sweep
    // (group-C deviance fold; bit-identical, the fused Σ equals the lean one).
    for iter in 0..=MAX_IRLS_ITERS {
        // η = X · β is already in `irls_eta`: seeded to 0 (β = 0) or the
        // truth start before the loop, then refreshed by the accept step below
        // after each β update. Reusing it here is bit-identical to recomputing
        // X·β (same β, same summation order).

        // p, W, and Σ log1pexp(η) in one fused vectorized pass; the working
        // response z is a scalar follow-up (it carries the f32 y and a division,
        // no transcendental) that also folds in the Σ y·η deviance half.
        let lp_sum = crate::simd_transcendental::pw_and_log1pexp_sum(
            &irls_eta[..n],
            &mut irls_p[..n],
            &mut irls_w[..n],
        );
        let mut yeta = 0.0;
        for i in 0..n {
            let yi = y[i] as f64;
            yeta += yi * irls_eta[i];
            irls_z[i] = irls_eta[i] + (yi - irls_p[i]) / irls_w[i];
        }

        // Adaptive early exit on |Δdeviance|, where deviance is the
        // Bernoulli −2·log-lik `2·(Σ log1pexp(η) − Σ y·η)` at the CURRENT β —
        // the one the previous pass's solve accepted. Pass 0 sees the seed β,
        // whose deviance v1 never measured: skip check and tracker so the first
        // real comparison stays β₂-vs-β₁, exactly v1's sequence.
        if iter > 0 {
            let deviance = 2.0 * (lp_sum - yeta);
            deviance_final = deviance;
            if (deviance - deviance_prev).abs() < DEVIANCE_TOL {
                converged = true;
                break;
            }
            deviance_prev = deviance;
        }
        // Solve budget spent — the `..=` pass exists only for the check above.
        if iter == MAX_IRLS_ITERS {
            break;
        }
        n_iter = iter + 1;

        // WX = W∘X into irls_wx (column-major, mirrors x_f64), then
        // X′WX (lower triangle — Cholesky reads Side::Lower only) and X′Wz
        // through faer GEMM (`Par::Seq`; per-fit parallelism is the outer
        // rayon loop). GEMM accumulation order, deliberately NOT the old
        // per-entry row-order dots: the serial FP-add chain was the latency
        // floor on wide p (measured 0.94× glm_wide) — group-H result-moving
        // change.
        {
            let wx = &mut irls_wx[..n * p];
            for j in 0..p {
                let xj = &x_f64[j * n..(j + 1) * n];
                let wxj = &mut wx[j * n..(j + 1) * n];
                for i in 0..n {
                    wxj[i] = irls_w[i] * xj[i];
                }
            }
        }
        let wx_ref = MatRef::from_column_major_slice(&irls_wx[..n * p], n, p);
        let x_ref = MatRef::from_column_major_slice(x_f64, n, p);
        triangular::matmul(
            irls_xtwx.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            x_ref.transpose(),
            BlockStructure::Rectangular,
            wx_ref,
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        matmul(
            MatMut::from_column_major_slice_mut(&mut irls_xtwz[..p], p, 1),
            Accum::Replace,
            wx_ref.transpose(),
            MatRef::from_column_major_slice(&irls_z[..n], n, 1),
            1.0,
            Par::Seq,
        );

        // Cholesky of X'WX on the lower triangle. faer's high-level API
        // returns an owned factor (it does not consume irls_xtwx, which is
        // rebuilt from scratch each iter anyway). The factor drives the in-place
        // β solve below; its L is materialised once after the loop.
        let chol = match irls_xtwx.rb().llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(_) => {
                had_pd_failure = true;
                break;
            }
        };

        // Solve β_new = L⁻ᵀ L⁻¹ · X'Wz using chol.solve_in_place.
        // First write xtwz into a 1-col view; chol.solve_in_place expects a
        // MatMut. Since irls_xtwz is &mut [f64] of length p, build a
        // temporary MatMut via faer's `from_column_major_slice_mut`.
        {
            use faer::linalg::solvers::Solve;
            let mut rhs = MatMut::from_column_major_slice_mut(irls_xtwz, p, 1usize);
            chol.solve_in_place(rhs.rb_mut());
        }
        irls_betas_new[..p].copy_from_slice(&irls_xtwz[..p]);

        // Stash this iteration's factor; the cached L is materialised once
        // after the loop (only the converged path reads it).
        last_chol = Some(chol);

        // Non-finite guard on β_new.
        let mut all_finite = true;
        for &b in &irls_betas_new[..p] {
            if !b.is_finite() {
                all_finite = false;
                break;
            }
        }
        if !all_finite {
            break;
        }

        // Accept β_new unconditionally and compute new deviance.
        //
        // β_new is accepted unconditionally — no step-halving: it causes
        // systematic power drift at small N (measured ~3.5% at N=50). The
        // DEVIANCE_TOL early-exit and MAX_IRLS_ITERS cap are sufficient
        // divergence guards.
        irls_betas[..p].copy_from_slice(&irls_betas_new[..p]);
        // η = X·β as a column sweep (axpy over contiguous x_f64 columns) — each
        // η_i still accumulates in the same j order from 0.0, bit-identical to
        // the strided per-row form. Mirrors the truth-start seed loop above —
        // change together.
        irls_eta[..n].fill(0.0);
        for j in 0..p {
            let xj = &x_f64[j * n..(j + 1) * n];
            let b_j = irls_betas[j];
            for i in 0..n {
                irls_eta[i] += xj[i] * b_j;
            }
        }

        // BETA_CAP divergence guard at iter ≥ 3. Fires before the next
        // pass's convergence check — a capped β never reports converged,
        // matching v1's check order.
        if iter >= 3 {
            let mut max_abs: f64 = 0.0;
            for &b in &irls_betas[..p] {
                let ab = b.abs();
                if ab > max_abs {
                    max_abs = ab;
                }
            }
            if max_abs > BETA_CAP {
                break;
            }
        }
    }

    if had_pd_failure {
        converged = false;
    }

    // Post-fit saturation guard. Evaluate p_i = σ(η_i) from the final irls_eta
    // to catch the case where the loop early-exits but β has drifted into a
    // saturated region.
    if converged {
        // irls_w already holds the FINAL η's weights: convergence only breaks
        // right after the top-of-pass fused kernel refilled p/W from the carried
        // η — no recompute needed. The clamp floor (WEIGHT_CLAMP = 1e-6) sits
        // below SATURATION_W (1e-5), so `w < SATURATION_W` is equivalent to the
        // raw `p(1-p) < SATURATION_W` test the scalar guard used.
        let saturated = irls_w[..n]
            .iter()
            .filter(|&&w_i| w_i < SATURATION_W)
            .count();
        if (saturated as f64) / (n as f64) > SATURATION_FRAC {
            converged = false;
        }
    }

    // If not converged: NaN the inference outputs but leave betas as fitted
    // (matches v1 which returns whatever β the loop left + NaN inference).
    if !converged {
        // Partial re-NaN: inference outputs only — `betas` are deliberately
        // left at the loop's last fitted value (v1 parity), so the 3-array
        // `nan_fill_ols_scratch` must NOT be used here.
        irls_var_diag[..t].fill(f64::NAN);
        irls_t_sq[..t].fill(f64::NAN);
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
        };
    }

    // Materialise the cached lower-triangular L of the last accepted X'WX —
    // once, now that the fit has converged (deferred from the IRLS loop). On
    // the converged path `last_chol` is always `Some`: `converged` is only set
    // after a successful factorization stashed it.
    if let Some(chol) = last_chol {
        let l = chol.L();
        for j in 0..p {
            for i in 0..p {
                irls_l[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
            }
        }
    }

    // Per-target Var(β̂_j) = ((X'WX)⁻¹)_jj via forward solve on L:
    //   L · u = e_{tj} → var_diag = ‖u‖²
    // (No σ̂² scaling — under Bernoulli the score covariance is (X'WX)⁻¹.)
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        irls_u_scratch[..p].fill(0.0);
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= irls_l[(i, k)] * irls_u_scratch[k];
            }
            let l_ii = irls_l[(i, i)];
            irls_u_scratch[i] = if l_ii.abs() < FLOAT_NEAR_ZERO {
                f64::NAN
            } else {
                acc / l_ii
            };
        }
        let mut norm_sq = 0.0;
        for &v in &irls_u_scratch[..p] {
            norm_sq += v * v;
        }
        irls_var_diag[out_idx] = norm_sq;
        if norm_sq > FLOAT_NEAR_ZERO && norm_sq.is_finite() {
            let beta_j = irls_betas[tj];
            irls_t_sq[out_idx] = (beta_j * beta_j) / norm_sq;
        } else {
            irls_t_sq[out_idx] = f64::NAN;
        }
    }

    GlmFitView {
        betas: &irls_betas[..p],
        var_diag: &irls_var_diag[..t],
        t_sq: &irls_t_sq[..t],
        l: irls_l.into_const(),
        n_iter,
        converged: true,
        deviance: deviance_final,
        deviance_null,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_gen::generate_sim_data;
    use crate::spec::{
        CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
        OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
    };
    use crate::workspace::SimWorkspace;
    use faer::Mat;

    /// Build a `GlmScratch` borrowing every IRLS field of `ws`. Used by tests
    /// to avoid duplicating the inline struct literal at each call site.
    fn glm_scratch(ws: &mut SimWorkspace) -> GlmScratch<'_> {
        GlmScratch {
            irls_eta: &mut ws.irls_eta,
            irls_p: &mut ws.irls_p,
            irls_w: &mut ws.irls_w,
            irls_z: &mut ws.irls_z,
            irls_betas: &mut ws.irls_betas,
            irls_betas_new: &mut ws.irls_betas_new,
            irls_var_diag: &mut ws.irls_var_diag,
            irls_t_sq: &mut ws.irls_t_sq,
            irls_u_scratch: &mut ws.irls_u_scratch,
            irls_xtwx: ws.irls_xtwx.as_mut(),
            irls_xtwz: &mut ws.irls_xtwz,
            irls_l: ws.irls_l.as_mut(),
            irls_x_f64: &mut ws.irls_x_f64,
            irls_wx: &mut ws.irls_wx,
        }
    }

    /// Helper: copy `ws.x_full[..n]` and `ws.y_full[..n]` out into owned
    /// `(Mat<f32>, Vec<f32>)` so the caller can re-borrow `ws` mutably for
    /// the glm scratch without violating aliasing.
    fn copy_xy(ws: &SimWorkspace, n: usize, p: usize) -> (Mat<f32>, Vec<f32>) {
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x[(i, j)] = ws.x_full[(i, j)];
            }
        }
        let y: Vec<f32> = ws.y_full[..n].to_vec();
        (x, y)
    }

    /// Build a Logit spec with `n_non_factor` continuous predictors,
    /// `effect_sizes` of length `1 + n_non_factor` (intercept + per-predictor).
    fn logit_spec(n_non_factor: u32, effect_sizes: Vec<f64>) -> SimulationSpec {
        let p = (n_non_factor as usize) * (n_non_factor as usize);
        let mut corr = vec![0.0; p];
        for j in 0..(n_non_factor as usize) {
            corr[j * n_non_factor as usize + j] = 1.0;
        }
        let intercept = effect_sizes[0];
        SimulationSpec {
            n_non_factor,
            n_factor_dummies: 0,
            correlation: corr,
            var_types: vec![Distribution::Normal; n_non_factor as usize],
            var_pinned: vec![],
            var_params: vec![0.0; n_non_factor as usize],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes,
            target_indices: (1..=n_non_factor).collect(),
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
            intercept,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    /// EST-15: IRLS converges within MAX_IRLS_ITERS on well-behaved binary
    /// data with non-trivial signal, returning finite β̂ / z². No recovery
    /// value is pinned — only that the iteration terminates as *converged*
    /// with a finite iteration count and finite inference.
    #[test]
    fn glm_converges_on_separable_signal() {
        let intercept = (0.3_f64 / 0.7).ln();
        let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
        let n = 5000;
        let p = 4;
        let mut ws = SimWorkspace::new(n, p, 3, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let (x, y) = copy_xy(&ws, n, p);

        let targets: Vec<u32> = vec![0, 1, 2, 3];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(fit.converged, "IRLS should converge on well-behaved data");
        assert!(
            fit.n_iter > 0 && fit.n_iter <= MAX_IRLS_ITERS,
            "n_iter {} must be in (0, MAX_IRLS_ITERS={}]",
            fit.n_iter,
            MAX_IRLS_ITERS
        );
        for &b in fit.betas.iter() {
            assert!(b.is_finite(), "β̂ must be finite on converged fit");
        }
        for &t in fit.t_sq.iter() {
            // Finiteness only: direction + magnitude are already pinned by the
            // sibling `glm_wald_z_sq_direction_and_ballpark` (same spec/seed,
            // asserts z²[β₁] > 50 at n=5000). A tautological `>= 0.0` on a
            // squared quantity adds nothing beyond the finite check here.
            assert!(t.is_finite(), "z² must be finite on converged fit");
        }
    }

    #[test]
    fn glm_all_zero_y_short_circuits() {
        let n = 100;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) / (n as f32) - 0.5;
        }
        let y = vec![0.0f32; n];
        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(!fit.converged);
        assert_eq!(fit.n_iter, 0);
    }

    #[test]
    fn glm_all_one_y_short_circuits() {
        let n = 100;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) / (n as f32) - 0.5;
        }
        let y = vec![1.0f32; n];
        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(!fit.converged);
        assert_eq!(fit.n_iter, 0);
    }

    #[test]
    fn glm_rank_deficient_design() {
        // X with two identical columns → X'WX is singular.
        let n = 100;
        let p = 3;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            // Column 1 and column 2 identical.
            x[(i, 1)] = ((i as f32) / (n as f32)) - 0.5;
            x[(i, 2)] = x[(i, 1)];
        }
        // Build y with mixed 0/1 to avoid the all-0/all-1 short circuit.
        let y: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mut ws = SimWorkspace::new(n, p, 0, 0, None);
        let targets: Vec<u32> = vec![1, 2];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(
            !fit.converged,
            "rank-deficient design must report non-converged"
        );
    }

    /// EST-16: GLM Wald z² is NaN on a non-converged fit (the variance is not
    /// recoverable). Error path for the z² shape rule — a broken kernel that
    /// emitted a finite garbage z² when the fit failed would be caught.
    #[test]
    fn glm_z_sq_nan_on_non_converged() {
        // All-zero y short-circuits to non-converged.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) / (n as f32) - 0.5;
        }
        let y = vec![0.0f32; n];
        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(!fit.converged);
        for &t in fit.t_sq.iter() {
            assert!(t.is_nan(), "z² must be NaN on non-converged fit, got {t}");
        }
    }

    #[test]
    fn glm_deviance_nan_on_non_converged() {
        // All-0 y short-circuit → non-converged.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) / (n as f32) - 0.5;
        }
        let y = vec![0.0f32; n];
        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(!fit.converged);
        assert!(
            fit.deviance.is_nan(),
            "deviance must be NaN on non-converged"
        );
        assert!(
            fit.deviance_null.is_nan(),
            "deviance_null must be NaN on all-0 short-circuit (sum_y = 0)"
        );
    }

    #[test]
    fn glm_separation_marks_non_converged() {
        // Build a fully-separated dataset: y = (x > 0). Logistic regression
        // diverges (β → ∞); BETA_CAP or saturation guard should fire.
        let n = 200;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            x[(i, 0)] = 1.0;
            let xi = (i as f32 - n as f32 / 2.0) / 10.0; // wide span
            x[(i, 1)] = xi;
            y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
        }
        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(
            !fit.converged,
            "fully separated data must report non-converged"
        );
    }

    /// W1 truth-start: warm (β₀ = spec.effect_sizes) and cold (β = 0) starts
    /// on the same bytes converge to the same |Δdev| < DEVIANCE_TOL fixpoint —
    /// β̂ within 1e-5 abs, z = √z² within 1e-4 abs (the campaign parity
    /// floors) — and the warm path spends no more iterations than the cold
    /// one. A warm start that shifted the fixpoint (rather than the path to
    /// it) or slowed convergence would fail here.
    #[test]
    fn glm_truth_start_matches_cold_fixpoint() {
        let intercept = (0.3_f64 / 0.7).ln();
        let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
        let n = 1000;
        let p = 4;
        let mut ws = SimWorkspace::new(n, p, 3, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let (x, y) = copy_xy(&ws, n, p);
        let targets: Vec<u32> = vec![0, 1, 2, 3];

        let cold = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(cold.converged, "cold fit must converge");
        let cold_betas = cold.betas.to_vec();
        let cold_t_sq = cold.t_sq.to_vec();
        let cold_iters = cold.n_iter;

        let warm = glm_irls_fit(
            x.as_ref(),
            &y,
            &targets,
            Some(&spec.effect_sizes),
            glm_scratch(&mut ws),
        );
        assert!(warm.converged, "warm fit must converge");
        for (j, (&bw, &bc)) in warm.betas.iter().zip(&cold_betas).enumerate() {
            assert!(
                (bw - bc).abs() < 1e-5,
                "β̂[{j}]: warm {bw} vs cold {bc} exceeds 1e-5"
            );
        }
        for (t, (&tw, &tc)) in warm.t_sq.iter().zip(&cold_t_sq).enumerate() {
            assert!(
                (tw.sqrt() - tc.sqrt()).abs() < 1e-4,
                "z[{t}]: warm {} vs cold {} exceeds 1e-4",
                tw.sqrt(),
                tc.sqrt()
            );
        }
        assert!(
            warm.n_iter <= cold_iters,
            "warm n_iter {} must not exceed cold {cold_iters}",
            warm.n_iter
        );
    }

    /// Warm-path allocation guard for the IRLS fit (mirrors `ols.rs`'s
    /// `fit_suff_stats_warm_path_bounded_alloc`). Per fit the allocations are
    /// faer's `Llt` internals for the in-place β solve (one per IRLS
    /// iteration) plus a single post-loop `.L()` that exposes the cached factor
    /// `L`. On faer 0.24 `.L()` returns a borrow (`MatRef`) instead of an owned
    /// `Mat`, so that post-loop materialisation no longer allocates: this fixture
    /// converges in 3 iterations and now allocates 8 blocks/fit (800 over
    /// `N_CALLS`), down from 9/fit on faer 0.20 (itself down from 12/fit before
    /// the in-loop deferral).
    ///
    /// `BOUND` locks the measured warm-path block count for this fixture; it is
    /// iteration-count-dependent, so a change here flags either a new
    /// per-iteration allocation or a shift in convergence behaviour. If a future
    /// faer version changes its Cholesky internals, update the bound — do not
    /// relax it to whatever passes today.
    ///
    /// `#[ignore]` because `dhat::Profiler` measures process-wide allocations
    /// and other concurrent tests in the same binary contaminate the count.
    /// Run explicitly with:
    ///   `cargo test -p engine-core glm_fit_warm_path_bounded_alloc -- --ignored --test-threads=1`
    #[test]
    #[ignore]
    fn glm_fit_warm_path_bounded_alloc() {
        const N: usize = 1000;
        const P: usize = 4;
        const N_CALLS: usize = 100;
        // 8 blocks/fit × 100 calls on faer 0.24 (was 9/fit on 0.20; the `.L()`
        // borrow dropped the post-loop factor materialisation). Re-measured
        // after the truth start landed: this well-signalled fixture converges
        // in the same iteration count warm as cold, so the count is unchanged
        // — the warm start pays off on slow-converging (e.g. rare-events)
        // fits, not here.
        const BOUND: u64 = 800;

        let intercept = (0.3_f64 / 0.7).ln();
        let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
        let mut ws = SimWorkspace::new(N, P, 3, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let (x, y) = copy_xy(&ws, N, P);
        let targets: Vec<u32> = vec![1, 2, 3];

        // Warmup outside the profiler window; assert convergence so the bench
        // measures the converged path (the only one that caches L). Truth-
        // started like the shipped hot loop — the bound is iteration-count-
        // dependent, so the fixture must walk the same warm path.
        let warm = glm_irls_fit(
            x.as_ref(),
            &y,
            &targets,
            Some(&spec.effect_sizes),
            glm_scratch(&mut ws),
        );
        assert!(warm.converged, "bench fixture must converge");

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            let _ = glm_irls_fit(
                x.as_ref(),
                &y,
                &targets,
                Some(&spec.effect_sizes),
                glm_scratch(&mut ws),
            );
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "glm_irls_fit allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }

    // -----------------------------------------------------------------
    // C6 — GLM deviance_null golden value (external oracle: R glm()$null.deviance)
    // -----------------------------------------------------------------

    #[test]
    fn glm_deviance_null_golden_value() {
        // null deviance = -2*(y_sum*ln(y_bar) + (n-y_sum)*ln(1-y_bar))
        // y_sum=40, n=100, y_bar=0.4
        // → expected = -2*(40*ln(0.4) + 60*ln(0.6)) ≈ 134.6023334
        // R: glm(c(rep(1,40),rep(0,60)) ~ 1, family=binomial)$null.deviance = 134.6023
        //
        // y pattern: every 5th group, first 2 are 1 and last 3 are 0.
        // This gives exactly 40 ones spread throughout, so there is no separation
        // between y and the linearly-increasing x predictor.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f32) / (n as f32) - 0.5; // arbitrary predictor
        }
        // y[i] = 1 for i % 5 < 2, else 0 → exactly 40 ones, not separated from x.
        let mut y = vec![0.0f32; n];
        for (i, v) in y.iter_mut().enumerate() {
            if i % 5 < 2 {
                *v = 1.0;
            }
        }

        let mut ws = SimWorkspace::new(n, p, 1, 0, None);
        let targets: Vec<u32> = vec![1];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(fit.converged, "must converge: non-separated y pattern");

        // External oracle: R glm(y ~ x, family=binomial)$null.deviance with y_sum=40/n=100
        // null deviance = -2*(40*ln(0.4) + 60*ln(0.6)) = 134.6023334 (depends only on y_bar)
        let expected = 134.6023334f64;
        let abs_err = (fit.deviance_null - expected).abs();
        assert!(
            abs_err < 0.001,
            "deviance_null = {}, expected {expected}, err = {abs_err}",
            fit.deviance_null
        );
    }

    // -----------------------------------------------------------------
    // C11 — GLM Wald z² direction and ballpark (n=5000, β₁=0.5)
    // -----------------------------------------------------------------

    #[test]
    fn glm_wald_z_sq_direction_and_ballpark() {
        // Extends glm_converges_on_separable_signal with direction + z² band assertions.
        // At n=5000 with β₁=0.5, β₂=-0.3, β₃=0.0, seed=42 (deterministic), the
        // fitted β̂ should be in the correct direction and the Wald z² for β₁ should
        // be in a realistic range (analytic SE ≈ 0.028 → z² ≈ 300 at n=5000).
        let intercept = (0.3_f64 / 0.7).ln();
        let spec = logit_spec(3, vec![intercept, 0.5, -0.3, 0.0]);
        let n = 5000;
        let p = 4;
        let mut ws = SimWorkspace::new(n, p, 3, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let (x, y) = copy_xy(&ws, n, p);
        let targets: Vec<u32> = vec![0, 1, 2, 3];
        let fit = glm_irls_fit(x.as_ref(), &y, &targets, None, glm_scratch(&mut ws));
        assert!(fit.converged, "IRLS should converge on well-behaved data");

        // β̂₁ should be near +0.5 (positive direction).
        assert!(
            fit.betas[1] > 0.35 && fit.betas[1] < 0.65,
            "β̂₁ = {} expected near 0.5",
            fit.betas[1]
        );
        // β̂₂ should be near -0.3 (negative direction).
        assert!(
            fit.betas[2] > -0.45 && fit.betas[2] < -0.15,
            "β̂₂ = {} expected near -0.3",
            fit.betas[2]
        );
        // Wald z² for target 1 (β₁): at n=5000 with β₁=0.5, z² should be substantial.
        // A biased IRLS returning z²≈0 or z²>1e6 would fail this bound.
        assert!(
            fit.t_sq[1] > 50.0 && fit.t_sq[1] < 2000.0,
            "z²[β₁] = {} expected in (50, 2000) at n=5000",
            fit.t_sq[1]
        );
    }
}
