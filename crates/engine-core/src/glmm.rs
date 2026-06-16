//! Clustered-logistic GLMM kernel (spec §3.4): glmer-faithful nAGQ=1. The outer
//! BOBYQA optimizes [θ | β] jointly; for each candidate a penalized-IRLS inner
//! loop solves the conditional modes ũ (β fixed), and the objective is the
//! Laplace deviance d(y,ũ) + ‖ũ‖² + log|L|². RE design Z and Λ_θ are dense
//! (spec-sanctioned within the regime; the block/sparse backend is the D3-0
//! hedge, not built). All scratch lives in `GlmmWorkspace`, allocated once per
//! (spec, max_n) shape — the warm path is zero-alloc (Bobyqa::new once).
//!
//! No σ² scale: binomial dispersion is fixed at 1, so D̂ = Λ̂Λ̂′ directly.

use bobyqa::{Bobyqa, Config, Status};
use faer::{Mat, MatRef};

use crate::lmm::{LmmGroupings, PIN_THETA, RHO_BEGIN, RHO_END, THETA0, THETA_TRUTH_FLOOR};

/// PIRLS inner-loop caps — the same as glm.rs's IRLS (PIRLS *is* that IRLS plus
/// the +I ridge).
pub const PIRLS_MAX_ITERS: usize = 50;
/// Adaptive PIRLS exit on |Δ penalized-deviance|, relative to the objective
/// scale (lme4's pwrss discipline): converged when
/// `|Δpen| < PIRLS_TOL_REL · (1 + |pen|)`. The penalized deviance is O(n), so
/// the former 1e-8 absolute gate demanded ~1e-12 *relative* accuracy and spent
/// 2–4 extra inner iterations per solve buying precision the outer BOBYQA
/// never sees (group-C result-moving change).
pub const PIRLS_TOL_REL: f64 = 1e-6;
/// Wide finite β box for the joint BOBYQA. Bound to `glm::BETA_CAP` (the log-odds
/// divergence guard, same magnitude) so the box and the cap can never drift apart.
pub const BETA_BOX: f64 = crate::glm::BETA_CAP;

/// Per-fit GLMM result (mirrors `LmmFit`; no σ² — dispersion is fixed at 1).
pub struct GlmmFit {
    pub converged: bool,
    /// 0 = interior, 1 = ≥1 diagonal θ pinned (converged), 2 = optimizer/Schur
    /// failure (non-converged).
    pub boundary_hit: u8,
    /// Bit k set iff diagonal variance component k pinned (order
    /// [intercept, slope_0, …, extra_1, …]). Mirrors `LmmFit.pinned_components`.
    pub pinned_components: u32,
    pub n_eval: usize,
    /// Estimated random-intercept variance D̂[0][0] (NaN on non-converged).
    pub tau_squared_hat: f64,
    /// Joint Wald-χ² over `target_indices` (NaN when empty / non-converged).
    pub joint_t_sq: f64,
}

/// All GLMM solver scratch — allocated once per (spec, max_n) shape.
pub struct GlmmWorkspace {
    pub groupings: LmmGroupings, // reused RE structure (estimator-agnostic)
    pub k: usize,                // total RE columns (groupings.k_total)
    pub p: usize,                // fixed-effect predictors
    pub n_theta: usize,
    pub z: Mat<f64>,             // max_n × k dense RE design (built per (spec,N) in Task 4)
    pub m: Mat<f64>,             // max_n × k = ZΛ (rebuilt per BOBYQA eval)
    pub solver: Bobyqa,          // sized n_theta + p
    pub params: Vec<f64>,        // [θ | β]
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub theta_truth: Vec<f64>,   // truth-start θ (= for_cluster_spec recipe)
    // PIRLS scratch (sized max_n / k):
    pub eta: Vec<f64>,
    pub prob: Vec<f64>,
    pub w: Vec<f64>,
    pub eta_fixed: Vec<f64>,     // Σ_j x·β, hoisted out of the PIRLS iteration (β fixed within a solve)
    pub m_buf: Vec<f64>,         // n×q_p row-major mᵢ = Λ_p'·zᵢ (blocked path) — filled once per PIRLS solve
    pub z_buf: Vec<f64>,         // n×(q_p−1) row-major f64 copy of x[:, slope_cols] — filled once per fit
    pub mu: Vec<f64>,            // (Mu)ᵢ per iteration via GEMV; overwritten in place by the IRLS residual W·Mu + (y−p) before the RHS GEMV
    pub u: Vec<f64>,
    pub u_seed: Vec<f64>,        // within-fit û warm-start incumbent (Phase 3); RESET to 0 each fit_glmm — never carried across fits
    pub a: Mat<f64>,             // k × k  M'WM + I
    pub wm: Mat<f64>,            // max_n × k = W∘M scratch for the dense-Gram GEMM (rebuilt per PIRLS iteration)
    pub a_rhs: Vec<f64>,         // length k
    pub a_blocks: Vec<f64>,      // s · q_p² packed per-cluster q_p×q_p blocks (no-extras path; Σ wᵢmᵢmᵢ'+I then Crout L)
    pub lam: Vec<f64>,           // q_p × q_p primary Λ_p scratch (row-major)
    // inference scratch:
    pub xtwx: Mat<f64>,          // p × p
    pub xtwm: Mat<f64>,          // p × k
    pub ainv_mtwx: Mat<f64>,     // k × p  = A⁻¹ M'WX
    pub schur: Mat<f64>,         // p × p  X'WX − X'WM A⁻¹ M'WX
    pub betas: Vec<f64>,         // length p (copied from params[n_theta..])
    pub var_diag: Vec<f64>,      // length p
    pub t_sq: Vec<f64>,          // length p
    pub fwd_solve: Vec<f64>,     // length p; Var(β̂)_jj forward-solve scratch (per-target)
    // joint Wald scratch (reuse lme::joint_wald_chi_sq):
    pub joint_k_inv: Mat<f64>,
    pub joint_sigma_t_chol: Mat<f64>,
    pub joint_rhs: Vec<f64>,
}

impl GlmmWorkspace {
    /// Build the GLMM workspace for a Glm+cluster spec. `slope_cols` are the
    /// x_full indices of the primary slopes (`spec.cluster_slope_design_cols`).
    pub fn for_cluster_spec(
        p: usize,
        cluster: &engine_contract::ClusterSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        let groupings = LmmGroupings::from_cluster_spec(cluster, max_n, slope_cols);
        let k = groupings.k_total;
        let n_theta = groupings.n_theta();
        let q = groupings.primary_q;
        let n_primary = groupings.n_primary;

        // θ truth-start = vech(chol(D_rel)), shared verbatim with the LMM via
        // lmm::cluster_theta_truth — the recipe is already σ=1 there.
        let theta_truth = crate::lmm::cluster_theta_truth(cluster);

        // Bounds: θ part from blind_theta_and_bounds; β part = [−BETA_BOX, BETA_BOX].
        let (theta0, mut lower, mut upper) = groupings.blind_theta_and_bounds();
        let mut params = theta0;
        params.extend(std::iter::repeat(0.0).take(p)); // β cold default; overwritten at fit
        lower.extend(std::iter::repeat(-BETA_BOX).take(p));
        upper.extend(std::iter::repeat(BETA_BOX).take(p));

        // ρ_begin ≤ RHO_BEGIN and ≤ 0.1·min diagonal θ₀ (mirror for_cluster_spec)
        // so the unit/truth start is not projected onto a bound (§3.7).
        let min_diag = groupings
            .diagonal_theta()
            .iter()
            .map(|&i| theta_truth[i].max(THETA_TRUTH_FLOOR))
            .fold(f64::INFINITY, f64::min);
        let rho_begin = (0.1 * min_diag).min(RHO_BEGIN);
        let config = Config {
            rho_begin,
            rho_end: RHO_END,
            ..Config::new(n_theta + p)
        };

        GlmmWorkspace {
            groupings,
            k,
            p,
            n_theta,
            z: Mat::zeros(max_n, k.max(1)),
            m: Mat::zeros(max_n, k.max(1)),
            solver: Bobyqa::new(n_theta + p, config)
                .expect("BOBYQA config constants are valid by construction"),
            params,
            lower,
            upper,
            theta_truth,
            eta: vec![0.0; max_n],
            prob: vec![0.0; max_n],
            w: vec![0.0; max_n],
            eta_fixed: vec![0.0; max_n],
            m_buf: vec![0.0; max_n * q],
            z_buf: vec![0.0; max_n * (q - 1)],
            mu: vec![0.0; max_n],
            u: vec![0.0; k.max(1)],
            u_seed: vec![0.0; k.max(1)],
            a: Mat::zeros(k.max(1), k.max(1)),
            wm: Mat::zeros(max_n, k.max(1)),
            a_rhs: vec![0.0; k.max(1)],
            a_blocks: vec![0.0; (q * q * n_primary).max(1)],
            lam: vec![0.0; q * q],
            xtwx: Mat::zeros(p, p),
            xtwm: Mat::zeros(p, k.max(1)),
            ainv_mtwx: Mat::zeros(k.max(1), p),
            schur: Mat::zeros(p, p),
            betas: vec![0.0; p],
            var_diag: vec![0.0; p],
            t_sq: vec![0.0; p],
            fwd_solve: vec![0.0; p],
            joint_k_inv: Mat::zeros(p, p),
            joint_sigma_t_chol: Mat::zeros(p, p),
            joint_rhs: vec![0.0; p],
        }
    }
}

/// Dispatch filter: Some iff Glm + cluster present. Mirrors `build_lmm_workspace`.
pub fn build_glmm_workspace(
    spec: &crate::spec::SimulationSpec,
    max_n: usize,
    n_predictors: usize,
) -> Option<Box<GlmmWorkspace>> {
    let cluster = spec.cluster.as_ref()?;
    if spec.estimator != engine_contract::EstimatorSpec::Glm {
        return None;
    }
    let slope_cols: Vec<usize> = spec
        .cluster_slope_design_cols
        .iter()
        .map(|&c| c as usize)
        .collect();
    // `test_formula` reduced fit: the FIXED design fits only `spec.fit_columns`
    // (ascending kernel cols, intercept always present). Size the β-dimension `p`
    // to the reduced count so the joint [θ|β] BOBYQA and every p-sized inference
    // scratch match. The RE design Z and θ-truth stay FULL — `slope_cols` index
    // the full generation design, and a random slope may reference a fixed term
    // the test formula drops. Empty / covers-all ⇒ full (current behaviour). The
    // batch GLMM branch gathers the matching reduced X/β/targets per (sim, N).
    let p_fit = if !spec.fit_columns.is_empty() && spec.fit_columns.len() < n_predictors {
        spec.fit_columns.len()
    } else {
        n_predictors
    };
    Some(Box::new(GlmmWorkspace::for_cluster_spec(
        p_fit,
        cluster,
        max_n,
        &slope_cols,
    )))
}

// Kernel — written as borrow-split FREE fns so the Task-5 BOBYQA closure can call
// them on destructured workspace fields without re-borrowing the whole workspace.

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. q ≤ MAX_PRIMARY_Q (8).
/// Mirrors the inline Crout in `lmm::reml_deviance`'s family loop — change together.
fn glmm_block_chol(blk: &mut [f64], q: usize) -> bool {
    for j in 0..q {
        let mut d = blk[j * q + j];
        for k in 0..j { d -= blk[j * q + k] * blk[j * q + k]; }
        if !(d.is_finite() && d > 0.0) { return false; }
        let l = d.sqrt();
        blk[j * q + j] = l;
        for i in (j + 1)..q {
            let mut v = blk[i * q + j];
            for k in 0..j { v -= blk[i * q + k] * blk[j * q + k]; }
            blk[i * q + j] = v / l;
        }
    }
    true
}

/// Solve `L Lᵀ x = b` in place (`b` overwritten with `x`) for the `q×q` lower
/// factor `l` produced by `glmm_block_chol` (row-major, diagonal = L pivots).
/// Forward `L y = b` then back `Lᵀ x = y`.
fn glmm_block_solve(l: &[f64], q: usize, b: &mut [f64]) {
    for r in 0..q {
        let mut v = b[r];
        for c in 0..r { v -= l[r * q + c] * b[c]; }
        b[r] = v / l[r * q + r];
    }
    for r in (0..q).rev() {
        let mut v = b[r];
        for c in (r + 1)..q { v -= l[c * q + r] * b[c]; }
        b[r] = v / l[r * q + r];
    }
}

/// Build the dense RE design Z (`max_n × k`, level-major) for one dataset.
///
/// Layout mirrors `LmmGroupings`'s RE-column convention (`from_cluster_spec`):
/// the primary block is `q_p · n_primary` wide, level-major within each
/// component (`[intercept 0..S | slope_0 … | slope_{q-2}]` → at level `lvl`,
/// component `c`, the column is `lvl·q_p + c`), then each extra grouping's
/// indicator columns at its ABSOLUTE `extra_offsets[e]` (already includes the
/// primary block width — do not add it again). `slope_cols` index `x`.
pub(crate) fn build_z(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    slope_cols: &[usize],
    n: usize,
) {
    let g = &ws.groupings;
    let q = g.primary_q;
    for c in 0..ws.k { for i in 0..n { ws.z[(i, c)] = 0.0; } }
    for i in 0..n {
        let lvl = cluster_ids[i] as usize;
        let base = lvl * q;
        ws.z[(i, base)] = 1.0; // intercept
        for (d, &sc) in slope_cols.iter().enumerate() {
            ws.z[(i, base + 1 + d)] = x[(i, sc)] as f64;
        }
    }
    for (e, ids) in extra_ids.iter().enumerate() {
        let off = g.extra_offsets[e]; // ABSOLUTE — do not add primary width again
        for i in 0..n {
            ws.z[(i, off + ids[i] as usize)] = 1.0;
        }
    }
}

/// Per-fit f64 widening of the primary-slope columns: `z_buf[i·(q−1)+d] =
/// x[i, slope_cols[d]]`. θ/β change per BOBYQA eval but `x` is fixed per fit,
/// so this hoists every f32 MatRef load out of the per-solve M fill — the fill
/// becomes a pure contiguous-f64 product. f32→f64 widening is value-exact, so
/// bit-identity is preserved. No-op at q_p = 1 (no slope columns).
fn fill_z_f64(g: &LmmGroupings, x: MatRef<f32>, z_buf: &mut [f64], n: usize) {
    let q = g.primary_q;
    for i in 0..n {
        for d in 0..q - 1 {
            z_buf[i * (q - 1) + d] = x[(i, g.primary_slope_cols[d])] as f64;
        }
    }
}

/// Form M = ZΛ in place, with Λ block-diagonal: a shared lower-triangular
/// primary block `Λ_p` (q_p×q_p) repeated per primary level, then one scalar
/// θ_e per extra grouping. `Λ_p` is the column-major vech θ prefix expanded by
/// `lmm::primary_lambda` (row-major lower-tri storage, so `lam[r*q + c]` is its
/// (r,c) entry). Each extra grouping's columns scale by its θ scalar.
pub(crate) fn apply_lambda(
    groupings: &LmmGroupings,
    params: &[f64],
    z: MatRef<f64>,
    m: &mut Mat<f64>,
    lam: &mut [f64],
    n: usize,
) {
    let q = groupings.primary_q;
    let s = groupings.n_primary;
    crate::lmm::primary_lambda(&params[..groupings.n_theta()], q, lam);
    for lvl in 0..s {
        let base = lvl * q;
        for i in 0..n {
            for c in 0..q {
                let mut acc = 0.0;
                for r in c..q {
                    acc += z[(i, base + r)] * lam[r * q + c];
                }
                m[(i, base + c)] = acc;
            }
        }
    }
    let base_theta = q * (q + 1) / 2;
    // Each extra grouping owns a CONTIGUOUS column block at its ABSOLUTE
    // `extra_offsets[e]`, scaled by its own scalar θ. Span the block by the
    // grouping's OWN width — NOT the gap to the next declaration's offset:
    // `extra_offsets` is non-monotonic (a nested grouping always sits at the low
    // `prim_width` slot, so a crossed-before-nested declaration makes offsets
    // decrease), so a "scale up to the next offset" loop empties one block and
    // over-scales another. A nested grouping spans `n_primary · nested_per_parent`
    // child columns; a crossed grouping spans its stored level count.
    for (e, &off) in groupings.extra_offsets.iter().enumerate() {
        let theta_e = params[base_theta + e];
        let width = if groupings.nested_theta == Some(base_theta + e) {
            s * groupings.nested_per_parent
        } else {
            groupings
                .crossed
                .iter()
                .find(|&&(ti, _)| ti == base_theta + e)
                .map(|&(_, cnt)| cnt)
                .expect("an extra grouping is either nested or crossed")
        };
        for col in off..off + width {
            for i in 0..n {
                m[(i, col)] = z[(i, col)] * theta_e;
            }
        }
    }
}

/// Penalized-IRLS inner solve: conditional modes ũ (β fixed) by Fisher scoring
/// on the penalized binomial likelihood with M = ZΛ the scaled RE design and a
/// +I ridge (the standard nAGQ=1 reparameterization — u ~ N(0, I)). At each
/// step A = M'WM + I and the IRLS RHS `M'(W·Mu + (y − p))` give the next u via a
/// dense Cholesky solve. Returns (deviance `2·d`, ‖ũ‖², `log|A|` at the converged
/// iterate, converged); a Cholesky failure surfaces as `(NaN, NaN, NaN, false)`.
/// `log|A|` is read off the same converged factor that solved for the final u —
/// the caller need not re-factor A (faer `llt` is deterministic).
/// Mu, A = M'(W∘M) + I, and the RHS all go through faer GEMM (`Par::Seq`) with
/// `wm` as the W∘M scratch — GEMM accumulation order, deliberately NOT the old
/// per-entry i-order dots (the serial FP-add chain was the dense path's latency
/// floor; group-H result-moving change). `eta_fixed`/`mu` are caller-owned
/// length-n scratch. `A` is left holding the FINAL-iterate A for the caller.
/// Iterates from the caller-provided `u` (the warm-start seed); the caller owns
/// resetting it per fit.
#[allow(clippy::too_many_arguments)]
fn pirls_solve(
    k: usize, p: usize, m: MatRef<f64>, x: MatRef<f32>, y: &[f32], beta: &[f64],
    eta: &mut [f64], prob: &mut [f64], w: &mut [f64], u: &mut [f64],
    eta_fixed: &mut [f64], mu: &mut [f64], wm: &mut Mat<f64>,
    a: &mut Mat<f64>, a_rhs: &mut [f64], n: usize,
) -> (f64, f64, f64, bool) {
    use faer::linalg::matmul::triangular::BlockStructure;
    use faer::linalg::matmul::{matmul, triangular};
    use faer::linalg::solvers::Solve;
    use faer::{Accum, MatMut, Par};
    // m arrives max_n × k with the first n rows live; GEMM needs exact dims.
    let m = m.subrows(0, n);
    // β is fixed within this solve, so η_fixed,ᵢ = Σ_j x[i,j]·β[j] is invariant
    // across PIRLS iterations — compute it once.
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p { e += x[(i, j)] as f64 * beta[j]; }
        eta_fixed[i] = e;
    }
    let mut pen_prev = f64::INFINITY;
    let mut converged = false;
    let mut dev = f64::NAN;
    let mut pen = f64::NAN;
    let mut logdet = 0.0;
    for _ in 0..PIRLS_MAX_ITERS {
        // (Mu)ᵢ once per iteration via GEMV — feeds both the η pass and,
        // residual-folded in place below, the IRLS RHS.
        matmul(
            MatMut::from_column_major_slice_mut(&mut mu[..n], n, 1),
            Accum::Replace, m, MatRef::from_column_major_slice(&u[..k], k, 1),
            1.0, Par::Seq,
        );
        let mut yeta = 0.0;
        for i in 0..n {
            let e = eta_fixed[i] + mu[i];
            eta[i] = e;
            yeta += y[i] as f64 * e;
        }
        let lp_sum =
            crate::simd_transcendental::pw_and_log1pexp_sum(&eta[..n], &mut prob[..n], &mut w[..n]);
        dev = 2.0 * (lp_sum - yeta);
        // WM = W∘M, then A = M′·(WM) + I — lower triangle only (the Cholesky
        // below reads Side::Lower).
        for c in 0..k {
            for i in 0..n { wm[(i, c)] = w[i] * m[(i, c)]; }
        }
        triangular::matmul(
            a.as_mut(), BlockStructure::TriangularLower, Accum::Replace,
            m.transpose(), BlockStructure::Rectangular,
            wm.as_ref().subrows(0, n), BlockStructure::Rectangular,
            1.0, Par::Seq,
        );
        for r in 0..k { a[(r, r)] += 1.0; }
        // IRLS RHS M′(W·Mu + (y − p)): fold the residual into mu in place
        // (mu is dead until next iteration's refill), then one GEMV.
        for i in 0..n {
            mu[i] = w[i] * mu[i] + (y[i] as f64 - prob[i]);
        }
        matmul(
            MatMut::from_column_major_slice_mut(&mut a_rhs[..k], k, 1),
            Accum::Replace, m.transpose(),
            MatRef::from_column_major_slice(&mu[..n], n, 1), 1.0, Par::Seq,
        );
        let ac = match a.as_ref().llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(_) => return (f64::NAN, f64::NAN, f64::NAN, false),
        };
        // Solve in place on the caller's a_rhs scratch (1-col view) — no
        // per-iteration alloc; the solve itself is unchanged.
        let rhs = MatMut::from_column_major_slice_mut(&mut a_rhs[..k], k, 1usize);
        ac.solve_in_place(rhs);
        pen = 0.0;
        for c in 0..k { u[c] = a_rhs[c]; pen += u[c] * u[c]; }
        let penalized = dev + pen;
        if (penalized - pen_prev).abs() < PIRLS_TOL_REL * (1.0 + penalized.abs()) {
            converged = true;
            // log|A| off the factor that just solved for the final u (= chol of the
            // final-iterate A the caller would otherwise re-factor). Reusing it
            // drops one O(k³) factorization per Laplace eval.
            for r in 0..k { logdet += ac.L()[(r, r)].ln(); }
            break;
        }
        pen_prev = penalized;
    }
    (dev, pen, logdet, converged)
}

/// Block-diagonal PIRLS for the no-extras regime (`groupings.extra_offsets` empty):
/// `A = M'WM + I` is `s` independent `q_p×q_p` blocks because each row loads exactly
/// one cluster's `q_p` columns. `m_buf` (row-major n×q_p) holds `mᵢ = Λ_p'·zᵢ`
/// (`zᵢ = [1, x[i, slope_cols]]`, slope columns pre-widened per fit into `z_buf`
/// by `fill_z_f64`), filled once per solve since Λ and x are fixed within one.
/// The η-pass forms ηᵢ and the deviance from it; the scatter-pass accumulates
/// `wᵢ·mᵢmᵢ'` into cluster `i`'s block plus `mᵢ·(yᵢ−pᵢ)` into its RHS — keeping
/// the dense path's `O(n·k²)` Gram/RHS collapsed to `O(n·q_p²)`. Then per block: `rhs_f = (A_f−I)·u_f + g_f`, Crout factor (log|A|
/// off the pivots), solve `u_f`. NOT bit-identical to `pirls_solve` (reordered
/// accumulation) but the same estimator. Mirrors `pirls_solve`'s half-step
/// (w/A from the pre-update u, u updated after) so the two agree to FP error.
/// `lam` is Λ_p row-major (`lam[r·q+c]`) from `primary_lambda`. Leaves `a_blocks`
/// FACTORED (per-block L of the final iterate) for `blocked_schur_fill` to reuse,
/// and eta/prob/w/u filled. Returns `(dev, ‖u‖², log|A|, converged)`; a non-PD
/// block ⇒ `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (the
/// warm-start seed); the caller owns resetting it per fit.
#[allow(clippy::too_many_arguments)]
fn pirls_solve_blocked(
    g: &crate::lmm::LmmGroupings, cluster_ids: &[u32], x: MatRef<f32>, y: &[f32],
    beta: &[f64], lam: &[f64], z_buf: &[f64], m_buf: &mut [f64],
    eta: &mut [f64], prob: &mut [f64], w: &mut [f64],
    u: &mut [f64], eta_fixed: &mut [f64], a_blocks: &mut [f64], a_rhs: &mut [f64],
    n: usize,
) -> (f64, f64, f64, bool) {
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p { e += x[(i, j)] as f64 * beta[j]; }
        eta_fixed[i] = e;
    }
    // M = ZΛ_p (mᵢ = Λ_p'·zᵢ, zᵢ = [1, x[i, slope_cols]] pre-widened into z_buf
    // per fit) is invariant within one solve — Λ and x are fixed; the iteration
    // only mutates u/η/prob/w/blocks. Fill it once per solve. Measured on the
    // glmm_slope profile (2026-06): the former per-row recompute closure was
    // ~57% of fit runtime, >90% of that MatRef indexing / bounds checks / 64-byte
    // return-by-value copies rather than FMA — buffering removes the overhead,
    // not the math. Bit-identical: the same `Σ_{r≥c} z_r·lam[r·q+c]` reduction
    // runs once per (i,c) in the same inner order, and both consumers below read
    // the same f64 values in the same order.
    for i in 0..n {
        for c in 0..q {
            let mut acc = 0.0;
            for r in c..q {
                let zr = if r == 0 { 1.0 } else { z_buf[i * (q - 1) + (r - 1)] };
                acc += zr * lam[r * q + c];
            }
            m_buf[i * q + c] = acc;
        }
    }
    let mut pen_prev = f64::INFINITY;
    let mut converged = false;
    let (mut dev, mut pen, mut logdet) = (f64::NAN, f64::NAN, 0.0);
    for _ in 0..PIRLS_MAX_ITERS {
        for v in a_blocks[..s * q * q].iter_mut() { *v = 0.0; }
        for v in a_rhs[..k].iter_mut() { *v = 0.0; }
        // Loop-split (Phase 4): the former interleaved transcendental+scatter sweep
        // becomes η-pass → SIMD pass → scatter-pass so the transcendental runs
        // vectorized over a materialized η[] with no gather/scatter data deps.
        // --- pass 1: η-pass (scalar gather): form ηᵢ, accumulate Σ y·η ---
        let mut yeta = 0.0;
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let ubase = cluster_ids[i] as usize * q;
            let mut e = eta_fixed[i];
            for c in 0..q { e += m_row[c] * u[ubase + c]; }
            eta[i] = e;
            yeta += y[i] as f64 * e;
        }
        // --- pass 2: SIMD pass: η[] → prob[]/w[]; Σ log1pexp (lane-wise reduce) ---
        let lp_sum =
            crate::simd_transcendental::pw_and_log1pexp_sum(&eta[..n], &mut prob[..n], &mut w[..n]);
        dev = 2.0 * (lp_sum - yeta);
        // --- pass 3: scatter-pass (scalar): wᵢmᵢmᵢ' and (yᵢ−pᵢ)·mᵢ into the blocks ---
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let f = cluster_ids[i] as usize;
            let ubase = f * q;
            let ablk = f * q * q;
            let wi = w[i];
            let resid = y[i] as f64 - prob[i];
            for r in 0..q {
                a_rhs[ubase + r] += m_row[r] * resid;
                let wr = wi * m_row[r];
                for c in 0..=r { a_blocks[ablk + r * q + c] += wr * m_row[c]; }
            }
        }
        // --- per block: rhs_f = (A_f−I)·u_old_f + g_f ; +I ; factor ; solve ---
        logdet = 0.0;
        pen = 0.0;
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
            // (A_f − I)·u_old_f added to g_f (in a_rhs), using the still-unfactored
            // symmetric lower triangle.
            for r in 0..q {
                let mut acc = a_rhs[ubase + r];
                for c in 0..q {
                    let (hi, lo) = if r >= c { (r, c) } else { (c, r) };
                    acc += a_blocks[ablk + hi * q + lo] * u[ubase + c];
                }
                a_rhs[ubase + r] = acc;
            }
            for r in 0..q { a_blocks[ablk + r * q + r] += 1.0; }
            if !glmm_block_chol(&mut a_blocks[ablk..ablk + q * q], q) {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            for r in 0..q { logdet += a_blocks[ablk + r * q + r].ln(); }
            // solve u_new_f = A_f⁻¹ rhs_f in place (rhs lives in a_rhs[ubase..], copy to u).
            for r in 0..q { u[ubase + r] = a_rhs[ubase + r]; }
            glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut u[ubase..ubase + q]);
            for r in 0..q { pen += u[ubase + r] * u[ubase + r]; }
        }
        let penalized = dev + pen;
        if (penalized - pen_prev).abs() < PIRLS_TOL_REL * (1.0 + penalized.abs()) {
            converged = true;
            break;
        }
        pen_prev = penalized;
    }
    (dev, pen, logdet, converged)
}

/// Laplace deviance at (θ, β): rebuild M = ZΛ, solve the PIRLS conditional
/// modes, then return `d(y,ũ) + ‖ũ‖² + log|A|` (A = M'WM + I at ũ). The +I in A
/// is the same ridge the penalty `‖ũ‖²` carries — this is glmer's nAGQ=1 Laplace
/// objective. Non-convergence / Cholesky failure ⇒ `f64::INFINITY` (the module's
/// failure surface, mirrors `lmm::reml_deviance`). `pirls_solve` returns `log|A|`
/// off its converged factor, so there is no re-factor here.
/// The blocked branch requires `z_buf` pre-filled for this fit's `x`
/// (`fill_z_f64`); the dense branch ignores `z_buf`/`m_buf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_deviance(
    groupings: &LmmGroupings, params: &[f64], z: MatRef<f64>, m: &mut Mat<f64>,
    lam: &mut [f64], z_buf: &[f64], m_buf: &mut [f64], x: MatRef<f32>, y: &[f32],
    cluster_ids: &[u32], eta: &mut [f64], prob: &mut [f64], w: &mut [f64], u: &mut [f64],
    eta_fixed: &mut [f64], mu: &mut [f64], wm: &mut Mat<f64>, a: &mut Mat<f64>, a_rhs: &mut [f64],
    a_blocks: &mut [f64], p: usize, n: usize,
) -> f64 {
    let k = groupings.k_total;
    let n_theta = groupings.n_theta();
    let (dev, pen, logdet, conv) = if groupings.extra_offsets.is_empty() {
        // No extras ⇒ A is block-diagonal: reconstruct mᵢ per row, never build Z/M.
        crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, lam);
        pirls_solve_blocked(
            groupings, cluster_ids, x, y, &params[n_theta..n_theta + p], lam, z_buf, m_buf,
            eta, prob, w, u, eta_fixed, a_blocks, a_rhs, n,
        )
    } else {
        // Crossed/nested ⇒ A genuinely dense: unchanged dense path.
        apply_lambda(groupings, params, z, m, lam, n);
        pirls_solve(
            k, p, m.as_ref(), x, y, &params[n_theta..n_theta + p],
            eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, n,
        )
    };
    if !conv || !dev.is_finite() { return f64::INFINITY; }
    dev + pen + 2.0 * logdet
}

/// Workspace-bound wrapper for `laplace_deviance`: copies `params` into the
/// workspace, then destructures it into disjoint borrows (z read; m/lam/a/etc.
/// written) for the borrow-split kernel. Test-only entry point — the production
/// fit (`fit_glmm`) destructures the workspace and calls `laplace_deviance`
/// directly (the BOBYQA closure and the pinned-γ̂ re-eval both inline it), so this
/// exists purely to drive the deviance from a `&[f64]` in tests.
#[cfg(test)]
pub(crate) fn glmm_laplace_deviance(
    params: &[f64], ws: &mut GlmmWorkspace, x: MatRef<f32>, y: &[f32], cluster_ids: &[u32], n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    let GlmmWorkspace {
        groupings, params: prm, p, z, m, lam, z_buf, m_buf, eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, a_blocks, ..
    } = ws;
    laplace_deviance(
        groupings, &prm[..], z.as_ref(), m, lam, z_buf, m_buf, x, y, cluster_ids,
        eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, a_blocks, *p, n,
    )
}

/// Fit the clustered-logistic GLMM. `ws.z` must already be built (build_z) for
/// this (X, ids, N). `beta_start` = spec.effect_sizes. Writes β̂/Var/z² into
/// ws.{betas,var_diag,t_sq}; returns the GlmmFit summary.
pub fn fit_glmm(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    beta_start: &[f64],
    n: usize,
) -> GlmmFit {
    let (k, p, n_theta) = (ws.k, ws.p, ws.n_theta);

    // γ₀ = [θ₀ | β₀].
    match theta_start {
        Some(ts) => for (t, &v) in ws.params[..n_theta].iter_mut().zip(ts) {
            *t = v.max(THETA_TRUTH_FLOOR);
        },
        None => for t in ws.params[..n_theta].iter_mut() { *t = THETA0; },
    }
    for (j, &b) in beta_start.iter().enumerate().take(p) {
        ws.params[n_theta + j] = b.clamp(-BETA_BOX, BETA_BOX);
    }

    // Within-fit warm-start resets per fit — the incumbent is NEVER carried across
    // fits (that is the rejected cross-sim warm-start; it would break the
    // plan_chunks/run_chunk merge and same-seed reproducibility).
    for v in ws.u_seed[..k].iter_mut() { *v = 0.0; }

    // Joint BOBYQA over [θ | β]. Borrow-split mirrors `glmm_laplace_deviance`:
    // `solver` held by `minimize`; the closure calls the shared `laplace_deviance`
    // on the disjoint scratch fields (groupings read; m/lam/eta/prob/w/u/a/a_rhs
    // written). The closure's `gamma` is BOBYQA's candidate point, not the bound
    // `params` (which `minimize` owns as its `x`).
    let GlmmWorkspace {
        solver, params, lower, upper, groupings,
        z, m, eta, prob, w, u, u_seed, eta_fixed, mu, wm, a, a_rhs, a_blocks, lam,
        z_buf, m_buf, p: pf, ..
    } = ws;
    // x is fixed for this fit: widen the slope columns to f64 once, so every
    // BOBYQA eval's per-solve M fill runs MatRef-free. Blocked path ONLY: the
    // dense (crossed/nested) branch never reads z_buf, and under a test_formula
    // reduced fit `x` is the REDUCED design while `primary_slope_cols` index the
    // FULL one — an unconditional fill could index past x's columns there.
    if groupings.extra_offsets.is_empty() {
        fill_z_f64(groupings, x, z_buf, n);
    }
    let mut best_obj = f64::INFINITY;
    let out = solver.minimize(
        |gamma| {
            // Within-fit û warm-start: seed PIRLS from the incumbent (best point so
            // far), not from 0. The conditional mode is point-determined, so the seed
            // only shifts the stopping iterate within the PIRLS exit band.
            u[..k].copy_from_slice(&u_seed[..k]);
            let obj = laplace_deviance(
                groupings, gamma, z.as_ref(), m, lam, z_buf, m_buf, x, y, cluster_ids,
                eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, a_blocks, *pf, n,
            );
            if obj < best_obj { best_obj = obj; u_seed[..k].copy_from_slice(&u[..k]); }
            obj
        },
        params, lower, upper,
    );

    debug_assert!(out.status != Status::InvalidArgs);
    let ok = matches!(out.status, Status::Converged);

    // Per-component diagonal pin (β never pins). `diag` borrows ws.groupings; the
    // loop mutates the disjoint field ws.params, so no clone is needed.
    let diag = ws.groupings.diagonal_theta();
    let mut pinned_components = 0u32;
    let mut pinned = false;
    if ok {
        for (kk, &ti) in diag.iter().enumerate() {
            if ws.params[ti] <= PIN_THETA {
                ws.params[ti] = 0.0;
                pinned = true;
                pinned_components |= 1 << kk;
            }
        }
    }

    // Re-evaluate at the (possibly pinned) γ̂ to refresh M, ũ, W̃. ws.params already
    // holds the pinned γ̂, so call the kernel on it directly — no params copy.
    if ok {
        // Warm-start the pinned re-eval from the incumbent (its modes are the
        // inference iterate); u_seed holds the BOBYQA incumbent after minimize.
        ws.u[..k].copy_from_slice(&ws.u_seed[..k]);
        let GlmmWorkspace {
            groupings, params, p, z, m, lam, z_buf, m_buf, eta, prob, w, u, eta_fixed, mu,
            wm, a, a_rhs, a_blocks, ..
        } = ws;
        // z_buf still holds this fit's slope copy — x is unchanged since fill_z_f64.
        let _ = laplace_deviance(
            groupings, &params[..], z.as_ref(), m, lam, z_buf, m_buf, x, y, cluster_ids,
            eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, a_blocks, *p, n,
        );
    }

    if !ok {
        return nan_fit(ws, target_indices, out.n_eval);
    }

    // β̂ from γ̂.
    for j in 0..p { ws.betas[j] = ws.params[n_theta + j]; }

    // Var(β̂) = Schur⁻¹. Blocked path (no extras) reuses the per-block factors the
    // blocked PIRLS left in ws.a_blocks; crossed/nested path factors the dense ws.a.
    let inf_ok = if ws.groupings.extra_offsets.is_empty() {
        blocked_schur_fill(ws, x, cluster_ids, n)
    } else {
        dense_schur_fill(ws, x, n)
    };
    if !inf_ok { return nan_fit(ws, target_indices, out.n_eval); }
    // Var(β̂)_jj from chol(Schur) forward-solve (mirrors fit_lmm's recovery).
    let sc = match ws.schur.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c, Err(_) => return nan_fit(ws, target_indices, out.n_eval),
    };
    let lschur = sc.L();
    for &tj in target_indices {
        let tj = tj as usize;
        // Forward-solve into reusable scratch; fwd_solve[i] is written before it is
        // read as fwd_solve[kk] (kk < i), so no per-target zero-fill is needed.
        for i in 0..p {
            let mut acc = if i == tj { 1.0 } else { 0.0 };
            for kk in 0..i { acc -= lschur[(i, kk)] * ws.fwd_solve[kk]; }
            ws.fwd_solve[i] = acc / lschur[(i, i)];
        }
        let vd: f64 = ws.fwd_solve[..p].iter().map(|v| v * v).sum();
        ws.var_diag[tj] = vd;
        ws.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
            ws.betas[tj] * ws.betas[tj] / vd
        } else { f64::NAN };
    }

    // Joint Wald-χ² via the lme helper (Schur is the β-information; scale 1.0).
    let joint_t_sq = if target_indices.is_empty() {
        f64::NAN
    } else {
        crate::lme::joint_wald_chi_sq(
            ws.schur.as_ref(), &ws.betas, 1.0, target_indices,
            ws.joint_k_inv.as_mut(), ws.joint_sigma_t_chol.as_mut(), &mut ws.joint_rhs,
        )
    };

    // τ̂² = D̂[0][0] = (Λ̂Λ̂')[0][0]. No σ² (binomial). For lower-tri Λ_p stored
    // row-major (lam[r*q + c]), row 0 has only the (0,0) entry nonzero, so
    // D̂[0][0] = Σ_c Λ[0,c]² = Λ[0,0]² — the random-INTERCEPT variance.
    crate::lmm::primary_lambda(&ws.params[..n_theta], ws.groupings.primary_q, &mut ws.lam);
    let q = ws.groupings.primary_q;
    let mut d00 = 0.0;
    for r in 0..q { d00 += ws.lam[r] * ws.lam[r]; }
    GlmmFit {
        converged: true,
        boundary_hit: u8::from(pinned),
        pinned_components,
        n_eval: out.n_eval,
        tau_squared_hat: d00,
        joint_t_sq,
    }
}

/// NaN-fill the inference outputs on a non-converged / Schur-failure fit, mirror
/// `fit_lmm`'s NaN-fill branch (boundary_hit = 2 = optimizer/Schur failure).
fn nan_fit(ws: &mut GlmmWorkspace, targets: &[u32], n_eval: usize) -> GlmmFit {
    for v in ws.betas.iter_mut() { *v = f64::NAN; }
    for &t in targets { ws.var_diag[t as usize] = f64::NAN; ws.t_sq[t as usize] = f64::NAN; }
    GlmmFit { converged: false, boundary_hit: 2, pinned_components: 0, n_eval,
        tau_squared_hat: f64::NAN, joint_t_sq: f64::NAN }
}

/// Dense Schur fill (crossed/nested path): X'W̃X, X'W̃M, A⁻¹M'W̃X via the `k×k`
/// `ws.a` LLT, and `ws.schur = X'W̃X − X'W̃M·A⁻¹M'W̃X`. Reads `ws.{a, m, w, x via
/// arg}`. Returns false on a non-PD `ws.a`. Unchanged from the pre-Phase-2 inline
/// inference — moved verbatim so the crossed path is byte-for-byte identical.
fn dense_schur_fill(ws: &mut GlmmWorkspace, x: MatRef<f32>, n: usize) -> bool {
    use faer::linalg::solvers::Solve;
    let (k, p) = (ws.k, ws.p);
    for r in 0..p { for c in 0..=r {
        let mut s = 0.0; for i in 0..n { s += x[(i, r)] as f64 * ws.w[i] * x[(i, c)] as f64; }
        ws.xtwx[(r, c)] = s; ws.xtwx[(c, r)] = s;
    }}
    for r in 0..p { for c in 0..k {
        let mut s = 0.0; for i in 0..n { s += x[(i, r)] as f64 * ws.w[i] * ws.m[(i, c)]; }
        ws.xtwm[(r, c)] = s;
    }}
    let ac = match ws.a.as_ref().llt(faer::Side::Lower) { Ok(c) => c, Err(_) => return false };
    for r in 0..k { for c in 0..p { ws.ainv_mtwx[(r, c)] = ws.xtwm[(c, r)]; } }
    ac.solve_in_place(ws.ainv_mtwx.as_mut());
    for r in 0..p { for c in 0..p {
        let mut s = ws.xtwx[(r, c)];
        for j in 0..k { s -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)]; }
        ws.schur[(r, c)] = s;
    }}
    true
}

/// Blocked Schur fill (no-extras path): reconstruct mᵢ = Λ_p'·zᵢ per row to build
/// X'W̃X (p×p, dense) and the per-cluster coupling X'W̃M (into `ws.xtwm` columns
/// `f·q_p..`), then solve `A_f T_f = (M'W̃X)_f` per block by REUSING the factored
/// `ws.a_blocks` the converged blocked PIRLS left behind (W̃ in `ws.w`, Λ̂ in
/// `ws.lam`), and `ws.schur = X'W̃X − Σ_f (X'W̃M)_f·T_f`. Only the trailing `p×p`
/// Schur LLT (done by the common code after this) stays dense. Returns false if a
/// stored block is not usable (defensive — the PIRLS already proved them PD).
fn blocked_schur_fill(ws: &mut GlmmWorkspace, x: MatRef<f32>, cluster_ids: &[u32], n: usize) -> bool {
    let (p, q, s) = (ws.p, ws.groupings.primary_q, ws.groupings.n_primary);
    let k = q * s;
    // X'W̃X (p×p).
    for r in 0..p { for c in 0..=r {
        let mut sm = 0.0; for i in 0..n { sm += x[(i, r)] as f64 * ws.w[i] * x[(i, c)] as f64; }
        ws.xtwx[(r, c)] = sm; ws.xtwx[(c, r)] = sm;
    }}
    // X'W̃M, blocked: zero then scatter the q_p coupling columns per row.
    for r in 0..p { for c in 0..k { ws.xtwm[(r, c)] = 0.0; } }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let mut m_row = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
        for c in 0..q {
            let mut acc = 0.0;
            for rr in c..q {
                let zr = if rr == 0 { 1.0 } else { x[(i, ws.groupings.primary_slope_cols[rr - 1])] as f64 };
                acc += zr * ws.lam[rr * q + c];
            }
            m_row[c] = acc;
        }
        let wi = ws.w[i];
        for r in 0..p {
            let xw = x[(i, r)] as f64 * wi;
            for c in 0..q { ws.xtwm[(r, f * q + c)] += xw * m_row[c]; }
        }
    }
    // T_f = A_f⁻¹ (M'W̃X)_f, per block, reusing the stored factor; ainv_mtwx rows
    // f·q_p.. hold T_f. (M'W̃X)_f[c, col] = (X'W̃M)_f[col, c] = ws.xtwm[(col, f·q+c)].
    for f in 0..s {
        let ablk = f * q * q;
        for col in 0..p {
            let mut rhs = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            for c in 0..q { rhs[c] = ws.xtwm[(col, f * q + c)]; }
            glmm_block_solve(&ws.a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
            for c in 0..q { ws.ainv_mtwx[(f * q + c, col)] = rhs[c]; }
        }
    }
    // Schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X). Exact: A is block-diagonal, so the per-block
    // solves above equal the full A⁻¹M'W̃X; the Σ_j over k is a full sum (every
    // column j belongs to one cluster and is populated — there are no zero columns).
    for r in 0..p { for c in 0..p {
        let mut sm = ws.xtwx[(r, c)];
        for j in 0..k { sm -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)]; }
        ws.schur[(r, c)] = sm;
    }}
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    // Tiny deterministic LCG → reproducible test data without RNG-determinism caveats.
    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Clustered-binary dataset: n=80, p=2 (intercept + x1), 8 clusters, a
    /// random intercept (q_p=1). Returns (X f64, y∈{0,1}, cluster_ids).
    /// `contiguous` selects the id layout: false = round-robin `i % nc`
    /// (the `FixedClusters` production layout, the historical fixture), true =
    /// block layout `i / per_cluster` (the `FixedSize`/DGEN-FS production
    /// layout). The blocked path must hold on both — layout-sensitive rewrites
    /// of its row loops are a live optimization direction.
    fn glmm_intercept_dataset_layout(contiguous: bool) -> (Mat<f64>, Vec<f32>, Vec<u32>) {
        let (n, nc) = (80usize, 8usize);
        let mut st = 7u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = if contiguous { i / (n / nc) } else { i % nc };
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0; x[(i, 1)] = x1;
            let eta = 0.2 + 0.8 * x1 + u0[c];
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        (x, y, ids)
    }

    fn glmm_intercept_dataset() -> (Mat<f64>, Vec<f32>, Vec<u32>) {
        glmm_intercept_dataset_layout(false)
    }

    /// q_p=2 (intercept + slope on col 0) primary grouping plus ONE crossed extra
    /// grouping. Returns (X f64 [1, x1], y, primary ids, crossed ids, spec). Every
    /// crossed level gets ≥1 obs (round-robin), and every primary level too.
    fn glmm_slope_crossed_dataset(
    ) -> (Mat<f64>, Vec<f32>, Vec<u32>, Vec<u32>, engine_contract::ClusterSpec) {
        let (n, n_prim, n_crossed) = (96usize, 8usize, 4usize);
        let mut st = 13u64;
        let u0: Vec<f64> = (0..n_prim).map(|_| 0.6 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..n_prim).map(|_| 0.4 * lcg(&mut st)).collect();
        let uc: Vec<f64> = (0..n_crossed).map(|_| 0.5 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        let mut crossed = vec![0u32; n];
        for i in 0..n {
            let c = i % n_prim;
            let cc = i % n_crossed;
            ids[i] = c as u32;
            crossed[i] = cc as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0; x[(i, 1)] = x1;
            let eta = 0.2 + 0.8 * x1 + u0[c] + u1[c] * x1 + uc[cc];
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        // Slope on design col 0 of the [1, x1] X passed to build_z (the slope_cols
        // arg is &[1] there — the x1 column index in the full X); the spec carries
        // ONE SlopeTerm + ONE crossed grouping so for_cluster_spec sizes q_p=2 + tail.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: n_prim as u32 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(0),
                variance: 0.16,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![engine_contract::GroupingSpec {
                relation: engine_contract::GroupingRelation::Crossed {
                    n_clusters: n_crossed as u32,
                },
                tau_squared: 0.25,
                slopes: vec![],
            }],
        };
        (x, y, ids, crossed, cluster)
    }

    /// Independent Laplace deviance: dense Z (intercept indicators), M = Z·λ
    /// (scalar λ = θ[0]), Newton-on-u PIRLS, deviance = d + ‖u‖² + logdet(M'WM+I).
    fn brute_force_intercept_laplace(
        theta0: f64, beta: &[f64], x: &Mat<f64>, y: &[f32], ids: &[u32], nc: usize,
    ) -> f64 {
        let (n, p) = (x.nrows(), x.ncols());
        let mut m = Mat::<f64>::zeros(n, nc);
        for i in 0..n { m[(i, ids[i] as usize)] = theta0; }
        let mut u = vec![0.0f64; nc];
        let pen_dev = |u: &[f64], w_out: Option<&mut [f64]>| -> f64 {
            let mut d = 0.0; let mut pen = 0.0;
            let mut wbuf = vec![0.0; n];
            for i in 0..n {
                let mut eta = 0.0; for j in 0..p { eta += x[(i, j)] * beta[j]; }
                for c in 0..nc { eta += m[(i, c)] * u[c]; }
                let pi = 1.0 / (1.0 + (-eta).exp());
                d += if eta > 0.0 { eta + (-eta).exp().ln_1p() } else { eta.exp().ln_1p() } - y[i] as f64 * eta;
                wbuf[i] = (pi * (1.0 - pi)).max(1e-6);
                let _ = pi;
            }
            for &uc in u { pen += uc * uc; }
            if let Some(w) = w_out { w.copy_from_slice(&wbuf); }
            2.0 * d + pen
        };
        let mut w = vec![0.0; n];
        for _ in 0..50 {
            let mut eta = vec![0.0; n]; let mut pvec = vec![0.0; n];
            for i in 0..n {
                let mut e = 0.0; for j in 0..p { e += x[(i, j)] * beta[j]; }
                for c in 0..nc { e += m[(i, c)] * u[c]; }
                eta[i] = e; let pi = 1.0 / (1.0 + (-e).exp());
                pvec[i] = pi; w[i] = (pi * (1.0 - pi)).max(1e-6);
            }
            let mut g = vec![0.0; nc];
            for c in 0..nc {
                let mut s = 0.0; for i in 0..n { s += m[(i, c)] * (y[i] as f64 - pvec[i]); }
                g[c] = 2.0 * u[c] - 2.0 * s;
            }
            let mut h = Mat::<f64>::zeros(nc, nc);
            for a in 0..nc { for b in 0..nc {
                let mut s = 0.0; for i in 0..n { s += m[(i, a)] * w[i] * m[(i, b)]; }
                h[(a, b)] = 2.0 * (s + if a == b { 1.0 } else { 0.0 });
            }}
            let hc = h.as_ref().llt(faer::Side::Lower).unwrap();
            let mut step = Mat::<f64>::zeros(nc, 1);
            for c in 0..nc { step[(c, 0)] = g[c]; }
            hc.solve_in_place(step.as_mut());
            let mut max = 0.0f64;
            for c in 0..nc { u[c] -= step[(c, 0)]; max = max.max(step[(c, 0)].abs()); }
            if max < 1e-10 { break; }
        }
        let _ = pen_dev(&u, Some(&mut w));
        let mut a = Mat::<f64>::zeros(nc, nc);
        for r in 0..nc { for c in 0..nc {
            let mut s = 0.0; for i in 0..n { s += m[(i, r)] * w[i] * m[(i, c)]; }
            a[(r, c)] = s + if r == c { 1.0 } else { 0.0 };
        }}
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut logdet = 0.0; for r in 0..nc { logdet += ac.L()[(r, r)].ln(); }
        pen_dev(&u, None) + 2.0 * logdet
    }

    #[test]
    fn laplace_deviance_matches_brute_force_intercept() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let beta = [0.2_f64, 0.8];
        let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5; ws.params[1] = beta[0]; ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!((got - want).abs() < 1e-6, "laplace dev: got {got}, want {want}");
    }

    #[test]
    fn laplace_deviance_collapses_to_glm_at_theta_zero() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let beta = [0.2_f64, 0.8];
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.0; ws.params[1] = beta[0]; ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        // Widen from the SAME f32 design the kernel consumes (f32-gen/f64-fit) — using xf64 here re-introduces a ~1e-7 f32→f64 round-trip gap.
        let mut d = 0.0;
        for i in 0..80 {
            let eta = beta[0] + beta[1] * xf32[(i, 1)] as f64;
            d += if eta > 0.0 { eta + (-eta).exp().ln_1p() } else { eta.exp().ln_1p() }
                - y[i] as f64 * eta;
        }
        let want = 2.0 * d;
        assert!((got - want).abs() < 1e-9, "collapse: got {got}, want {want}");
    }

    #[test]
    fn build_z_width_general_populates_all_columns() {
        let (xf64, _y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
        let n = ids.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[crossed_ids.clone()], &[1], n);
        let mut touched = vec![false; ws.k];
        for c in 0..ws.k { for i in 0..n { if ws.z[(i, c)] != 0.0 { touched[c] = true; } } }
        assert!(touched.iter().all(|&t| t), "every RE column must be populated — offset wiring");
        // The fit_glmm width-general assertions (β̂/τ̂ recovery on this same fixture)
        // are added in Task 5, reusing `glmm_slope_crossed_dataset`.
    }

    /// `apply_lambda` must scale each extra grouping's columns by ITS OWN θ over
    /// ITS OWN width, even when `extra_offsets` is non-monotonic. A
    /// crossed-before-nested declaration places the nested block at the low
    /// `prim_width` slot (small offset) AFTER the crossed block (large offset),
    /// so a "span to the next declaration's offset" loop would leave the crossed
    /// block unscaled and over-scale the nested block with the crossed θ.
    /// Regression guard for that extra-offset span bug.
    #[test]
    fn apply_lambda_handles_nonmonotonic_extra_offsets() {
        let (n_prim, n_crossed, n_per_parent) = (4usize, 3usize, 2usize);
        // Declaration order [Crossed, Nested] ⇒ extra_offsets[0] (crossed, high) >
        // extra_offsets[1] (nested, low) — the non-monotonic precondition.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: n_prim as u32 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::Crossed { n_clusters: n_crossed as u32 },
                    tau_squared: 0.16,
                    slopes: vec![],
                },
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::NestedWithin { n_per_parent: n_per_parent as u32 },
                    tau_squared: 0.09,
                    slopes: vec![],
                },
            ],
        };
        let n = 8usize;
        let ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        let g = &ws.groupings;
        assert!(g.extra_offsets[0] > g.extra_offsets[1],
            "fixture must produce non-monotonic offsets, got {:?}", g.extra_offsets);
        // q_p = 1 ⇒ base_theta = 1; θ_crossed at idx 1, θ_nested at idx 2.
        let base_theta = 1usize;
        let (theta_crossed, theta_nested) = (2.0_f64, 3.0_f64);
        let mut params = vec![0.0; g.n_theta()];
        params[0] = 1.0; // primary Λ (irrelevant to the extra blocks)
        params[base_theta] = theta_crossed;
        params[base_theta + 1] = theta_nested;
        // z = ones everywhere so M directly reveals the per-column scale.
        let mut z = Mat::<f64>::zeros(n, g.k_total);
        for i in 0..n { for c in 0..g.k_total { z[(i, c)] = 1.0; } }
        let mut m = Mat::<f64>::zeros(n, g.k_total);
        let mut lam = vec![0.0; g.primary_q * g.primary_q];
        apply_lambda(g, &params, z.as_ref(), &mut m, &mut lam, n);
        let coff = g.extra_offsets[0];
        for c in coff..coff + n_crossed {
            assert!((m[(0, c)] - theta_crossed).abs() < 1e-12,
                "crossed col {c} = {}, want {theta_crossed}", m[(0, c)]);
        }
        let noff = g.extra_offsets[1];
        for c in noff..noff + n_prim * n_per_parent {
            assert!((m[(0, c)] - theta_nested).abs() < 1e-12,
                "nested col {c} = {}, want {theta_nested}", m[(0, c)]);
        }
    }

    #[test]
    fn fit_glmm_recovers_direction_and_finite_inference() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        let targets = [1u32];
        let beta_truth = [0.2_f64, 0.8];
        let fit = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &targets, Some(&[0.5]), &beta_truth, 80);
        assert!(fit.converged);
        assert!(ws.betas[1] > 0.3, "β̂₁ should be positive (truth 0.8), got {}", ws.betas[1]);
        assert!(ws.t_sq[1].is_finite() && ws.t_sq[1] >= 0.0);
        assert!(fit.tau_squared_hat.is_finite() && fit.tau_squared_hat >= 0.0);
    }

    /// τ²→0 collapse-to-glm.rs (standing gate, L1 form).
    #[test]
    fn fit_glmm_collapses_to_plain_irls_when_tau_negligible() {
        use crate::glm::{glm_irls_fit, GlmScratch};
        let (n, nc) = (200usize, 10usize);
        let mut st = 99u64;
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % nc) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0; x[(i, 1)] = x1;
            let eta = 0.1 + 0.7 * x1;                 // NO u term ⇒ τ²→0 truth
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = x[(i, j)] as f32; } }

        // Plain logistic on the same bytes (adapt GlmScratch fields to the REAL struct).
        let mut sw = crate::workspace::SimWorkspace::new(n, 2, 1, 0, None);
        let irls = {
            let s = GlmScratch {
                irls_eta: &mut sw.irls_eta[..n], irls_p: &mut sw.irls_p[..n],
                irls_w: &mut sw.irls_w[..n], irls_z: &mut sw.irls_z[..n],
                irls_betas: &mut sw.irls_betas[..2], irls_betas_new: &mut sw.irls_betas_new[..2],
                irls_var_diag: &mut sw.irls_var_diag[..1], irls_t_sq: &mut sw.irls_t_sq[..1],
                irls_u_scratch: &mut sw.irls_u_scratch[..2],
                irls_xtwx: sw.irls_xtwx.as_mut().submatrix_mut(0, 0, 2, 2),
                irls_xtwz: &mut sw.irls_xtwz[..2],
                irls_l: sw.irls_l.as_mut().submatrix_mut(0, 0, 2, 2),
                irls_x_f64: &mut sw.irls_x_f64[..n * 2],
                irls_wx: &mut sw.irls_wx[..n * 2],
            };
            let f = glm_irls_fit(xf32.as_ref(), &y, &[1], None, s);
            (f.betas.to_vec(), f.t_sq.to_vec(), f.converged)
        };
        assert!(irls.2, "plain IRLS must converge");

        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: nc as u32 }, 0.1);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], n);
        let fit = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.05]), &[0.1, 0.7], n);
        assert!(fit.converged);
        assert!((ws.betas[1] - irls.0[1]).abs() < 1e-2, "β̂₁ glmm {} vs irls {}", ws.betas[1], irls.0[1]);
        // GLMM z² is indexed by coefficient (ws.t_sq[1] = slope = target 1); IRLS
        // z² is indexed by target position (irls.1[0] = first target = slope).
        assert!((ws.t_sq[1].sqrt() - irls.1[0].sqrt()).abs() < 5e-2,
            "z glmm {} vs irls {}", ws.t_sq[1].sqrt(), irls.1[0].sqrt());
    }

    /// Deferred from Task 4.7: fit_glmm on the q_p=2 slope + crossed fixture.
    #[test]
    fn fit_glmm_width_general_slope_and_crossed() {
        let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
        let n = y.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[crossed_ids.clone()], &[1], n);
        let fit = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], None, &[0.2, 0.8], n);
        assert!(fit.converged);
        // Planted slope 0.8; small balanced binary GLMM (n=96) inflates β̂₁ to ≈3.2 with
        // z²≈13 — direction + significance are the robust claims, not the magnitude. τ̂²
        // collapses to 0 on this draw (valid), so only bound it finite + non-blown. The
        // old `>= 0.0` clauses were tautological on squared quantities.
        assert!(ws.betas[1] > 0.5, "slope must be strongly positive, got {}", ws.betas[1]);
        assert!(
            ws.t_sq[1].is_finite() && ws.t_sq[1] > 3.84,
            "z² must clear the α=0.05 bar (3.84), got {}",
            ws.t_sq[1]
        );
        assert!(
            fit.tau_squared_hat.is_finite() && (0.0..5.0).contains(&fit.tau_squared_hat),
            "τ̂² {}",
            fit.tau_squared_hat
        );
    }

    /// Warm-path zero-alloc lock — `#[ignore]` because `dhat::Profiler` measures
    /// process-wide allocations and concurrent tests contaminate the count. faer's
    /// rayon parallelism also jitters the count run-to-run on a multi-core box, so
    /// pin it to one thread for a deterministic measurement:
    /// Run: `RAYON_NUM_THREADS=1 cargo test -p engine-core fit_glmm_warm_path_bounded_alloc -- --ignored --test-threads=1`
    #[test]
    #[ignore]
    fn fit_glmm_warm_path_bounded_alloc() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        let _ = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.5]), &[0.2, 0.8], 80); // warmup
        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..20 {
            let _ = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.5]), &[0.2, 0.8], 80);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        // Measured 120 (this machine) — 6 blocks/fit on the no-extras intercept
        // fixture, down from 7780. Phase 2 routes it through the BLOCKED PIRLS, so
        // the dense per-iteration k×1 rhs Mat and the per-eval dense `llt` internals
        // are gone; the per-block Crout factor/solve work on pre-allocated a_blocks
        // and stack-sized m_row scratch. What remains is the joint [θ|β] BOBYQA's
        // own per-eval scratch. Block COUNT is deterministic for a fixed code path
        // under `RAYON_NUM_THREADS=1` — a change to that floor flags a new allocation
        // or a shifted eval/iteration trajectory. If faer changes its Cholesky
        // internals, update — do not relax. Phase 3's within-fit warm-start allocates
        // nothing itself (u_seed copy/reset hit pre-allocated buffers) but its
        // within-exit-band objective shift nudges the BOBYQA trajectory by a few evals:
        // 120 → 124. Phase 4 (SIMD fit-path transcendentals) keeps the floor flat:
        // the `pulp` dispatch is alloc-free and the loop-split uses only the
        // pre-allocated eta/prob/w scratch + stack-sized m_row, so the bound holds.
        // Re-measured after the relative PIRLS exit (group C): still exactly 124
        // on this fixture — fewer inner iterations, same eval-scratch count.
        const BOUND: u64 = 124;
        assert!(stats.total_blocks <= BOUND,
            "warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
            stats.total_blocks, BOUND);
    }

    /// `glmm_block_chol` factors a q×q SPD block in place to its lower Crout L,
    /// and `glmm_block_solve` then solves L Lᵀ x = b — checked against a faer LLT
    /// solve on the same matrix. q=3 with a deliberately non-trivial SPD block.
    #[test]
    fn block_chol_and_solve_match_faer() {
        let q = 3usize;
        // SPD A (row-major, lower triangle is what the helper reads).
        let a = [
            4.0, 0.0, 0.0,
            2.0, 5.0, 0.0,
            1.0, 3.0, 6.0,
        ];
        let b = [1.0_f64, -2.0, 0.5];

        // Reference: faer LLT solve.
        let mut af = Mat::<f64>::zeros(q, q);
        for r in 0..q { for c in 0..=r { af[(r, c)] = a[r * q + c]; af[(c, r)] = a[r * q + c]; } }
        let ac = af.as_ref().llt(faer::Side::Lower).unwrap();
        let mut rhs = Mat::<f64>::zeros(q, 1);
        for r in 0..q { rhs[(r, 0)] = b[r]; }
        ac.solve_in_place(rhs.as_mut());

        // Under test.
        let mut blk = a;
        assert!(super::glmm_block_chol(&mut blk, q), "block should be PD");
        let mut x = b;
        super::glmm_block_solve(&blk, q, &mut x);
        for r in 0..q {
            assert!((x[r] - rhs[(r, 0)]).abs() < 1e-12, "x[{r}] = {}, faer {}", x[r], rhs[(r, 0)]);
        }
        // log|A| from pivots = 2·Σ ln L[r,r] should match faer's.
        let logdet_helper: f64 = (0..q).map(|r| blk[r * q + r].ln()).sum::<f64>() * 2.0;
        let logdet_faer: f64 = (0..q).map(|r| ac.L()[(r, r)].ln()).sum::<f64>() * 2.0;
        assert!((logdet_helper - logdet_faer).abs() < 1e-12);
    }

    /// Non-PD block ⇒ `glmm_block_chol` returns false (the module's failure surface).
    #[test]
    fn block_chol_rejects_non_pd() {
        let q = 2usize;
        let mut blk = [1.0_f64, 0.0, 2.0, 1.0]; // [[1,·],[2,1]] ⇒ pivot 1 − 4 < 0
        assert!(!super::glmm_block_chol(&mut blk, q));
    }

    /// No-extras q_p=2 (intercept + slope) clustered-binary dataset — exercises the
    /// blocked SLOPE path (no crossed/nested ⇒ extra_offsets empty ⇒ blocked dispatch).
    /// `contiguous` selects the id layout: false = round-robin `i % nc`
    /// (the `FixedClusters` production layout, the historical fixture), true =
    /// block layout `i / per_cluster` (the `FixedSize`/DGEN-FS production
    /// layout). The blocked path must hold on both — layout-sensitive rewrites
    /// of its row loops are a live optimization direction.
    fn glmm_slope_noextra_dataset_layout(contiguous: bool) -> (Mat<f64>, Vec<f32>, Vec<u32>) {
        let (n, nc) = (96usize, 8usize);
        let mut st = 21u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.4 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = if contiguous { i / (n / nc) } else { i % nc };
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0; x[(i, 1)] = x1;
            let eta = 0.2 + 0.8 * x1 + u0[c] + u1[c] * x1;
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        (x, y, ids)
    }

    fn glmm_slope_noextra_dataset() -> (Mat<f64>, Vec<f32>, Vec<u32>) {
        glmm_slope_noextra_dataset_layout(false)
    }

    /// The blocked intercept path computes the same Laplace deviance as the
    /// independent brute force (reorders accumulation ⇒ FP-close, not bit-equal).
    /// After Phase 2 this routes through `pirls_solve_blocked` (extra_offsets empty).
    #[test]
    fn blocked_laplace_matches_brute_force_intercept() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let beta = [0.2_f64, 0.8];
        let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        assert!(ws.groupings.extra_offsets.is_empty(), "fixture must route blocked");
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5; ws.params[1] = beta[0]; ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!((got - want).abs() < 1e-6, "blocked laplace: got {got}, want {want}");
    }

    /// `FixedSize`/DGEN-FS production layout (contiguous id runs) through the
    /// blocked intercept path. The round-robin fixtures only cover
    /// `FixedClusters`-style interleaved ids; a layout-sensitive rewrite of
    /// the blocked row loops behaves differently per layout, so the
    /// brute-force oracle must hold here too.
    #[test]
    fn blocked_laplace_matches_brute_force_intercept_contiguous() {
        let (xf64, y, ids) = glmm_intercept_dataset_layout(true);
        let beta = [0.2_f64, 0.8];
        let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        assert!(ws.groupings.extra_offsets.is_empty(), "fixture must route blocked");
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5; ws.params[1] = beta[0]; ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!((got - want).abs() < 1e-6, "blocked laplace (contiguous): got {got}, want {want}");
    }

    /// Blocked deviance == dense deviance on the no-extras slope fixture, to FP
    /// error. Drives both kernels directly on the same M / Λ_p (dev-time
    /// equivalence smoke test; a wild divergence is a coding bug, per the spec).
    #[test]
    fn blocked_pirls_matches_dense_slope_noextra() {
        let (xf64, y, ids) = glmm_slope_noextra_dataset();
        let n = y.len();
        // slope on design col 1 ⇒ slope_cols = &[1]; q_p = 2.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(1), variance: 0.16,
                corr_with_intercept: 0.0, corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        assert!(ws.groupings.extra_offsets.is_empty());
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
        let beta = [0.2_f64, 0.8];
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

        // Dense reference: apply_lambda → pirls_solve.
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        let GlmmWorkspace { groupings, z, m, lam, eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, .. } = &mut ws;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
        let dense = pirls_solve(
            k, p, m.as_ref(), xf32.as_ref(), &y, &params[nt..],
            eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, n,
        );

        // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        build_z(&mut ws2, xf32.as_ref(), &ids, &[], &[1], n);
        crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
        fill_z_f64(&ws2.groupings, xf32.as_ref(), &mut ws2.z_buf, n);
        let GlmmWorkspace { groupings, lam, z_buf, m_buf, eta, prob, w, u, eta_fixed, a_blocks, a_rhs, .. } = &mut ws2;
        let blocked = pirls_solve_blocked(
            groupings, &ids, xf32.as_ref(), &y, &beta, lam, z_buf, m_buf,
            eta, prob, w, u, eta_fixed, a_blocks, a_rhs, n,
        );

        assert_eq!(dense.3, blocked.3, "convergence flag");
        assert!((dense.0 - blocked.0).abs() < 1e-9, "dev: dense {} blocked {}", dense.0, blocked.0);
        assert!((dense.1 - blocked.1).abs() < 1e-9, "pen: dense {} blocked {}", dense.1, blocked.1);
        assert!((dense.2 - blocked.2).abs() < 1e-9, "logdet: dense {} blocked {}", dense.2, blocked.2);
        for c in 0..k {
            assert!((ws.u[c] - ws2.u[c]).abs() < 1e-7, "u[{c}]: dense {} blocked {}", ws.u[c], ws2.u[c]);
        }
    }

    /// Blocked == dense on the cluster-contiguous (`FixedSize`/DGEN-FS) id
    /// layout — the q_p=2 twin of `blocked_pirls_matches_dense_slope_noextra`,
    /// which only exercises round-robin ids. Same FP bands; if a reordered
    /// accumulation ever moves them, re-derive the band with documentation —
    /// never widen silently.
    #[test]
    fn blocked_pirls_matches_dense_slope_contiguous() {
        let (xf64, y, ids) = glmm_slope_noextra_dataset_layout(true);
        let n = y.len();
        // slope on design col 1 ⇒ slope_cols = &[1]; q_p = 2.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(1), variance: 0.16,
                corr_with_intercept: 0.0, corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        assert!(ws.groupings.extra_offsets.is_empty());
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
        let beta = [0.2_f64, 0.8];
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

        // Dense reference: apply_lambda → pirls_solve.
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        let GlmmWorkspace { groupings, z, m, lam, eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, .. } = &mut ws;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
        let dense = pirls_solve(
            k, p, m.as_ref(), xf32.as_ref(), &y, &params[nt..],
            eta, prob, w, u, eta_fixed, mu, wm, a, a_rhs, n,
        );

        // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        build_z(&mut ws2, xf32.as_ref(), &ids, &[], &[1], n);
        crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
        fill_z_f64(&ws2.groupings, xf32.as_ref(), &mut ws2.z_buf, n);
        let GlmmWorkspace { groupings, lam, z_buf, m_buf, eta, prob, w, u, eta_fixed, a_blocks, a_rhs, .. } = &mut ws2;
        let blocked = pirls_solve_blocked(
            groupings, &ids, xf32.as_ref(), &y, &beta, lam, z_buf, m_buf,
            eta, prob, w, u, eta_fixed, a_blocks, a_rhs, n,
        );

        assert_eq!(dense.3, blocked.3, "convergence flag");
        assert!((dense.0 - blocked.0).abs() < 1e-9, "dev: dense {} blocked {}", dense.0, blocked.0);
        assert!((dense.1 - blocked.1).abs() < 1e-9, "pen: dense {} blocked {}", dense.1, blocked.1);
        assert!((dense.2 - blocked.2).abs() < 1e-9, "logdet: dense {} blocked {}", dense.2, blocked.2);
        for c in 0..k {
            assert!((ws.u[c] - ws2.u[c]).abs() < 1e-7, "u[{c}]: dense {} blocked {}", ws.u[c], ws2.u[c]);
        }
    }

    /// Blocked inference (β̂, Var(β̂)_jj, z²) matches the dense inference on the
    /// no-extras slope fixture, to tight FP tolerance — the inference reorder is
    /// the same estimator. Runs the full `fit_glmm` (blocked) and an explicit
    /// dense recomputation of the Schur Var on the same converged state.
    #[test]
    fn blocked_inference_matches_dense_slope_noextra() {
        let (xf64, y, ids) = glmm_slope_noextra_dataset();
        let n = y.len();
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(1), variance: 0.16,
                corr_with_intercept: 0.0, corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let fit = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.5, 0.1, 0.4]), &[0.2, 0.8], n);
        assert!(fit.converged);
        // Recompute Var(β̂) densely from the converged ws.{w, lam, params} via a
        // freshly-built M and ws.a, and compare β̂ / var_diag / t_sq.
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let beta_blocked: Vec<f64> = ws.betas[..p].to_vec();
        let var_blocked = ws.var_diag[1];
        let tsq_blocked = ws.t_sq[1];
        // Dense M = ZΛ̂ at the converged θ̂.
        crate::lmm::primary_lambda(&ws.params[..nt], ws.groupings.primary_q, &mut ws.lam);
        {
            let GlmmWorkspace { groupings, params, z, m, lam, .. } = &mut ws;
            apply_lambda(groupings, &params[..], z.as_ref(), m, lam, n);
        }
        // X'W̃X, X'W̃M, A = M'W̃M + I (dense), Schur, Var(β̂)_11.
        let mut xtwx = Mat::<f64>::zeros(p, p);
        for r in 0..p { for c in 0..p {
            let mut sm = 0.0; for i in 0..n { sm += xf32[(i, r)] as f64 * ws.w[i] * xf32[(i, c)] as f64; }
            xtwx[(r, c)] = sm;
        }}
        let mut xtwm = Mat::<f64>::zeros(p, k);
        for r in 0..p { for c in 0..k {
            let mut sm = 0.0; for i in 0..n { sm += xf32[(i, r)] as f64 * ws.w[i] * ws.m[(i, c)]; }
            xtwm[(r, c)] = sm;
        }}
        let mut a = Mat::<f64>::zeros(k, k);
        for r in 0..k { for c in 0..k {
            let mut sm = if r == c { 1.0 } else { 0.0 };
            for i in 0..n { sm += ws.m[(i, r)] * ws.w[i] * ws.m[(i, c)]; }
            a[(r, c)] = sm;
        }}
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ainv = Mat::<f64>::zeros(k, p);
        for r in 0..k { for c in 0..p { ainv[(r, c)] = xtwm[(c, r)]; } }
        ac.solve_in_place(ainv.as_mut());
        let mut schur = Mat::<f64>::zeros(p, p);
        for r in 0..p { for c in 0..p {
            let mut sm = xtwx[(r, c)];
            for j in 0..k { sm -= xtwm[(r, j)] * ainv[(j, c)]; }
            schur[(r, c)] = sm;
        }}
        let sc = schur.as_ref().llt(faer::Side::Lower).unwrap();
        let mut fwd = vec![0.0; p];
        for i in 0..p {
            let mut acc = if i == 1 { 1.0 } else { 0.0 };
            for kk in 0..i { acc -= sc.L()[(i, kk)] * fwd[kk]; }
            fwd[i] = acc / sc.L()[(i, i)];
        }
        let var_dense: f64 = fwd.iter().map(|v| v * v).sum();
        assert!((var_blocked - var_dense).abs() < 1e-8, "var: blocked {var_blocked} dense {var_dense}");
        let tsq_dense = beta_blocked[1] * beta_blocked[1] / var_dense;
        assert!((tsq_blocked - tsq_dense).abs() < 1e-6, "z²: blocked {tsq_blocked} dense {tsq_dense}");
    }

    /// Within-fit û warm-start MUST reset per fit. A reused-workspace re-fit must
    /// match a fresh-workspace (canonically cold) fit BIT-FOR-BIT. Carrying the
    /// incumbent across fits is the rejected cross-sim warm-start that breaks
    /// merge / same-seed reproducibility. This is the per-fit-reset guard.
    #[test]
    fn warm_start_is_per_fit_deterministic() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mk = || {
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
            let mut xf32 = Mat::<f32>::zeros(80, 2);
            for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
            build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
            (ws, xf32)
        };
        // Canonical: a fresh workspace's first (cold) fit.
        let (mut ws_ref, xref) = mk();
        let _ = fit_glmm(&mut ws_ref, xref.as_ref(), &y, &ids, &[1], Some(&[0.5]), &[0.2, 0.8], 80);
        let ref_beta = ws_ref.betas[1].to_bits();
        // Reused workspace: a throwaway fit pollutes u_seed, then the measured fit
        // must match the canonical cold result bit-for-bit (only if u_seed resets).
        let (mut ws, xf32) = mk();
        let _ = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.5]), &[0.2, 0.8], 80);
        let _ = fit_glmm(&mut ws, xf32.as_ref(), &y, &ids, &[1], Some(&[0.5]), &[0.2, 0.8], 80);
        assert_eq!(ws.betas[1].to_bits(), ref_beta, "re-fit β̂ must match the fresh cold fit bit-for-bit");
    }

    /// The Laplace objective is a (near-)pure function of (θ,β): the same point
    /// solved from u=0 vs from a perturbed seed agrees within ≪ the estimate floor
    /// — the conditional mode is unique, only the stopping iterate is seed-dependent.
    #[test]
    fn warm_start_objective_is_seed_independent() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 }, 0.25);
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 { for j in 0..2 { xf32[(i, j)] = xf64[(i, j)] as f32; } }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5; ws.params[1] = 0.2; ws.params[2] = 0.8;
        for v in ws.u.iter_mut() { *v = 0.0; }
        let cold = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        for (c, v) in ws.u.iter_mut().enumerate() { *v = 0.05 * (c as f64 - 4.0); }
        let warm = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!((cold - warm).abs() < 1e-6, "objective seed-dependent: cold {cold} warm {warm}");
    }

}
