//! Clustered-logistic GLMM kernel: glmer-faithful nAGQ=1. The outer
//! BOBYQA optimizes [θ | β] jointly; for each candidate a penalized-IRLS inner
//! loop solves the conditional modes ũ (β fixed), and the objective is the
//! Laplace deviance d(y,ũ) + ‖ũ‖² + log|L|². RE design Z and Λ_θ are dense
//! (spec-sanctioned within the regime; the block/sparse backend is the D3-0
//! hedge, not built). All scratch lives in `GlmmWorkspace`, allocated once per
//! (spec, max_n) shape — the warm path is zero-alloc (Bobyqa::new once).
//!
//! No σ² scale: binomial dispersion is fixed at 1, so D̂ = Λ̂Λ̂′ directly.
//!
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use bobyqa::{Bobyqa, Config, Status};
use engine_contract::WaldSe;
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
/// Relative FD step for `fd_hessian_cov`'s joint-deviance Hessian:
/// `h_k = FD_STEP_REL·max(1, |γ̂_k|)`. The Hessian is step-invariant over h ∈
/// [1e-4, 1e-1] on the committed fixture (the deviance is smooth/deterministic
/// enough that 1/h² noise amplification is negligible), so 1e-2 is a comfortable
/// mid-band choice — see `fd_hessian_cov`'s doc comment for the full diagnostic.
pub const FD_STEP_REL: f64 = 1e-2;

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
    /// Set iff the per-fit FD-Hessian covariance fell back to the RX/Schur block
    /// (non-PD joint Hessian / non-finite perturbed deviance). Always `false`
    /// here — `fit_glmm` does not yet run the `hessian`-mode kernel; Task 6 wires
    /// it to `fd_hessian_cov`'s `NonPdFellBackToRx` status.
    pub hessian_fallback: bool,
}

/// All GLMM solver scratch — allocated once per (spec, max_n) shape.
pub struct GlmmWorkspace {
    pub groupings: LmmGroupings, // reused RE structure (estimator-agnostic)
    pub k: usize,                // total RE columns (groupings.k_total)
    pub p: usize,                // fixed-effect predictors
    pub n_theta: usize,
    pub z: Mat<f64>,      // max_n × k dense RE design (built per (spec,N) in Task 4)
    pub m: Mat<f64>,      // max_n × k = ZΛ (rebuilt per BOBYQA eval)
    pub solver: Bobyqa,   // sized n_theta + p
    pub params: Vec<f64>, // [θ | β]
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub theta_truth: Vec<f64>, // truth-start θ (= for_cluster_spec recipe)
    // PIRLS scratch (sized max_n / k):
    pub eta: Vec<f64>,
    pub prob: Vec<f64>,
    pub w: Vec<f64>,
    pub eta_fixed: Vec<f64>, // Σ_j x·β, hoisted out of the PIRLS iteration (β fixed within a solve)
    pub m_buf: Vec<f64>, // n×q_p row-major mᵢ = Λ_p'·zᵢ (blocked path) — filled once per PIRLS solve
    pub z_buf: Vec<f64>, // n×(q_p−1) row-major f64 copy of x[:, slope_cols] — filled once per fit
    pub mu: Vec<f64>, // (Mu)ᵢ per iteration via GEMV; overwritten in place by the IRLS residual W·Mu + (y−p) before the RHS GEMV
    pub u: Vec<f64>,
    pub u_seed: Vec<f64>, // within-fit û warm-start incumbent (Phase 3); RESET to 0 each fit_glmm — never carried across fits
    pub a: Mat<f64>,      // k × k  M'WM + I
    pub wm: Mat<f64>, // max_n × k = W∘M scratch for the dense-Gram GEMM (rebuilt per PIRLS iteration)
    pub a_rhs: Vec<f64>, // length k
    pub a_blocks: Vec<f64>, // s · q_p² packed per-cluster q_p×q_p blocks (no-extras path; Σ wᵢmᵢmᵢ'+I then Crout L)
    // structured (block-diagonal core + Schur) path scratch — see
    // `pirls_solve_blocked_extras`. Sized to the worst-case grid shape; left
    // FACTORED (core L's + Schur L) after a converged structured PIRLS so
    // `structured_schur_fill` reuses them. `q_core = primary_q + nested_per_parent`,
    // `e = k_crossed`, `s = n_primary`.
    pub core_blocks: Vec<f64>, // s · q_core² packed per-cluster core blocks (D_f+I then Crout L)
    pub coupling: Vec<f64>,    // s · q_core · e core↔crossed coupling C_f (row-major per cluster)
    pub schur_blk: Vec<f64>, // e × e Schur S = (E+I) − Σ_f C_f'A_f⁻¹C_f (row-major; Crout L in place)
    pub lam: Vec<f64>,       // q_p × q_p primary Λ_p scratch (row-major)
    // Packed M = ZΛ nonzeros for the STRUCTURED path — filled once per deviance
    // eval by `build_packed_m` (replaces `apply_lambda` there), then read by the
    // structured PIRLS passes and `structured_schur_fill` so they never touch the
    // dense faer `m`. `q_core = primary_q + nested_per_parent`, `G_cap =
    // MAX_EXTRA_GROUPINGS`. Sized once at construction — no per-solve alloc.
    pub m_core_buf: Vec<f64>, // max_n · q_core row-major; [i·q_core+local] = M[(i, core_col(f,local))]
    pub cross_val: Vec<f64>,  // max_n · G_cap row-major; nonzero M value (z·θ) per crossed grouping
    pub cross_col: Vec<u32>,  // max_n · G_cap row-major; its crossed-block-local index b (0..e)
    pub n_cross: Vec<u8>, // max_n; #crossed nonzeros for row i (≤ G ≤ MAX_EXTRA_GROUPINGS < 256)
    // inference scratch:
    pub xtwx: Mat<f64>,      // p × p
    pub xtwm: Mat<f64>,      // p × k
    pub ainv_mtwx: Mat<f64>, // k × p  = A⁻¹ M'WX
    pub schur: Mat<f64>,     // p × p  X'WX − X'WM A⁻¹ M'WX
    pub betas: Vec<f64>,     // length p (copied from params[n_theta..])
    pub var_diag: Vec<f64>,  // length p
    pub t_sq: Vec<f64>,      // length p
    pub fwd_solve: Vec<f64>, // length p; Var(β̂)_jj forward-solve scratch (per-target)
    // joint Wald scratch (reuse lme::joint_wald_chi_sq):
    pub joint_k_inv: Mat<f64>,
    pub joint_sigma_t_chol: Mat<f64>,
    pub joint_rhs: Vec<f64>,
    // FD-Hessian SE scratch (`fd_hessian_cov`), allocated once so the per-fit
    // hessian path reuses them. `m = n_theta + p = params.len()`.
    pub hess_scratch: Mat<f64>, // m × m joint-deviance Hessian
    pub fd_saved: Vec<f64>,     // length m; converged γ̂ snapshot restored each return
    pub fd_steps: Vec<f64>,     // length m; per-coordinate FD step h_k
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
        // Structured-path block sizes: core width q_core = q_p + nested children,
        // crossed width e. Buffers stay 1-sized minima when the shape has no
        // extras (the no-extras blocked path never touches them).
        let q_core = q + groupings.nested_per_parent;
        let e_crossed = groupings.k_crossed();

        // θ truth-start = vech(chol(D_rel)), shared verbatim with the LMM via
        // lmm::cluster_theta_truth — the recipe is already σ=1 there.
        let theta_truth = crate::lmm::cluster_theta_truth(cluster);

        // Bounds: θ part from blind_theta_and_bounds; β part = [−BETA_BOX, BETA_BOX].
        let (theta0, mut lower, mut upper) = groupings.blind_theta_and_bounds();
        let mut params = theta0;
        params.extend(std::iter::repeat_n(0.0, p)); // β cold default; overwritten at fit
        lower.extend(std::iter::repeat_n(-BETA_BOX, p));
        upper.extend(std::iter::repeat_n(BETA_BOX, p));

        // ρ_begin ≤ RHO_BEGIN and ≤ 0.1·min diagonal θ₀ (mirror for_cluster_spec)
        // so the unit/truth start is not projected onto a bound.
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
            core_blocks: vec![0.0; (q_core * q_core * n_primary).max(1)],
            coupling: vec![0.0; (q_core * n_primary * e_crossed).max(1)],
            schur_blk: vec![0.0; (e_crossed * e_crossed).max(1)],
            lam: vec![0.0; q * q],
            // Packed-M buffers (structured path). `q_core = q + nested_per_parent`,
            // `G_cap = MAX_EXTRA_GROUPINGS`. `.max(1)` keeps a valid (never-read)
            // allocation on the no-extras shapes that route elsewhere.
            m_core_buf: vec![0.0; (max_n * q_core).max(1)],
            cross_val: vec![0.0; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
            cross_col: vec![0u32; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
            n_cross: vec![0u8; max_n.max(1)],
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
            hess_scratch: Mat::zeros((n_theta + p).max(1), (n_theta + p).max(1)),
            fd_saved: vec![0.0; n_theta + p],
            fd_steps: vec![0.0; n_theta + p],
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
        for k in 0..j {
            d -= blk[j * q + k] * blk[j * q + k];
        }
        if !(d.is_finite() && d > 0.0) {
            return false;
        }
        let l = d.sqrt();
        blk[j * q + j] = l;
        for i in (j + 1)..q {
            let mut v = blk[i * q + j];
            for k in 0..j {
                v -= blk[i * q + k] * blk[j * q + k];
            }
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
        for c in 0..r {
            v -= l[r * q + c] * b[c];
        }
        b[r] = v / l[r * q + r];
    }
    for r in (0..q).rev() {
        let mut v = b[r];
        for c in (r + 1)..q {
            v -= l[c * q + r] * b[c];
        }
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
///
/// Builds the GLMM design-`Z` (`ws.z`) for the dense-extras path; the
/// block-diagonal / structured fits reconstruct `mᵢ` per row instead.
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
    for c in 0..ws.k {
        for i in 0..n {
            ws.z[(i, c)] = 0.0;
        }
    }
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
        #[allow(clippy::needless_range_loop)]
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

/// Pack the STRUCTURED-path nonzeros of `M = ZΛ` into the workspace's packed
/// buffers, once per deviance eval — the structured analogue of `apply_lambda`,
/// which it replaces on this path (`apply_lambda` writes the full dense `n×k`
/// every eval; this writes only the `q_core` core + ≤`G` crossed nonzeros each
/// row reads). `m_core_buf[i·q_core+local]` = the `Λ`-scaled core value
/// `M[(i, core_col(f,local))]` for row `i`'s primary cluster `f = cluster_ids[i]`
/// (primary `local<q`: `Σ_{r≥local} z_r·lam[r·q+local]`, mirroring `apply_lambda`'s
/// core write and the blocked-path fill; nested `local≥q`: the nested indicator
/// scaled by its θ). For each crossed grouping with `θ≠0`, the row's single active
/// level contributes one nonzero: `cross_val = z·θ`, `cross_col = b` (the crossed
/// block-local index, `0..e`), with `n_cross[i]` the count (`≤ G`). A θ-pinned
/// (θ=0) grouping is skipped, mirroring `apply_lambda`'s `z·θ=0` ⇒ no nonzero. The
/// crossed-column scan over `z` is O(n·e) but runs ONCE per eval; the per-PIRLS-
/// iteration passes then read O(n·G). Reads `z` (the dense design `build_z` left),
/// `lam` (filled here via `primary_lambda`), `params`, and `cluster_ids`.
#[allow(clippy::too_many_arguments)]
fn build_packed_m(
    g: &LmmGroupings,
    params: &[f64],
    z: MatRef<f64>,
    lam: &mut [f64],
    cluster_ids: &[u32],
    m_core_buf: &mut [f64],
    cross_val: &mut [f64],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    n: usize,
) {
    let q = g.primary_q;
    let s = g.n_primary;
    let np = g.nested_per_parent;
    let qc = q + np;
    let prim_width = q * s;
    let k_family = qc * s;
    let base_theta = q * (q + 1) / 2;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    crate::lmm::primary_lambda(&params[..g.n_theta()], q, lam);
    let theta_nested = g.nested_theta.map(|ti| params[ti]).unwrap_or(0.0);
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        // Core primary block: same `Σ_{r≥c} z_r·lam[r·q+c]` reduction `apply_lambda`
        // writes at M[(i, f·q+c)], packed to the local layout the passes read.
        for c in 0..q {
            let mut acc = 0.0;
            for r in c..q {
                acc += z[(i, f * q + r)] * lam[r * q + c];
            }
            m_core_buf[i * qc + c] = acc;
        }
        // Core nested children of parent f: the indicator scaled by θ_nested (one of
        // the np slots is 1, the rest 0 — kept as written zeros so the passes sum a
        // contiguous q_core slice).
        for j in 0..np {
            let col = prim_width + f * np + j;
            m_core_buf[i * qc + q + j] = z[(i, col)] * theta_nested;
        }
        // Crossed: one nonzero per crossed grouping (its single active level), θ-pinned
        // groupings skipped.
        let mut cnt = 0usize;
        for &(theta_idx, count) in &g.crossed {
            let theta = params[theta_idx];
            if theta == 0.0 {
                continue;
            }
            let off = g.extra_offsets[theta_idx - base_theta];
            for col in off..off + count {
                let zv = z[(i, col)];
                if zv != 0.0 {
                    cross_col[i * g_cap + cnt] = (col - k_family) as u32;
                    cross_val[i * g_cap + cnt] = zv * theta;
                    cnt += 1;
                    break;
                }
            }
        }
        n_cross[i] = cnt as u8;
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
    k: usize,
    p: usize,
    m: MatRef<f64>,
    x: MatRef<f32>,
    y: &[f32],
    beta: &[f64],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    eta_fixed: &mut [f64],
    mu: &mut [f64],
    wm: &mut Mat<f64>,
    a: &mut Mat<f64>,
    a_rhs: &mut [f64],
    n: usize,
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
        for j in 0..p {
            e += x[(i, j)] as f64 * beta[j];
        }
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
            Accum::Replace,
            m,
            MatRef::from_column_major_slice(&u[..k], k, 1),
            1.0,
            Par::Seq,
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
            for i in 0..n {
                wm[(i, c)] = w[i] * m[(i, c)];
            }
        }
        triangular::matmul(
            a.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            m.transpose(),
            BlockStructure::Rectangular,
            wm.as_ref().subrows(0, n),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        for r in 0..k {
            a[(r, r)] += 1.0;
        }
        // IRLS RHS M′(W·Mu + (y − p)): fold the residual into mu in place
        // (mu is dead until next iteration's refill), then one GEMV.
        for i in 0..n {
            mu[i] = w[i] * mu[i] + (y[i] as f64 - prob[i]);
        }
        matmul(
            MatMut::from_column_major_slice_mut(&mut a_rhs[..k], k, 1),
            Accum::Replace,
            m.transpose(),
            MatRef::from_column_major_slice(&mu[..n], n, 1),
            1.0,
            Par::Seq,
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
        for c in 0..k {
            u[c] = a_rhs[c];
            pen += u[c] * u[c];
        }
        let penalized = dev + pen;
        if (penalized - pen_prev).abs() < PIRLS_TOL_REL * (1.0 + penalized.abs()) {
            converged = true;
            // log|A| off the factor that just solved for the final u (= chol of the
            // final-iterate A the caller would otherwise re-factor). Reusing it
            // drops one O(k³) factorization per Laplace eval.
            for r in 0..k {
                logdet += ac.L()[(r, r)].ln();
            }
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
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    x: MatRef<f32>,
    y: &[f32],
    beta: &[f64],
    lam: &[f64],
    z_buf: &[f64],
    m_buf: &mut [f64],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    eta_fixed: &mut [f64],
    a_blocks: &mut [f64],
    a_rhs: &mut [f64],
    n: usize,
) -> (f64, f64, f64, bool) {
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += x[(i, j)] as f64 * beta[j];
        }
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
                let zr = if r == 0 {
                    1.0
                } else {
                    z_buf[i * (q - 1) + (r - 1)]
                };
                acc += zr * lam[r * q + c];
            }
            m_buf[i * q + c] = acc;
        }
    }
    let mut pen_prev = f64::INFINITY;
    let mut converged = false;
    let (mut dev, mut pen, mut logdet) = (f64::NAN, f64::NAN, 0.0);
    for _ in 0..PIRLS_MAX_ITERS {
        for v in a_blocks[..s * q * q].iter_mut() {
            *v = 0.0;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = 0.0;
        }
        // Loop-split (Phase 4): the former interleaved transcendental+scatter sweep
        // becomes η-pass → SIMD pass → scatter-pass so the transcendental runs
        // vectorized over a materialized η[] with no gather/scatter data deps.
        // --- pass 1: η-pass (scalar gather): form ηᵢ, accumulate Σ y·η ---
        let mut yeta = 0.0;
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let ubase = cluster_ids[i] as usize * q;
            let mut e = eta_fixed[i];
            for c in 0..q {
                e += m_row[c] * u[ubase + c];
            }
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
                for c in 0..=r {
                    a_blocks[ablk + r * q + c] += wr * m_row[c];
                }
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
            for r in 0..q {
                a_blocks[ablk + r * q + r] += 1.0;
            }
            if !glmm_block_chol(&mut a_blocks[ablk..ablk + q * q], q) {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            for r in 0..q {
                logdet += a_blocks[ablk + r * q + r].ln();
            }
            // solve u_new_f = A_f⁻¹ rhs_f in place (rhs lives in a_rhs[ubase..], copy to u).
            u[ubase..ubase + q].copy_from_slice(&a_rhs[ubase..ubase + q]);
            glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut u[ubase..ubase + q]);
            for r in 0..q {
                pen += u[ubase + r] * u[ubase + r];
            }
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

/// Factor the structured `A = [[D, C], [C', E]]` in place: Crout-factor each
/// core block `core_blocks[f]` (holding `D_f + I` on entry, its lower `L` on
/// return) and build + factor the Schur complement `schur_blk` (holding `E + I`
/// on entry, `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f` then its `L` on return). Returns
/// `Some(log|A|) = Σ_f log|A_f| + log|S|` (Schur determinant identity), or `None`
/// on a non-PD core block / Schur. `coupling[f]` holds `C_f` (q_core×e row-major),
/// unchanged. Shared by `pirls_solve_blocked_extras` (per iteration) and
/// `structured_schur_fill` (reusing the converged factors). `q_core ≤ MAX_PRIMARY_Q`.
fn structured_factor(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &mut [f64],
    coupling: &[f64],
    schur_blk: &mut [f64],
) -> Option<f64> {
    use crate::lmm::MAX_PRIMARY_Q;
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let mut logdet = 0.0;
    for f in 0..s {
        let cb = f * qc * qc;
        if !glmm_block_chol(&mut core_blocks[cb..cb + qc * qc], qc) {
            return None;
        }
        for r in 0..qc {
            logdet += core_blocks[cb + r * qc + r].ln();
        }
        // S −= C_f' A_f⁻¹ C_f (lower triangle), one crossed column at a time.
        let coup = f * qc * e;
        let mut ycol = [0.0_f64; MAX_PRIMARY_Q];
        for b in 0..e {
            for local in 0..qc {
                ycol[local] = coupling[coup + local * e + b];
            }
            glmm_block_solve(&core_blocks[cb..cb + qc * qc], qc, &mut ycol[..qc]);
            // y = A_f⁻¹ C_f[:,b]; S[a][b] −= Σ_local C_f[local][a]·y[local].
            #[allow(clippy::needless_range_loop)]
            for a in b..e {
                let mut acc = 0.0;
                for local in 0..qc {
                    acc += coupling[coup + local * e + a] * ycol[local];
                }
                schur_blk[a * e + b] -= acc;
            }
        }
    }
    if e > 0 {
        if !glmm_block_chol(&mut schur_blk[..e * e], e) {
            return None;
        }
        for b in 0..e {
            logdet += schur_blk[b * e + b].ln();
        }
    }
    Some(logdet)
}

/// Apply `A⁻¹` to a packed RHS in place, using the factors `structured_factor`
/// left in `core_blocks`/`schur_blk` and the coupling `C_f`. `a_rhs` arrives
/// packed `[g_core in (f,local) order | g_e]` (core part `s·q_core` then crossed
/// `e`) and returns `[u_core | u_e]`: `t_f = A_f⁻¹ g_{core,f}`,
/// `u_e = S⁻¹(g_e − Σ_f C_f' t_f)`, `u_{core,f} = t_f − A_f⁻¹(C_f u_e)`. `e=0`
/// (nested only) stops after `t_f`. Shared by the structured PIRLS solve and the
/// inference Schur fill.
fn structured_ainv_solve(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &[f64],
    coupling: &[f64],
    schur_blk: &[f64],
    a_rhs: &mut [f64],
) {
    use crate::lmm::MAX_PRIMARY_Q;
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let k_family = qc * s;
    // t_f = A_f⁻¹ g_{core,f} (overwrites a_rhs core); g_e −= Σ_f C_f' t_f.
    for f in 0..s {
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = f * qc * e;
        glmm_block_solve(
            &core_blocks[cb..cb + qc * qc],
            qc,
            &mut a_rhs[gcb..gcb + qc],
        );
        for b in 0..e {
            let mut acc = 0.0;
            for local in 0..qc {
                acc += coupling[coup + local * e + b] * a_rhs[gcb + local];
            }
            a_rhs[k_family + b] -= acc;
        }
    }
    if e == 0 {
        return;
    }
    glmm_block_solve(&schur_blk[..e * e], e, &mut a_rhs[k_family..k_family + e]);
    // u_{core,f} = t_f − A_f⁻¹(C_f u_e).
    for f in 0..s {
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = f * qc * e;
        let mut v = [0.0_f64; MAX_PRIMARY_Q];
        #[allow(clippy::needless_range_loop)]
        for local in 0..qc {
            let mut acc = 0.0;
            for b in 0..e {
                acc += coupling[coup + local * e + b] * a_rhs[k_family + b];
            }
            v[local] = acc;
        }
        glmm_block_solve(&core_blocks[cb..cb + qc * qc], qc, &mut v[..qc]);
        for local in 0..qc {
            a_rhs[gcb + local] -= v[local];
        }
    }
}

/// Structured PIRLS for the intercept-only crossed/nested regime
/// (`groupings.structured_extras_eligible()`): `A = M'WM + I` is
/// `[[D, C], [C', E]]` where `D` is block-diagonal over the primary clusters
/// (each `q_core×q_core` core block = the primary RE column + its
/// nested-within-primary children) and the only dense coupling `C` is to the thin
/// crossed width `e`. A per-row scatter (each row touches ONE primary cluster's
/// core columns + one crossed level per crossed grouping — `M = ZΛ` sparsity, NOT
/// row contiguity, so this is layout-independent) assembles the core blocks, the
/// coupling `C_f`, the crossed `E`, and the RHS `g = M'(W·Mu + (y−p))`; then a
/// Schur complement on `e` solves it: factor each `A_f = chol(D_f+I)`, form
/// `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f`, back-substitute `u_e = S⁻¹(g_e − Σ_f C_f'
/// A_f⁻¹ g_{core,f})` then `u_{core,f} = A_f⁻¹ g_{core,f} − A_f⁻¹ C_f u_e`, and
/// `log|A| = Σ_f log|A_f| + log|S|` (Schur determinant identity). `e=0` (nested
/// only) skips the Schur entirely. Collapses the dense `O(n·k²)` Gram + `O(k³)`
/// factor to `O(n·q_core²)` scatter + `O(s·q_core³ + e³)` factor. NOT bit-identical
/// to `pirls_solve` (scatter vs GEMM accumulation) but the same estimator.
/// The `M = ZΛ` nonzeros arrive PACKED (`m_core_buf` core slice + `cross_*`/
/// `n_cross` crossed entries the caller filled via `build_packed_m`), never the
/// dense faer `m`. Leaves
/// `core_blocks` (per-cluster L) + `schur_blk` (Schur L) + `coupling` FACTORED for
/// `structured_schur_fill` to reuse, and eta/prob/w/u filled. Returns
/// `(dev, ‖u‖², log|A|, converged)`; a non-PD core block / Schur ⇒
/// `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (warm-start
/// seed); the caller owns resetting it per fit. `a_rhs` (length ≥ k_total) is the
/// RHS scratch, packed `[g_core in (f,local) order | g_e]`.
#[allow(clippy::too_many_arguments)]
fn pirls_solve_blocked_extras(
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    m_core_buf: &[f64],
    cross_val: &[f64],
    cross_col: &[u32],
    n_cross: &[u8],
    x: MatRef<f32>,
    y: &[f32],
    beta: &[f64],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    eta_fixed: &mut [f64],
    mu: &mut [f64],
    core_blocks: &mut [f64],
    coupling: &mut [f64],
    schur_blk: &mut [f64],
    a_rhs: &mut [f64],
    n: usize,
) -> (f64, f64, f64, bool) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np; // core-block width; ≤ MAX_PRIMARY_Q by eligibility
    let s = g.n_primary;
    let prim_width = q * s;
    let k_family = qc * s; // = prim_width + s·np
    let e = g.k_crossed(); // crossed width; 0 ⇒ no Schur
    let k = k_family + e; // = g.k_total
    let p = beta.len();
    // The j-th core-block-local column of cluster f maps to RE column:
    // local < q ⇒ primary component (f·q + local); else ⇒ nested child
    // (prim_width + f·np + (local−q)). Single source for the gather/scatter.
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut ef = 0.0;
        for j in 0..p {
            ef += x[(i, j)] as f64 * beta[j];
        }
        eta_fixed[i] = ef;
    }
    let mut pen_prev = f64::INFINITY;
    let mut converged = false;
    let (mut dev, mut pen, mut logdet) = (f64::NAN, f64::NAN, 0.0);
    for _ in 0..PIRLS_MAX_ITERS {
        // --- pass 1: η-pass — ηᵢ = η_fixed,ᵢ + (Mu)ᵢ over the row's nonzeros ---
        // Reads the packed M nonzeros (contiguous q_core core slice + n_cross[i]
        // crossed entries) `build_packed_m` filled — no faer indexing, crossed term
        // O(G) not O(e).
        let mut yeta = 0.0;
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let mut mui = 0.0;
            for local in 0..qc {
                mui += m_core[local] * u[core_col(f, local)];
            }
            let cbase = i * g_cap;
            for z in 0..n_cross[i] as usize {
                let b = cross_col[cbase + z] as usize;
                mui += cross_val[cbase + z] * u[k_family + b];
            }
            eta[i] = eta_fixed[i] + mui;
            mu[i] = mui; // keep (Mu)ᵢ for the IRLS residual below
            yeta += y[i] as f64 * eta[i];
        }
        // --- pass 2: SIMD — η[] → prob[]/w[]; Σ log1pexp (lane-wise reduce) ---
        let lp_sum =
            crate::simd_transcendental::pw_and_log1pexp_sum(&eta[..n], &mut prob[..n], &mut w[..n]);
        dev = 2.0 * (lp_sum - yeta);
        // IRLS effective residual rᵢ = wᵢ·(Mu)ᵢ + (yᵢ−pᵢ), so the scattered RHS is
        // M'(W·Mu + (y−p)) = (A−I)u + M'(y−p) — matching `pirls_solve`'s GEMV RHS.
        for i in 0..n {
            mu[i] = w[i] * mu[i] + (y[i] as f64 - prob[i]);
        }
        // --- pass 3: scatter — wᵢmᵢmᵢ' into D_f/C_f/E (lower tri), rᵢmᵢ into g ---
        for v in core_blocks[..s * qc * qc].iter_mut() {
            *v = 0.0;
        }
        for v in coupling[..s * qc * e].iter_mut() {
            *v = 0.0;
        }
        for v in schur_blk[..e * e].iter_mut() {
            *v = 0.0;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = 0.0;
        }
        // Reads the packed core slice + crossed nonzeros directly — no per-row stack
        // copy / rescan; the `m_core`/`cz_*` rebuild the dense path needed is gone.
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let wi = w[i];
            let ri = mu[i]; // effective residual
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let cbase = i * g_cap;
            let ncz = n_cross[i] as usize;
            let cb = f * qc * qc;
            let gcb = f * qc;
            let coup = f * qc * e;
            for r in 0..qc {
                let mr = m_core[r];
                a_rhs[gcb + r] += mr * ri;
                let wmr = wi * mr;
                for c in 0..=r {
                    core_blocks[cb + r * qc + c] += wmr * m_core[c];
                }
                for z in 0..ncz {
                    coupling[coup + r * e + cross_col[cbase + z] as usize] +=
                        wmr * cross_val[cbase + z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let vb = cross_val[cbase + z];
                a_rhs[k_family + b] += vb * ri;
                let wvb = wi * vb;
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        schur_blk[b * e + b2] += wvb * cross_val[cbase + z2];
                    }
                }
            }
        }
        // --- +I ridge (per RE column) on the core diagonals and the E diagonal ---
        for f in 0..s {
            let cb = f * qc * qc;
            for r in 0..qc {
                core_blocks[cb + r * qc + r] += 1.0;
            }
        }
        for b in 0..e {
            schur_blk[b * e + b] += 1.0;
        }
        // --- factor (core blocks + Schur), then apply A⁻¹ to the scattered RHS ---
        logdet = match structured_factor(g, core_blocks, coupling, schur_blk) {
            Some(ld) => ld,
            None => return (f64::NAN, f64::NAN, f64::NAN, false),
        };
        structured_ainv_solve(g, core_blocks, coupling, schur_blk, a_rhs);
        // a_rhs[gcb..] now holds u_{core,f}; a_rhs[k_family+b] holds u_e. Scatter to
        // u (RE-column order) and accumulate ‖u‖².
        pen = 0.0;
        for f in 0..s {
            let gcb = f * qc;
            for local in 0..qc {
                let val = a_rhs[gcb + local];
                u[core_col(f, local)] = val;
                pen += val * val;
            }
        }
        for b in 0..e {
            let val = a_rhs[k_family + b];
            u[k_family + b] = val;
            pen += val * val;
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
    groupings: &LmmGroupings,
    params: &[f64],
    z: MatRef<f64>,
    m: &mut Mat<f64>,
    lam: &mut [f64],
    z_buf: &[f64],
    m_buf: &mut [f64],
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    eta_fixed: &mut [f64],
    mu: &mut [f64],
    wm: &mut Mat<f64>,
    a: &mut Mat<f64>,
    a_rhs: &mut [f64],
    a_blocks: &mut [f64],
    core_blocks: &mut [f64],
    coupling: &mut [f64],
    schur_blk: &mut [f64],
    m_core_buf: &mut [f64],
    cross_val: &mut [f64],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    p: usize,
    n: usize,
) -> f64 {
    let k = groupings.k_total;
    let n_theta = groupings.n_theta();
    let (dev, pen, logdet, conv) = if groupings.extra_offsets.is_empty() {
        // No extras ⇒ A is block-diagonal: reconstruct mᵢ per row, never build Z/M.
        crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, lam);
        pirls_solve_blocked(
            groupings,
            cluster_ids,
            x,
            y,
            &params[n_theta..n_theta + p],
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            a_blocks,
            a_rhs,
            n,
        )
    } else if groupings.structured_extras_eligible() {
        // Intercept-only crossed/nested ⇒ block-diagonal core + Schur on the
        // crossed width. The M = ZΛ nonzeros are packed once here (core slice +
        // crossed entries) instead of materializing the dense n×k M every eval; the
        // structured passes read the packed buffers. `z`/`m` are untouched on this
        // path now (the dense `m` only feeds the genuinely-dense fallback below).
        build_packed_m(
            groupings,
            params,
            z,
            lam,
            cluster_ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            n,
        );
        pirls_solve_blocked_extras(
            groupings,
            cluster_ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            x,
            y,
            &params[n_theta..n_theta + p],
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            a_rhs,
            n,
        )
    } else {
        // Non-eligible extras (oversized core) ⇒ A genuinely dense: dense fallback.
        apply_lambda(groupings, params, z, m, lam, n);
        pirls_solve(
            k,
            p,
            m.as_ref(),
            x,
            y,
            &params[n_theta..n_theta + p],
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            n,
        )
    };
    if !conv || !dev.is_finite() {
        return f64::INFINITY;
    }
    dev + pen + 2.0 * logdet
}

/// Evaluate the joint Laplace deviance at the params CURRENTLY in `ws.params`
/// (the FD loop in `fd_hessian_cov` writes them before each call). Borrow-split
/// twin of `fit_glmm`'s BOBYQA-closure body: destructures the workspace into the
/// disjoint borrows `laplace_deviance` needs (z read; m/lam/a/etc. written) and
/// calls it. Seeds the PIRLS conditional modes from u = 0 each call so the
/// returned deviance is independent of evaluation order — the FD second
/// differences need each `f(γ)` to depend only on γ, not on the warm-start
/// history of neighbouring perturbations.
///
/// Caller must have filled `ws.z_buf` for this fit's `x` (blocked path) — `x` is
/// constant across all FD perturbations, so fill it ONCE before the FD loop, not
/// per eval (`fd_hessian_cov` does; `glmm_laplace_deviance` does it inline).
pub(crate) fn laplace_deviance_at(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let k = ws.k;
    for v in ws.u[..k.max(1)].iter_mut() {
        *v = 0.0;
    }
    let GlmmWorkspace {
        groupings,
        params: prm,
        p,
        z,
        m,
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        eta_fixed,
        mu,
        wm,
        a,
        a_rhs,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        ..
    } = ws;
    laplace_deviance(
        groupings,
        &prm[..],
        z.as_ref(),
        m,
        lam,
        z_buf,
        m_buf,
        x,
        y,
        cluster_ids,
        eta,
        prob,
        w,
        u,
        eta_fixed,
        mu,
        wm,
        a,
        a_rhs,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        *p,
        n,
    )
}

/// Workspace-bound wrapper for `laplace_deviance`: copies `params` into the
/// workspace, fills `z_buf`, then delegates to the shared `laplace_deviance_at`.
/// Test-only entry point — the production fit (`fit_glmm`) destructures the
/// workspace and calls `laplace_deviance` directly (the BOBYQA closure and the
/// pinned-γ̂ re-eval both inline it), so this exists purely to drive the deviance
/// from a `&[f64]` in tests.
#[cfg(test)]
pub(crate) fn glmm_laplace_deviance(
    params: &[f64],
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    laplace_deviance_at(ws, x, y, cluster_ids, n)
}

/// Outcome of the FD-Hessian fixed-effect covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdHessianStatus {
    /// The joint-deviance Hessian was PD and every perturbed eval finite; the
    /// returned covariance is `2·(H_dev⁻¹)_ββ` (lme4 `vcov(use.hessian = TRUE)`).
    Ok,
    /// The joint Hessian was non-PD (or a perturbed deviance was non-finite — the
    /// few-cluster failure mode); the returned covariance is the RX/Schur block.
    NonPdFellBackToRx,
}

/// Evaluate the joint Laplace deviance at `fd_saved + Σ deltaₖ·e_{coordₖ}`,
/// reusing `ws.fd_saved` (distinct field from `ws.params`, so the disjoint
/// field borrows are legal). `coords`/`deltas` are ≤ 2 long (a diagonal or a
/// mixed partial). Leaves `ws.params` perturbed — callers restore from
/// `ws.fd_saved` between the directional evals via this same write.
fn fd_eval(
    ws: &mut GlmmWorkspace,
    coords: &[usize],
    deltas: &[f64],
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let m = ws.fd_saved.len();
    ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
    for (&c, &d) in coords.iter().zip(deltas) {
        ws.params[c] += d;
    }
    laplace_deviance_at(ws, x, y, cluster_ids, n)
}

/// Fill `out_cov` (p×p) with the RX/Schur fixed-effect covariance `inv(ws.schur)`
/// — `ws.schur` is the β-INFORMATION matrix, so the inverse is the covariance
/// directly (NO factor of 2; that factor only applies to the deviance Hessian,
/// where info = H_dev/2). Reuses `fit_glmm`'s inference-block Schur-fill dispatch
/// (`blocked`/`structured`/`dense`), so it requires `ws.{w, lam, a_blocks, …}` to
/// hold the factors a converged PIRLS at the current `ws.params` left behind.
/// Returns false on a non-PD Schur. Shared by the `fd_hessian_cov` fallback and
/// (later) the Rx production path.
pub(crate) fn rx_cov_into(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> bool {
    use faer::linalg::solvers::Solve;
    let inf_ok = if ws.groupings.extra_offsets.is_empty() {
        blocked_schur_fill(ws, x, cluster_ids, n)
    } else if ws.groupings.structured_extras_eligible() {
        structured_schur_fill(ws, x, cluster_ids, n)
    } else {
        dense_schur_fill(ws, x, n)
    };
    if !inf_ok {
        return false;
    }
    let chol = match ws.schur.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let mut inv = Mat::<f64>::identity(p, p);
    chol.solve_in_place(inv.as_mut());
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = inv[(a, b)];
        }
    }
    true
}

/// Finite-difference Hessian of the Laplace deviance over the joint (θ,β) at the
/// converged point in `ws.params`; inverts and writes the p×p fixed-effect
/// covariance block into `out_cov`. On a non-PD joint Hessian (or a non-finite
/// perturbed deviance — the few-cluster failure mode) writes the RX/Schur
/// covariance instead and returns the fallback status. Restores `ws.params` on
/// return. Matches glmer `vcov(use.hessian = TRUE)` (factor of 2: deviance =
/// −2logL, so observed info = H_dev/2 and cov = info⁻¹ = 2·H_dev⁻¹).
///
/// FD scheme (tuned against `tests/fixtures/glmm_hessian_vcov.json`, the n=96 /
/// 12-cluster `y ~ x1 + (1|grp)` glmer fit): central differences inside a
/// 2-level Richardson extrapolation (combine the step-h and step-h/2 second
/// differences as `(4·D(h/2) − D(h))/3`, cancelling the O(h²) truncation term to
/// match numDeriv's Richardson precision). Step `h_k = FD_STEP_REL·max(1, |γ̂_k|)`.
/// On this fixture the result is STEP-INVARIANT (the inverted covariance is
/// constant to ~7 sig figs across h ∈ [1e-4, 1e-1]) and PIRLS-tolerance-invariant
/// — i.e. our deviance is smooth/precise enough that this is the TRUE Hessian of
/// our Laplace deviance, not an FD approximation with residual bias.
///
/// Achieved match vs lme4 `vcov(use.hessian=TRUE)`: worst per-entry gap ~1.7e-3
/// (diagonal, ~1.3% rel) / ~3.2e-4 (off-diagonal), measured at OUR converged θ̂
/// (the production path; see the unit test). The gap is the θ↔β cross-curvature
/// term, and three facts pin its cause (none is a hand-wave):
///   1. NOT off-stationarity. Our `fit_glmm` lands on lme4's θ̂ to ~0.07% and β̂
///      to ~0.1%, and the diagonal gap is the SAME ~1.7e-3 whether the FD Hessian
///      is taken at our θ̂ or at lme4's exact θ̂ — a 0.07% θ offset cannot inflate
///      a stationary-point cross-term.
///   2. NOT our FD error. Our inverted covariance is STEP-INVARIANT to ~7 sig
///      figs across h ∈ [1e-4, 1e-1] — it is the true curvature of our deviance.
///   3. H_ββ is exact. Our closed-form RX/Schur β-covariance (`rx_cov_into`, never
///      through numDeriv) matches lme4's `vcov(use.hessian=FALSE)` to ~3.6e-6.
///
/// `use.hessian` TRUE vs FALSE differ ONLY in the θ↔β cross-curvature correction;
/// with H_ββ exact and the point θ-stationary, the residual ~1.3% lives entirely
/// in that cross-term — where lme4's numDeriv differentiates its PIRLS deviance
/// (~1e-7 pwrss noise floor amplified through small-step Hessians) while our
/// large-step Richardson FD on a deterministic deviance recovers cleaner
/// curvature. The 1% band is immaterial for power (the implied β correlation is
/// ~-0.05 either way).
///
/// `m = ws.params.len() = n_theta + p`; the β block is rows/cols `n_theta..m`.
/// Precondition: `ws` is at a CONVERGED fit and `ws.z_buf`-eligible scratch is
/// valid for (x, ids, n) (the deviance evals re-solve PIRLS).
pub fn fd_hessian_cov(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> FdHessianStatus {
    use faer::linalg::solvers::Solve;
    let m = ws.params.len();
    let n_theta = ws.n_theta;

    // Snapshot γ̂ and the per-coordinate FD step; fill z_buf once (blocked path).
    ws.fd_saved[..m].copy_from_slice(&ws.params[..m]);
    for k in 0..m {
        ws.fd_steps[k] = FD_STEP_REL * ws.fd_saved[k].abs().max(1.0);
    }
    if ws.groupings.extra_offsets.is_empty() {
        let GlmmWorkspace {
            groupings, z_buf, ..
        } = &mut *ws;
        fill_z_f64(groupings, x, z_buf, n);
    }

    // Restore γ̂ and take the RX/Schur fallback: re-eval the central deviance to
    // repopulate W̃/Λ̂/block factors at γ̂, then invert the β information.
    macro_rules! fallback {
        () => {{
            let _ = fd_eval(ws, &[], &[], x, y, cluster_ids, n);
            let ok = rx_cov_into(ws, x, cluster_ids, p, n, out_cov);
            debug_assert!(ok, "RX fallback Schur must be PD at a converged fit");
            // Double failure (joint Hessian AND RX Schur both non-PD): rx_cov_into
            // leaves out_cov UNTOUCHED on `false`, so in release it would keep stale
            // data while we still report NonPdFellBackToRx. NaN-fill so the caller
            // (Task 6 routes this to nan_fit) can detect it via is_nan().
            if !ok {
                for a in 0..p {
                    for b in 0..p {
                        out_cov[(a, b)] = f64::NAN;
                    }
                }
            }
            ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
            return FdHessianStatus::NonPdFellBackToRx;
        }};
    }

    let f0 = fd_eval(ws, &[], &[], x, y, cluster_ids, n);
    if !f0.is_finite() {
        fallback!();
    }

    // Second difference of coord k at step s, central: (f(+s)−2f0+f(−s))/s².
    macro_rules! second_diff {
        ($k:expr, $s:expr) => {{
            let s = $s;
            let fp = fd_eval(ws, &[$k], &[s], x, y, cluster_ids, n);
            let fm = fd_eval(ws, &[$k], &[-s], x, y, cluster_ids, n);
            if !(fp.is_finite() && fm.is_finite()) {
                fallback!();
            }
            (fp - 2.0 * f0 + fm) / (s * s)
        }};
    }
    // Symmetric 4-point mixed partial of (i,j) at steps (si, sj):
    // (f(+si,+sj) − f(+si,−sj) − f(−si,+sj) + f(−si,−sj)) / (4·si·sj).
    macro_rules! mixed_diff {
        ($i:expr, $j:expr, $si:expr, $sj:expr) => {{
            let (si, sj) = ($si, $sj);
            let fpp = fd_eval(ws, &[$i, $j], &[si, sj], x, y, cluster_ids, n);
            let fpm = fd_eval(ws, &[$i, $j], &[si, -sj], x, y, cluster_ids, n);
            let fmp = fd_eval(ws, &[$i, $j], &[-si, sj], x, y, cluster_ids, n);
            let fmm = fd_eval(ws, &[$i, $j], &[-si, -sj], x, y, cluster_ids, n);
            if !(fpp.is_finite() && fpm.is_finite() && fmp.is_finite() && fmm.is_finite()) {
                fallback!();
            }
            (fpp - fpm - fmp + fmm) / (4.0 * si * sj)
        }};
    }

    // Build the symmetric m×m Hessian into ws.hess_scratch (upper, then mirror).
    for i in 0..m {
        let hi = ws.fd_steps[i];
        // Diagonal: Richardson on the central second difference.
        let d_full = second_diff!(i, hi);
        let d_half = second_diff!(i, hi * 0.5);
        let hii = (4.0 * d_half - d_full) / 3.0;
        ws.hess_scratch[(i, i)] = hii;
        for j in (i + 1)..m {
            let hj = ws.fd_steps[j];
            let mx_full = mixed_diff!(i, j, hi, hj);
            let mx_half = mixed_diff!(i, j, hi * 0.5, hj * 0.5);
            let hij = (4.0 * mx_half - mx_full) / 3.0;
            ws.hess_scratch[(i, j)] = hij;
            ws.hess_scratch[(j, i)] = hij;
        }
    }

    // Invert the joint Hessian; non-PD ⇒ RX fallback. cov = 2·(H⁻¹)_ββ.
    let chol = match ws.hess_scratch.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => fallback!(),
    };
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = 2.0 * inv[(n_theta + a, n_theta + b)];
        }
    }

    ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
    FdHessianStatus::Ok
}

/// Fit the clustered-logistic GLMM. `ws.z` must already be built (build_z) for
/// this (X, ids, N). `beta_start` = spec.effect_sizes. Writes β̂/Var/z² into
/// ws.{betas,var_diag,t_sq}; returns the GlmmFit summary.
#[allow(clippy::too_many_arguments)]
pub fn fit_glmm(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    y: &[f32],
    cluster_ids: &[u32],
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    beta_start: &[f64],
    n: usize,
    wald_se: WaldSe,
) -> GlmmFit {
    let (k, p, n_theta) = (ws.k, ws.p, ws.n_theta);

    // γ₀ = [θ₀ | β₀].
    match theta_start {
        Some(ts) => {
            for (t, &v) in ws.params[..n_theta].iter_mut().zip(ts) {
                *t = v.max(THETA_TRUTH_FLOOR);
            }
        }
        None => {
            for t in ws.params[..n_theta].iter_mut() {
                *t = THETA0;
            }
        }
    }
    for (j, &b) in beta_start.iter().enumerate().take(p) {
        ws.params[n_theta + j] = b.clamp(-BETA_BOX, BETA_BOX);
    }

    // Within-fit warm-start resets per fit — the incumbent is NEVER carried across
    // fits (that is the rejected cross-sim warm-start; it would break the
    // plan_chunks/run_chunk merge and same-seed reproducibility).
    for v in ws.u_seed[..k].iter_mut() {
        *v = 0.0;
    }

    // Joint BOBYQA over [θ | β]. Borrow-split mirrors `glmm_laplace_deviance`:
    // `solver` held by `minimize`; the closure calls the shared `laplace_deviance`
    // on the disjoint scratch fields (groupings read; m/lam/eta/prob/w/u/a/a_rhs
    // written). The closure's `gamma` is BOBYQA's candidate point, not the bound
    // `params` (which `minimize` owns as its `x`).
    let GlmmWorkspace {
        solver,
        params,
        lower,
        upper,
        groupings,
        z,
        m,
        eta,
        prob,
        w,
        u,
        u_seed,
        eta_fixed,
        mu,
        wm,
        a,
        a_rhs,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        lam,
        z_buf,
        m_buf,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        p: pf,
        ..
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
                groupings,
                gamma,
                z.as_ref(),
                m,
                lam,
                z_buf,
                m_buf,
                x,
                y,
                cluster_ids,
                eta,
                prob,
                w,
                u,
                eta_fixed,
                mu,
                wm,
                a,
                a_rhs,
                a_blocks,
                core_blocks,
                coupling,
                schur_blk,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                *pf,
                n,
            );
            if obj < best_obj {
                best_obj = obj;
                u_seed[..k].copy_from_slice(&u[..k]);
            }
            obj
        },
        params,
        lower,
        upper,
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
            groupings,
            params,
            p,
            z,
            m,
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            a_blocks,
            core_blocks,
            coupling,
            schur_blk,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            ..
        } = ws;
        // z_buf still holds this fit's slope copy — x is unchanged since fill_z_f64.
        // On the structured path this re-eval re-packs m_core_buf/cross_* at γ̂, which
        // `structured_schur_fill` then reads (the dense `m` it formerly read is no
        // longer maintained here).
        let _ = laplace_deviance(
            groupings,
            &params[..],
            z.as_ref(),
            m,
            lam,
            z_buf,
            m_buf,
            x,
            y,
            cluster_ids,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            a_blocks,
            core_blocks,
            coupling,
            schur_blk,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            *p,
            n,
        );
    }

    if !ok {
        return nan_fit(ws, target_indices, out.n_eval);
    }

    // β̂ from γ̂.
    for j in 0..p {
        ws.betas[j] = ws.params[n_theta + j];
    }

    // Var(β̂): `Rx` inverts the β-information (Schur) directly — fast, but the
    // expected-information Schur complement assumes β–θ orthogonality (exact for
    // the Gaussian LMM, anticonservative for the GLMM where IRLS weights couple
    // β,θ). `Hessian` (default) sources Var(β̂) from the FD-Hessian of the joint
    // Laplace deviance (glmer `use.hessian = TRUE`), the lme4 "correct" denom.
    let mut hessian_fallback = false;
    let joint_t_sq = match wald_se {
        WaldSe::Rx => {
            // Schur fill. No-extras reuses the per-block factors the blocked PIRLS
            // left in ws.a_blocks; structured-eligible extras reuse the core+Schur
            // factors the structured PIRLS left in ws.{core_blocks, schur_blk,
            // coupling}; the dense fallback factors ws.a.
            let inf_ok = if ws.groupings.extra_offsets.is_empty() {
                blocked_schur_fill(ws, x, cluster_ids, n)
            } else if ws.groupings.structured_extras_eligible() {
                structured_schur_fill(ws, x, cluster_ids, n)
            } else {
                dense_schur_fill(ws, x, n)
            };
            if !inf_ok {
                return nan_fit(ws, target_indices, out.n_eval);
            }
            // Var(β̂)_jj from chol(Schur) forward-solve (mirrors fit_lmm's recovery).
            let sc = match ws.schur.as_ref().llt(faer::Side::Lower) {
                Ok(c) => c,
                Err(_) => return nan_fit(ws, target_indices, out.n_eval),
            };
            let lschur = sc.L();
            for &tj in target_indices {
                let tj = tj as usize;
                // Forward-solve into reusable scratch; fwd_solve[i] is written before
                // it is read as fwd_solve[kk] (kk < i), so no per-target zero-fill is
                // needed.
                for i in 0..p {
                    let mut acc = if i == tj { 1.0 } else { 0.0 };
                    for kk in 0..i {
                        acc -= lschur[(i, kk)] * ws.fwd_solve[kk];
                    }
                    ws.fwd_solve[i] = acc / lschur[(i, i)];
                }
                let vd: f64 = ws.fwd_solve[..p].iter().map(|v| v * v).sum();
                ws.var_diag[tj] = vd;
                ws.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
                    ws.betas[tj] * ws.betas[tj] / vd
                } else {
                    f64::NAN
                };
            }
            // Joint Wald-χ² via the lme helper (Schur is the β-information; scale 1.0).
            if target_indices.is_empty() {
                f64::NAN
            } else {
                crate::lme::joint_wald_chi_sq(
                    ws.schur.as_ref(),
                    &ws.betas,
                    1.0,
                    target_indices,
                    ws.joint_k_inv.as_mut(),
                    ws.joint_sigma_t_chol.as_mut(),
                    &mut ws.joint_rhs,
                )
            }
        }
        WaldSe::Hessian => {
            // FD-Hessian covariance into a LOCAL p×p Mat — NOT a ws field: the kernel
            // takes `&mut ws`, so `&mut ws.<field>` for out_cov would alias it. The
            // allocation is acceptable on this default path (the zero-alloc gate
            // pins the Rx warm path). The kernel re-evals PIRLS itself and its
            // RX fallback runs schur_fill internally, so skip the schur_fill above.
            let mut cov = Mat::<f64>::zeros(p, p);
            let status = fd_hessian_cov(ws, x, y, cluster_ids, p, n, &mut cov);
            // Double-failure sentinel: the kernel NaN-fills `cov` when BOTH the joint
            // Hessian and the RX fallback fail. Treat as a failed fit.
            if !cov[(0, 0)].is_finite() {
                return nan_fit(ws, target_indices, out.n_eval);
            }
            hessian_fallback = matches!(status, FdHessianStatus::NonPdFellBackToRx);
            // Marginal var/t² straight off the covariance diagonal.
            for &tj in target_indices {
                let tj = tj as usize;
                let vd = cov[(tj, tj)];
                ws.var_diag[tj] = vd;
                ws.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
                    ws.betas[tj] * ws.betas[tj] / vd
                } else {
                    f64::NAN
                };
            }
            // Joint Wald-χ²: `joint_wald_chi_sq` expects the β-INFORMATION (it inverts
            // and sub-blocks internally), so pass info = cov⁻¹. Write cov⁻¹ into the
            // now-free ws.schur (schur_fill was skipped on this arm) and reuse the
            // helper verbatim — same faer LLT-inverse idiom as `rx_cov_into`.
            if target_indices.is_empty() {
                f64::NAN
            } else {
                use faer::linalg::solvers::Solve;
                match cov.as_ref().llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let mut inv = Mat::<f64>::identity(p, p);
                        chol.solve_in_place(inv.as_mut());
                        for a in 0..p {
                            for b in 0..p {
                                ws.schur[(a, b)] = inv[(a, b)];
                            }
                        }
                        crate::lme::joint_wald_chi_sq(
                            ws.schur.as_ref(),
                            &ws.betas,
                            1.0,
                            target_indices,
                            ws.joint_k_inv.as_mut(),
                            ws.joint_sigma_t_chol.as_mut(),
                            &mut ws.joint_rhs,
                        )
                    }
                    Err(_) => f64::NAN,
                }
            }
        }
    };

    // τ̂² = D̂[0][0] = (Λ̂Λ̂')[0][0]. No σ² (binomial). For lower-tri Λ_p stored
    // row-major (lam[r*q + c]), row 0 has only the (0,0) entry nonzero, so
    // D̂[0][0] = Σ_c Λ[0,c]² = Λ[0,0]² — the random-INTERCEPT variance.
    crate::lmm::primary_lambda(&ws.params[..n_theta], ws.groupings.primary_q, &mut ws.lam);
    let q = ws.groupings.primary_q;
    let mut d00 = 0.0;
    for r in 0..q {
        d00 += ws.lam[r] * ws.lam[r];
    }
    GlmmFit {
        converged: true,
        boundary_hit: u8::from(pinned),
        pinned_components,
        n_eval: out.n_eval,
        tau_squared_hat: d00,
        joint_t_sq,
        hessian_fallback,
    }
}

/// NaN-fill the inference outputs on a non-converged / Schur-failure fit, mirror
/// `fit_lmm`'s NaN-fill branch (boundary_hit = 2 = optimizer/Schur failure).
fn nan_fit(ws: &mut GlmmWorkspace, targets: &[u32], n_eval: usize) -> GlmmFit {
    for v in ws.betas.iter_mut() {
        *v = f64::NAN;
    }
    for &t in targets {
        ws.var_diag[t as usize] = f64::NAN;
        ws.t_sq[t as usize] = f64::NAN;
    }
    GlmmFit {
        converged: false,
        boundary_hit: 2,
        pinned_components: 0,
        n_eval,
        tau_squared_hat: f64::NAN,
        joint_t_sq: f64::NAN,
        hessian_fallback: false,
    }
}

/// Dense Schur fill (crossed/nested path): X'W̃X, X'W̃M, A⁻¹M'W̃X via the `k×k`
/// `ws.a` LLT, and `ws.schur = X'W̃X − X'W̃M·A⁻¹M'W̃X`. Reads `ws.{a, m, w, x via
/// arg}`. Returns false on a non-PD `ws.a`. Unchanged from the pre-Phase-2 inline
/// inference — moved verbatim so the crossed path is byte-for-byte identical.
fn dense_schur_fill(ws: &mut GlmmWorkspace, x: MatRef<f32>, n: usize) -> bool {
    use faer::linalg::solvers::Solve;
    let (k, p) = (ws.k, ws.p);
    for r in 0..p {
        for c in 0..=r {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] as f64 * ws.w[i] * x[(i, c)] as f64;
            }
            ws.xtwx[(r, c)] = s;
            ws.xtwx[(c, r)] = s;
        }
    }
    for r in 0..p {
        for c in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] as f64 * ws.w[i] * ws.m[(i, c)];
            }
            ws.xtwm[(r, c)] = s;
        }
    }
    let ac = match ws.a.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return false,
    };
    for r in 0..k {
        for c in 0..p {
            ws.ainv_mtwx[(r, c)] = ws.xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ws.ainv_mtwx.as_mut());
    for r in 0..p {
        for c in 0..p {
            let mut s = ws.xtwx[(r, c)];
            for j in 0..k {
                s -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = s;
        }
    }
    true
}

/// Blocked Schur fill (no-extras path): reconstruct mᵢ = Λ_p'·zᵢ per row to build
/// X'W̃X (p×p, dense) and the per-cluster coupling X'W̃M (into `ws.xtwm` columns
/// `f·q_p..`), then solve `A_f T_f = (M'W̃X)_f` per block by REUSING the factored
/// `ws.a_blocks` the converged blocked PIRLS left behind (W̃ in `ws.w`, Λ̂ in
/// `ws.lam`), and `ws.schur = X'W̃X − Σ_f (X'W̃M)_f·T_f`. Only the trailing `p×p`
/// Schur LLT (done by the common code after this) stays dense. Returns false if a
/// stored block is not usable (defensive — the PIRLS already proved them PD).
fn blocked_schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let (p, q, s) = (ws.p, ws.groupings.primary_q, ws.groupings.n_primary);
    let k = q * s;
    // X'W̃X (p×p).
    for r in 0..p {
        for c in 0..=r {
            let mut sm = 0.0;
            for i in 0..n {
                sm += x[(i, r)] as f64 * ws.w[i] * x[(i, c)] as f64;
            }
            ws.xtwx[(r, c)] = sm;
            ws.xtwx[(c, r)] = sm;
        }
    }
    // X'W̃M, blocked: zero then scatter the q_p coupling columns per row.
    for r in 0..p {
        for c in 0..k {
            ws.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let mut m_row = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
        #[allow(clippy::needless_range_loop)]
        for c in 0..q {
            let mut acc = 0.0;
            for rr in c..q {
                let zr = if rr == 0 {
                    1.0
                } else {
                    x[(i, ws.groupings.primary_slope_cols[rr - 1])] as f64
                };
                acc += zr * ws.lam[rr * q + c];
            }
            m_row[c] = acc;
        }
        let wi = ws.w[i];
        for r in 0..p {
            let xw = x[(i, r)] as f64 * wi;
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.xtwm[(r, f * q + c)] += xw * m_row[c];
            }
        }
    }
    // T_f = A_f⁻¹ (M'W̃X)_f, per block, reusing the stored factor; ainv_mtwx rows
    // f·q_p.. hold T_f. (M'W̃X)_f[c, col] = (X'W̃M)_f[col, c] = ws.xtwm[(col, f·q+c)].
    for f in 0..s {
        let ablk = f * q * q;
        for col in 0..p {
            let mut rhs = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                rhs[c] = ws.xtwm[(col, f * q + c)];
            }
            glmm_block_solve(&ws.a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.ainv_mtwx[(f * q + c, col)] = rhs[c];
            }
        }
    }
    // Schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X). Exact: A is block-diagonal, so the per-block
    // solves above equal the full A⁻¹M'W̃X; the Σ_j over k is a full sum (every
    // column j belongs to one cluster and is populated — there are no zero columns).
    for r in 0..p {
        for c in 0..p {
            let mut sm = ws.xtwx[(r, c)];
            for j in 0..k {
                sm -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = sm;
        }
    }
    true
}

/// Structured Schur fill (intercept-only crossed/nested path): builds `X'W̃X`
/// (p×p) and `X'W̃M` (p×k, by per-row scatter into each row's core + crossed
/// columns), then applies `A⁻¹` to each of the `p` columns of `M'W̃X` by REUSING
/// the core-block + Schur factors the converged structured PIRLS left in
/// `ws.{core_blocks, schur_blk, coupling}` (via `structured_ainv_solve`), and
/// `ws.schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X)`. Mirrors `blocked_schur_fill`; the only
/// difference is the `A⁻¹` apply uses the block+Schur back-substitution instead of
/// per-block solves alone. Returns false on nothing (the factors were already
/// proven PD by the PIRLS) — kept `-> bool` to match the dispatch arms.
fn structured_schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f32>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let p = ws.p;
    let g = &ws.groupings;
    let (q, np, s) = (g.primary_q, g.nested_per_parent, g.n_primary);
    let qc = q + np;
    let e = g.k_crossed();
    let prim_width = q * s;
    let k_family = qc * s;
    let k = ws.k;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    // X'W̃X (p×p).
    for r in 0..p {
        for c in 0..=r {
            let mut sm = 0.0;
            for i in 0..n {
                sm += x[(i, r)] as f64 * ws.w[i] * x[(i, c)] as f64;
            }
            ws.xtwx[(r, c)] = sm;
            ws.xtwx[(c, r)] = sm;
        }
    }
    // X'W̃M: zero then scatter each row's core + crossed columns. Reads the PACKED
    // M nonzeros (`m_core_buf` core slice + `cross_*`/`n_cross` crossed entries) the
    // converged re-eval's `build_packed_m` left behind — the dense `ws.m` is no
    // longer maintained on the structured path.
    for r in 0..p {
        for c in 0..k {
            ws.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let wi = ws.w[i];
        let cbase = i * g_cap;
        let ncz = ws.n_cross[i] as usize;
        for r in 0..p {
            let xw = x[(i, r)] as f64 * wi;
            for local in 0..qc {
                ws.xtwm[(r, core_col(f, local))] += xw * ws.m_core_buf[i * qc + local];
            }
            for z in 0..ncz {
                let b = ws.cross_col[cbase + z] as usize;
                ws.xtwm[(r, k_family + b)] += xw * ws.cross_val[cbase + z];
            }
        }
    }
    // ainv_mtwx[:, c] = A⁻¹ (M'W̃X)[:, c], one fixed-effect column at a time.
    for c in 0..p {
        for f in 0..s {
            for local in 0..qc {
                ws.a_rhs[f * qc + local] = ws.xtwm[(c, core_col(f, local))];
            }
        }
        for b in 0..e {
            ws.a_rhs[k_family + b] = ws.xtwm[(c, k_family + b)];
        }
        structured_ainv_solve(
            &ws.groupings,
            &ws.core_blocks,
            &ws.coupling,
            &ws.schur_blk,
            &mut ws.a_rhs,
        );
        for f in 0..s {
            for local in 0..qc {
                ws.ainv_mtwx[(core_col(f, local), c)] = ws.a_rhs[f * qc + local];
            }
        }
        for b in 0..e {
            ws.ainv_mtwx[(k_family + b, c)] = ws.a_rhs[k_family + b];
        }
    }
    // Schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X). Every RE column belongs to a core block or
    // the crossed tail and is populated, so the Σ_j over k is a full sum.
    for r in 0..p {
        for c in 0..p {
            let mut sm = ws.xtwx[(r, c)];
            for j in 0..k {
                sm -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = sm;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    // Tiny deterministic LCG → reproducible test data without RNG-determinism caveats.
    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
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
    fn glmm_slope_crossed_dataset() -> (
        Mat<f64>,
        Vec<f32>,
        Vec<u32>,
        Vec<u32>,
        engine_contract::ClusterSpec,
    ) {
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
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            let eta = 0.2 + 0.8 * x1 + u0[c] + u1[c] * x1 + uc[cc];
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        // Slope on design col 0 of the [1, x1] X passed to build_z (the slope_cols
        // arg is &[1] there — the x1 column index in the full X); the spec carries
        // ONE SlopeTerm + ONE crossed grouping so for_cluster_spec sizes q_p=2 + tail.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
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
        theta0: f64,
        beta: &[f64],
        x: &Mat<f64>,
        y: &[f32],
        ids: &[u32],
        nc: usize,
    ) -> f64 {
        let (n, p) = (x.nrows(), x.ncols());
        let mut m = Mat::<f64>::zeros(n, nc);
        for i in 0..n {
            m[(i, ids[i] as usize)] = theta0;
        }
        let mut u = vec![0.0f64; nc];
        let pen_dev = |u: &[f64], w_out: Option<&mut [f64]>| -> f64 {
            let mut d = 0.0;
            let mut pen = 0.0;
            let mut wbuf = vec![0.0; n];
            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..p {
                    eta += x[(i, j)] * beta[j];
                }
                for c in 0..nc {
                    eta += m[(i, c)] * u[c];
                }
                let pi = 1.0 / (1.0 + (-eta).exp());
                d += if eta > 0.0 {
                    eta + (-eta).exp().ln_1p()
                } else {
                    eta.exp().ln_1p()
                } - y[i] as f64 * eta;
                wbuf[i] = (pi * (1.0 - pi)).max(1e-6);
                let _ = pi;
            }
            for &uc in u {
                pen += uc * uc;
            }
            if let Some(w) = w_out {
                w.copy_from_slice(&wbuf);
            }
            2.0 * d + pen
        };
        let mut w = vec![0.0; n];
        for _ in 0..50 {
            let mut eta = vec![0.0; n];
            let mut pvec = vec![0.0; n];
            for i in 0..n {
                let mut e = 0.0;
                for j in 0..p {
                    e += x[(i, j)] * beta[j];
                }
                for c in 0..nc {
                    e += m[(i, c)] * u[c];
                }
                eta[i] = e;
                let pi = 1.0 / (1.0 + (-e).exp());
                pvec[i] = pi;
                w[i] = (pi * (1.0 - pi)).max(1e-6);
            }
            let mut g = vec![0.0; nc];
            for c in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, c)] * (y[i] as f64 - pvec[i]);
                }
                g[c] = 2.0 * u[c] - 2.0 * s;
            }
            let mut h = Mat::<f64>::zeros(nc, nc);
            for a in 0..nc {
                for b in 0..nc {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += m[(i, a)] * w[i] * m[(i, b)];
                    }
                    h[(a, b)] = 2.0 * (s + if a == b { 1.0 } else { 0.0 });
                }
            }
            let hc = h.as_ref().llt(faer::Side::Lower).unwrap();
            let mut step = Mat::<f64>::zeros(nc, 1);
            for c in 0..nc {
                step[(c, 0)] = g[c];
            }
            hc.solve_in_place(step.as_mut());
            let mut max = 0.0f64;
            for c in 0..nc {
                u[c] -= step[(c, 0)];
                max = max.max(step[(c, 0)].abs());
            }
            if max < 1e-10 {
                break;
            }
        }
        let _ = pen_dev(&u, Some(&mut w));
        let mut a = Mat::<f64>::zeros(nc, nc);
        for r in 0..nc {
            for c in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, r)] * w[i] * m[(i, c)];
                }
                a[(r, c)] = s + if r == c { 1.0 } else { 0.0 };
            }
        }
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut logdet = 0.0;
        for r in 0..nc {
            logdet += ac.L()[(r, r)].ln();
        }
        pen_dev(&u, None) + 2.0 * logdet
    }

    #[test]
    fn laplace_deviance_matches_brute_force_intercept() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let beta = [0.2_f64, 0.8];
        let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5;
        ws.params[1] = beta[0];
        ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!(
            (got - want).abs() < 1e-6,
            "laplace dev: got {got}, want {want}"
        );
    }

    #[test]
    fn laplace_deviance_collapses_to_glm_at_theta_zero() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let beta = [0.2_f64, 0.8];
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.0;
        ws.params[1] = beta[0];
        ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        // Widen from the SAME f32 design the kernel consumes (f32-gen/f64-fit) — using xf64 here re-introduces a ~1e-7 f32→f64 round-trip gap.
        let mut d = 0.0;
        for i in 0..80 {
            let eta = beta[0] + beta[1] * xf32[(i, 1)] as f64;
            d += if eta > 0.0 {
                eta + (-eta).exp().ln_1p()
            } else {
                eta.exp().ln_1p()
            } - y[i] as f64 * eta;
        }
        let want = 2.0 * d;
        assert!(
            (got - want).abs() < 1e-9,
            "collapse: got {got}, want {want}"
        );
    }

    #[test]
    fn build_z_width_general_populates_all_columns() {
        let (xf64, _y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
        let n = ids.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(
            &mut ws,
            xf32.as_ref(),
            &ids,
            std::slice::from_ref(&crossed_ids),
            &[1],
            n,
        );
        let mut touched = vec![false; ws.k];
        #[allow(clippy::needless_range_loop)]
        for c in 0..ws.k {
            for i in 0..n {
                if ws.z[(i, c)] != 0.0 {
                    touched[c] = true;
                }
            }
        }
        assert!(
            touched.iter().all(|&t| t),
            "every RE column must be populated — offset wiring"
        );
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
            sizing: engine_contract::ClusterSizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::Crossed {
                        n_clusters: n_crossed as u32,
                    },
                    tau_squared: 0.16,
                    slopes: vec![],
                },
                engine_contract::GroupingSpec {
                    relation: engine_contract::GroupingRelation::NestedWithin {
                        n_per_parent: n_per_parent as u32,
                    },
                    tau_squared: 0.09,
                    slopes: vec![],
                },
            ],
        };
        let n = 8usize;
        let ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        let g = &ws.groupings;
        assert!(
            g.extra_offsets[0] > g.extra_offsets[1],
            "fixture must produce non-monotonic offsets, got {:?}",
            g.extra_offsets
        );
        // q_p = 1 ⇒ base_theta = 1; θ_crossed at idx 1, θ_nested at idx 2.
        let base_theta = 1usize;
        let (theta_crossed, theta_nested) = (2.0_f64, 3.0_f64);
        let mut params = vec![0.0; g.n_theta()];
        params[0] = 1.0; // primary Λ (irrelevant to the extra blocks)
        params[base_theta] = theta_crossed;
        params[base_theta + 1] = theta_nested;
        // z = ones everywhere so M directly reveals the per-column scale.
        let mut z = Mat::<f64>::zeros(n, g.k_total);
        for i in 0..n {
            for c in 0..g.k_total {
                z[(i, c)] = 1.0;
            }
        }
        let mut m = Mat::<f64>::zeros(n, g.k_total);
        let mut lam = vec![0.0; g.primary_q * g.primary_q];
        apply_lambda(g, &params, z.as_ref(), &mut m, &mut lam, n);
        let coff = g.extra_offsets[0];
        for c in coff..coff + n_crossed {
            assert!(
                (m[(0, c)] - theta_crossed).abs() < 1e-12,
                "crossed col {c} = {}, want {theta_crossed}",
                m[(0, c)]
            );
        }
        let noff = g.extra_offsets[1];
        for c in noff..noff + n_prim * n_per_parent {
            assert!(
                (m[(0, c)] - theta_nested).abs() < 1e-12,
                "nested col {c} = {}, want {theta_nested}",
                m[(0, c)]
            );
        }
    }

    #[test]
    fn fit_glmm_recovers_direction_and_finite_inference() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        let targets = [1u32];
        let beta_truth = [0.2_f64, 0.8];
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &targets,
            Some(&[0.5]),
            &beta_truth,
            80,
            WaldSe::Rx,
        );
        assert!(fit.converged);
        assert!(
            ws.betas[1] > 0.3,
            "β̂₁ should be positive (truth 0.8), got {}",
            ws.betas[1]
        );
        // Strictly positive, not merely >= 0.0 (a squared quantity is trivially
        // non-negative): a zero Wald statistic means the SE diverged or β̂
        // collapsed. Direction is guarded by the β̂₁ > 0.3 check above; this is
        // one fixed binary dataset (n=80), not a power sim, so the single draw
        // need not clear significance.
        assert!(
            ws.t_sq[1].is_finite() && ws.t_sq[1] > 0.0,
            "t²[1] = {} must be finite and strictly positive",
            ws.t_sq[1]
        );
        assert!(fit.tau_squared_hat.is_finite() && fit.tau_squared_hat >= 0.0);
    }

    #[derive(serde::Deserialize)]
    struct HessianFixture {
        n: usize,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        cluster_ids: Vec<u32>,
        theta: f64,
        beta: Vec<f64>,
        vcov_hessian: Vec<Vec<f64>>,
        #[allow(dead_code)]
        vcov_rx: Vec<Vec<f64>>,
    }

    fn load_hessian_fixture() -> HessianFixture {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/glmm_hessian_vcov.json"
        );
        let s = std::fs::read_to_string(path).expect("read hessian fixture");
        serde_json::from_str(&s).expect("parse hessian fixture")
    }

    /// FD-Hessian fixed-effect covariance matches lme4 `vcov(use.hessian = TRUE)`
    /// on the committed n=96 / 12-cluster `y ~ x1 + (1|grp)` glmer fit. Runs OUR
    /// `fit_glmm` to convergence (the production code path) and takes the FD Hessian
    /// at OUR own θ̂/β̂ — NOT at lme4's fixture params. Pins both the kernel
    /// convention and the load-bearing factor of 2 (deviance = −2logL ⇒
    /// cov = 2·inv(H_dev)).
    ///
    /// Why fit first (test rigor — GATE-1 I1): the FD Hessian's β-block invariance
    /// to the θ↔β cross-curvature only holds AT a θ-stationary point. Evaluating at
    /// our own converged θ̂ both (a) reflects production and (b) lets us assert our
    /// solver agrees with lme4's θ̂ — proving the residual vcov gap below is genuine
    /// curvature, not an off-stationarity artifact. Measured here: our θ̂ matches
    /// lme4's to ~0.07% and β̂ to ~0.1%, and the diagonal vcov gap is the SAME
    /// magnitude (~1.7e-3) as when evaluated at lme4's exact θ̂ — i.e. the 0.07% θ
    /// offset does NOT inflate the gap, refuting the off-stationarity hypothesis.
    #[test]
    fn fd_hessian_cov_matches_glmer_use_hessian_true() {
        let fx = load_hessian_fixture();
        let n = fx.n;
        let p = fx.beta.len();
        let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[]);
        let mut xf32 = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf32[(i, j)] = fx.x[i][j] as f32;
            }
        }
        let y: Vec<f32> = fx.y.iter().map(|&v| v as f32).collect();
        let ids = fx.cluster_ids.clone();
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], n);

        // RX/Schur machinery-exactness check, evaluated at lme4's EXACT θ̂/β̂ so the
        // comparison is point-matched: rx_cov_into reproduces lme4 vcov(use.hessian
        // =FALSE) to ~3.6e-6 (this path never goes through numDeriv — it inverts the
        // closed-form β information). Pins that the PIRLS / W̃ / Schur machinery the
        // FD Hessian shares is exact, isolating any vcov_hessian gap to the FD ↔
        // numDeriv comparison. Requires a converged central PIRLS at those params.
        ws.params[0] = fx.theta;
        for j in 0..p {
            ws.params[1 + j] = fx.beta[j];
        }
        let _ = laplace_deviance_at(&mut ws, xf32.as_ref(), &y, &ids, n);
        let mut rx = Mat::<f64>::zeros(p, p);
        assert!(rx_cov_into(&mut ws, xf32.as_ref(), &ids, p, n, &mut rx));
        for i in 0..p {
            for j in 0..p {
                let (got, want) = (rx[(i, j)], fx.vcov_rx[i][j]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "rx[{i}][{j}] got {got} want {want} (gap {})",
                    (got - want).abs()
                );
            }
        }

        // Production path: converge OUR fit (params overwritten with our θ̂/β̂).
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged, "fit_glmm must converge on the fixture");
        // Our solver vs lme4: θ̂ to ~0.07%, β̂ to ~0.1% (measured). Proves the two
        // optimisers land on the same stationary point, so the FD Hessian below is
        // taken at a genuine θ-stationary point — the residual vcov gap is NOT an
        // off-stationarity artifact. Tol = a few % (well above the achieved band);
        // a MATERIAL divergence here would be a real engine finding, not noise.
        assert!(
            (ws.params[0] - fx.theta).abs() / fx.theta < 0.01,
            "our θ̂ {} vs lme4 θ̂ {} ({}% rel)",
            ws.params[0],
            fx.theta,
            100.0 * (ws.params[0] - fx.theta).abs() / fx.theta
        );
        for j in 0..p {
            assert!(
                (ws.params[1 + j] - fx.beta[j]).abs() < 5e-3,
                "our β̂[{j}] {} vs lme4 β̂[{j}] {} (gap {})",
                ws.params[1 + j],
                fx.beta[j],
                (ws.params[1 + j] - fx.beta[j]).abs()
            );
        }
        let our_theta = ws.params[0];

        let mut cov = Mat::<f64>::zeros(p, p);
        let status = fd_hessian_cov(&mut ws, xf32.as_ref(), &y, &ids, p, n, &mut cov);
        assert_eq!(status, FdHessianStatus::Ok);
        // ws.params restored to OUR converged snapshot on return.
        assert!((ws.params[0] - our_theta).abs() < 1e-15);
        // Achieved FD-vs-lme4(use.hessian=TRUE) band at OUR converged fit: worst
        // entry gap ~1.7e-3 (diagonal, ~1.3% rel) / ~3.2e-4 (off-diagonal). This is
        // the θ↔β cross-curvature term (the ONLY part where use.hessian TRUE vs
        // FALSE differ — and our RX above matches the FALSE block to 3.6e-6, so
        // H_ββ is exact). It is NOT off-stationarity (our θ̂ ≈ lme4's, asserted
        // above) and NOT our FD error (our value is step-invariant): it is lme4's
        // numDeriv noise on its PIRLS deviance's ~1e-7 pwrss floor vs our clean
        // Richardson FD on a deterministic deviance. tol = achieved band + margin;
        // it pins the convention + the load-bearing factor of 2.
        let tol = 2e-3;
        for i in 0..p {
            for j in 0..p {
                let (got, want) = (cov[(i, j)], fx.vcov_hessian[i][j]);
                assert!(
                    (got - want).abs() < tol,
                    "vcov[{i}][{j}] got {got} want {want} (gap {})",
                    (got - want).abs()
                );
            }
        }
    }

    /// `fit_glmm(.., WaldSe::Hessian)` sources the per-fit marginal SE from
    /// `fd_hessian_cov` (glmer `use.hessian = TRUE`) instead of the Schur
    /// forward-solve: on the committed fixture the x1 Hessian SE EXCEEDS the Rx
    /// SE and matches the fixture's `vcov_hessian` diagonal. `WaldSe::Rx` keeps
    /// the unchanged Schur path. Pins that the dispatch reads the FD-Hessian
    /// covariance into `ws.var_diag` end-to-end (not just the standalone kernel).
    #[test]
    fn hessian_mode_t_sq_uses_fd_hessian_cov() {
        let fx = load_hessian_fixture();
        let n = fx.n;
        let p = fx.beta.len();
        let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters },
            0.25,
        );
        let mut xf32 = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf32[(i, j)] = fx.x[i][j] as f32;
            }
        }
        let y: Vec<f32> = fx.y.iter().map(|&v| v as f32).collect();
        let ids = fx.cluster_ids.clone();
        let t1 = 1usize; // x1 column

        let mut ws_h = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[]);
        build_z(&mut ws_h, xf32.as_ref(), &ids, &[], &[], n);
        let fit_h = fit_glmm(
            &mut ws_h,
            xf32.as_ref(),
            &y,
            &ids,
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Hessian,
        );
        assert!(fit_h.converged, "hessian-mode fit must converge");
        let se_h = ws_h.var_diag[t1].sqrt();

        let mut ws_rx = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[]);
        build_z(&mut ws_rx, xf32.as_ref(), &ids, &[], &[], n);
        let fit_rx = fit_glmm(
            &mut ws_rx,
            xf32.as_ref(),
            &y,
            &ids,
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
        );
        assert!(fit_rx.converged, "rx-mode fit must converge");
        let se_rx = ws_rx.var_diag[t1].sqrt();

        assert!(se_h > se_rx, "hessian SE {se_h} must exceed rx SE {se_rx}");
        // Match Task 4's kernel band (2e-3 ABSOLUTE on the covariance entries) on
        // the variance the dispatch wrote into ws.var_diag — the FD-vs-lme4
        // use.hessian=TRUE gap is the θ↔β cross-curvature term (see
        // `fd_hessian_cov_matches_glmer_use_hessian_true`).
        let want_var = fx.vcov_hessian[t1][t1];
        assert!(
            (ws_h.var_diag[t1] - want_var).abs() < 2e-3,
            "hessian var {} must match fixture vcov_hessian diag {want_var}",
            ws_h.var_diag[t1]
        );
    }

    /// A HIGH-variance converged fit drives the joint (θ,β) Hessian non-PD while
    /// the β-only Schur stays PD, so `fd_hessian_cov` falls back to the RX/Schur
    /// covariance (`NonPdFellBackToRx`) and the produced cov equals `rx_cov_into`'s.
    ///
    /// Which boundary actually trips it (empirically scanned): NOT the θ→0 floor.
    /// The intercept-only Laplace deviance is EVEN in θ (D ∝ θ²), so at θ̂→0 its
    /// θ-curvature is structurally POSITIVE and the θ↔β cross-partials vanish by
    /// symmetry — the joint Hessian stays block-diagonal and PD there. The deviance
    /// goes non-convex at the OTHER end: at a LARGE random-intercept variance the
    /// Laplace `2·log|L|²` term grows like `log(variance)` (concave) while the data
    /// deviance saturates, so the θ-curvature turns NEGATIVE ⇒ the joint (θ,β)
    /// Hessian LLT fails. The β fixed-effect information (`rx_cov_into`'s Schur,
    /// computed from W̃ at the conditional modes — a different matrix than the FD
    /// Hessian) stays PD throughout. A strong-clustering (high-ICC) binary dataset
    /// converges to exactly this large-θ̂ regime. Per
    /// `[[faer-llt-rank-deficiency-grey-zone]]` the β design is well-conditioned
    /// (intercept + a within-cluster-varying continuous x1, NOT separable, NOT a
    /// duplicate column) — only the θ/variance direction is degenerate.
    ///
    /// Confirmed branch (asserted below): the FULLY-ASSEMBLED joint Hessian is
    /// finite yet fails LLT — i.e. the non-PD-Hessian fallback, NOT the
    /// non-finite-deviance fallback (this kernel's log1pexp deviance + `+I` ridge
    /// keep every perturbed deviance finite, so the non-finite branch is effectively
    /// unreachable with well-posed data).
    #[test]
    fn fd_hessian_non_pd_falls_back_to_rx_and_counts() {
        let (n, nc) = (80usize, 8usize);
        let per = n / nc;
        // Strong between-cluster intercept offsets (SD ≈ 2.5) ⇒ high ICC ⇒ the fit
        // converges to a LARGE τ̂² (θ̂ ≳ 2), the concave region where the joint
        // Hessian goes non-PD. Noisy (logistic-sampled) labels — no separation.
        let mut st = 4242u64;
        let mut xf64 = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i / per; // block layout
            ids[i] = c as u32;
            let u_c = 10.0 * (2.0 * (c as f64) / ((nc - 1) as f64) - 1.0); // ∈ [-10, 10]
            let x1 = lcg(&mut st); // within-cluster-varying ⇒ β slope identified
            xf64[(i, 0)] = 1.0;
            xf64[(i, 1)] = x1;
            let eta = 0.0 + 0.8 * x1 + u_c;
            let pr = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
        }
        let p = 2usize;
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters {
                n_clusters: nc as u32,
            },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[]);
        let mut xf32 = Mat::<f32>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], n);

        // Converge a fit (kernel precondition for fd_hessian_cov).
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1u32],
            None,
            &[0.0, 0.8],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged, "fixture fit must converge");
        // The non-PD region for this intercept-only fixture begins around θ ≈ 5
        // (empirically scanned); this high-ICC data converges well past it (θ̂ ≈ 30).
        assert!(
            ws.params[0] > 5.0,
            "fit must reach the high-variance non-PD regime (θ̂ = {})",
            ws.params[0]
        );

        let mut cov = Mat::<f64>::zeros(p, p);
        let status = fd_hessian_cov(&mut ws, xf32.as_ref(), &y, &ids, p, n, &mut cov);
        assert_eq!(
            status,
            FdHessianStatus::NonPdFellBackToRx,
            "high-variance joint Hessian must be non-PD ⇒ RX fallback"
        );
        // Confirm this is the non-PD-Hessian branch, not the non-finite-deviance
        // branch: the joint Hessian was FULLY assembled (every perturbed deviance
        // finite) AND is non-PD (LLT errors). The fallback macro re-evals only the
        // central deviance, leaving ws.hess_scratch holding the complete Hessian.
        let m = ws.params.len();
        for i in 0..m {
            for j in 0..m {
                assert!(
                    ws.hess_scratch[(i, j)].is_finite(),
                    "assembled Hessian must be finite (non-finite branch NOT taken): H[{i}][{j}]"
                );
            }
        }
        assert!(
            ws.hess_scratch.as_ref().llt(faer::Side::Lower).is_err(),
            "assembled joint Hessian must be non-PD (LLT must fail)"
        );

        // Fallback cov must equal the RX/Schur cov (a real inverse, not NaN). The
        // central deviance was re-evaluated inside fd_hessian_cov's fallback, so
        // the β-information factors are valid for rx_cov_into here too.
        let mut rx = Mat::<f64>::zeros(p, p);
        assert!(
            rx_cov_into(&mut ws, xf32.as_ref(), &ids, p, n, &mut rx),
            "β-only Schur must stay PD (well-conditioned β design)"
        );
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (cov[(i, j)] - rx[(i, j)]).abs() < 1e-10,
                    "cov[{i}][{j}] {} vs rx {}",
                    cov[(i, j)],
                    rx[(i, j)]
                );
            }
        }
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
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            let eta = 0.1 + 0.7 * x1; // NO u term ⇒ τ²→0 truth
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = x[(i, j)] as f32;
            }
        }

        // Plain logistic on the same bytes (adapt GlmScratch fields to the REAL struct).
        let mut sw = crate::workspace::SimWorkspace::new(n, 2, 1, 0, None);
        let irls = {
            let s = GlmScratch {
                irls_eta: &mut sw.irls_eta[..n],
                irls_p: &mut sw.irls_p[..n],
                irls_w: &mut sw.irls_w[..n],
                irls_z: &mut sw.irls_z[..n],
                irls_betas: &mut sw.irls_betas[..2],
                irls_betas_new: &mut sw.irls_betas_new[..2],
                irls_var_diag: &mut sw.irls_var_diag[..1],
                irls_t_sq: &mut sw.irls_t_sq[..1],
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
            engine_contract::ClusterSizing::FixedClusters {
                n_clusters: nc as u32,
            },
            0.1,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], n);
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.05]),
            &[0.1, 0.7],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged);
        assert!(
            (ws.betas[1] - irls.0[1]).abs() < 1e-2,
            "β̂₁ glmm {} vs irls {}",
            ws.betas[1],
            irls.0[1]
        );
        // GLMM z² is indexed by coefficient (ws.t_sq[1] = slope = target 1); IRLS
        // z² is indexed by target position (irls.1[0] = first target = slope).
        assert!(
            (ws.t_sq[1].sqrt() - irls.1[0].sqrt()).abs() < 5e-2,
            "z glmm {} vs irls {}",
            ws.t_sq[1].sqrt(),
            irls.1[0].sqrt()
        );
    }

    /// Deferred from Task 4.7: fit_glmm on the q_p=2 slope + crossed fixture.
    #[test]
    fn fit_glmm_width_general_slope_and_crossed() {
        let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
        let n = y.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(
            &mut ws,
            xf32.as_ref(),
            &ids,
            std::slice::from_ref(&crossed_ids),
            &[1],
            n,
        );
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            None,
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged);
        // Planted slope 0.8; small balanced binary GLMM (n=96) inflates β̂₁ to ≈3.2 with
        // z²≈13 — direction + significance are the robust claims, not the magnitude. τ̂²
        // collapses to 0 on this draw (valid), so only bound it finite + non-blown. The
        // old `>= 0.0` clauses were tautological on squared quantities.
        assert!(
            ws.betas[1] > 0.5,
            "slope must be strongly positive, got {}",
            ws.betas[1]
        );
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
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        let _ = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
        ); // warmup
        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..20 {
            let _ = fit_glmm(
                &mut ws,
                xf32.as_ref(),
                &y,
                &ids,
                &[1],
                Some(&[0.5]),
                &[0.2, 0.8],
                80,
                WaldSe::Rx,
            );
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
        assert!(
            stats.total_blocks <= BOUND,
            "warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
            stats.total_blocks,
            BOUND
        );
    }

    /// Structured-path warm zero-alloc lock (Group G), the crossed/nested twin of
    /// `fit_glmm_warm_path_bounded_alloc`. There was no such gate before — the dense
    /// crossed/nested path allocated inside faer's per-eval `llt`. The structured
    /// path replaces that with `glmm_block_chol`/`glmm_block_solve` on the
    /// pre-allocated `core_blocks`/`schur_blk`/`coupling` + stack-sized per-block
    /// scratch, so the only per-eval blocks left are the joint [θ|β] BOBYQA's own
    /// scratch (M is built into the pre-allocated `ws.m`). `#[ignore]` + one-thread
    /// for the same reasons as the no-extras gate. Run:
    /// `RAYON_NUM_THREADS=1 cargo test -p engine-core fit_glmm_structured_warm_path_bounded_alloc -- --ignored --test-threads=1`
    #[test]
    #[ignore]
    fn fit_glmm_structured_warm_path_bounded_alloc() {
        // crossed_nested q_p=1: 8 core blocks of 3×3 + a 6×6 Schur (the richest
        // structured shape; exercises both the block factor and the Schur).
        let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(2, 6);
        let n = y.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        assert!(ws.groupings.structured_extras_eligible());
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &extra_ids, &[], n);
        let theta = [0.5_f64, 0.4, 0.45];
        let _ = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&theta),
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
        ); // warmup
        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..20 {
            let _ = fit_glmm(
                &mut ws,
                xf32.as_ref(),
                &y,
                &ids,
                &[1],
                Some(&theta),
                &[0.2, 0.8],
                n,
                WaldSe::Rx,
            );
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        // Measured floor on this machine (RAYON_NUM_THREADS=1): exactly 124 blocks
        // across 20 fits — the SAME count as the no-extras blocked gate. The
        // structured path factors/solves on pre-allocated buffers
        // (core_blocks/schur_blk/coupling) with stack-sized per-block scratch, so it
        // adds NO per-fit faer `llt` allocation; what remains is purely the joint
        // [θ|β] BOBYQA's own per-eval scratch. A change here flags a new allocation
        // or a shifted eval trajectory; do not relax — find the alloc.
        const BOUND: u64 = 124;
        assert!(
            stats.total_blocks <= BOUND,
            "structured warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
            stats.total_blocks,
            BOUND
        );
    }

    /// `glmm_block_chol` factors a q×q SPD block in place to its lower Crout L,
    /// and `glmm_block_solve` then solves L Lᵀ x = b — checked against a faer LLT
    /// solve on the same matrix. q=3 with a deliberately non-trivial SPD block.
    #[test]
    fn block_chol_and_solve_match_faer() {
        let q = 3usize;
        // SPD A (row-major, lower triangle is what the helper reads).
        let a = [4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 3.0, 6.0];
        let b = [1.0_f64, -2.0, 0.5];

        // Reference: faer LLT solve.
        let mut af = Mat::<f64>::zeros(q, q);
        for r in 0..q {
            for c in 0..=r {
                af[(r, c)] = a[r * q + c];
                af[(c, r)] = a[r * q + c];
            }
        }
        let ac = af.as_ref().llt(faer::Side::Lower).unwrap();
        let mut rhs = Mat::<f64>::zeros(q, 1);
        for r in 0..q {
            rhs[(r, 0)] = b[r];
        }
        ac.solve_in_place(rhs.as_mut());

        // Under test.
        let mut blk = a;
        assert!(super::glmm_block_chol(&mut blk, q), "block should be PD");
        let mut x = b;
        super::glmm_block_solve(&blk, q, &mut x);
        for r in 0..q {
            assert!(
                (x[r] - rhs[(r, 0)]).abs() < 1e-12,
                "x[{r}] = {}, faer {}",
                x[r],
                rhs[(r, 0)]
            );
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
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
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
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        assert!(
            ws.groupings.extra_offsets.is_empty(),
            "fixture must route blocked"
        );
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5;
        ws.params[1] = beta[0];
        ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!(
            (got - want).abs() < 1e-6,
            "blocked laplace: got {got}, want {want}"
        );
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
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        assert!(
            ws.groupings.extra_offsets.is_empty(),
            "fixture must route blocked"
        );
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5;
        ws.params[1] = beta[0];
        ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!(
            (got - want).abs() < 1e-6,
            "blocked laplace (contiguous): got {got}, want {want}"
        );
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
                column: engine_contract::ColumnId(1),
                variance: 0.16,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        assert!(ws.groupings.extra_offsets.is_empty());
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
        let beta = [0.2_f64, 0.8];
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

        // Dense reference: apply_lambda → pirls_solve.
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        let GlmmWorkspace {
            groupings,
            z,
            m,
            lam,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            ..
        } = &mut ws;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
        let dense = pirls_solve(
            k,
            p,
            m.as_ref(),
            xf32.as_ref(),
            &y,
            &params[nt..],
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            n,
        );

        // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        build_z(&mut ws2, xf32.as_ref(), &ids, &[], &[1], n);
        crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
        fill_z_f64(&ws2.groupings, xf32.as_ref(), &mut ws2.z_buf, n);
        let GlmmWorkspace {
            groupings,
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            a_blocks,
            a_rhs,
            ..
        } = &mut ws2;
        let blocked = pirls_solve_blocked(
            groupings,
            &ids,
            xf32.as_ref(),
            &y,
            &beta,
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            a_blocks,
            a_rhs,
            n,
        );

        assert_eq!(dense.3, blocked.3, "convergence flag");
        assert!(
            (dense.0 - blocked.0).abs() < 1e-9,
            "dev: dense {} blocked {}",
            dense.0,
            blocked.0
        );
        assert!(
            (dense.1 - blocked.1).abs() < 1e-9,
            "pen: dense {} blocked {}",
            dense.1,
            blocked.1
        );
        assert!(
            (dense.2 - blocked.2).abs() < 1e-9,
            "logdet: dense {} blocked {}",
            dense.2,
            blocked.2
        );
        for c in 0..k {
            assert!(
                (ws.u[c] - ws2.u[c]).abs() < 1e-7,
                "u[{c}]: dense {} blocked {}",
                ws.u[c],
                ws2.u[c]
            );
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
                column: engine_contract::ColumnId(1),
                variance: 0.16,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        assert!(ws.groupings.extra_offsets.is_empty());
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
        let beta = [0.2_f64, 0.8];
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

        // Dense reference: apply_lambda → pirls_solve.
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        let GlmmWorkspace {
            groupings,
            z,
            m,
            lam,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            ..
        } = &mut ws;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
        let dense = pirls_solve(
            k,
            p,
            m.as_ref(),
            xf32.as_ref(),
            &y,
            &params[nt..],
            eta,
            prob,
            w,
            u,
            eta_fixed,
            mu,
            wm,
            a,
            a_rhs,
            n,
        );

        // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        build_z(&mut ws2, xf32.as_ref(), &ids, &[], &[1], n);
        crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
        fill_z_f64(&ws2.groupings, xf32.as_ref(), &mut ws2.z_buf, n);
        let GlmmWorkspace {
            groupings,
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            a_blocks,
            a_rhs,
            ..
        } = &mut ws2;
        let blocked = pirls_solve_blocked(
            groupings,
            &ids,
            xf32.as_ref(),
            &y,
            &beta,
            lam,
            z_buf,
            m_buf,
            eta,
            prob,
            w,
            u,
            eta_fixed,
            a_blocks,
            a_rhs,
            n,
        );

        assert_eq!(dense.3, blocked.3, "convergence flag");
        assert!(
            (dense.0 - blocked.0).abs() < 1e-9,
            "dev: dense {} blocked {}",
            dense.0,
            blocked.0
        );
        assert!(
            (dense.1 - blocked.1).abs() < 1e-9,
            "pen: dense {} blocked {}",
            dense.1,
            blocked.1
        );
        assert!(
            (dense.2 - blocked.2).abs() < 1e-9,
            "logdet: dense {} blocked {}",
            dense.2,
            blocked.2
        );
        for c in 0..k {
            assert!(
                (ws.u[c] - ws2.u[c]).abs() < 1e-7,
                "u[{c}]: dense {} blocked {}",
                ws.u[c],
                ws2.u[c]
            );
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
                column: engine_contract::ColumnId(1),
                variance: 0.16,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1]);
        let mut xf32 = Mat::<f32>::zeros(n, 2);
        for i in 0..n {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[1], n);
        let fit = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5, 0.1, 0.4]),
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
        );
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
            let GlmmWorkspace {
                groupings,
                params,
                z,
                m,
                lam,
                ..
            } = &mut ws;
            apply_lambda(groupings, &params[..], z.as_ref(), m, lam, n);
        }
        // X'W̃X, X'W̃M, A = M'W̃M + I (dense), Schur, Var(β̂)_11.
        let mut xtwx = Mat::<f64>::zeros(p, p);
        for r in 0..p {
            for c in 0..p {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf32[(i, r)] as f64 * ws.w[i] * xf32[(i, c)] as f64;
                }
                xtwx[(r, c)] = sm;
            }
        }
        let mut xtwm = Mat::<f64>::zeros(p, k);
        for r in 0..p {
            for c in 0..k {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf32[(i, r)] as f64 * ws.w[i] * ws.m[(i, c)];
                }
                xtwm[(r, c)] = sm;
            }
        }
        let mut a = Mat::<f64>::zeros(k, k);
        for r in 0..k {
            for c in 0..k {
                let mut sm = if r == c { 1.0 } else { 0.0 };
                for i in 0..n {
                    sm += ws.m[(i, r)] * ws.w[i] * ws.m[(i, c)];
                }
                a[(r, c)] = sm;
            }
        }
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ainv = Mat::<f64>::zeros(k, p);
        for r in 0..k {
            for c in 0..p {
                ainv[(r, c)] = xtwm[(c, r)];
            }
        }
        ac.solve_in_place(ainv.as_mut());
        let mut schur = Mat::<f64>::zeros(p, p);
        for r in 0..p {
            for c in 0..p {
                let mut sm = xtwx[(r, c)];
                for j in 0..k {
                    sm -= xtwm[(r, j)] * ainv[(j, c)];
                }
                schur[(r, c)] = sm;
            }
        }
        let sc = schur.as_ref().llt(faer::Side::Lower).unwrap();
        let mut fwd = vec![0.0; p];
        for i in 0..p {
            let mut acc = if i == 1 { 1.0 } else { 0.0 };
            #[allow(clippy::needless_range_loop)]
            for kk in 0..i {
                acc -= sc.L()[(i, kk)] * fwd[kk];
            }
            fwd[i] = acc / sc.L()[(i, i)];
        }
        let var_dense: f64 = fwd.iter().map(|v| v * v).sum();
        assert!(
            (var_blocked - var_dense).abs() < 1e-8,
            "var: blocked {var_blocked} dense {var_dense}"
        );
        let tsq_dense = beta_blocked[1] * beta_blocked[1] / var_dense;
        assert!(
            (tsq_blocked - tsq_dense).abs() < 1e-6,
            "z²: blocked {tsq_blocked} dense {tsq_dense}"
        );
    }

    /// Within-fit û warm-start MUST reset per fit. A reused-workspace re-fit must
    /// match a fresh-workspace (canonically cold) fit BIT-FOR-BIT. Carrying the
    /// incumbent across fits is the rejected cross-sim warm-start that breaks
    /// merge / same-seed reproducibility. This is the per-fit-reset guard.
    #[test]
    fn warm_start_is_per_fit_deterministic() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mk = || {
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
            let mut xf32 = Mat::<f32>::zeros(80, 2);
            for i in 0..80 {
                for j in 0..2 {
                    xf32[(i, j)] = xf64[(i, j)] as f32;
                }
            }
            build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
            (ws, xf32)
        };
        // Canonical: a fresh workspace's first (cold) fit.
        let (mut ws_ref, xref) = mk();
        let _ = fit_glmm(
            &mut ws_ref,
            xref.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
        );
        let ref_beta = ws_ref.betas[1].to_bits();
        // Reused workspace: a throwaway fit pollutes u_seed, then the measured fit
        // must match the canonical cold result bit-for-bit (only if u_seed resets).
        let (mut ws, xf32) = mk();
        let _ = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
        );
        let _ = fit_glmm(
            &mut ws,
            xf32.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
        );
        assert_eq!(
            ws.betas[1].to_bits(),
            ref_beta,
            "re-fit β̂ must match the fresh cold fit bit-for-bit"
        );
    }

    /// The Laplace objective is a (near-)pure function of (θ,β): the same point
    /// solved from u=0 vs from a perturbed seed agrees within ≪ the estimate floor
    /// — the conditional mode is unique, only the stopping iterate is seed-dependent.
    #[test]
    fn warm_start_objective_is_seed_independent() {
        let (xf64, y, ids) = glmm_intercept_dataset();
        let cluster = engine_contract::ClusterSpec::intercept_only(
            engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            0.25,
        );
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[]);
        let mut xf32 = Mat::<f32>::zeros(80, 2);
        for i in 0..80 {
            for j in 0..2 {
                xf32[(i, j)] = xf64[(i, j)] as f32;
            }
        }
        build_z(&mut ws, xf32.as_ref(), &ids, &[], &[], 80);
        ws.params[0] = 0.5;
        ws.params[1] = 0.2;
        ws.params[2] = 0.8;
        for v in ws.u.iter_mut() {
            *v = 0.0;
        }
        let cold = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        for (c, v) in ws.u.iter_mut().enumerate() {
            *v = 0.05 * (c as f64 - 4.0);
        }
        let warm = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, 80);
        assert!(
            (cold - warm).abs() < 1e-6,
            "objective seed-dependent: cold {cold} warm {warm}"
        );
    }

    // -- Group G structured-solve armor (crossed/nested/crossed_nested, q_p=1) ----

    /// q_p=1 clustered-binary dataset with intercept-only extra groupings, in the
    /// three Group-G bench shapes. `np` = nested children per parent (0 = none),
    /// `n_crossed` = crossed levels (0 = none). Returns (X f64 [1, x1], y∈{0,1},
    /// primary ids, extra ids in DECLARATION order `[nested?, crossed?]`, spec).
    /// Mirrors `glmm_slope_crossed_dataset`'s construction (round-robin ids so
    /// every level is populated); the spec declares nested before crossed so the
    /// θ vector is `[θ_p, θ_nested?, θ_crossed?]` and the engine's `extra_offsets`
    /// land nested at `prim_width`, crossed in the trailing block.
    #[allow(clippy::type_complexity)]
    fn glmm_extras_q1_dataset(
        np: usize,
        n_crossed: usize,
    ) -> (
        Mat<f64>,
        Vec<f32>,
        Vec<u32>,
        Vec<Vec<u32>>,
        engine_contract::ClusterSpec,
    ) {
        let (n, n_prim) = (96usize, 8usize);
        let mut st = 29u64;
        let u0: Vec<f64> = (0..n_prim).map(|_| 0.6 * lcg(&mut st)).collect();
        // Nested children are globalized: parent·np + within ⇒ n_prim·np draws.
        let un: Vec<f64> = (0..n_prim * np.max(1))
            .map(|_| 0.4 * lcg(&mut st))
            .collect();
        let uc: Vec<f64> = (0..n_crossed.max(1)).map(|_| 0.5 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        let mut nested = vec![0u32; n];
        let mut crossed = vec![0u32; n];
        for i in 0..n {
            let c = i % n_prim;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            let mut eta = 0.2 + 0.8 * x1 + u0[c];
            if np > 0 {
                // within-parent child cycles through 0..np; globalized id = c·np + within.
                let within = (i / n_prim) % np;
                let gid = c * np + within;
                nested[i] = gid as u32;
                eta += un[gid];
            }
            if n_crossed > 0 {
                let cc = i % n_crossed;
                crossed[i] = cc as u32;
                eta += uc[cc];
            }
            let p = 1.0 / (1.0 + (-eta).exp());
            y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
        }
        let mut extra_groupings = Vec::new();
        let mut extra_ids = Vec::new();
        if np > 0 {
            extra_groupings.push(engine_contract::GroupingSpec {
                relation: engine_contract::GroupingRelation::NestedWithin {
                    n_per_parent: np as u32,
                },
                tau_squared: 0.16,
                slopes: vec![],
            });
            extra_ids.push(nested);
        }
        if n_crossed > 0 {
            extra_groupings.push(engine_contract::GroupingSpec {
                relation: engine_contract::GroupingRelation::Crossed {
                    n_clusters: n_crossed as u32,
                },
                tau_squared: 0.25,
                slopes: vec![],
            });
            extra_ids.push(crossed);
        }
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings,
        };
        (x, y, ids, extra_ids, cluster)
    }

    /// Independent dense Laplace deviance for the intercept-only extras shapes:
    /// builds a fully-dense Z `[primary | nested children | crossed]` in its OWN
    /// column order (permutation-invariant to the engine's layout), `M = Z·θ`
    /// (θ_p on primary cols, θ_nested on nested, θ_crossed on crossed), runs
    /// Newton-on-u to the conditional mode, and returns `d(y,ũ) + ‖ũ‖² + log|A|`
    /// (A = M'WM + I). The ground-truth oracle for `pirls_solve_blocked_extras`:
    /// a parity test only proves "same as dense"; this proves "correct".
    /// Mirrors `brute_force_intercept_laplace` generalized to the extra columns.
    #[allow(clippy::too_many_arguments)]
    fn brute_force_extras_laplace(
        theta_p: f64,
        theta_n: f64,
        theta_c: f64,
        beta: &[f64],
        x: &Mat<f64>,
        y: &[f32],
        ids: &[u32],
        nested: &[u32],
        crossed: &[u32],
        n_prim: usize,
        np: usize,
        n_crossed: usize,
    ) -> f64 {
        let (n, p) = (x.nrows(), x.ncols());
        let nc = n_prim + n_prim * np + n_crossed;
        let mut m = Mat::<f64>::zeros(n, nc);
        let nest_base = n_prim;
        let cross_base = n_prim + n_prim * np;
        for i in 0..n {
            m[(i, ids[i] as usize)] = theta_p;
            if np > 0 {
                m[(i, nest_base + nested[i] as usize)] = theta_n;
            }
            if n_crossed > 0 {
                m[(i, cross_base + crossed[i] as usize)] = theta_c;
            }
        }
        let eta_of = |u: &[f64], i: usize| -> f64 {
            let mut e = 0.0;
            for j in 0..p {
                e += x[(i, j)] * beta[j];
            }
            for c in 0..nc {
                e += m[(i, c)] * u[c];
            }
            e
        };
        let mut u = vec![0.0f64; nc];
        // Newton on the penalized binomial: H = 2(M'WM + I), g = 2(u − M'(y−p)).
        for _ in 0..80 {
            let mut pvec = vec![0.0; n];
            let mut w = vec![0.0; n];
            for i in 0..n {
                let e = eta_of(&u, i);
                let pi = 1.0 / (1.0 + (-e).exp());
                pvec[i] = pi;
                w[i] = (pi * (1.0 - pi)).max(1e-6);
            }
            let mut g = vec![0.0; nc];
            for c in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, c)] * (y[i] as f64 - pvec[i]);
                }
                g[c] = 2.0 * u[c] - 2.0 * s;
            }
            let mut h = Mat::<f64>::zeros(nc, nc);
            for a in 0..nc {
                for b in 0..nc {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += m[(i, a)] * w[i] * m[(i, b)];
                    }
                    h[(a, b)] = 2.0 * (s + if a == b { 1.0 } else { 0.0 });
                }
            }
            let hc = h.as_ref().llt(faer::Side::Lower).unwrap();
            let mut step = Mat::<f64>::zeros(nc, 1);
            for c in 0..nc {
                step[(c, 0)] = g[c];
            }
            hc.solve_in_place(step.as_mut());
            let mut max = 0.0f64;
            for c in 0..nc {
                u[c] -= step[(c, 0)];
                max = max.max(step[(c, 0)].abs());
            }
            if max < 1e-11 {
                break;
            }
        }
        // Laplace at the mode: d + ‖u‖² + log|A|.
        let mut d = 0.0;
        let mut w = vec![0.0; n];
        for i in 0..n {
            let e = eta_of(&u, i);
            let pi = 1.0 / (1.0 + (-e).exp());
            w[i] = (pi * (1.0 - pi)).max(1e-6);
            d += if e > 0.0 {
                e + (-e).exp().ln_1p()
            } else {
                e.exp().ln_1p()
            } - y[i] as f64 * e;
        }
        let pen: f64 = u.iter().map(|v| v * v).sum();
        let mut a = Mat::<f64>::zeros(nc, nc);
        for r in 0..nc {
            for c in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, r)] * w[i] * m[(i, c)];
                }
                a[(r, c)] = s + if r == c { 1.0 } else { 0.0 };
            }
        }
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut logdet = 0.0;
        for r in 0..nc {
            logdet += ac.L()[(r, r)].ln();
        }
        2.0 * d + pen + 2.0 * logdet
    }

    /// `laplace_deviance` matches the independent dense brute-force oracle on all
    /// three Group-G shapes. With the structured kernel in place the dispatch
    /// routes these eligible shapes through `pirls_solve_blocked_extras`, so this
    /// is the structured-vs-oracle ground-truth test: a parity test only proves
    /// "same as dense", this proves the Schur assembly is *correct*.
    #[test]
    fn structured_extras_laplace_matches_brute_force() {
        // (np, n_crossed, label)
        for (np, ncr, label) in [
            (0usize, 6usize, "crossed"),
            (2, 0, "nested"),
            (2, 6, "crossed_nested"),
        ] {
            let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
            let n = y.len();
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
            assert!(
                !ws.groupings.extra_offsets.is_empty(),
                "{label}: fixture must route through the extras path"
            );
            let mut xf32 = Mat::<f32>::zeros(n, 2);
            for i in 0..n {
                for j in 0..2 {
                    xf32[(i, j)] = xf64[(i, j)] as f32;
                }
            }
            let extra_refs: Vec<&[u32]> = extra_ids.iter().map(|v| v.as_slice()).collect();
            build_z(&mut ws, xf32.as_ref(), &ids, &extra_ids, &[], n);
            // θ = [θ_p, θ_nested?, θ_crossed?] in declaration order [nested, crossed].
            let theta_p = 0.5;
            let theta_n = 0.4;
            let theta_c = 0.45;
            let beta = [0.2_f64, 0.8];
            let nt = ws.n_theta;
            ws.params[0] = theta_p;
            let mut ti = 1;
            if np > 0 {
                ws.params[ti] = theta_n;
                ti += 1;
            }
            if ncr > 0 {
                ws.params[ti] = theta_c;
            }
            ws.params[nt] = beta[0];
            ws.params[nt + 1] = beta[1];
            let got =
                glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf32.as_ref(), &y, &ids, n);
            // Build dense f64 from the SAME f32 design the kernel consumes.
            let mut xb = Mat::<f64>::zeros(n, 2);
            for i in 0..n {
                xb[(i, 0)] = xf32[(i, 0)] as f64;
                xb[(i, 1)] = xf32[(i, 1)] as f64;
            }
            let nested = if np > 0 { extra_refs[0] } else { &[][..] };
            let crossed = if ncr > 0 {
                extra_refs[extra_refs.len() - 1]
            } else {
                &[][..]
            };
            let want = brute_force_extras_laplace(
                theta_p, theta_n, theta_c, &beta, &xb, &y, &ids, nested, crossed, 8, np, ncr,
            );
            assert!(
                (got - want).abs() < 1e-6,
                "{label} dense laplace: got {got}, want {want}"
            );
        }
    }

    /// Structured == dense on all three Group-G shapes, to FP error: drives the
    /// dense `pirls_solve` and `pirls_solve_blocked_extras` on the same M / θ / β
    /// from fresh scratch. The Schur reassociation is the same estimator. Bands
    /// mirror the no-extras blocked parity (`blocked_pirls_matches_dense_slope_noextra`):
    /// dev/pen/logdet 1e-9, u 1e-7.
    #[test]
    fn structured_extras_matches_dense() {
        for (np, ncr, label) in [
            (0usize, 6usize, "crossed"),
            (2, 0, "nested"),
            (2, 6, "crossed_nested"),
        ] {
            let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
            let n = y.len();
            let mut xf32 = Mat::<f32>::zeros(n, 2);
            for i in 0..n {
                for j in 0..2 {
                    xf32[(i, j)] = xf64[(i, j)] as f32;
                }
            }
            // θ = [θ_p, θ_nested?, θ_crossed?] in declaration order [nested, crossed].
            let mut theta = vec![0.5_f64];
            if np > 0 {
                theta.push(0.4);
            }
            if ncr > 0 {
                theta.push(0.45);
            }
            let beta = [0.2_f64, 0.8];

            // Dense reference: apply_lambda → pirls_solve.
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
            assert!(
                ws.groupings.structured_extras_eligible(),
                "{label}: fixture must be structured-eligible"
            );
            build_z(&mut ws, xf32.as_ref(), &ids, &extra_ids, &[], n);
            let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
            let mut params = vec![0.0; nt + p];
            params[..nt].copy_from_slice(&theta);
            params[nt..].copy_from_slice(&beta);
            let GlmmWorkspace {
                groupings,
                z,
                m,
                lam,
                eta,
                prob,
                w,
                u,
                eta_fixed,
                mu,
                wm,
                a,
                a_rhs,
                ..
            } = &mut ws;
            apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
            let dense = pirls_solve(
                k,
                p,
                m.as_ref(),
                xf32.as_ref(),
                &y,
                &params[nt..],
                eta,
                prob,
                w,
                u,
                eta_fixed,
                mu,
                wm,
                a,
                a_rhs,
                n,
            );

            // Structured: build_packed_m → pirls_solve_blocked_extras, fresh scratch.
            let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
            build_z(&mut ws2, xf32.as_ref(), &ids, &extra_ids, &[], n);
            {
                let GlmmWorkspace {
                    groupings,
                    params: prm,
                    z,
                    lam,
                    m_core_buf,
                    cross_val,
                    cross_col,
                    n_cross,
                    ..
                } = &mut ws2;
                prm[..nt].copy_from_slice(&theta);
                prm[nt..nt + p].copy_from_slice(&beta);
                build_packed_m(
                    groupings,
                    &prm[..],
                    z.as_ref(),
                    lam,
                    &ids,
                    m_core_buf,
                    cross_val,
                    cross_col,
                    n_cross,
                    n,
                );
            }
            let structured = {
                let GlmmWorkspace {
                    groupings,
                    m_core_buf,
                    cross_val,
                    cross_col,
                    n_cross,
                    eta,
                    prob,
                    w,
                    u,
                    eta_fixed,
                    mu,
                    core_blocks,
                    coupling,
                    schur_blk,
                    a_rhs,
                    ..
                } = &mut ws2;
                pirls_solve_blocked_extras(
                    groupings,
                    &ids,
                    m_core_buf,
                    cross_val,
                    cross_col,
                    n_cross,
                    xf32.as_ref(),
                    &y,
                    &beta,
                    eta,
                    prob,
                    w,
                    u,
                    eta_fixed,
                    mu,
                    core_blocks,
                    coupling,
                    schur_blk,
                    a_rhs,
                    n,
                )
            };

            assert_eq!(dense.3, structured.3, "{label}: convergence flag");
            assert!(
                (dense.0 - structured.0).abs() < 1e-9,
                "{label} dev: dense {} structured {}",
                dense.0,
                structured.0
            );
            assert!(
                (dense.1 - structured.1).abs() < 1e-9,
                "{label} pen: dense {} structured {}",
                dense.1,
                structured.1
            );
            assert!(
                (dense.2 - structured.2).abs() < 1e-9,
                "{label} logdet: dense {} structured {}",
                dense.2,
                structured.2
            );
            for c in 0..k {
                assert!(
                    (ws.u[c] - ws2.u[c]).abs() < 1e-7,
                    "{label} u[{c}]: dense {} structured {}",
                    ws.u[c],
                    ws2.u[c]
                );
            }
        }
    }

    /// Structured inference (Var(β̂)_jj, z²) matches a dense Schur recomputation on
    /// all three Group-G shapes — the structured `A⁻¹` apply in `structured_schur_fill`
    /// is the same estimator as the dense `dense_schur_fill`. Runs the full
    /// `fit_glmm` (structured) then recomputes Var(β̂) densely from the converged
    /// ws.{w, lam, params}. Mirrors `blocked_inference_matches_dense_slope_noextra`.
    #[test]
    fn structured_inference_matches_dense() {
        for (np, ncr, label) in [
            (0usize, 6usize, "crossed"),
            (2, 0, "nested"),
            (2, 6, "crossed_nested"),
        ] {
            let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
            let n = y.len();
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
            assert!(
                ws.groupings.structured_extras_eligible(),
                "{label}: eligible"
            );
            let mut xf32 = Mat::<f32>::zeros(n, 2);
            for i in 0..n {
                for j in 0..2 {
                    xf32[(i, j)] = xf64[(i, j)] as f32;
                }
            }
            build_z(&mut ws, xf32.as_ref(), &ids, &extra_ids, &[], n);
            let fit = fit_glmm(
                &mut ws,
                xf32.as_ref(),
                &y,
                &ids,
                &[1],
                None,
                &[0.2, 0.8],
                n,
                WaldSe::Rx,
            );
            assert!(fit.converged, "{label}: fit must converge");
            let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
            let var_structured = ws.var_diag[1];
            let tsq_structured = ws.t_sq[1];
            let beta1 = ws.betas[1];
            // Dense recompute of Var(β̂)_11 from the converged state.
            crate::lmm::primary_lambda(&ws.params[..nt], ws.groupings.primary_q, &mut ws.lam);
            {
                let GlmmWorkspace {
                    groupings,
                    params,
                    z,
                    m,
                    lam,
                    ..
                } = &mut ws;
                apply_lambda(groupings, &params[..], z.as_ref(), m, lam, n);
            }
            let mut xtwx = Mat::<f64>::zeros(p, p);
            for r in 0..p {
                for c in 0..p {
                    let mut sm = 0.0;
                    for i in 0..n {
                        sm += xf32[(i, r)] as f64 * ws.w[i] * xf32[(i, c)] as f64;
                    }
                    xtwx[(r, c)] = sm;
                }
            }
            let mut xtwm = Mat::<f64>::zeros(p, k);
            for r in 0..p {
                for c in 0..k {
                    let mut sm = 0.0;
                    for i in 0..n {
                        sm += xf32[(i, r)] as f64 * ws.w[i] * ws.m[(i, c)];
                    }
                    xtwm[(r, c)] = sm;
                }
            }
            let mut a = Mat::<f64>::zeros(k, k);
            for r in 0..k {
                for c in 0..k {
                    let mut sm = if r == c { 1.0 } else { 0.0 };
                    for i in 0..n {
                        sm += ws.m[(i, r)] * ws.w[i] * ws.m[(i, c)];
                    }
                    a[(r, c)] = sm;
                }
            }
            let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
            let mut ainv = Mat::<f64>::zeros(k, p);
            for r in 0..k {
                for c in 0..p {
                    ainv[(r, c)] = xtwm[(c, r)];
                }
            }
            ac.solve_in_place(ainv.as_mut());
            let mut schur = Mat::<f64>::zeros(p, p);
            for r in 0..p {
                for c in 0..p {
                    let mut sm = xtwx[(r, c)];
                    for j in 0..k {
                        sm -= xtwm[(r, j)] * ainv[(j, c)];
                    }
                    schur[(r, c)] = sm;
                }
            }
            let sc = schur.as_ref().llt(faer::Side::Lower).unwrap();
            let mut fwd = vec![0.0; p];
            for i in 0..p {
                let mut acc = if i == 1 { 1.0 } else { 0.0 };
                #[allow(clippy::needless_range_loop)]
                for kk in 0..i {
                    acc -= sc.L()[(i, kk)] * fwd[kk];
                }
                fwd[i] = acc / sc.L()[(i, i)];
            }
            let var_dense: f64 = fwd.iter().map(|v| v * v).sum();
            assert!(
                (var_structured - var_dense).abs() < 1e-8,
                "{label} var: structured {var_structured} dense {var_dense}"
            );
            let tsq_dense = beta1 * beta1 / var_dense;
            assert!(
                (tsq_structured - tsq_dense).abs() < 1e-6,
                "{label} z²: structured {tsq_structured} dense {tsq_dense}"
            );
        }
    }
}
