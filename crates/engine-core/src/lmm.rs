//! General-machine LMM solver core — family-blocked profiled-REML deviance
//! (primary + nested children eliminated family-by-family, crossed factors in
//! a dense tail Cholesky with [X y]) + BOBYQA θ-search over one diagonal θ
//! component per grouping. A degenerate single-intercept `ClusterSpec`
//! collapses to M1's per-cluster shrink-downdate arithmetic (up to FP
//! reassociation), so the q=1 parity corpus re-proves on this machine.
//!
//! Engine-resident: non-degenerate (extra-grouping) Mle specs dispatch here
//! from `batch.rs` / `introspect.rs` via `build_lmm_workspace`; every shipped
//! (degenerate single-intercept) `ClusterSpec` keeps routing to the scalar
//! Brent path in `lme.rs`.
//!
//! Hot-loop invariants (mirror `lme.rs`):
//!  * Bounded allocations on the warm path (twin test in `lmm::tests`): all
//!    scratch and the BOBYQA solver live in `LmmWorkspace`, allocated once
//!    per (p, max_clusters) shape; the only per-call allocations are faer
//!    `llt` internals — the same acceptance the shipped path carries.
//!  * Inference is squared statistics (`t_sq = β̂²/Var(β̂)`); never sqrt the
//!    SE, never call a CDF on the per-fit path.
//!  * `f64::INFINITY` is the deviance failure surface.
//!
//! **NR** = Press, Teukolsky, Vetterling & Flannery (2007), *Numerical Recipes:
//! The Art of Scientific Computing*, 3rd ed., Cambridge University Press.
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use crate::ols::chol_rank_deficient;
use bobyqa::{Bobyqa, Config, Status};
use faer::{Mat, MatRef};

/// θ start — unit diagonal (lme4/MixedModels.jl default), the
/// `theta_start: None` blind default. Cold start per fit; no warm-start
/// across sims (would re-import cross-grid-point path dependence).
pub const THETA0: f64 = 1.0;
/// Per-component θ upper box — mirrors the shipped Brent reach (θ ≤ 1e3).
pub const THETA_HI: f64 = 1e3;
/// Initial trust radius. Must be ≤ 1.0: PRIMA start-projection silently moves
/// an x₀ within rho_begin of a bound; 0.5 keeps θ₀ = 1.0 strictly clear of
/// the 0 lower bound. Box width 1e3 ≥ 2·rho_begin is the crate's up-front
/// validity requirement.
pub const RHO_BEGIN: f64 = 0.5;
/// Final trust radius = θ̂ target accuracy. 1e-6 measured equivalent to 1e-8
/// on every Gate-0 parity gate under the amended abs floors (stat 1e-4 /
/// β̂ 1e-5), at 15.1–15.7 vs 19.5–20.7 evals/fit — a ~25% eval cut for free.
pub const RHO_END: f64 = 1e-6;
/// Truth-start floor: a `Some(θ₀)` start is clamped to max(θ₀, this) so a
/// zero/near-zero true θ never starts the search on the boundary itself.
/// Keep ≥ 10·RHO_END: the future scaled schedule derives
/// rho_begin = 0.1·θ₀, and the crate requires rho_end ≤ rho_begin.
pub const THETA_TRUTH_FLOOR: f64 = 0.01;
/// Pin threshold: a Converged diagonal component ≤ this is deterministically
/// pinned at exactly 0 and counted converged. 1e-4 aligns the class boundary
/// with the shipped τ̂≈0 detection (`lme.rs` pins boundary_hit=1 fits at
/// θ = 1e-4).
pub const PIN_THETA: f64 = 1e-4;
/// Rank guard on the p×p block of the factor — mirrors `lme.rs` EPS_RANK.
pub const EPS_RANK: f64 = 1e-8;

/// BOBYQA config for an n_theta-dimensional θ-search. `Config::new` supplies
/// the PRIMA defaults (npt = 2n+1, max_fun = 500·n) — at n = 1 exactly the
/// Gate-0 spike's npt = 3 / max_fun = 500.
pub fn bobyqa_config(n_theta: usize) -> Config {
    Config {
        rho_begin: RHO_BEGIN,
        rho_end: RHO_END,
        ..Config::new(n_theta)
    }
}

/// Re-exported from `engine-contract`, where `validate()` (invariant 20)
/// enforces it. The per-fit id buffer (`gid` in `add_rows_multi`) and the
/// truth-start buffer are sized `1 + MAX_EXTRA_GROUPINGS` — the ceiling lives in
/// the contract so oversized specs are rejected before a fit is ever built.
pub use engine_contract::MAX_EXTRA_GROUPINGS;

/// Re-exported from `engine-contract`, where `validate()` (invariant 21)
/// enforces it. `q_p = 1 + #slopes ≤ MAX_PRIMARY_Q`.
pub use engine_contract::MAX_PRIMARY_Q;

/// Maximum θ count: the vech of the primary q_p×q_p block has
/// `MAX_PRIMARY_Q*(MAX_PRIMARY_Q+1)/2` entries plus one per extra grouping.
/// Sizes the stack-allocated truth-start buffer `tbuf` in `batch.rs`; both
/// the hot path and the cold/excluded path use this. Change together with the
/// two `tbuf` sites (marked "mirrors MAX_THETA — change together").
pub const MAX_THETA: usize = MAX_PRIMARY_Q * (MAX_PRIMARY_Q + 1) / 2 + MAX_EXTRA_GROUPINGS;

// ---------------------------------------------------------------------------
// LmmGroupings — grouping-structure metadata shared by suff stats + deviance.
// ---------------------------------------------------------------------------

/// Grouping-structure metadata the suff stats and deviance share.
///
/// RE column order is the ELIMINATION order — `[primary 0..S | nested
/// children (parent-contiguous: child id = parent·n_per + within) | crossed
/// factors last]` — decoupled from θ order, which stays `[primary, extras in
/// declaration order]` (matching data-gen draw order and the truth-start
/// vector).
pub struct LmmGroupings {
    /// Primary level capacity at the sized max_n.
    pub n_primary: usize,
    /// Children per parent; 0 = no nested extra (family width 1).
    pub nested_per_parent: usize,
    /// θ component index of the nested extra, if any.
    pub nested_theta: Option<usize>,
    /// Crossed extras in declaration order: (θ component index, level count).
    pub crossed: Vec<(usize, usize)>,
    /// RE-column offset of extra g (declaration order) — where its globalized
    /// level ids land in `s`/`counts`.
    pub extra_offsets: Vec<usize>,
    /// Total RE columns K.
    pub k_total: usize,
    /// Primary RE block width `q_p = 1 + #slopes` (1 = intercept only).
    pub primary_q: usize,
    /// `[X y]` row indices of the slope covariates (their x_full design columns),
    /// one per slope in declaration order, used to recover the per-level `q_p×q_p`
    /// Gram from `s`. Empty iff `primary_q == 1`.
    pub primary_slope_cols: Vec<usize>,
    /// θ-indices of the diagonal variance components (pinnable), in
    /// `boundary_rate_per_component` order: the `q_p` primary vech diagonals
    /// (column-major — the diagonal of column d sits at offset `Σ_{j<d}(q_p−j)`)
    /// then the extra-grouping scalars. q_p=1 ⇒ `[0, extras…]`; q_p=2 ⇒
    /// `[0, 2, extras…]`; off-diagonal vech entries excluded. Computed once per
    /// workspace by `compute_diagonal_theta` (the vech-diagonal walk lives there,
    /// single-sourced) so the per-fit pin loop borrows instead of reallocating.
    pub diagonal_theta: Vec<usize>,
}

/// The vech-diagonal θ-index walk, single source of truth for
/// `LmmGroupings::diagonal_theta`. `primary_q` diagonal entries (column-major:
/// the diagonal of column d sits at offset `Σ_{j<d}(primary_q−j)`) then one
/// scalar per extra grouping at `primary_q(primary_q+1)/2 + e`.
fn compute_diagonal_theta(primary_q: usize, n_extras: usize) -> Vec<usize> {
    let q = primary_q;
    let mut idx = Vec::with_capacity(q + n_extras);
    let mut off = 0usize;
    for d in 0..q {
        idx.push(off);
        off += q - d; // advance past column d's vech block (length q−d)
    }
    let base = q * (q + 1) / 2;
    for e in 0..n_extras {
        idx.push(base + e);
    }
    idx
}

impl LmmGroupings {
    /// Single q=1 grouping — the M1 shape.
    pub fn single(max_clusters: usize) -> Self {
        LmmGroupings {
            n_primary: max_clusters,
            nested_per_parent: 0,
            nested_theta: None,
            crossed: vec![],
            extra_offsets: vec![],
            k_total: max_clusters,
            primary_q: 1,
            primary_slope_cols: vec![],
            diagonal_theta: compute_diagonal_theta(1, 0), // [0]
        }
    }

    /// Structure for a (validated) ClusterSpec at workspace size max_n.
    /// validate() guarantees ≤ 1 nested entry and crossed ⇒ FixedClusters.
    /// `slope_cols` are the x_full column indices for the primary slopes
    /// (`spec.cluster_slope_design_cols` as usize); pass `&[]` for M1/M2 callers.
    pub fn from_cluster_spec(
        cluster: &engine_contract::ClusterSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        use engine_contract::{ClusterSizing, GroupingRelation};
        let n_primary = match &cluster.sizing {
            ClusterSizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            // div_ceil keeps a partial trailing parent's ids in range
            // (production N is an atom multiple; tests may not be).
            ClusterSizing::FixedSize { cluster_size } => {
                max_n.div_ceil((*cluster_size).max(1) as usize)
            }
        };
        let q_p = 1 + slope_cols.len();
        // Width-general layout: the primary block is `q_p · n_primary` wide
        // ([intercept 0..S | slope_0 S..2S | … | slope_{q-2}]), and the M2
        // nested children + crossed tail follow exactly as before, shifted up by
        // the (q_p−1)·n_primary slope columns. q_p=1 ⇒ `prim_width == n_primary`,
        // so every offset (and k_total) is byte-identical to the M2 path.
        // OWNING site for the RE-column layout: `add_rows_multi`'s zx_slope/s fills
        // and `primary_gram`/`reml_deviance`'s reads use the same `d·n_primary + f`
        // (slope) / `prim_width + f·np + c` (nested-child) convention — change together.
        // `glmm::pirls_solve_blocked_extras` and `glmm::structured_schur_fill` also
        // read the `prim_width + f·np + c` nested-child convention to gather each
        // primary cluster's core-block columns — change together.
        let prim_width = q_p * n_primary;
        let n_extras = cluster.extra_groupings.len();
        debug_assert!(n_extras <= MAX_EXTRA_GROUPINGS);
        let mut nested_per_parent = 0usize;
        let mut nested_theta = None;
        let mut extra_offsets = vec![0usize; n_extras];
        for (g, gs) in cluster.extra_groupings.iter().enumerate() {
            if let GroupingRelation::NestedWithin { n_per_parent } = gs.relation {
                nested_per_parent = (n_per_parent).max(1) as usize;
                nested_theta = Some(q_p * (q_p + 1) / 2 + g); // scalar after the primary vech
                extra_offsets[g] = prim_width; // nested children begin after the primary block
            }
        }
        let mut off = prim_width + n_primary * nested_per_parent;
        let mut crossed = Vec::new();
        for (g, gs) in cluster.extra_groupings.iter().enumerate() {
            if let GroupingRelation::Crossed { n_clusters } = gs.relation {
                let k = (n_clusters).max(1) as usize;
                crossed.push((q_p * (q_p + 1) / 2 + g, k)); // scalar after the primary vech
                extra_offsets[g] = off;
                off += k;
            }
        }
        LmmGroupings {
            n_primary,
            nested_per_parent,
            nested_theta,
            crossed,
            extra_offsets,
            k_total: off,
            primary_q: q_p,
            primary_slope_cols: slope_cols.to_vec(),
            diagonal_theta: compute_diagonal_theta(q_p, n_extras),
        }
    }

    /// Primary vech (`q_p(q_p+1)/2`) + one scalar per extra grouping.
    /// q_p=1 ⇒ `1 + extra_offsets.len()` (the M2 shape).
    pub fn n_theta(&self) -> usize {
        self.primary_q * (self.primary_q + 1) / 2 + self.extra_offsets.len()
    }
    /// Columns eliminated family-by-family: the `q_p` primary RE cols per level
    /// plus nested children. (`k_crossed = k_total − k_family` is the dense tail.)
    pub fn k_family(&self) -> usize {
        self.n_primary * (self.primary_q + self.nested_per_parent)
    }
    pub fn k_crossed(&self) -> usize {
        self.k_total - self.k_family()
    }
    /// True iff the structured block+Schur GLMM PIRLS path
    /// (`glmm::pirls_solve_blocked_extras`) applies: extra groupings are present
    /// (an empty-extras shape routes to the no-extras *blocked* path instead) and
    /// the per-primary core-block width `q_core = primary_q + nested_per_parent`
    /// fits the `MAX_PRIMARY_Q` stack scratch the per-block Crout solve uses.
    /// Extra groupings are intercept-only by contract (`invariant_19`), so no
    /// slopes-on-extras check is needed here. A non-eligible extras shape
    /// (oversized core) falls through to the dense `glmm::pirls_solve`.
    pub fn structured_extras_eligible(&self) -> bool {
        !self.extra_offsets.is_empty() && self.primary_q + self.nested_per_parent <= MAX_PRIMARY_Q
    }
    /// Borrow of the cached diagonal θ-index map (computed once per workspace by
    /// `compute_diagonal_theta`). Zero-alloc: the per-fit pin loop and the
    /// rho-schedule fold read this slice without reallocating. See the
    /// `diagonal_theta` field doc for the column-major vech layout.
    pub fn diagonal_theta(&self) -> &[usize] {
        &self.diagonal_theta
    }

    /// Blind θ₀ and per-component boxes. Diagonal vech entries (the q_p primary
    /// variances + extra scalars) start at THETA0 with box [0, HI]; off-diagonal
    /// vech entries start at 0 with the signed box [−HI, HI]. q_p=1 ⇒ the M2
    /// shape (every entry diagonal: θ₀ = [THETA0;n], box [0, HI]).
    pub fn blind_theta_and_bounds(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n_theta();
        let mut theta = vec![THETA0; n];
        let mut lower = vec![0.0; n];
        let upper = vec![THETA_HI; n];
        let diag = self.diagonal_theta();
        let prim_vech = self.primary_q * (self.primary_q + 1) / 2;
        for i in 0..prim_vech {
            if !diag.contains(&i) {
                theta[i] = 0.0; // off-diagonal blind start
                lower[i] = -THETA_HI; // signed box
            }
        }
        (theta, lower, upper)
    }
}

// ---------------------------------------------------------------------------
// LmmSuffStats — augmented per-RE-column sufficient statistics.
// ---------------------------------------------------------------------------

/// w_i = [x_i ; y_i] (length m = p+1): `c` = Σ w wᵀ (lower triangle),
/// `s[:, a]` = Σ_{i in RE column a} w_i, `counts[a]` = n_a, over the full
/// RE-column set (`[primary | nested children | crossed]`, elimination order).
/// `zx` holds cross-counts only when crossed factors exist (crossed ⇒ Regime A
/// ⇒ K moderate); nested-only designs derive parent↔child coupling from
/// `counts` + the id/n_per parent map, so Regime-B-nested memory stays O(K·m).
///
/// Layout invariant: `s` stays per-RE-column-addressable with `counts`
/// alongside — the balanced-design collapse slots in at exactly this
/// granularity later; don't fold columns at accumulation time.
pub struct LmmSuffStats {
    /// Augmented width m = p + 1 (y in the last slot).
    pub m: usize,
    pub n_rows: usize,
    /// Highest primary cluster id + 1 seen since the last reset.
    pub n_clusters: usize,
    pub groupings: LmmGroupings,
    /// m×m Σ w wᵀ (lower triangle; upper never read).
    pub c: Mat<f64>,
    /// m × k_total per-RE-column Σ w.
    pub s: Mat<f64>,
    /// Per-RE-column row counts.
    pub counts: Vec<u32>,
    /// Crossed cross-counts: zx[(a, b)] = #rows where RE column `a` and
    /// crossed column `k_family + b` co-occur. 0×0 when no crossed factors;
    /// nested↔primary coupling is derived from `counts` + the id/n_per parent
    /// map instead. Same-factor crossed pairs never co-occur (level-disjoint),
    /// so those entries stay 0 and the Ω assembly can read unconditionally.
    pub zx: Mat<f64>,
    /// Slope-weighted twin of `zx` (Task 6 composition): for a primary slope RE
    /// column `scol = (d+1)·n_primary + f`, `zx_slope[(scol, b)] = Σ_{i ∈ f ∩
    /// crossed_b} x_{slope_d}` — the covariate-weighted co-occurrence the
    /// slope↔crossed coupling in `fam_b` reads (plain `zx` is unweighted, fit for
    /// the intercept row only). Same shape as `zx` (`k_total × k_crossed`); only
    /// the slope-RE-col rows are filled. 0×0 when no crossed factor; left all-zero
    /// when `primary_q == 1` (no slopes).
    pub zx_slope: Mat<f64>,
    /// Per-row widened [X y] (len m) — filled once per row so the c-triangle
    /// and s scatter read contiguous f64 instead of re-indexing the f32 data
    /// plane per (i, j). Scratch, not a statistic: reset leaves it alone.
    pub w_buf: Vec<f64>,
}

impl LmmSuffStats {
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, LmmGroupings::single(max_clusters))
    }

    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let m = p + 1;
        let k = groupings.k_total;
        let kx = groupings.k_crossed();
        LmmSuffStats {
            m,
            n_rows: 0,
            n_clusters: 0,
            c: Mat::zeros(m, m),
            s: Mat::zeros(m, k),
            counts: vec![0; k],
            zx: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            zx_slope: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            w_buf: vec![0.0; m],
            groupings,
        }
    }

    /// Reset to "no rows seen", reusing storage.
    pub fn reset(&mut self) {
        let m = self.m;
        for j in 0..m {
            for i in 0..m {
                self.c[(i, j)] = 0.0;
            }
        }
        for a in 0..self.counts.len() {
            for j in 0..m {
                self.s[(j, a)] = 0.0;
            }
            self.counts[a] = 0;
        }
        let (zr, zc) = (self.zx.nrows(), self.zx.ncols());
        for j in 0..zc {
            for i in 0..zr {
                self.zx[(i, j)] = 0.0;
                self.zx_slope[(i, j)] = 0.0;
            }
        }
        self.n_rows = 0;
        self.n_clusters = 0;
    }

    /// Primary-only convenience — the M1 shape.
    pub fn add_rows(&mut self, x: MatRef<'_, f32>, y: &[f32], cluster_ids: &[u32]) {
        self.add_rows_multi(x, y, cluster_ids, &[]);
    }

    /// Accumulate a block of rows for every grouping. `extra_ids[g]` holds
    /// extra grouping g's GLOBALIZED level ids (workspace layout — crossed
    /// 0..I, nested parent·n_per+within), declaration order; this routine maps
    /// them onto the elimination-order column offsets.
    pub fn add_rows_multi(
        &mut self,
        x: MatRef<'_, f32>,
        y: &[f32],
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
    ) {
        debug_assert_eq!(x.nrows(), y.len());
        debug_assert_eq!(x.nrows(), cluster_ids.len());
        debug_assert_eq!(extra_ids.len(), self.groupings.extra_offsets.len());
        let p = self.m - 1;
        debug_assert_eq!(x.ncols(), p);
        let kf = self.groupings.k_family();
        let n_g = 1 + extra_ids.len();
        let mut gid = [0usize; 1 + MAX_EXTRA_GROUPINGS];
        for row in 0..x.nrows() {
            gid[0] = cluster_ids[row] as usize;
            for (e, ids) in extra_ids.iter().enumerate() {
                gid[1 + e] = self.groupings.extra_offsets[e] + ids[row] as usize;
            }
            debug_assert!(gid[..n_g].iter().all(|&a| a < self.counts.len()));
            for &a in &gid[..n_g] {
                self.counts[a] += 1;
            }
            // Widen this row's [X y] once (f32 data plane → f64 accumulators);
            // every scatter below reads w_buf.
            for j in 0..p {
                self.w_buf[j] = x[(row, j)] as f64;
            }
            self.w_buf[p] = y[row] as f64;
            for &a in &gid[..n_g] {
                let scol = self
                    .s
                    .col_mut(a)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                #[allow(clippy::needless_range_loop)]
                for j in 0..self.m {
                    scol[j] += self.w_buf[j];
                }
            }
            for j in 0..self.m {
                let wj = self.w_buf[j];
                let ccol = self
                    .c
                    .col_mut(j)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                #[allow(clippy::needless_range_loop)]
                for i in j..self.m {
                    ccol[i] += self.w_buf[i] * wj;
                }
            }
            if self.groupings.k_crossed() > 0 {
                let slope = self.groupings.primary_q > 1;
                let n_prim = self.groupings.n_primary;
                for bi in 0..n_g {
                    let b = gid[bi];
                    if b >= kf {
                        let bl = b - kf;
                        #[allow(clippy::needless_range_loop)]
                        for ai in 0..n_g {
                            if ai != bi {
                                self.zx[(gid[ai], bl)] += 1.0;
                            }
                        }
                        // Slope-weighted twin for the slope↔crossed coupling
                        // (Task 6). The intercept row is `zx`'s gid[0]; each slope
                        // component d's RE col at this row's primary level gid[0]
                        // is (d+1)·n_primary + gid[0]. Reuses this crossed col `bl`
                        // — no re-derivation of crossed memberships. x widens
                        // f32→f64. Only the primary's crossed co-occurrence
                        // matters: a slope lives on the primary grouping, so the
                        // weight is x_{slope}; nested/other-crossed groupings carry
                        // no slope, so they contribute nothing here.
                        if slope {
                            for (d, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                                let z = self.w_buf[sc];
                                // scol mirrors from_cluster_spec's RE-column layout — change together.
                                let scol = (d + 1) * n_prim + gid[0];
                                self.zx_slope[(scol, bl)] += z;
                            }
                        }
                    }
                }
            }
            // Primary slopes: each slope k's RE column at level gid[0] (offset
            // (k+1)·n_primary + gid[0]) accumulates z = x_{slope_k} weighted sums
            // into `s`; the intercept subcol (gid[0]) is already filled with z=1
            // above. counts is NOT incremented for slope subcols (the Gram reads
            // `s`, not counts). z and the [X y] weights widen f32→f64.
            if self.groupings.primary_q > 1 {
                let n_prim = self.groupings.n_primary;
                for (k, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                    let z = self.w_buf[sc];
                    let scol = (k + 1) * n_prim + gid[0];
                    let scol_mut = self
                        .s
                        .col_mut(scol)
                        .try_as_col_major_mut()
                        .unwrap()
                        .as_slice_mut();
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..self.m {
                        scol_mut[j] += z * self.w_buf[j];
                    }
                }
            }
            if gid[0] + 1 > self.n_clusters {
                self.n_clusters = gid[0] + 1;
            }
        }
        self.n_rows += x.nrows();
    }
}

// ---------------------------------------------------------------------------
// LmmFitScratch — per-fit scratch, allocated once per (p, max_clusters).
// ---------------------------------------------------------------------------

pub struct LmmFitScratch {
    /// Row-major w×w family block (w = q_p + n_per) — assembled and
    /// Crout-factored in place per family; rows contiguous for the Crout.
    pub fam_a: Vec<f64>,
    /// Stacked forward-solved family couplings, t_dim × W_tot column-major
    /// (W_tot = n_primary·w): column f·w+r is family f's L_f⁻¹B_f row r,
    /// contiguous. Filled and solved per family, consumed by ONE triangular
    /// GEMM downdate after the family loop — the per-family tail re-traversals
    /// are gone (the F/G post-mortem cure: traffic/chain shape, not indexing).
    pub bt: Vec<f64>,
    /// (k_crossed+m)² tail [[H, B_x],[B_xᵀ, C]] over [crossed | X y],
    /// column-major lower triangle (entry (i,j) at j·t_dim+i); GEMM-downdated
    /// once per eval, then one dense faer llt over a MatRef view.
    pub tail: Vec<f64>,
    /// λ per local crossed column (θ of the owning factor), refreshed per eval.
    pub lam_x: Vec<f64>,
    /// q_p×q_p primary-slope scratch (row-major), refreshed per eval/family: the
    /// lower-tri Λ_p and the per-level Gram G_f. Empty on the q_p=1 path. Kept in
    /// scratch so the deviance hot loop stays zero-alloc (the warm-path invariant).
    pub prim_lam: Vec<f64>,
    pub prim_gram: Vec<f64>,
    /// Balanced-collapse Grams: pair-major r ≤ r′ blocks, w(w+1)/2 of
    /// them, each a FULL t_dim×t_dim column-major G_rr′ = Σ_f raw_r(f)·raw_r′(f)ᵀ
    /// over the active balanced prefix — θ-independent, refreshed once per fit
    /// by `precompute_balanced_collapse`. Empty on the slope path (collapse
    /// never applies there).
    pub fam_gram: Vec<f64>,
    /// t_dim² combine scratch for the collapse downdate (lower triangle used);
    /// its first w slots double as the A⁻¹ forward-solve temp.
    pub comb: Vec<f64>,
    /// w×w row-major A(θ)⁻¹, rebuilt per eval on the collapse path.
    pub a_inv: Vec<f64>,
    /// Active balanced families (prefix length). 0 = collapse off → the
    /// per-family loop runs (the fallback and the pre-F behaviour).
    pub collapse_n_active: usize,
    /// m×m trailing block of the tail factor — identical semantics to M1's
    /// augmented [X y] factor; every recovery step reads only this.
    pub factor: Mat<f64>,
    pub betas: Vec<f64>,
    pub var_diag: Vec<f64>,
    pub t_sq: Vec<f64>,
    pub u: Vec<f64>,
    pub sigma_sq: f64,
    /// p×p X'V⁻¹X rebuild (L_XX·L_XXᵀ) + the shared joint-Wald scratch
    /// (mirrors the lme workspace triple the promoted helper expects).
    pub joint_xtvix: Mat<f64>,
    pub joint_k_inv: Mat<f64>,
    pub joint_sigma_t_chol: Mat<f64>,
    pub joint_rhs: Vec<f64>,
}

impl LmmFitScratch {
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, &LmmGroupings::single(max_clusters))
    }

    pub fn with_groupings(p: usize, g: &LmmGroupings) -> Self {
        let m = p + 1;
        let w = g.primary_q + g.nested_per_parent; // q_p primary cols + nested children
        let t_dim = g.k_crossed() + m;
        let q2 = if g.primary_q > 1 {
            g.primary_q * g.primary_q
        } else {
            0
        };
        // Collapse scratch only on the intercept-primary path; slope w would
        // mis-size it and the path never collapses.
        let npairs = if g.primary_q == 1 { w * (w + 1) / 2 } else { 0 };
        LmmFitScratch {
            fam_a: vec![0.0; w * w],
            bt: vec![0.0; g.n_primary * w * t_dim],
            tail: vec![0.0; t_dim * t_dim],
            lam_x: vec![0.0; g.k_crossed()],
            prim_lam: vec![0.0; q2],
            prim_gram: vec![0.0; q2],
            fam_gram: vec![0.0; npairs * t_dim * t_dim],
            // max(t_dim², w): the first w slots double as the A⁻¹ forward-solve
            // temp, and deep nesting can push w past t_dim² (tiny p, large n_per).
            comb: vec![
                0.0;
                if npairs > 0 {
                    (t_dim * t_dim).max(w)
                } else {
                    0
                }
            ],
            a_inv: vec![0.0; if npairs > 0 { w * w } else { 0 }],
            collapse_n_active: 0,
            factor: Mat::zeros(m, m),
            betas: vec![0.0; p],
            var_diag: vec![0.0; p],
            t_sq: vec![0.0; p],
            u: vec![0.0; p],
            sigma_sq: f64::NAN,
            joint_xtvix: Mat::zeros(p, p),
            joint_k_inv: Mat::zeros(p, p),
            joint_sigma_t_chol: Mat::zeros(p, p),
            joint_rhs: vec![0.0; p],
        }
    }
}

// ---------------------------------------------------------------------------
// cluster_theta_truth — spec-derived θ truth-start (shared with GLMM).
// ---------------------------------------------------------------------------

/// Build the θ truth-start vector for any `ClusterSpec`. Always uses the
/// vech(chol(D_rel)) recipe — for q=1 (no slopes) this is `[√τ²]` for the
/// primary entry plus `[√τ²_g]` per extra, which is bit-identical to the
/// prior direct-sqrt branch: `√((√τ²)²) == √τ²`. Diagonal vech entries are
/// floored at `THETA_TRUTH_FLOOR`; off-diagonals (signed, q_p ≥ 2 only) are
/// not floored. Extra-grouping scalars always follow the primary vech, keeping
/// the output length `q_p(q_p+1)/2 + n_extras`. Shared verbatim by
/// `LmmWorkspace::for_cluster_spec` and `GlmmWorkspace::for_cluster_spec` —
/// change together (mirrors data_gen slope draw's D assembly).
pub(crate) fn cluster_theta_truth(cluster: &engine_contract::ClusterSpec) -> Vec<f64> {
    let q = 1 + cluster.slopes.len();
    let mut tau = vec![cluster.tau_squared.max(0.0).sqrt()];
    for s in &cluster.slopes {
        tau.push(s.variance.max(0.0).sqrt());
    }
    // Build D_rel = diag(τ)·R·diag(τ) — same assembly as data_gen's slope draw.
    let (_, r) = cluster.re_correlation_matrix();
    let mut d = vec![0.0f64; q * q];
    for i in 0..q {
        for j in 0..q {
            d[i * q + j] = tau[i] * r[i * q + j] * tau[j];
        }
    }
    let lam = crate::data_gen::chol_lower(&d, q);
    let mut tt = Vec::with_capacity(q * (q + 1) / 2 + cluster.extra_groupings.len());
    for c in 0..q {
        for rr in c..q {
            let v = lam[rr * q + c];
            tt.push(if rr == c { v.max(THETA_TRUTH_FLOOR) } else { v });
        }
    }
    for g in &cluster.extra_groupings {
        tt.push(g.tau_squared.max(0.0).sqrt().max(THETA_TRUTH_FLOOR));
    }
    tt
}

// ---------------------------------------------------------------------------
// LmmWorkspace — everything a fit needs, allocated once per problem shape.
// ---------------------------------------------------------------------------

pub struct LmmWorkspace {
    pub suff: LmmSuffStats,
    pub fit: LmmFitScratch,
    /// BOBYQA solver state — `Bobyqa::new` is the crate's only allocation
    /// site; `minimize` is zero-alloc on the warm path.
    pub solver: Bobyqa,
    /// θ in/out buffer for `minimize`; holds θ̂ (post-pin) after `fit_lmm`.
    pub theta: Vec<f64>,
    /// Per-component box bounds. Diagonal entries: [0, THETA_HI].
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// Spec-derived truth start, [primary, extras in declaration order]:
    /// θ₀_g = √τ²_g (unit residual σ by data_gen construction; Mle+scenarios
    /// is rejected, so the spec τ² IS the block τ²), clamped at
    /// THETA_TRUTH_FLOOR. Empty until built by `for_cluster_spec`. The
    /// DGP-derived hint is the recorded deliberate exception to the
    /// generation↔estimation split.
    pub theta_truth: Vec<f64>,
}

impl LmmWorkspace {
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, LmmGroupings::single(max_clusters))
    }

    /// Workspace for a validated non-degenerate ClusterSpec at max_n. Carries
    /// the spec-derived truth start and a scaled BOBYQA schedule (P1).
    /// `slope_cols` are the x_full column indices for the primary slopes
    /// (`spec.cluster_slope_design_cols` as usize); pass `&[]` for M1/M2 callers.
    pub fn for_cluster_spec(
        p: usize,
        cluster: &engine_contract::ClusterSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        let groupings = LmmGroupings::from_cluster_spec(cluster, max_n, slope_cols);
        let n_theta = groupings.n_theta();
        // θ truth-start via the shared helper — always-vech path (behaviour-
        // preserving: for q=1 chol([[τ²]]) = [[√τ²]], vech = [√τ²], bit-identical
        // to the prior direct-sqrt branch). Mirrors GlmmWorkspace::for_cluster_spec.
        let theta_truth = cluster_theta_truth(cluster);
        debug_assert_eq!(theta_truth.len(), n_theta);
        // Scaled schedule (P1): rho_begin = 0.1·min θ₀ — the eval count is
        // dominated by rho shrinkage, not travel distance. Floor 0.01 ⇒
        // rho_begin ≥ 1e-3 ≥ 10·RHO_END; cap 0.5 keeps the blind-start
        // conservatism for large τ. PRIMA start-projection: θ₀ = 10·rho_begin
        // by construction, clear of the push-out rule. Fold over DIAGONAL
        // truth-start entries only — a signed off-diagonal λ_{d,j} near 0 must
        // not drive the start radius.
        let rho_begin = (0.1
            * groupings
                .diagonal_theta()
                .iter()
                .map(|&i| theta_truth[i])
                .fold(f64::INFINITY, f64::min))
        .min(RHO_BEGIN);
        // npt: ⌈1.5n⌉+1 from n_theta = 3 up, Powell's 2n+1 below (E2 npt sweep,
        // 2026-06-12, clock-locked): the mid model wins on every measured dim ≥ 3
        // (n=3 lmm_slope 1.06x / crossed_nested 1.05x, n=6 multislope 1.10x —
        // mostly smaller kernel inner dims, evals/fit flat), while at n=2 the
        // range collapses to n+2, which loses (lmm_nested 0.88x, evals 21.8→26.6).
        // GLMM keeps 2n+1 — its sweep was mixed-to-negative (glmm.rs). LMM-only.
        let npt = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let config = Config {
            rho_begin,
            rho_end: RHO_END,
            npt,
            ..Config::new(n_theta)
        };
        let fit = LmmFitScratch::with_groupings(p, &groupings);
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        LmmWorkspace {
            suff: LmmSuffStats::with_groupings(p, groupings),
            fit,
            solver: Bobyqa::new(n_theta, config)
                .expect("BOBYQA config constants are valid by construction"),
            theta,
            lower,
            upper,
            theta_truth,
        }
    }

    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let n_theta = groupings.n_theta();
        let fit = LmmFitScratch::with_groupings(p, &groupings);
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        LmmWorkspace {
            suff: LmmSuffStats::with_groupings(p, groupings),
            fit,
            // The constants are valid by construction for the crate's checks
            // (npt default within bounds; box width 1e3 ≥ 2·RHO_BEGIN), so a
            // failure here is an engine bug, not a runtime branch.
            solver: Bobyqa::new(n_theta, bobyqa_config(n_theta))
                .expect("BOBYQA config constants are valid by construction"),
            theta,
            lower,
            upper,
            theta_truth: vec![], // blind path — hint unused
        }
    }
}

/// Dispatch helper: Some iff this spec routes to the general path
/// (estimator Mle + non-degenerate ClusterSpec). Degenerate specs keep the
/// scalar Brent path — shipped workloads never enter the new code.
pub fn build_lmm_workspace(
    spec: &crate::spec::SimulationSpec,
    max_n: usize,
    n_predictors: usize,
) -> Option<Box<LmmWorkspace>> {
    let cluster = spec.cluster.as_ref()?;
    // General path = Mle + non-degenerate ClusterSpec, where non-degenerate means
    // extra groupings OR primary slopes. A degenerate single-intercept spec keeps
    // the scalar Brent path in `lme.rs`.
    if spec.estimator != engine_contract::EstimatorSpec::Mle
        || (cluster.extra_groupings.is_empty() && cluster.slopes.is_empty())
    {
        return None;
    }
    let slope_cols: Vec<usize> = spec
        .cluster_slope_design_cols
        .iter()
        .map(|&c| c as usize)
        .collect();
    Some(Box::new(LmmWorkspace::for_cluster_spec(
        n_predictors,
        cluster,
        max_n,
        &slope_cols,
    )))
}

// ---------------------------------------------------------------------------
// Primary slope block helpers (q_p×q_p) — free fns; q_p is tiny.
// ---------------------------------------------------------------------------

/// Unpack the primary q×q lower-triangular Λ from the column-major vech θ prefix
/// into `lam` (row-major, len q·q; upper triangle zeroed). `pub(crate)` — Task 9
/// reuses it to reconstruct the RE covariance D = ΛΛ′ for the introspection
/// surface. Caller owns `lam` so the deviance hot loop stays zero-alloc.
pub(crate) fn primary_lambda(theta: &[f64], q: usize, lam: &mut [f64]) {
    for v in lam[..q * q].iter_mut() {
        *v = 0.0;
    }
    let mut t = 0;
    for c in 0..q {
        for r in c..q {
            lam[r * q + c] = theta[t];
            t += 1;
        }
    }
}

/// Per-level primary Gram G_f (q×q, row-major) recovered from suff stats into
/// `gram`, no new accumulator: G[0][0]=n_f; G[0][a]=G[a][0]=Σ x_{a-1} over f;
/// G[a][b]=Σ x_{a-1} x_{b-1} over f. The slope covariates are [X y] rows, so
/// every entry sits in `s`. Component d's RE col at level f is `d·n_primary + f`
/// (mirrors `from_cluster_spec`'s RE-column layout — change together).
fn primary_gram(suff: &LmmSuffStats, g: &LmmGroupings, f: usize, q: usize, gram: &mut [f64]) {
    let n_prim = g.n_primary;
    for v in gram[..q * q].iter_mut() {
        *v = 0.0;
    }
    gram[0] = f64::from(suff.counts[f]); // G[0][0]
    for a in 1..q {
        let sa = suff.s[(g.primary_slope_cols[a - 1], f)]; // Σ x_{a-1} over f
        gram[a] = sa;
        gram[a * q] = sa;
        for b in 1..=a {
            // Σ x_{a-1} x_{b-1} over f — slope_{a-1}'s subcol against slope_{b-1}'s level.
            let v = suff.s[(g.primary_slope_cols[a - 1], b * n_prim + f)];
            gram[a * q + b] = v;
            gram[b * q + a] = v;
        }
    }
}

/// A_f = I_q + Λ′ G Λ into the lower triangle of the row-major `fam_a` block
/// (`stride` = family width w; what Crout reads). Λ lower-tri row-major,
/// G symmetric row-major. (Λ′G)[r][e] = Σ_{d≥r} Λ[d][r] G[d][e]; A[r][c] =
/// δ_{rc} + Σ_{e≥c} (Λ′G)[r][e] Λ[e][c].
fn assemble_primary_a(fam_a: &mut [f64], stride: usize, lam: &[f64], gram: &[f64], q: usize) {
    for r in 0..q {
        for c in 0..=r {
            let mut s = 0.0;
            for e in c..q {
                let mut m_re = 0.0;
                for d in r..q {
                    m_re += lam[d * q + r] * gram[d * q + e];
                }
                s += m_re * lam[e * q + c];
            }
            fam_a[r * stride + c] = if r == c { 1.0 + s } else { s };
        }
    }
}

/// Balanced-collapse precompute: detect a balanced active prefix and
/// accumulate the θ-independent cross-Grams G_rr′ from the suff stats. Returns
/// false (and arms the fallback loop) when the design is unbalanced, has a
/// slope primary, or is empty. Balance = counts[f] equal over an active prefix
/// and zero after, per child slot c equal across active families (the
/// grid-atom-snapped layout; non-prefix actives are conservatively rejected).
/// `fit.bt` is per-eval scratch, free here — its first w·t_dim slots stage each
/// family's raw rows.
pub(crate) fn precompute_balanced_collapse(suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> bool {
    let g = &suff.groupings;
    fit.collapse_n_active = 0;
    if g.primary_q != 1 || g.n_primary == 0 || suff.n_rows == 0 {
        return false;
    }
    let np = g.nested_per_parent;
    let w = 1 + np;
    let kx = g.k_crossed();
    let m = suff.m;
    let t_dim = kx + m;
    let n0 = suff.counts[0];
    if n0 == 0 {
        return false;
    }
    let mut n_active = 1;
    while n_active < g.n_primary && suff.counts[n_active] == n0 {
        n_active += 1;
    }
    if suff.counts[n_active..g.n_primary].iter().any(|&c| c != 0) {
        return false; // hole or non-prefix layout — fall back
    }
    for c in 0..np {
        let c0 = suff.counts[g.n_primary + c]; // family 0, child slot c
        for f in 0..g.n_primary {
            let cc = suff.counts[g.n_primary + f * np + c];
            if (f < n_active && cc != c0) || (f >= n_active && cc != 0) {
                return false;
            }
        }
    }
    // Grams over the active prefix (inactive families are all-zero rows and
    // would contribute nothing anyway).
    let blk = t_dim * t_dim;
    let npairs = w * (w + 1) / 2;
    fit.fam_gram[..npairs * blk].fill(0.0);
    for f in 0..n_active {
        for r in 0..w {
            let gcol = if r == 0 {
                f
            } else {
                g.n_primary + f * np + (r - 1)
            };
            let dst = &mut fit.bt[r * t_dim..(r + 1) * t_dim];
            for (b, slot) in dst[..kx].iter_mut().enumerate() {
                *slot = suff.zx[(gcol, b)];
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            dst[kx..kx + m].copy_from_slice(scol);
        }
        let (bt, gram) = (&fit.bt, &mut fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            for rp in r..w {
                let gblk = &mut gram[pidx * blk..(pidx + 1) * blk];
                for j in 0..t_dim {
                    let vj = bt[rp * t_dim + j];
                    if vj != 0.0 {
                        for i in 0..t_dim {
                            gblk[j * t_dim + i] += bt[r * t_dim + i] * vj;
                        }
                    }
                }
                pidx += 1;
            }
        }
    }
    fit.collapse_n_active = n_active;
    true
}

// ---------------------------------------------------------------------------
// reml_deviance — the blocked-Cholesky objective.
// ---------------------------------------------------------------------------

/// REML profiled deviance at θ via the family-blocked augmented Cholesky.
///
/// Ω_θ over [primary | nested children | crossed | X y]. The leading block is
/// block-diagonal per FAMILY (a primary level + its nested children — nested
/// children never co-occur across parents), so it is eliminated family-by-
/// family: factor the (1+n_per)² A_f, forward-solve its coupling to the
/// [crossed | X y] tail — cost linear in cluster count. The per-family tail
/// downdates are stacked into ONE triangular GEMM after the family loop
/// (Tail −= Bt·Bt′ over the solved couplings in `bt`; result-moving vs the
/// old sequential per-family subtraction — the F/G chain-latency cure).
/// Crossed factors couple everything (the dense Z_a′Z_b coupling, sanctioned
/// dense within the stated regime), so they stay in the tail with [X y]: one
/// dense (k_crossed+m) faer llt per evaluation. With no extras this is M1's
/// per-cluster shrink downdate up to FP reassociation, and with no crossed
/// factors the tail is just the m×m [X y] block.
///
/// Balanced collapse (intercept-only primary): when the per-fit precompute
/// (`precompute_balanced_collapse`) finds a balanced active prefix — grid
/// atom-snapping guarantees one at production N — the family loop is replaced
/// by ONE Crout of the common A(θ), log|L_ZZ|'s family part by
/// n_active·log|L|, and the stacked-GEMM downdate by a θ-independent Gram
/// combine Σ_{r,r′} A⁻¹[r,r′]·scale_r·scale_r′·G_rr′, column-scaled by
/// diag(λ_x | 1). Reassociation-level result movement vs the loop; unbalanced
/// counts and the slope path (data-dependent A_f) keep the loop.
///
/// The deviance reads OFF THE FACTORS — log|L_ZZ|² from the family pivots +
/// the crossed tail diagonal, log|L_XX|², r² = L[p,p]² from the trailing m×m
/// block — no β backsolve per evaluation. Normalization matches
/// `lme.rs::profiled_deviance` exactly:
///   dev(θ) = log|V| + log|X'V⁻¹X| + (N−P)·log(σ̂²),
/// so general-vs-shipped deviance values agree to FP error, not up to a
/// constant. Returns INFINITY on any Cholesky failure / non-positive σ̂².
///
/// θ is vech-packed per grouping — [primary, extras in declaration order]. The
/// primary block is width-general: `Λ_p` is the column-major vech θ prefix
/// (`q_p(q_p+1)/2` entries), and the per-level Gram `G_f` is recovered from `s`
/// with no new accumulator.
///
/// Composition (Task 6): the q_p primary block coexists with the intercept-only
/// crossed/nested extra tail in one family-blocked elimination. The family block
/// is `q_p + nested_per_parent` wide; the new primary-slope↔nested-child
/// off-diagonal falls out of `s` (free), and the primary-slope↔crossed-factor
/// coupling reads the slope-weighted `zx_slope` twin (each slope row d at level f
/// is `zx_slope[(d·n_primary+f, b)]`, vs the intercept's unweighted `zx[(f, b)]`).
/// The extra-grouping scalars keep q_g = 1.
pub fn reml_deviance(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> f64 {
    let g = &suff.groupings;
    debug_assert_eq!(theta.len(), g.n_theta());
    let m = suff.m;
    let p = m - 1;
    if suff.n_rows <= p || p == 0 {
        return f64::INFINITY;
    }
    let kf = g.k_family();
    let kx = g.k_crossed();
    let t_dim = kx + m;
    let np = g.nested_per_parent;
    let w = g.primary_q + np; // width-general family width: q_p primary cols + nested children
    let th_p = theta[0];
    let th_n = g.nested_theta.map(|t| theta[t]).unwrap_or(0.0);

    // Width-general primary factor (q_p ≥ 2 ⇒ slope path; q_p == 1 ⇒ M2 scalar,
    // kept byte-identical). The slope path may now carry a crossed/nested tail
    // (Task 6 composition). Λ_p is the vech-packed θ prefix, refreshed into
    // scratch (`fit.prim_lam`) so the hot loop stays zero-alloc.
    let slope = g.primary_q > 1;
    if slope {
        primary_lambda(theta, g.primary_q, &mut fit.prim_lam);
    }

    // λ per local crossed column.
    {
        let mut b = 0usize;
        for &(ti, k) in &g.crossed {
            for _ in 0..k {
                fit.lam_x[b] = theta[ti];
                b += 1;
            }
        }
    }

    // --- tail init: [[H, ·],[B_x, C]] (lower triangle, column-major) ---
    fit.tail[..t_dim * t_dim].fill(0.0);
    for b in 0..kx {
        let lam = fit.lam_x[b];
        let gcol = kf + b;
        // Cross-factor coupling (row b in earlier columns a < b); same-factor
        // zx entries are structurally 0.
        let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
        for a in 0..b {
            fit.tail[a * t_dim + b] = lam * fit.lam_x[a] * zxb[kf + a];
        }
        let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[b * t_dim..(b + 1) * t_dim];
        tcol[b] = 1.0 + lam * lam * f64::from(suff.counts[gcol]);
        for j in 0..m {
            tcol[kx + j] = lam * scol[j];
        }
    }
    for j in 0..m {
        let ccol = suff.c.col(j).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[(kx + j) * t_dim..(kx + j + 1) * t_dim];
        tcol[kx + j..kx + m].copy_from_slice(&ccol[j..m]);
    }

    // --- family elimination ---
    let collapse = !slope && fit.collapse_n_active > 0;
    let mut log_lzz_half = 0.0_f64; // hoisted — single binding both arms write
    if collapse {
        let n_active = fit.collapse_n_active;
        // One representative A from the balanced prefix (family 0) — the
        // legacy q=1 fill verbatim.
        let n_f = f64::from(suff.counts[0]);
        fit.fam_a[0] = 1.0 + th_p * th_p * n_f;
        for c in 0..np {
            let n_c = f64::from(suff.counts[g.n_primary + c]);
            for c2 in 0..np {
                fit.fam_a[(1 + c) * w + (1 + c2)] = 0.0;
            }
            fit.fam_a[(1 + c) * w] = th_p * th_n * n_c;
            fit.fam_a[(1 + c) * w + (1 + c)] = 1.0 + th_n * th_n * n_c;
        }
        // Crout — the legacy in-place loop, one factor for all families.
        let mut log_l_half = 0.0_f64;
        for j in 0..w {
            let mut d = fit.fam_a[j * w + j];
            for k in 0..j {
                let v = fit.fam_a[j * w + k];
                d -= v * v;
            }
            if !(d.is_finite() && d > 0.0) {
                return f64::INFINITY;
            }
            let l = d.sqrt();
            fit.fam_a[j * w + j] = l;
            log_l_half += l.ln();
            for i in (j + 1)..w {
                let mut v = fit.fam_a[i * w + j];
                for k in 0..j {
                    v -= fit.fam_a[i * w + k] * fit.fam_a[j * w + k];
                }
                fit.fam_a[i * w + j] = v / l;
            }
        }
        log_lzz_half = (n_active as f64) * log_l_half;
        // A⁻¹ = L⁻ᵀL⁻¹ column by column (w ≤ 1+n_per — hand-rolled). comb's
        // first w slots are the forward-solve temp; comb is refilled below.
        for r in 0..w {
            for i in 0..w {
                let mut acc = if i == r { 1.0 } else { 0.0 };
                for k in 0..i {
                    acc -= fit.fam_a[i * w + k] * fit.comb[k];
                }
                fit.comb[i] = acc / fit.fam_a[i * w + i];
            }
            for i in (0..w).rev() {
                let mut acc = fit.comb[i];
                for k in (i + 1)..w {
                    acc -= fit.fam_a[k * w + i] * fit.a_inv[k * w + r];
                }
                fit.a_inv[i * w + r] = acc / fit.fam_a[i * w + i];
            }
        }
        // Combine: comb(lower) = Σ_{r≤r′} scale_r·scale_r′·A⁻¹[r,r′]·(G + [r≠r′]Gᵀ).
        let t2 = t_dim * t_dim;
        fit.comb[..t2].fill(0.0);
        let (comb, gram) = (&mut fit.comb, &fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            let sr = if r == 0 { th_p } else { th_n };
            for rp in r..w {
                let srp = if rp == 0 { th_p } else { th_n };
                let coeff = sr * srp * fit.a_inv[r * w + rp];
                let gblk = &gram[pidx * t2..(pidx + 1) * t2];
                if coeff != 0.0 {
                    if r == rp {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] += coeff * gblk[j * t_dim + i];
                            }
                        }
                    } else {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] +=
                                    coeff * (gblk[j * t_dim + i] + gblk[i * t_dim + j]);
                            }
                        }
                    }
                }
                pidx += 1;
            }
        }
        // Tail −= D·comb·D, D = diag(λ_x | 1_m) — column scaling folded here.
        for j in 0..t_dim {
            let dj = if j < kx { fit.lam_x[j] } else { 1.0 };
            for i in j..t_dim {
                let di = if i < kx { fit.lam_x[i] } else { 1.0 };
                fit.tail[j * t_dim + i] -= di * dj * fit.comb[j * t_dim + i];
            }
        }
    } else {
        for f in 0..g.n_primary {
            // A_f (w×w lower): the primary q_p×q_p block A_p = I + Λ′GΛ, then (on the
            // intercept-only M2 path) nested-child diags + parent–child counts. The
            // slope branch additionally carries the composed nested children
            // (Task 6); the M2 `else` stays byte-identical (q_p=1 parity).
            if slope {
                let q = g.primary_q;
                primary_gram(suff, g, f, q, &mut fit.prim_gram);
                // Disjoint field borrows keep this zero-alloc and borrow-checked.
                assemble_primary_a(&mut fit.fam_a, w, &fit.prim_lam, &fit.prim_gram, q); // I + Λ′GΛ
                                                                                         // Composed nested children (rows/cols q..q+np). Scalar child λ = θ_n
                                                                                         // (M2 build); child–child off-diagonals are 0 (children never
                                                                                         // co-occur). The primary↔child off-diagonal A[(q+c, e)] folds the raw
                                                                                         // cross-Gram (intercept = counts[child]; slope d = s[(slope_col_d,
                                                                                         // child_re_col)]) through Λ_p, mirroring how M2 reads counts for the
                                                                                         // intercept↔child term. n_primary = primary level count (slope RE
                                                                                         // stride); np = children per parent (nested width) — kept distinct.
                for c in 0..np {
                    // Nested child RE col = prim_width + f·np + c (prim_width = q_p·n_primary).
                    let gcol = g.n_primary * g.primary_q + f * np + c;
                    let n_c = f64::from(suff.counts[gcol]);
                    for c2 in 0..np {
                        fit.fam_a[(q + c) * w + (q + c2)] = 0.0;
                    }
                    fit.fam_a[(q + c) * w + (q + c)] = 1.0 + th_n * th_n * n_c;
                    // Primary↔child: A[(q+c, e)] = θ_n · Σ_{d≥e} Λ_p[d,e] · Graw_d,
                    // Graw_0 = n_c (intercept), Graw_d = Σ_{i∈child} x_{slope_{d-1}}.
                    for e in 0..q {
                        let mut acc = 0.0;
                        for d in e..q {
                            let graw_d = if d == 0 {
                                n_c
                            } else {
                                suff.s[(g.primary_slope_cols[d - 1], gcol)]
                            };
                            acc += fit.prim_lam[d * q + e] * graw_d;
                        }
                        fit.fam_a[(q + c) * w + e] = th_n * acc;
                    }
                }
            } else {
                // parent–child counts = child row counts (a child's rows all lie
                // inside its parent).
                let n_f = f64::from(suff.counts[f]);
                fit.fam_a[0] = 1.0 + th_p * th_p * n_f;
                for c in 0..np {
                    let gcol = g.n_primary + f * np + c;
                    let n_c = f64::from(suff.counts[gcol]);
                    for c2 in 0..np {
                        fit.fam_a[(1 + c) * w + (1 + c2)] = 0.0;
                    }
                    fit.fam_a[(1 + c) * w] = th_p * th_n * n_c;
                    fit.fam_a[(1 + c) * w + (1 + c)] = 1.0 + th_n * th_n * n_c;
                }
            }
            // In-place Crout Cholesky over the row-major w×w block, w ≤ 1+n_per —
            // hand-rolled (zero-alloc; INFINITY on a non-positive pivot, the
            // module's failure surface). Chains are ≤ w links — not chain-sick.
            for j in 0..w {
                let mut d = fit.fam_a[j * w + j];
                for k in 0..j {
                    let v = fit.fam_a[j * w + k];
                    d -= v * v;
                }
                if !(d.is_finite() && d > 0.0) {
                    return f64::INFINITY;
                }
                let l = d.sqrt();
                fit.fam_a[j * w + j] = l;
                log_lzz_half += l.ln();
                for i in (j + 1)..w {
                    let mut v = fit.fam_a[i * w + j];
                    for k in 0..j {
                        v -= fit.fam_a[i * w + k] * fit.fam_a[j * w + k];
                    }
                    fit.fam_a[i * w + j] = v / l;
                }
            }
            // B_f (rows = Bt columns f·w..f·w+w, each contiguous): cols [crossed | X y].
            let fb = f * w;
            if slope {
                // Primary rows folded through Λ_p; nested-child rows scaled by θ_n
                // (M2 build at the shifted child offset). n_prim is the primary level
                // count (slope RE stride: slope d-1's col at level f = d·n_prim+f);
                // np is the nested width — kept distinct.
                let q = g.primary_q;
                let n_prim = g.n_primary;
                // Primary rows ↔ crossed tail: intercept (d=0) reads zx[(f,b)];
                // slope d reads zx_slope[(d·n_prim+f, b)]; both folded through Λ_p,
                // scaled by the crossed λ_b. Column-b slices hoisted (unit-stride).
                for b in 0..kx {
                    let lam_b = fit.lam_x[b];
                    let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
                    let zxsb = suff.zx_slope.col(b).try_as_col_major().unwrap().as_slice();
                    for r in 0..q {
                        let mut brb = 0.0;
                        for d in r..q {
                            let zeta = if d == 0 { zxb[f] } else { zxsb[d * n_prim + f] };
                            brb += fit.prim_lam[d * q + r] * zeta;
                        }
                        fit.bt[(fb + r) * t_dim + b] = lam_b * brb;
                    }
                }
                // Primary rows ↔ [X y] tail: Z_f′[Xy] row d at col j is s[(j, d·n_prim+f)]
                // (intercept d=0 at col f), folded through Λ_p. Level-f s-columns
                // hoisted once per family (unit-stride faer columns).
                let mut s_cols: [&[f64]; MAX_PRIMARY_Q] = [&[]; MAX_PRIMARY_Q];
                for (d, sc) in s_cols.iter_mut().enumerate().take(q) {
                    *sc = suff
                        .s
                        .col(d * n_prim + f)
                        .try_as_col_major()
                        .unwrap()
                        .as_slice();
                }
                for r in 0..q {
                    let bcol = &mut fit.bt[(fb + r) * t_dim + kx..(fb + r) * t_dim + kx + m];
                    for j in 0..m {
                        let mut brj = 0.0;
                        #[allow(clippy::needless_range_loop)]
                        for d in r..q {
                            brj += fit.prim_lam[d * q + r] * s_cols[d][j];
                        }
                        bcol[j] = brj;
                    }
                }
                // Nested-child rows (q..q+np) — M2 build at the shifted child RE col.
                for c in 0..np {
                    let gcol = n_prim * q + f * np + c; // prim_width + f·np + c
                    let off = (fb + q + c) * t_dim;
                    for b in 0..kx {
                        fit.bt[off + b] = th_n * fit.lam_x[b] * suff.zx[(gcol, b)];
                    }
                    let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
                    let bcol = &mut fit.bt[off + kx..off + kx + m];
                    for j in 0..m {
                        bcol[j] = th_n * scol[j];
                    }
                }
            } else {
                let s_f = suff.s.col(f).try_as_col_major().unwrap().as_slice();
                let b0 = fb * t_dim;
                for b in 0..kx {
                    fit.bt[b0 + b] = th_p * fit.lam_x[b] * suff.zx[(f, b)];
                }
                {
                    let bcol = &mut fit.bt[b0 + kx..b0 + kx + m];
                    for j in 0..m {
                        bcol[j] = th_p * s_f[j];
                    }
                }
                for c in 0..np {
                    let gcol = g.n_primary + f * np + c;
                    let off = (fb + 1 + c) * t_dim;
                    for b in 0..kx {
                        fit.bt[off + b] = th_n * fit.lam_x[b] * suff.zx[(gcol, b)];
                    }
                    let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
                    let bcol = &mut fit.bt[off + kx..off + kx + m];
                    for j in 0..m {
                        bcol[j] = th_n * scol[j];
                    }
                }
            }
            // Forward-solve L_f⁻¹ B_f in place on this family's Bt columns — axpy
            // over contiguous t_dim-slices; per element the k-order subtractions
            // and the final divide are unchanged from the old row-sweep (solved
            // k<r values are final in both orders).
            for r in 0..w {
                let (done, rest) = fit.bt.split_at_mut((fb + r) * t_dim);
                let col_r = &mut rest[..t_dim];
                for k in 0..r {
                    let l_rk = fit.fam_a[r * w + k];
                    let col_k = &done[(fb + k) * t_dim..(fb + k + 1) * t_dim];
                    for t in 0..t_dim {
                        col_r[t] -= l_rk * col_k[t];
                    }
                }
                let l_rr = fit.fam_a[r * w + r];
                #[allow(clippy::needless_range_loop)]
                for t in 0..t_dim {
                    col_r[t] /= l_rr;
                }
            }
        }

        // --- one stacked downdate: Tail −= Σ_f B_f′B_f = Bt·Bt′ (lower) ---
        // The n_primary per-family rank-w tail re-traversals collapse into ONE
        // triangular GEMM through faer's blocked multi-accumulator FMA kernels
        // (Par::Seq — per-fit parallelism is the outer rayon loop). RESULT-MOVING:
        // GEMM accumulation order replaces the per-family sequential subtraction;
        // sanctioned (rides the golden re-freeze campaign), verified against the
        // brute-force oracle + Gate-0 parity bands which are orders wider than the
        // reorder's last-ulp footprint.
        let w_tot = g.n_primary * w;
        {
            let bt = faer::MatRef::from_column_major_slice(&fit.bt[..t_dim * w_tot], t_dim, w_tot);
            let tail = faer::MatMut::from_column_major_slice_mut(
                &mut fit.tail[..t_dim * t_dim],
                t_dim,
                t_dim,
            );
            faer::linalg::matmul::triangular::matmul(
                tail,
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Add,
                bt,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                bt.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                -1.0,
                faer::Par::Seq,
            );
        }
    }

    // --- dense tail factorization (faer llt on a MatRef view of the tail
    // scratch — same call/FP exposure as before) ---
    let tail_ref = faer::MatRef::from_column_major_slice(&fit.tail[..t_dim * t_dim], t_dim, t_dim);
    let chol = match tail_ref.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::INFINITY,
    };
    let l = chol.L();
    for b in 0..kx {
        let lbb = l[(b, b)];
        if !(lbb.is_finite() && lbb > 0.0) {
            return f64::INFINITY;
        }
        log_lzz_half += lbb.ln();
    }
    let log_lzz_sq = 2.0 * log_lzz_half;
    // Trailing m×m → fit.factor (M1 semantics; recovery reads only this).
    for j in 0..m {
        let lcol = l.col(kx + j).try_as_col_major().unwrap().as_slice();
        for i in 0..m {
            fit.factor[(i, j)] = if i >= j { lcol[kx + i] } else { 0.0 };
        }
    }

    let mut log_lxx_sq = 0.0_f64;
    for j in 0..p {
        let ljj = fit.factor[(j, j)];
        if !(ljj.is_finite() && ljj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= 2.0;

    let lyy = fit.factor[(p, p)];
    let r_sq = lyy * lyy;
    let df = (suff.n_rows - p) as f64;
    let sigma_sq = r_sq / df;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    fit.sigma_sq = sigma_sq;

    log_lzz_sq + log_lxx_sq + df * sigma_sq.ln()
}

// ---------------------------------------------------------------------------
// fit_lmm — BOBYQA θ-search + once-at-θ̂ recovery.
// ---------------------------------------------------------------------------

/// One general-path fit summary. θ̂ (post-pin) stays in `ws.theta`; β̂/Var/t²
/// land in `ws.fit`'s target slots — no per-fit allocation.
pub struct LmmFit {
    pub sigma_sq: f64,
    pub converged: bool,
    /// Shipped `lme.rs` coding: 0 = interior min, 1 = pinned at a variance
    /// boundary (counted converged), 2 = optimizer/numerical failure
    /// (NaN-filled, non-converged).
    pub boundary_hit: u8,
    /// Objective evaluations consumed (diagnostics only).
    pub n_eval: usize,
    /// Joint Wald-χ² over the target set (the shared `lme.rs` helper). Under
    /// H₀: β_T = 0, asymptotically χ²(k). NaN on non-converged / degenerate
    /// fits or an empty target set.
    pub joint_t_sq: f64,
    /// Bit k set iff diagonal variance component k (in `diagonal_theta()`
    /// order) pinned at 0. 0 on non-converged fits.
    pub pinned_components: u32,
}

/// Fit by BOBYQA minimisation of the REML profiled deviance over the box-
/// bounded θ, with β̂ / σ̂² / Var(β̂_target) recovered once at θ̂.
///
/// Caller contract: `ws.suff` holds the accumulated rows (reset + add_rows
/// per dataset); `target_indices` index design columns.
///
/// `theta_start`: `None` → blind THETA0 per component (the default for
/// arbitrary provided bytes); `Some(θ₀)` → per-component spec-derived truth
/// start, `[primary, extras in declaration order]` (Y is always synthetic, so
/// true θ_g = τ_g/σ is known), each component clamped to THETA_TRUTH_FLOOR. A
/// per-scenario constant — determinism and chunk merging are unaffected. The
/// DGP-derived hint is a deliberate, recorded exception to the
/// generation↔estimation split.
pub fn fit_lmm(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
) -> LmmFit {
    let LmmWorkspace {
        suff,
        fit,
        solver,
        theta,
        lower,
        upper,
        theta_truth: _, // the hint arrives via `theta_start`, never read here
    } = ws;
    let p = suff.m - 1;

    // Arm the balanced collapse for this dataset's counts (cheap —
    // O(n_primary·w²·t_dim²) once per fit; sets collapse_n_active = 0 on any
    // unbalanced/slope shape, which keeps the per-family loop).
    precompute_balanced_collapse(suff, fit);

    // Cold start per fit (no warm-start across sims — would re-import
    // cross-grid-point path dependence). A Some-start is clamped to the
    // floor; under the fixed RHO_BEGIN, PRIMA's start-projection may still
    // move a small start to rho_begin off the 0 bound — benign and
    // deterministic. The scaled schedule (rho_begin = 0.1·θ₀) that makes
    // small starts pay off is M2 activation: rho lives in the solver's
    // construction-time Config and θ₀ is per-scenario, so it belongs where
    // M2 builds the workspace per workload.
    match theta_start {
        Some(ts) => {
            debug_assert_eq!(ts.len(), theta.len());
            for (t, &v) in theta.iter_mut().zip(ts) {
                *t = v.max(THETA_TRUTH_FLOOR);
            }
        }
        None => {
            for t in theta.iter_mut() {
                *t = THETA0;
            }
        }
    }
    let out = solver.minimize(|xs| reml_deviance(xs, suff, fit), theta, lower, upper);

    // Status mapping: Converged ⇒ candidate fit; MaxFunReached /
    // ModelDegenerate ⇒ optimizer failure ⇒ NaN-fill, non-converged (the
    // generalized boundary_hit == 2). TargetReached unreachable (f_target
    // stays -inf); InvalidArgs would be an engine bug — the workspace fixes
    // shapes and bounds.
    debug_assert!(out.status != Status::InvalidArgs);
    let ok = matches!(out.status, Status::Converged);

    // Per-component deterministic pin: every DIAGONAL variance component ≤
    // PIN_THETA collapses to exactly 0 — FP-stable across platforms, counted
    // converged. Off-diagonal vech entries (signed slope covariances) are never
    // pinned: a corr → ±1 boundary presents as the *diagonal* λ_{dd} → 0 under
    // the Cholesky parameterization, so pinning the diagonal is the whole
    // policy. `pinned_components` records the bit per `diagonal_theta()` index.
    let diag = suff.groupings.diagonal_theta();
    let mut pinned = false;
    let mut pinned_components = 0u32;
    if ok {
        for (k, &ti) in diag.iter().enumerate() {
            if theta[ti] <= PIN_THETA {
                theta[ti] = 0.0;
                pinned = true;
                pinned_components |= 1 << k;
            }
        }
    }

    // Pin eval at θ̂ — refreshes factor/σ̂² at the accepted point (the shipped
    // path's "pin Cholesky at θ̂" step).
    let dev = if ok {
        reml_deviance(theta, suff, fit)
    } else {
        f64::INFINITY
    };

    // Rank guard on the p×p block — mirrors lme.rs's EPS_RANK min/max-diag
    // test on the pinning factor.
    let degenerate = !dev.is_finite() || chol_rank_deficient(fit.factor.as_ref(), p, EPS_RANK);
    if !ok || degenerate {
        for v in fit.betas.iter_mut() {
            *v = f64::NAN;
        }
        for &t in target_indices {
            fit.var_diag[t as usize] = f64::NAN;
            fit.t_sq[t as usize] = f64::NAN;
        }
        return LmmFit {
            sigma_sq: f64::NAN,
            converged: false,
            boundary_hit: 2,
            n_eval: out.n_eval,
            joint_t_sq: f64::NAN,
            pinned_components: 0,
        };
    }

    // β̂: backward solve L_XXᵀ β̂ = l_yX, where l_yX[j] = factor[(p, j)] (the
    // y-row of the augmented factor) — the once-at-θ̂ backsolve.
    for j in (0..p).rev() {
        let mut acc = fit.factor[(p, j)];
        for k in (j + 1)..p {
            acc -= fit.factor[(k, j)] * fit.betas[k];
        }
        fit.betas[j] = acc / fit.factor[(j, j)];
    }

    // Var(β̂_j) = σ̂²·‖L_XX⁻¹e_j‖² per target; t² = β̂²/Var — the lme.rs
    // step-7 forward-solve recipe on this factor.
    let sigma_sq = fit.sigma_sq;
    for &tj in target_indices {
        let tj = tj as usize;
        for v in fit.u[..p].iter_mut() {
            *v = 0.0;
        }
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= fit.factor[(i, k)] * fit.u[k];
            }
            fit.u[i] = acc / fit.factor[(i, i)];
        }
        let norm_sq: f64 = fit.u[..p].iter().map(|v| v * v).sum();
        let vd = sigma_sq * norm_sq;
        fit.var_diag[tj] = vd;
        fit.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
            (fit.betas[tj] * fit.betas[tj]) / vd
        } else {
            f64::NAN
        };
    }

    // Joint Wald-χ² over the target set — the shared lme.rs helper (promoted
    // pub(crate)). It re-Choleskys X'V⁻¹X internally, so hand it the product
    // the augmented factor already encodes: X'V⁻¹X = L_XX·L_XXᵀ (leading p×p
    // of fit.factor; the y row is index p).
    let joint_t_sq = if target_indices.is_empty() {
        f64::NAN
    } else {
        for j in 0..p {
            for i in 0..p {
                let mut acc = 0.0;
                for k in 0..=i.min(j) {
                    acc += fit.factor[(i, k)] * fit.factor[(j, k)];
                }
                fit.joint_xtvix[(i, j)] = acc;
            }
        }
        crate::lme::joint_wald_chi_sq(
            fit.joint_xtvix.as_ref(),
            &fit.betas,
            sigma_sq,
            target_indices,
            fit.joint_k_inv.as_mut(),
            fit.joint_sigma_t_chol.as_mut(),
            &mut fit.joint_rhs,
        )
    };

    LmmFit {
        sigma_sq,
        converged: true,
        boundary_hit: u8::from(pinned),
        n_eval: out.n_eval,
        joint_t_sq,
        pinned_components,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lme::{profiled_deviance, LmeScratch, LmeSuffStats};
    use crate::ols::OlsScratch;
    use crate::workspace::SimWorkspace;
    use faer::reborrow::IntoConst;

    /// Deterministic pseudo-data (NR LCG), uniform in (−1, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// n=48, p=3 (intercept + x1 + x2), 6 clusters,
    /// y = 0.5 + 0.4·x1 − 0.2·x2 + u_c + 0.8·e.
    fn hand_dataset() -> (Mat<f32>, Vec<f32>, Vec<u32>) {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 42u64;
        let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 3);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % n_clusters;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            x[(i, 2)] = x2 as f32;
            y[i] = (0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st)) as f32;
        }
        (x, y, ids)
    }

    /// Build the shipped-path scratch over a populated workspace — the same
    /// inline-reborrow shape `batch.rs`'s Mle arm and `lme::tests` use.
    fn build_lme_scratch<'w>(
        ws: &'w mut SimWorkspace,
        n_rows: u32,
        n_clusters: u32,
    ) -> LmeScratch<'w> {
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
        }
    }

    /// Populate a fresh SimWorkspace's lme suff-stats from a dataset and
    /// return it (helper shared by the deviance + fit parity tests).
    fn shipped_workspace(x: &Mat<f32>, y: &[f32], ids: &[u32], n_clusters: u32) -> SimWorkspace {
        let mut ws = SimWorkspace::new(
            x.nrows(),
            x.ncols(),
            x.ncols() - 1,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                engine_contract::ClusterSizing::FixedClusters { n_clusters },
                0.25,
            )),
        );
        ws.reset_lme_suff_stats();
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
        suff.add_rows(x.as_ref(), y, ids);
        ws
    }

    /// Same quantity, two factorizations — both return
    /// log|V| + log|X'V⁻¹X| + (N−P)·log σ̂², so agreement is FP-level
    /// (≤ 1e-9 rel), not up-to-a-constant. THE formulation proof; the Gate-0
    /// spike held this on every θ probed.
    #[test]
    fn deviance_matches_shipped_across_theta() {
        let (x, y, ids) = hand_dataset();
        let mut ws = shipped_workspace(&x, &y, &ids, 6);
        let mut scratch = build_lme_scratch(&mut ws, 48, 6);

        let mut suff = LmmSuffStats::new(3, 6);
        suff.add_rows(x.as_ref(), &y, &ids);
        let mut fit = LmmFitScratch::new(3, 6);
        let mut fit_c = LmmFitScratch::new(3, 6);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));

        for &theta in &[0.0, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let dev_ship = profiled_deviance(theta, &mut scratch);
            let dev_gen = reml_deviance(&[theta], &suff, &mut fit);
            assert!(dev_ship.is_finite() && dev_gen.is_finite(), "θ={theta}");
            let tol = 1e-9 * dev_ship.abs().max(1.0);
            assert!(
                (dev_ship - dev_gen).abs() <= tol,
                "θ={theta}: shipped {dev_ship} vs general {dev_gen}"
            );
            // Collapse arm — reassociation band vs the general loop incl. θ=0.
            let dev_c = reml_deviance(&[theta], &suff, &mut fit_c);
            let band = 1e-9 * dev_gen.abs().max(1.0);
            assert!(
                (dev_c - dev_gen).abs() <= band,
                "θ={theta}: collapse {dev_c} vs general {dev_gen}"
            );
        }
    }

    /// All scratch is overwritten per call — re-evaluating a θ after an
    /// intervening different-θ call reproduces bit-identical deviance and σ̂²
    /// (mirrors lme.rs's EST-20 stale-state test).
    #[test]
    fn reml_deviance_overwrites_state() {
        let (x, y, ids) = hand_dataset();
        let mut suff = LmmSuffStats::new(3, 6);
        suff.add_rows(x.as_ref(), &y, &ids);
        let mut fit = LmmFitScratch::new(3, 6);

        let dev_a = reml_deviance(&[1.0], &suff, &mut fit);
        let sig_a = fit.sigma_sq;
        let _ = reml_deviance(&[2.0], &suff, &mut fit);
        let dev_b = reml_deviance(&[1.0], &suff, &mut fit);
        let sig_b = fit.sigma_sq;
        assert_eq!(dev_a, dev_b, "deviance(θ=1) must be reproducible");
        assert_eq!(sig_a, sig_b, "σ̂²(θ=1) must be reproducible");
    }

    /// End-to-end q=1 parity on the hand dataset: the general machine vs the
    /// shipped `lme_fit` on the same suff-stats bytes, at the Gate-0 amended
    /// tolerances (rel 1e-4, abs floors β̂ 1e-5 / stat 1e-4 — the measured Brent
    /// θ̂-placement-noise floor).
    #[test]
    fn fit_matches_shipped_lme_fit_on_hand_dataset() {
        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws_ship = shipped_workspace(&x, &y, &ids, 6);
        let scratch = build_lme_scratch(&mut ws_ship, 48, 6);
        let ship = crate::lme::lme_fit(x.as_ref(), &y, &ids, &targets, None, scratch);
        assert!(ship.converged);

        let mut ws = LmmWorkspace::new(3, 6);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &targets, None);
        assert!(fit.converged);
        assert!(fit.boundary_hit <= 1);

        for j in 0..3 {
            let (a, b) = (ship.betas[j], ws.fit.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                "β[{j}]: {a} vs {b}"
            );
        }
        for &tj in &targets {
            let a = ship.t_sq[tj as usize].sqrt();
            let b = ws.fit.t_sq[tj as usize].sqrt();
            let d = (a - b).abs();
            assert!(
                d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
                "stat[{tj}]: {a} vs {b}"
            );
        }
        let (a, b) = (ship.joint_t_sq, fit.joint_t_sq);
        let d = (a - b).abs();
        assert!(
            d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
            "joint: {a} vs {b}"
        );
    }

    /// Deterministic pin: y carries NO between-cluster signal by construction —
    /// residuals alternate ±0.8 within each cluster with equal counts, so every
    /// cluster's residual sum is exactly 0 and the REML deviance is minimized at
    /// θ = 0. The fit must pin (boundary_hit == 1), write θ̂ = exactly 0.0, and
    /// count as converged — the Q7 deterministic-pin policy.
    #[test]
    fn zero_between_cluster_variance_pins_at_exactly_zero() {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 7u64;
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
            // the ±0.8 residuals cancel exactly per cluster.
            let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
            y[i] = (0.5 + 0.4 * x1 + e) as f32;
        }
        let mut ws = LmmWorkspace::new(2, n_clusters);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert_eq!(fit.boundary_hit, 1);
        assert_eq!(ws.theta[0], 0.0, "pin must be exact 0.0, not merely small");
        assert!(ws.fit.betas[1].is_finite());
    }

    /// Rank deficiency fails cleanly: x2 = 0.1·x1 (the scaled-duplicate fixture —
    /// exact duplicates can slip through faer's llt grey zone) must produce a
    /// non-converged, NaN-filled fit with boundary_hit == 2.
    #[test]
    fn rank_deficient_design_fails_cleanly() {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 11u64;
        let mut x = Mat::<f32>::zeros(n, 3);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            x[(i, 2)] = (0.1 * x1) as f32; // 0.1-scaled duplicate → guaranteed non-convergence
            y[i] = (0.5 + 0.4 * x1 + 0.8 * lcg(&mut st)) as f32;
        }
        let mut ws = LmmWorkspace::new(3, n_clusters);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(!fit.converged);
        assert_eq!(fit.boundary_hit, 2);
        assert!(ws.fit.betas.iter().all(|b| b.is_nan()));
        assert!(ws.fit.t_sq[1].is_nan() && ws.fit.t_sq[2].is_nan());
    }

    /// θ-start seam (P1): a truth-started fit reaches the same answer as the
    /// blind fit on the same bytes — and Some(0.0) exercises the
    /// THETA_TRUTH_FLOOR clamp rather than starting on the 0 boundary.
    /// Bands are the Gate-0 amended floors: two BOBYQA runs from different
    /// starts each place θ̂ within the rho_end band of the same minimum.
    #[test]
    fn theta_start_some_matches_blind_fit() {
        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws_blind = LmmWorkspace::new(3, 6);
        ws_blind.suff.add_rows(x.as_ref(), &y, &ids);
        let blind = fit_lmm(&mut ws_blind, &targets, None);
        assert!(blind.converged);

        for start in [[0.0], [0.6]] {
            let mut ws = LmmWorkspace::new(3, 6);
            ws.suff.add_rows(x.as_ref(), &y, &ids);
            let fit = fit_lmm(&mut ws, &targets, Some(&start));
            assert!(fit.converged, "start {start:?}");
            for j in 0..3 {
                let (a, b) = (ws_blind.fit.betas[j], ws.fit.betas[j]);
                let d = (a - b).abs();
                assert!(
                    d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                    "start {start:?} β[{j}]: blind {a} vs started {b}"
                );
            }
        }
    }

    /// Bounded-allocation warm-path twin of lme.rs's
    /// `lme_fit_warm_path_bounded_alloc`. Marked #[ignore] because dhat measures
    /// process-wide allocations and concurrent tests contaminate the count:
    ///   cargo test -p engine-core lmm_fit_warm_path_bounded_alloc -- --ignored --test-threads=1
    ///
    /// BOUND locks the measured warm-path block count. LmmWorkspace itself is
    /// allocation-free across fits (Bobyqa::new is the only solver allocation,
    /// done once). On the faer kernel the per-call blocks are `llt` internals —
    /// ~2 per deviance evaluation (15.1–15.7 evals/fit at rho_end 1e-6, the
    /// Gate-0 measured mean), the same acceptance the shipped path's 26
    /// blocks/call carry; if a future faer version changes its Cholesky
    /// internals, update the bound — do not relax it. The owned-kernel
    /// alternative (P3) was spiked and rejected — wasm `f64::ln` ULP forks,
    /// not the factorization, broke bit-equality — so the faer bound is the
    /// locked steady state.
    #[test]
    #[ignore]
    fn lmm_fit_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND: u64 = 4800; // Measured 4600 (this machine) — ~46 blocks/fit of faer `llt` internals on the family-blocked q=1 path (one m×m tail llt per eval). `fit_lmm` no longer allocates per fit (the diagonal_theta index map is cached once on LmmGroupings), so this count is purely faer's Cholesky internals — faer-version/machine specific. q=1 deviance is byte-identical to M1's (held by the lmm_parity corpus + golden_rng), so the eval trajectory is unchanged; the count differs from M1's 3804 only because faer's blocked llt allocates more per eval than M1's hand-rolled augmented factor. If faer changes its Cholesky internals, update — do not relax.

        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];
        let mut ws = LmmWorkspace::new(3, 6);

        // Warmup drives one-time setup outside the profiler window.
        ws.suff.reset();
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let _ = fit_lmm(&mut ws, &targets, None);

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows(x.as_ref(), &y, &ids);
            let _ = fit_lmm(&mut ws, &targets, None);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }

    // -----------------------------------------------------------------------
    // M2 multi-grouping: layout-true datasets, suff-stats, family-blocked
    // deviance vs a brute-force n×n oracle, and end-to-end fits.
    // -----------------------------------------------------------------------

    /// Layout-true multi-grouping dataset: primary S=6, crossed I=4, nested
    /// np=2 (optional), p=3, n = n_blocks·atom rows. Ids come from the
    /// contract layout helpers — the same functions the workspace uses.
    #[allow(clippy::type_complexity)]
    fn multi_dataset(
        with_nested: bool,
        n_blocks: usize,
    ) -> (
        Mat<f32>,
        Vec<f32>,
        Vec<u32>,
        Vec<Vec<u32>>,
        engine_contract::ClusterSpec,
    ) {
        use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, 0.25);
        cluster.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: 0.15,
            slopes: vec![],
        });
        if with_nested {
            cluster.extra_groupings.push(GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                tau_squared: 0.08,
                slopes: vec![],
            });
        }
        let n = n_blocks * cluster.atom();
        let mut st = 99u64;
        let u_p: Vec<f64> = (0..6).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_x: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
        let u_n: Vec<f64> = (0..12).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 3);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let n_extras = cluster.extra_groupings.len();
        let mut eids: Vec<Vec<u32>> = vec![vec![0u32; n]; n_extras];
        for i in 0..n {
            pid[i] = cluster.sizing.cluster_of_row(i) as u32;
            #[allow(clippy::needless_range_loop)]
            for g in 0..n_extras {
                eids[g][i] = cluster.extra_level_of_row(g, i) as u32;
            }
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            x[(i, 2)] = x2 as f32;
            y[i] = (0.5 + 0.4 * x1 - 0.2 * x2
                + u_p[pid[i] as usize]
                + u_x[eids[0][i] as usize]
                + if with_nested {
                    u_n[eids[1][i] as usize]
                } else {
                    0.0
                }
                + 0.8 * lcg(&mut st)) as f32;
        }
        (x, y, pid, eids, cluster)
    }

    /// `diagonal_theta` / `n_theta` / `k_family` at q_p ∈ {1, 2, 3} — locks
    /// the column-major vech ordering. q_p=1 must reproduce M2 values; q_p>1
    /// tests the standalone slope branch (no extras, k_crossed=0).
    #[test]
    fn groupings_vech_layout() {
        use engine_contract::{ClusterSizing, ClusterSpec, SlopeTerm};
        let sizing = ClusterSizing::FixedClusters { n_clusters: 4 };
        let base = ClusterSpec::intercept_only(sizing.clone(), 0.25);

        // q_p = 1 (intercept-only): M2 shape must be unchanged.
        let g1 = LmmGroupings::from_cluster_spec(&base, 40, &[]);
        assert_eq!(g1.n_theta(), 1);
        assert_eq!(g1.k_family(), 4); // 4 clusters × 1
        assert_eq!(g1.diagonal_theta(), &[0][..]);

        // q_p = 2 (1 slope): vech([σ_00, σ_10, σ_11]) length 3; diagonals at 0, 2.
        let mut spec2 = base.clone();
        spec2.slopes.push(SlopeTerm {
            column: engine_contract::ColumnId(1),
            variance: 0.1,
            corr_with_intercept: 0.0,
            corr_with: vec![],
        });
        let g2 = LmmGroupings::from_cluster_spec(&spec2, 40, &[1]);
        assert_eq!(g2.primary_q, 2);
        assert_eq!(g2.n_theta(), 3); // 2·3/2 = 3
        assert_eq!(g2.k_family(), 8); // 4 clusters × 2
        assert_eq!(g2.k_total, 8);
        assert_eq!(g2.diagonal_theta(), &[0, 2][..]); // off-diagonal vech[1]=1 excluded

        // q_p = 3 (2 slopes): vech([σ_00, σ_10, σ_11, σ_20, σ_21, σ_22]) length 6; diagonals at 0, 3, 5.
        let mut spec3 = base.clone();
        spec3.slopes.push(SlopeTerm {
            column: engine_contract::ColumnId(1),
            variance: 0.1,
            corr_with_intercept: 0.0,
            corr_with: vec![],
        });
        spec3.slopes.push(SlopeTerm {
            column: engine_contract::ColumnId(2),
            variance: 0.15,
            corr_with_intercept: 0.0,
            corr_with: vec![0.0],
        });
        let g3 = LmmGroupings::from_cluster_spec(&spec3, 40, &[1, 2]);
        assert_eq!(g3.primary_q, 3);
        assert_eq!(g3.n_theta(), 6); // 3·4/2 = 6
        assert_eq!(g3.k_family(), 12); // 4 clusters × 3
        assert_eq!(g3.k_total, 12);
        assert_eq!(g3.diagonal_theta(), &[0, 3, 5][..]);
    }

    /// Suff-stats bookkeeping on a hand-checkable block: counts per RE column,
    /// per-column sums, crossed cross-counts.
    #[test]
    fn suff_stats_multi_accumulators() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 1); // one atom block, n=48
        let g = LmmGroupings::from_cluster_spec(&cluster, 48, &[]);
        assert_eq!(g.n_primary, 6);
        assert_eq!(g.nested_per_parent, 2);
        assert_eq!(g.k_family(), 18); // 6 + 6·2
        assert_eq!(g.k_total, 22); // + 4 crossed
        assert_eq!(g.n_theta(), 3);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        // One full-factorial block: every primary level has 8 rows, every
        // child 4, every crossed level 12.
        for f in 0..6 {
            assert_eq!(suff.counts[f], 8);
        }
        for c in 6..18 {
            assert_eq!(suff.counts[c], 4);
        }
        for b in 18..22 {
            assert_eq!(suff.counts[b], 12);
        }
        // Crossed co-occurrence: each (primary, crossed) pair shares exactly
        // 2 rows in a full factorial of 6·4·2.
        assert_eq!(suff.zx[(0, 0)], 2.0);
        assert_eq!(suff.zx[(5, 3)], 2.0);
        // Same-factor crossed pairs never co-occur.
        assert_eq!(suff.zx[(18, 1)], 0.0);
        // Intercept column sum = row count per level.
        assert!((suff.s[(0, 0)] - 8.0).abs() < 1e-12);
    }

    /// Textbook REML deviance on the explicit n×n V — the oracle for the
    /// family-blocked elimination. dev = ln|V| + ln|X'V⁻¹X| + (N−P)·ln σ̂²,
    /// σ̂² = (y'V⁻¹y − β̂'X'V⁻¹y)/(N−P).  `groups[g]` = grouping g's global
    /// level ids (primary first); `theta[g]` the matching component.
    fn brute_force_deviance(theta: &[f64], x: &Mat<f32>, y: &[f32], groups: &[&[u32]]) -> f64 {
        use faer::linalg::solvers::Solve;
        let n = x.nrows();
        let p = x.ncols();
        // Widen to f64 for all oracle arithmetic.
        let mut x64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x64[(i, j)] = x[(i, j)] as f64;
            }
        }
        let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = 1.0;
        }
        for (g, ids) in groups.iter().enumerate() {
            let t2 = theta[g] * theta[g];
            for i in 0..n {
                for j in 0..n {
                    if ids[i] == ids[j] {
                        v[(i, j)] += t2;
                    }
                }
            }
        }
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut log_det_v = 0.0;
        for i in 0..n {
            log_det_v += vc.L()[(i, i)].ln();
        }
        let log_det_v = 2.0 * log_det_v;
        let mut vix = x64.clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y64[i];
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += x64[(i, a)] * vix[(i, b)];
                }
                xtvix[(a, b)] = acc;
            }
            let mut acc = 0.0;
            for i in 0..n {
                acc += x64[(i, a)] * viy[(i, 0)];
            }
            xtviy[a] = acc;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut log_det_k = 0.0;
        for a in 0..p {
            log_det_k += kc.L()[(a, a)].ln();
        }
        let log_det_k = 2.0 * log_det_k;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for a in 0..p {
            beta[(a, 0)] = xtviy[a];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += y64[i] * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for a in 0..p {
            bxy += beta[(a, 0)] * xtviy[a];
        }
        let df = (n - p) as f64;
        let sigma_sq = (ytviy - bxy) / df;
        log_det_v + log_det_k + df * sigma_sq.ln()
    }

    fn assert_deviance_matches_oracle(with_nested: bool, thetas: &[Vec<f64>]) {
        let (x, y, pid, eids, cluster) = multi_dataset(with_nested, 2);
        let n = x.nrows();
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(3, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(3, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        // Oracle wants global ids per grouping.
        let mut groups: Vec<&[u32]> = vec![&pid];
        for e in &eids {
            groups.push(e);
        }
        for th in thetas {
            let dev = reml_deviance(th, &suff, &mut fit);
            let oracle = brute_force_deviance(th, &x, &y, &groups);
            assert!(dev.is_finite(), "θ={th:?}");
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!(
                (dev - oracle).abs() <= tol,
                "θ={th:?}: family-blocked {dev} vs oracle {oracle}"
            );
            // Collapse arm: same θ through the balanced path — reassociation
            // band vs the loop, oracle band absolute.
            let dev_c = reml_deviance(th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs loop {dev}"
            );
            assert!(
                (dev_c - oracle).abs() <= tol,
                "θ={th:?}: collapse vs oracle"
            );
        }
    }

    #[test]
    fn crossed_deviance_matches_brute_force() {
        assert_deviance_matches_oracle(
            false,
            &[
                vec![0.5, 0.3],
                vec![1.0, 1.0],
                vec![2.0, 0.1],
                vec![0.0, 0.7],
                vec![1e-3, 1e-3],
            ],
        );
    }

    #[test]
    fn crossed_plus_nested_deviance_matches_brute_force() {
        assert_deviance_matches_oracle(
            true,
            &[
                vec![0.5, 0.3, 0.2],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.5, 0.9],
                vec![2.0, 0.05, 0.4],
            ],
        );
    }

    /// Unbalanced counts must take the legacy loop byte-for-byte: a failed
    /// precompute leaves collapse_n_active = 0 and the eval path untouched.
    #[test]
    fn unbalanced_counts_fall_back_byte_identical() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
        let n = x.nrows() - 1; // truncate one row — last cluster short
        let g = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        let eids_t: Vec<Vec<u32>> = eids.iter().map(|e| e[..n].to_vec()).collect();
        suff.add_rows_multi(x.as_ref().subrows(0, n), &y[..n], &pid[..n], &eids_t);
        let gref = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
        let mut fit_a = LmmFitScratch::with_groupings(3, &gref);
        let mut fit_b = LmmFitScratch::with_groupings(3, &gref);
        assert!(!precompute_balanced_collapse(&suff, &mut fit_b));
        for th in [[0.5, 0.3, 0.2], [1.0, 1.0, 1.0], [0.0, 0.5, 0.9]] {
            let a = reml_deviance(&th, &suff, &mut fit_a);
            let b = reml_deviance(&th, &suff, &mut fit_b);
            assert_eq!(a.to_bits(), b.to_bits(), "θ={th:?}");
        }
    }

    /// Nested-only in Regime B — the path with NO crossed tail (zx is 0×0)
    /// and parents that grow with N.
    #[test]
    fn nested_regime_b_deviance_matches_brute_force() {
        use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedSize { cluster_size: 8 }, 0.25);
        cluster.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.1,
            slopes: vec![],
        });
        let n = 4 * cluster.atom(); // 64
        let mut st = 7u64;
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n];
        let u_p: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_c: Vec<f64> = (0..16).map(|_| 0.3 * lcg(&mut st)).collect();
        for i in 0..n {
            pid[i] = cluster.sizing.cluster_of_row(i) as u32;
            cid[i] = cluster.extra_level_of_row(0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] =
                (0.5 + 0.4 * x1 + u_p[pid[i] as usize] + u_c[cid[i] as usize] + 0.8 * lcg(&mut st))
                    as f32;
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![cid.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        for th in [[0.6, 0.4], [1.0, 1.0], [0.2, 0.0], [0.0, 0.0]] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &cid]);
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
            // Collapse arm — reassociation band vs the loop incl. the θ=0 edge.
            let dev_c = reml_deviance(&th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs {dev}"
            );
        }
    }

    /// Balanced-collapse applicability: balanced intercept designs precompute,
    /// slope groupings and unbalanced counts fall back.
    #[test]
    fn balanced_collapse_applicability() {
        // Balanced: the regime-B nested dataset (atom-multiple by construction).
        use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedSize { cluster_size: 8 }, 0.25);
        cluster.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            tau_squared: 0.1,
            slopes: vec![],
        });
        let n = 4 * cluster.atom(); // 64
        let max_n = 2 * n; // workspace sized for a larger grid top — active PREFIX
        let mut st = 7u64;
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n];
        for i in 0..n {
            pid[i] = cluster.sizing.cluster_of_row(i) as u32;
            cid[i] = cluster.extra_level_of_row(0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] = (0.5 + 0.4 * x1 + 0.8 * lcg(&mut st)) as f32;
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
        let n_primary = g.n_primary;
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![cid.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let gref = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit));
        assert_eq!(fit.collapse_n_active, n / 8);
        assert!(fit.collapse_n_active < n_primary); // genuinely a prefix

        // Unbalanced: drop the last row — the trailing cluster is short.
        let mut suff_u =
            LmmSuffStats::with_groupings(2, LmmGroupings::from_cluster_spec(&cluster, max_n, &[]));
        let eids_u = vec![cid[..n - 1].to_vec()];
        suff_u.add_rows_multi(
            x.as_ref().subrows(0, n - 1),
            &y[..n - 1],
            &pid[..n - 1],
            &eids_u,
        );
        assert!(!precompute_balanced_collapse(&suff_u, &mut fit));
        assert_eq!(fit.collapse_n_active, 0);

        // Slope path: never applicable — populated, balanced data, so the
        // rejection is the q_p guard, not the empty-suff early-out (balanced
        // slope counts would otherwise pass the count checks).
        let (xs, ys, ids_s) = slope_dataset();
        let gs = slope_groupings();
        let mut suff_s = LmmSuffStats::with_groupings(2, slope_groupings());
        suff_s.add_rows_multi(xs.as_ref(), &ys, &ids_s, &[]);
        let mut fit_s = LmmFitScratch::with_groupings(2, &gs);
        assert!(!precompute_balanced_collapse(&suff_s, &mut fit_s));
    }

    /// Two crossed factors — the dense cross-factor coupling block.
    #[test]
    fn two_crossed_factors_deviance_matches_brute_force() {
        use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 3 }, 0.25);
        for k in [4u32, 2u32] {
            cluster.extra_groupings.push(GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: k },
                tau_squared: 0.1,
                slopes: vec![],
            });
        }
        let n = 2 * cluster.atom(); // 48
        let mut st = 21u64;
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let mut e0 = vec![0u32; n];
        let mut e1 = vec![0u32; n];
        let u_p: Vec<f64> = (0..3).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_a: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
        let u_b: Vec<f64> = (0..2).map(|_| 0.3 * lcg(&mut st)).collect();
        for i in 0..n {
            pid[i] = cluster.sizing.cluster_of_row(i) as u32;
            e0[i] = cluster.extra_level_of_row(0, i) as u32;
            e1[i] = cluster.extra_level_of_row(1, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] = (0.5
                + 0.4 * x1
                + u_p[pid[i] as usize]
                + u_a[e0[i] as usize]
                + u_b[e1[i] as usize]
                + 0.8 * lcg(&mut st)) as f32;
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![e0.clone(), e1.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        for th in [[0.5, 0.4, 0.3], [1.0, 1.0, 1.0], [0.3, 0.0, 0.8]] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &e0, &e1]);
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
            // Collapse arm — reassociation band vs the loop.
            let dev_c = reml_deviance(&th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs {dev}"
            );
        }
    }

    /// Per-component pin: items carry NO between-level signal by construction
    /// (each item sees every subject equally, and the ±0.8 residual pattern is
    /// block-constant so item means cancel exactly), while subjects carry a
    /// real u_p. The crossed component must pin at exactly 0 (boundary_hit
    /// == 1) with the primary component interior.
    #[test]
    fn zero_crossed_variance_pins_only_that_component() {
        use engine_contract::{ClusterSizing, ClusterSpec, GroupingRelation, GroupingSpec};
        let s_cl = 4usize;
        let i_cl = 3usize;
        let mut cluster = ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters {
                n_clusters: s_cl as u32,
            },
            0.25,
        );
        cluster.extra_groupings.push(GroupingSpec {
            relation: GroupingRelation::Crossed {
                n_clusters: i_cl as u32,
            },
            tau_squared: 0.0,
            slopes: vec![],
        });
        let n = 4 * cluster.atom(); // 48: 4 blocks ⇒ ±0.8 cancels per item
        let mut st = 5u64;
        let u_p: Vec<f64> = (0..s_cl).map(|_| 0.8 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            pid[i] = cluster.sizing.cluster_of_row(i) as u32;
            eid[i] = cluster.extra_level_of_row(0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            let e = if (i / cluster.atom()) % 2 == 0 {
                0.8
            } else {
                -0.8
            };
            y[i] = (0.5 + 0.4 * x1 + u_p[pid[i] as usize] + e) as f32;
        }
        let mut ws = LmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        let eids = vec![eid];
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert_eq!(fit.boundary_hit, 1);
        assert_eq!(ws.theta[1], 0.0, "crossed component must pin at exact 0.0");
        assert!(
            ws.theta[0] > PIN_THETA,
            "primary component must stay interior"
        );
        assert!(fit.joint_t_sq.is_finite());
    }

    /// End-to-end crossed+nested fit recovers the generating β within wide
    /// sanity bands and produces finite Wald machinery — the L1 smoke for the
    /// full multi-grouping pipeline (the statistical gates live in L3).
    #[test]
    fn crossed_nested_fit_recovers_betas() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 4); // n = 192
        let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(fit.converged);
        assert!((ws.fit.betas[1] - 0.4).abs() < 0.15);
        assert!((ws.fit.betas[2] + 0.2).abs() < 0.15);
        assert!(ws.fit.t_sq[1].is_finite() && ws.fit.t_sq[2].is_finite());
        assert!(fit.joint_t_sq.is_finite() && fit.joint_t_sq > 0.0);
        assert_eq!(ws.theta.len(), 3);
    }

    /// General-path twin of lmm_fit_warm_path_bounded_alloc: crossed+nested
    /// workspace. Per-call blocks are the tail-llt faer internals (the family
    /// loop is hand-rolled, zero-alloc) — the same acceptance class as q=1.
    /// Truth-started, matching the production general path (batch/introspect
    /// thread `theta_truth` into every general fit, and `for_cluster_spec`'s
    /// scaled rho schedule is tuned for that start).
    #[test]
    #[ignore]
    fn lmm_fit_general_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND_GENERAL: u64 = 8400; // Measured 8000 (this machine) — ~80 blocks/fit truth-started (scaled rho + spec-derived start; the few-eval regime the production path runs). Per-eval faer `llt` internals only: the family loop is hand-rolled zero-alloc and the cached diagonal_theta map removed the per-fit Vec, so this count is faer-version/machine specific. If faer changes its Cholesky internals, update — do not relax.

        let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
        let targets: Vec<u32> = vec![1, 2];
        let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
        let truth = ws.theta_truth.clone();

        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let _ = fit_lmm(&mut ws, &targets, Some(&truth));

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
            let _ = fit_lmm(&mut ws, &targets, Some(&truth));
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND_GENERAL,
            "general fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND_GENERAL
        );
    }

    /// Truth-start lands in the same minimum as blind on multi-grouping
    /// surfaces (the P1 multimodality probe at M2's n_theta).
    #[test]
    fn truth_start_matches_blind_on_crossed_nested_bytes() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 4);
        let targets: Vec<u32> = vec![1, 2];
        let mut blind_ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
        blind_ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let blind = fit_lmm(&mut blind_ws, &targets, None);
        assert!(blind.converged);

        let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids);
        let truth = ws.theta_truth.clone();
        let fit = fit_lmm(&mut ws, &targets, Some(&truth));
        assert!(fit.converged);
        for j in 0..3 {
            let (a, b) = (blind_ws.fit.betas[j], ws.fit.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                "β[{j}]: {a} vs {b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // M3 standalone primary slopes: q_p×q_p primary block, oracle deviance,
    // diagonal-only pin. Data lives on the engine's f32 plane (mirrors the M2
    // oracle convention); the brute force widens the identical bytes to f64, so
    // the 1e-8 match is exact, not modulo an f32↔f64 roundtrip.
    // -----------------------------------------------------------------------

    /// n=64, p=2 (intercept + x1), 8 clusters, y carries u₀ + u₁·x1.
    fn slope_dataset() -> (Mat<f32>, Vec<f32>, Vec<u32>) {
        let (n, nc) = (64usize, 8usize);
        let mut st = 71u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % nc;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] = (0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + 0.8 * lcg(&mut st)) as f32;
        }
        (x, y, ids)
    }

    /// n=96, p=3 (intercept + x1 + x2), 8 clusters, y carries u₀ + u₁·x1 + u₂·x2.
    fn multislope_dataset() -> (Mat<f32>, Vec<f32>, Vec<u32>) {
        let (n, nc) = (96usize, 8usize);
        let mut st = 91u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let u2: Vec<f64> = (0..nc).map(|_| 0.25 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 3);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % nc;
            ids[i] = c as u32;
            let (x1, x2) = (lcg(&mut st), lcg(&mut st));
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            x[(i, 2)] = x2 as f32;
            y[i] =
                (0.5 + 0.4 * x1 + 0.2 * x2 + u0[c] + u1[c] * x1 + u2[c] * x2 + 0.8 * lcg(&mut st))
                    as f32;
        }
        (x, y, ids)
    }

    /// Textbook REML deviance with a q×q D over the slope columns of Z_p.
    /// `theta` is the column-major vech of Λ (q×q lower-tri); D_rel = ΛΛ′
    /// (σ-relative); V = I + Z·D_rel·Z′ with Z_i = [1, x[i, slope_cols]]. The
    /// f32 data is widened to f64 so the oracle reads the same bytes the suff
    /// stats accumulated.
    fn brute_force_slope_deviance(
        theta: &[f64],
        x: &Mat<f32>,
        y: &[f32],
        ids: &[u32],
        slope_cols: &[usize],
        q: usize,
    ) -> f64 {
        use faer::linalg::solvers::Solve;
        let (n, p) = (x.nrows(), x.ncols());
        // Λ (q×q lower-tri) from column-major vech, then D = ΛΛ′.
        let mut lam = vec![0.0f64; q * q];
        let mut t = 0;
        for c in 0..q {
            for r in c..q {
                lam[r * q + c] = theta[t];
                t += 1;
            }
        }
        let mut d = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                let mut s = 0.0;
                for k in 0..q {
                    s += lam[i * q + k] * lam[j * q + k];
                }
                d[i * q + j] = s;
            }
        }
        let zrow = |i: usize| -> Vec<f64> {
            let mut z = vec![1.0];
            for &sc in slope_cols {
                z.push(x[(i, sc)] as f64);
            }
            z
        };
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for i in 0..n {
            let zi = zrow(i);
            for j in 0..n {
                if ids[i] == ids[j] {
                    let zj = zrow(j);
                    let mut acc = 0.0;
                    for a in 0..q {
                        for b in 0..q {
                            acc += zi[a] * d[a * q + b] * zj[b];
                        }
                    }
                    v[(i, j)] += acc;
                }
            }
        }
        // REML profile (unchanged from the M2 oracle): ldv + ldk + df·ln s².
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldv = 0.0;
        for i in 0..n {
            ldv += vc.L()[(i, i)].ln();
        }
        let ldv = 2.0 * ldv;
        let mut x64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x64[(i, j)] = x[(i, j)] as f64;
            }
        }
        let mut vix = x64.clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y[i] as f64;
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for aa in 0..p {
            for bb in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x64[(i, aa)] * vix[(i, bb)];
                }
                xtvix[(aa, bb)] = s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += x64[(i, aa)] * viy[(i, 0)];
            }
            xtviy[aa] = s;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldk = 0.0;
        for aa in 0..p {
            ldk += kc.L()[(aa, aa)].ln();
        }
        let ldk = 2.0 * ldk;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for aa in 0..p {
            beta[(aa, 0)] = xtviy[aa];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += (y[i] as f64) * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for aa in 0..p {
            bxy += beta[(aa, 0)] * xtviy[aa];
        }
        let df = (n - p) as f64;
        let s2 = (ytviy - bxy) / df;
        ldv + ldk + df * s2.ln()
    }

    fn slope_groupings() -> LmmGroupings {
        // 8 primary clusters, one slope on x_full col 1; no extras.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(0),
                variance: 0.1,
                corr_with_intercept: 0.2,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        };
        LmmGroupings::from_cluster_spec(&cluster, 64, &[1])
    }

    fn multislope_groupings() -> LmmGroupings {
        // 8 primary clusters, two slopes on x_full cols 1,2; no extras.
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![
                engine_contract::SlopeTerm {
                    column: engine_contract::ColumnId(0),
                    variance: 0.10,
                    corr_with_intercept: 0.2,
                    corr_with: vec![],
                },
                engine_contract::SlopeTerm {
                    column: engine_contract::ColumnId(1),
                    variance: 0.08,
                    corr_with_intercept: 0.1,
                    corr_with: vec![0.15],
                },
            ],
            extra_groupings: vec![],
        };
        LmmGroupings::from_cluster_spec(&cluster, 96, &[1, 2])
    }

    #[test]
    fn slope_deviance_matches_brute_force() {
        let (x, y, ids) = slope_dataset();
        let mut suff = LmmSuffStats::with_groupings(2, slope_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &slope_groupings());
        // θ = vech(Λ), q=2: [λ₀₀, λ₁₀, λ₁₁].
        for th in [
            vec![1.0, 0.0, 1.0],
            vec![0.5, 0.2, 0.4],
            vec![2.0, -0.5, 0.7],
            vec![1e-3, 1e-3, 1e-3],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1], 2);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    #[test]
    fn multislope_deviance_matches_brute_force() {
        let (x, y, ids) = multislope_dataset();
        let mut suff = LmmSuffStats::with_groupings(3, multislope_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let mut fit = LmmFitScratch::with_groupings(3, &multislope_groupings());
        // θ = vech(Λ), q=3: [λ₀₀, λ₁₀, λ₂₀, λ₁₁, λ₂₁, λ₂₂].
        for th in [
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.6, 0.2, -0.1, 0.4, 0.15, 0.3],
            vec![1.5, -0.4, 0.3, 0.7, -0.2, 0.5],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1, 2], 3);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// End-to-end single-slope fit recovers the planted structure within BOBYQA
    /// bands and pins nothing on a well-identified design.
    #[test]
    fn slope_fit_converges_interior() {
        let (x, y, ids) = slope_dataset();
        let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        // Planted [intercept 0.5, slope 0.4]; small (n=64, 8 clusters) REML draw
        // recovers ≈[0.46, 0.20] — directionally correct, finite-sample attenuated.
        // Pin sign + a band tight enough to catch a sign flip, a collapse to 0, or a
        // blow-up (mere `is_finite` passed any of those).
        assert!(
            (0.2..0.8).contains(&ws.fit.betas[0]),
            "intercept {}",
            ws.fit.betas[0]
        );
        assert!(
            (0.05..0.6).contains(&ws.fit.betas[1]),
            "slope {}",
            ws.fit.betas[1]
        );
        assert_eq!(fit.pinned_components & !0b11, 0); // only 2 components exist
    }

    /// End-to-end two-slope fit: 3 components (intercept + 2 slopes), interior.
    #[test]
    fn multislope_fit_converges_interior() {
        let (x, y, ids) = multislope_dataset();
        let mut ws = LmmWorkspace::with_groupings(3, multislope_groupings());
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(fit.converged);
        // Planted [0.5, 0.4, 0.2]; recovered ≈[0.51, 0.64, 0.28]. Both slopes positive
        // with β̂₁ > β̂₂ (planted ordering preserved) — pin that, so a β₁/β₂ swap or a
        // scale collapse fails where the old `is_finite` pair passed.
        assert!(
            (0.2..0.9).contains(&ws.fit.betas[0]),
            "intercept {}",
            ws.fit.betas[0]
        );
        assert!(
            (0.2..1.1).contains(&ws.fit.betas[1]),
            "slope x1 {}",
            ws.fit.betas[1]
        );
        assert!(
            (0.0..0.7).contains(&ws.fit.betas[2]),
            "slope x2 {}",
            ws.fit.betas[2]
        );
        assert!(
            ws.fit.betas[1] > ws.fit.betas[2],
            "x1 slope must exceed x2 slope"
        );
        assert_eq!(fit.pinned_components & !0b111, 0); // only 3 components exist
    }

    /// Slope-variance collapse pins the SLOPE component (bit 1), not the
    /// intercept. x1 is a within-cluster antithetic ±1 pattern that carries a
    /// real fixed slope but ZERO cluster-varying slope, and the residual is a
    /// ±0.8 period-4 quadrature block (+,+,−,− against x1's +,−,+,−) so every
    /// cluster has Σ resid = 0 AND Σ x1·resid = 0 exactly — the REML
    /// slope-variance MLE is 0, so λ₁₁ pins (bit 1) while the planted u₀ keeps
    /// λ₀₀ interior. (The original lockstep ±0.8 pattern made resid ≡ 0.8·x1 —
    /// collinear with the slope covariate, so σ̂²→0 once large θ₀ absorbed the
    /// exactly-identified cluster means, the deviance ran unbounded to the θ₀
    /// box bound, and the λ₁₁ pin rode FP noise on the degenerate surface; the
    /// quadrature pattern keeps σ̂² positive and θ̂₀ genuinely interior.) Large
    /// balanced design (16 clusters × 16 rows) so finite-sample REML does not
    /// overfit a spurious slope RE the way a small noisy draw does.
    #[test]
    fn zero_slope_variance_pins_slope_component() {
        let (nc, per) = (16usize, 16usize);
        let n = nc * per;
        let mut st = 5u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut ids = vec![0u32; n];
        #[allow(clippy::needless_range_loop)]
        for c in 0..nc {
            for k in 0..per {
                let i = c * per + k;
                ids[i] = c as u32;
                // x1: identical antithetic pattern in every cluster (±1
                // alternating) — no between-cluster slope signal.
                let x1 = if k % 2 == 0 { 1.0 } else { -1.0 };
                // residual: ±0.8 period-4 quadrature against x1, so per cluster
                // Σ x1·resid = 0 AND Σ resid = 0 (no slope/intercept RE pull
                // from the noise; only the planted u₀ moves intercepts).
                let e = if (k / 2) % 2 == 0 { 0.8 } else { -0.8 };
                x[(i, 0)] = 1.0;
                x[(i, 1)] = x1 as f32;
                y[i] = (0.5 + 0.4 * x1 + u0[c] + e) as f32;
            }
        }
        let mut ws = LmmWorkspace::with_groupings(
            2,
            LmmGroupings::from_cluster_spec(
                &engine_contract::ClusterSpec {
                    sizing: engine_contract::ClusterSizing::FixedClusters {
                        n_clusters: nc as u32,
                    },
                    tau_squared: 0.25,
                    slopes: vec![engine_contract::SlopeTerm {
                        column: engine_contract::ColumnId(0),
                        variance: 0.1,
                        corr_with_intercept: 0.2,
                        corr_with: vec![],
                    }],
                    extra_groupings: vec![],
                },
                n,
                &[1],
            ),
        );
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert!(
            ws.theta[2] == 0.0,
            "slope λ₁₁ must pin to exactly 0, got {:e}",
            ws.theta[2]
        );
        assert!(fit.pinned_components & 0b10 != 0, "slope component bit set");
        assert!(
            ws.theta[0] > PIN_THETA,
            "intercept component must stay interior"
        );
        assert!(
            ws.theta[0] < THETA_HI,
            "intercept component must be off the box bound"
        );
    }

    // -----------------------------------------------------------------------
    // M3 composition: primary slope (1 + x1 | g) co-existing with an
    // intercept-only crossed (1 | item) / nested (1 | g:sub) extra. The
    // family-blocked deviance must match a brute-force V = I + Z_p D_p Z_p′ +
    // τ_e² Z_e Z_e′. Data on the f32 plane (the suff-stats input convention);
    // the oracle widens the identical bytes, so the 1e-8 match is exact.
    // -----------------------------------------------------------------------

    /// n=80, p=2 (intercept + x1), 8 primary clusters crossed with 5 items;
    /// y carries u₀ + u₁·x1 (primary) + v (item intercept).
    fn composed_dataset() -> (Mat<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
        let (n, nc, ni) = (80usize, 8usize, 5usize);
        let mut st = 41u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let v: Vec<f64> = (0..ni).map(|_| 0.4 * lcg(&mut st)).collect();
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let (mut pid, mut iid) = (vec![0u32; n], vec![0u32; n]);
        for i in 0..n {
            let (c, it) = (i % nc, i % ni);
            pid[i] = c as u32;
            iid[i] = it as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] = (0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + v[it] + 0.8 * lcg(&mut st)) as f32;
        }
        (x, y, pid, iid)
    }

    /// primary (1 + x1 | g), crossed (1 | item); slope on x_full col 1.
    fn composed_groupings() -> LmmGroupings {
        let cluster = engine_contract::ClusterSpec {
            sizing: engine_contract::ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![engine_contract::SlopeTerm {
                column: engine_contract::ColumnId(0),
                variance: 0.1,
                corr_with_intercept: 0.2,
                corr_with: vec![],
            }],
            extra_groupings: vec![engine_contract::GroupingSpec {
                relation: engine_contract::GroupingRelation::Crossed { n_clusters: 5 },
                tau_squared: 0.16,
                slopes: vec![],
            }],
        };
        LmmGroupings::from_cluster_spec(&cluster, 80, &[1])
    }

    /// REML deviance on the explicit n×n V for the composed model: the 2×2
    /// primary slope block (D_p = ΛΛ′ over [1, x1]) PLUS the extra-grouping
    /// intercept block (θ_e² when the extra ids match). The f32 data is widened
    /// to f64 so the oracle reads the same bytes the suff stats accumulated.
    /// `eid` is the extra grouping's level id per row (item, or nested child).
    /// θ = [primary vech λ₀₀, λ₁₀, λ₁₁ ; extra scalar θ_e].
    fn brute_force_composed_deviance(
        theta: &[f64],
        x: &Mat<f32>,
        y: &[f32],
        pid: &[u32],
        eid: &[u32],
    ) -> f64 {
        use faer::linalg::solvers::Solve;
        let (n, p) = (x.nrows(), x.ncols());
        let (a, b, c) = (theta[0], theta[1], theta[2]);
        // D_p = ΛΛ′, Λ = [[a,0],[b,c]] (column-major vech).
        let (d00, d01, d11) = (a * a, a * b, b * b + c * c);
        let te2 = theta[3] * theta[3];
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                if pid[i] == pid[j] {
                    let (zi1, zj1) = (x[(i, 1)] as f64, x[(j, 1)] as f64);
                    v[(i, j)] += d00 + d01 * (zi1 + zj1) + d11 * zi1 * zj1;
                }
                if eid[i] == eid[j] {
                    v[(i, j)] += te2;
                }
            }
        }
        // REML profile (identical to the other oracles): ldv + ldk + df·ln s².
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldv = 0.0;
        for i in 0..n {
            ldv += vc.L()[(i, i)].ln();
        }
        let ldv = 2.0 * ldv;
        let mut x64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x64[(i, j)] = x[(i, j)] as f64;
            }
        }
        let mut vix = x64.clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y[i] as f64;
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for aa in 0..p {
            for bb in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x64[(i, aa)] * vix[(i, bb)];
                }
                xtvix[(aa, bb)] = s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += x64[(i, aa)] * viy[(i, 0)];
            }
            xtviy[aa] = s;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldk = 0.0;
        for aa in 0..p {
            ldk += kc.L()[(aa, aa)].ln();
        }
        let ldk = 2.0 * ldk;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for aa in 0..p {
            beta[(aa, 0)] = xtviy[aa];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += (y[i] as f64) * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for aa in 0..p {
            bxy += beta[(aa, 0)] * xtviy[aa];
        }
        let df = (n - p) as f64;
        let s2 = (ytviy - bxy) / df;
        ldv + ldk + df * s2.ln()
    }

    /// Slope + crossed: the composed deviance matches the brute-force oracle to
    /// 1e-8 — the Task 6 composition gate. zx_slope carries the slope↔crossed
    /// coupling; the primary 2×2 block and the item intercept block are coupled
    /// through the shared family-blocked tail.
    #[test]
    fn composed_deviance_matches_brute_force() {
        let (x, y, pid, iid) = composed_dataset();
        let mut suff = LmmSuffStats::with_groupings(2, composed_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&iid)); // item ids as the single extra grouping
        let mut fit = LmmFitScratch::with_groupings(2, &composed_groupings());
        // θ = [λ₀₀, λ₁₀, λ₁₁, θ_c].
        for th in [
            vec![1.0, 0.0, 1.0, 0.5],
            vec![0.6, 0.2, 0.4, 0.3],
            vec![1.5, -0.4, 0.7, 0.8],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &iid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// Slope + NESTED: `(1 + x1 | g) + (1 | g:sub)` — the composed deviance with
    /// a nested child tail (vs the crossed tail above). Exercises the
    /// primary-slope↔child off-diagonal (read from `s`) and the shifted nested
    /// offset `q_p·n_primary + f·np + c`. The nested child ids are globalized
    /// (parent·np + within) — the workspace layout the contract helpers produce.
    #[test]
    fn composed_nested_deviance_matches_brute_force() {
        use engine_contract::{
            ClusterSizing, ClusterSpec, ColumnId, GroupingRelation, GroupingSpec, SlopeTerm,
        };
        // 8 primary clusters × 2 children each, fixed-size 8 ⇒ 64 rows / 4 blocks.
        let cluster = ClusterSpec {
            sizing: ClusterSizing::FixedSize { cluster_size: 8 },
            tau_squared: 0.25,
            slopes: vec![SlopeTerm {
                column: ColumnId(0),
                variance: 0.1,
                corr_with_intercept: 0.2,
                corr_with: vec![],
            }],
            extra_groupings: vec![GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                tau_squared: 0.09,
                slopes: vec![],
            }],
        };
        let n = 4 * cluster.atom(); // 64
        let mut st = 47u64;
        let u0: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..8).map(|_| 0.3 * lcg(&mut st)).collect();
        let u_c: Vec<f64> = (0..16).map(|_| 0.35 * lcg(&mut st)).collect(); // 8 parents × 2 children
        let mut x = Mat::<f32>::zeros(n, 2);
        let mut y = vec![0.0f32; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n]; // globalized child id (parent·np + within)
        for i in 0..n {
            let par = cluster.sizing.cluster_of_row(i);
            let child = cluster.extra_level_of_row(0, i); // already globalized par·np + within
            pid[i] = par as u32;
            cid[i] = child as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1 as f32;
            y[i] =
                (0.5 + 0.4 * x1 + u0[par] + u1[par] * x1 + u_c[child] + 0.8 * lcg(&mut st)) as f32;
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &[cid.clone()]);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        // The brute-force oracle is V-shape-agnostic: the nested child block adds
        // θ_n² when the (globalized) child ids match — same form as the crossed.
        for th in [
            vec![1.0, 0.0, 1.0, 0.5],
            vec![0.7, 0.25, 0.5, 0.4],
            vec![1.3, -0.3, 0.6, 0.2],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &cid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// Regression (composition truth-start length): a slopes + extra-groupings
    /// spec must yield `theta_truth.len() == n_theta` — the slope vech replaces
    /// only the primary part, with the extra √τ² scalars appended. Before the fix
    /// `for_cluster_spec`'s rho_begin fold indexed past a too-short theta_truth
    /// (q_p=2 + 1 extra ⇒ index 3 in len 3) and panicked at construction.
    #[test]
    fn composition_truth_start_keeps_extra_scalars() {
        use engine_contract::{
            ClusterSizing, ClusterSpec, ColumnId, GroupingRelation, GroupingSpec, SlopeTerm,
        };
        let cluster = ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 8 },
            tau_squared: 0.25,
            slopes: vec![SlopeTerm {
                column: ColumnId(0),
                variance: 0.1,
                corr_with_intercept: 0.2,
                corr_with: vec![],
            }],
            extra_groupings: vec![GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                tau_squared: 0.16,
                slopes: vec![],
            }],
        };
        // Constructs without panic (the bug panicked in the rho_begin fold).
        let ws = LmmWorkspace::for_cluster_spec(2, &cluster, 80, &[1]);
        assert_eq!(ws.theta_truth.len(), ws.suff.groupings.n_theta()); // 3 vech + 1 extra
    }

    /// Bounded-allocation twin (Step 5.11) — the standalone slope workspace
    /// allocates only faer `llt` internals on the warm `fit_lmm` loop, the same
    /// acceptance class as the q=1 / general twins.
    ///   cargo test -p engine-core lmm_fit_slope_warm_path_bounded_alloc -- --ignored --test-threads=1
    #[test]
    #[ignore]
    fn lmm_fit_slope_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND_SLOPE: u64 = 12000; // Measured 11400 (this machine) — ~114 blocks/fit of faer `llt` internals (one m×m tail llt per eval × ~54 evals on the blind 3-D q_p=2 surface; the family loop + primary Λ/Gram are zero-alloc scratch, and the cached diagonal_theta map removed the per-fit Vec). Higher total than q=1's 4600 only via the larger blind eval count, not a richer per-eval alloc — faer-version/machine specific. If faer's Cholesky internals change, update — do not relax.

        let (x, y, ids) = slope_dataset();
        let targets: Vec<u32> = vec![1];
        let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());

        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
        let _ = fit_lmm(&mut ws, &targets, None);

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[]);
            let _ = fit_lmm(&mut ws, &targets, None);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND_SLOPE,
            "slope fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND_SLOPE
        );
    }
}
