//! Data-generation spec types: `GenerationSpec`, `ColumnSpec`, `Correlations`, `ClusterSizing`, `UploadedFrame`.

use serde::{Deserialize, Serialize};

use crate::ids::ColumnId;

/// Predictor-generation plan: one `ColumnSpec` per predictor (order = the
/// engine's column layout), correlation structure over the continuous
/// columns, optional clustering, and the uploaded frame the `Resampled*` /
/// `FactorFromFrame` variants draw from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationSpec {
    pub columns: Vec<ColumnSpec>,
    pub correlations: Correlations,
    pub cluster: Option<ClusterSpec>,
    pub uploaded_frame: Option<UploadedFrame>,
    /// Predictor columns (continuous or factor) generated CONSTANT within each
    /// cluster — one draw per cluster, broadcast to all its rows. Indexes
    /// `columns`. Empty ⇒ every predictor varies per row (today's behaviour).
    /// A factor entry marks the whole factor (all its dummies) cluster-level.
    #[serde(default)]
    pub cluster_level_columns: Vec<ColumnId>,
}

/// How one predictor column is produced.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColumnSpec {
    /// Drawn from a parametric marginal.
    Synthetic {
        kind: SyntheticKind,
        /// Explicitly chosen by the user (incl. explicit normal) — scenario
        /// distribution swaps leave it alone. Unpinned `Normal` is the only
        /// swap-eligible state; serde default keeps old payloads valid.
        #[serde(default)]
        pinned: bool,
    },
    /// Continuous column resampled from `uploaded_frame` column `frame_column`.
    Resampled { frame_column: u32 },
    /// Uploaded binary (0/1) column. The engine resamples the column from the
    /// uploaded frame and applies a binary marginal transformation using
    /// `proportion` as the threshold. `proportion` is the empirical mean of
    /// the raw 0/1 values (computed at ingest, not re-derived at runtime).
    ResampledBinary { frame_column: u32, proportion: f64 },
    /// Categorical with `n_levels`, allocated to `proportions` (sum 1).
    FactorSynthetic {
        n_levels: u32,
        proportions: Vec<f64>,
        /// Per-factor override for proportion sampling, overriding the
        /// scenario-level `sampled_factor_proportions` default. `None` (default,
        /// absent in older payloads) inherits the scenario; `Some(true)` forces
        /// per-row categorical draw (Multinomial jitter); `Some(false)` forces the
        /// deterministic largest-remainder walk (exact counts, no RNG). Additive.
        #[serde(default)]
        sampled_proportions: Option<bool>,
    },
    /// Categorical resampled from the uploaded frame column; `proportions`
    /// are the level shares recorded at ingest.
    FactorFromFrame {
        frame_column: u32,
        n_levels: u32,
        proportions: Vec<f64>,
        /// See `FactorSynthetic::sampled_proportions`. Applies to the uploaded
        /// factor's allocation the same way (exact walk vs per-row draw).
        #[serde(default)]
        sampled_proportions: Option<bool>,
    },
}

/// Parametric marginal families for `ColumnSpec::Synthetic`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyntheticKind {
    Normal,
    Binary { p: f64 },
    RightSkewed,
    LeftSkewed,
    HighKurtosis,
    Uniform,
}

/// Correlation structure over the *continuous* columns: independent
/// (`Identity`) or a full matrix. `values` is the flat `k×k` matrix over
/// `continuous_columns` (k = its length), in that list's order; symmetric,
/// so row- vs column-major is moot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Correlations {
    Identity,
    Matrix {
        continuous_columns: Vec<ColumnId>,
        values: Vec<f64>,
    },
}

/// Host-validated numeric frame backing the `Resampled*` / `FactorFromFrame`
/// columns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UploadedFrame {
    /// Row-major values, `n_rows × n_cols` (element (row, col) at
    /// `data[row * n_cols + col]`); `len() == n_rows * n_cols`. Columns are
    /// addressed by the `frame_column` index in the `Resampled` /
    /// `ResampledBinary` / `FactorFromFrame` variants of `ColumnSpec`.
    pub data: Vec<f64>,
    /// Frame height — the rows available to resample from.
    pub n_rows: u32,
    /// Frame width; valid `frame_column` indices are `0..n_cols`.
    pub n_cols: u32,
    /// Strict (bootstrap) mode signal. When true, the adapter fills `upload_data` +
    /// `bootstrap_frame_map` and leaves `upload_normal` empty; the kernel's
    /// bootstrap arm (when present) row-samples whole frame rows from `upload_data`.
    /// Set by the spec-builder only for `UploadMode::Strict`.
    #[serde(default)]
    pub bootstrap: bool,
}

/// How cluster membership scales with total N.
///
/// `FixedClusters` (Regime A): the number of clusters is fixed; cluster size
/// grows with N. Row `i` joins cluster `i % n_clusters` (round-robin). The grid
/// atom is `n_clusters` — a nested prefix is balanced iff N is a multiple of it.
///
/// `FixedSize` (Regime B): each cluster has exactly `cluster_size` rows; the
/// number of clusters grows with N. Row `i` joins cluster `i / cluster_size`
/// (contiguous block). The grid atom is `cluster_size` — a nested prefix is a
/// set of complete clusters iff N is a multiple of it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterSizing {
    FixedClusters { n_clusters: u32 },
    FixedSize { cluster_size: u32 },
}

impl ClusterSizing {
    /// The grid atom: the smallest legal increment in total N. Snapping N to a
    /// multiple of the atom is what makes every nested prefix a valid balanced
    /// (Regime A) / complete-cluster (Regime B) dataset.
    pub fn atom(&self) -> usize {
        match self {
            ClusterSizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            ClusterSizing::FixedSize { cluster_size } => (*cluster_size).max(1) as usize,
        }
    }

    /// Number of distinct clusters present in the first `n` rows.
    /// FixedClusters: constant `n_clusters`. FixedSize: `n / cluster_size`.
    pub fn n_clusters_at(&self, n: usize) -> usize {
        match self {
            ClusterSizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            ClusterSizing::FixedSize { cluster_size } => n / (*cluster_size).max(1) as usize,
        }
    }

    /// Cluster index that row `i` belongs to.
    /// FixedClusters: `i % n_clusters` (round-robin). FixedSize: `i / cluster_size` (block).
    pub fn cluster_of_row(&self, i: usize) -> usize {
        match self {
            ClusterSizing::FixedClusters { n_clusters } => i % (*n_clusters).max(1) as usize,
            ClusterSizing::FixedSize { cluster_size } => i / (*cluster_size).max(1) as usize,
        }
    }
}

/// Cluster structure for mixed designs: sizing regime plus the
/// random-intercept variance τ².
///
/// The growth fields (`slopes`, `extra_groupings`) are contract-landed ahead
/// of capability: `validate()` rejects non-empty values until the milestones
/// that implement them, and the kernel never reads them. Defaults reproduce
/// today's payloads. The primary/extra asymmetry is deliberate, not
/// provisional: row layout must anchor on one factor (the `ClusterSizing`
/// regimes); nested factors derive their layout from the parent and crossed
/// factors carry fixed counts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClusterSpec {
    pub sizing: ClusterSizing,
    pub tau_squared: f64,
    /// Random slopes on the primary grouping factor.
    #[serde(default)]
    pub slopes: Vec<SlopeTerm>,
    /// Additional grouping factors (crossed / nested), positioned relative to
    /// the primary.
    #[serde(default)]
    pub extra_groupings: Vec<GroupingSpec>,
}

/// One random slope on a grouping factor. The per-group covariance with the
/// intercept is D = [[τ₀², ρτ₀τ₁], [ρτ₀τ₁, τ₁²]] (τ₀² = the owning spec's
/// tau_squared, τ₁² = variance, ρ = corr_with_intercept). For multi-slope
/// designs the full q_p×q_p RE covariance D is assembled in `re_covariance_is_psd`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlopeTerm {
    pub column: ColumnId,
    pub variance: f64,
    pub corr_with_intercept: f64,
    /// Correlations with EARLIER-declared slopes (those at a lower index in
    /// `slopes`), in declaration order. Length must equal this slope's own index
    /// (`slopes[0].corr_with` is empty). Fills the slope↔slope off-diagonals of
    /// the RE covariance `D`; empty ⇒ those off-diagonals are 0. The
    /// intercept↔slope correlation stays in `corr_with_intercept`. Additive.
    #[serde(default)]
    pub corr_with: Vec<f64>,
}

/// An additional grouping factor, positioned relative to the primary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroupingSpec {
    pub relation: GroupingRelation,
    pub tau_squared: f64,
    #[serde(default)]
    pub slopes: Vec<SlopeTerm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GroupingRelation {
    /// Fully crossed with the primary factor (e.g. (1|subject)+(1|item)).
    /// Count is fixed; total N scales by replicating the full factorial block.
    Crossed { n_clusters: u32 },
    /// Nested within the primary factor (e.g. (1|site/class)): n_per_parent
    /// child clusters inside each primary cluster. One nesting level below
    /// the primary; deeper chains are a future additive variant.
    NestedWithin { n_per_parent: u32 },
}

/// Maximum number of extra groupings a spec may declare. The solver lays each
/// fit's per-row level ids into a fixed-size stack buffer of width
/// `1 + MAX_EXTRA_GROUPINGS` (primary + extras) — `gid` in
/// `lmm::add_rows_multi` and the truth-start `tbuf` — so this is a HARD ceiling.
/// `validate()` (invariant 20) rejects any spec above it before a fit is built;
/// the constant lives here, not in `engine-core`, so the contract boundary can
/// enforce it. Mirrors those two buffer sizes — change together. Real designs
/// sit far below (≤ 1 nested + a handful of crossed factors).
pub const MAX_EXTRA_GROUPINGS: usize = 7;

/// Maximum primary RE width `q_p = 1 + #slopes`, i.e. at most 7 random slopes.
/// Bounds the stack-allocated θ truth-start buffer in the solver (the buffer is
/// sized `MAX_PRIMARY_Q*(MAX_PRIMARY_Q+1)/2 + MAX_EXTRA_GROUPINGS`).
/// `validate()` (invariant 21) rejects `1 + cluster.slopes.len() > MAX_PRIMARY_Q`
/// before a fit is built. Realistic random-slope models have 1–3 slopes;
/// 8 is a generous hard cap.
pub const MAX_PRIMARY_Q: usize = 8;

impl GroupingRelation {
    /// Levels this grouping contributes to one atom block: the fixed crossed
    /// count, or the per-parent child count.
    pub fn block_levels(&self) -> usize {
        match self {
            GroupingRelation::Crossed { n_clusters } => (*n_clusters).max(1) as usize,
            GroupingRelation::NestedWithin { n_per_parent } => (*n_per_parent).max(1) as usize,
        }
    }
}

impl ClusterSpec {
    /// Intercept-only spec — today's single-grouping shape. Keeps the many
    /// construction sites that predate the growth fields to two arguments.
    pub fn intercept_only(sizing: ClusterSizing, tau_squared: f64) -> Self {
        ClusterSpec {
            sizing,
            tau_squared,
            slopes: vec![],
            extra_groupings: vec![],
        }
    }

    /// Grid atom for the full grouping structure: the primary regime's atom
    /// times every extra grouping's block levels. Snapping N to a multiple of
    /// this is what makes every nested grid prefix a balanced design for
    /// EVERY factor simultaneously (each atom block is one full factorial —
    /// see `extra_level_of_row`). Coarser than the single-grouping atom by
    /// design: the documented cost of guaranteed balance and zero
    /// level-confounding.
    pub fn atom(&self) -> usize {
        self.extra_groupings
            .iter()
            .fold(self.sizing.atom(), |a, g| a * g.relation.block_levels())
    }

    /// Number of distinct levels extra grouping `g` has within the first `n`
    /// rows. Crossed: the fixed count. NestedWithin: parents at `n` ×
    /// n_per_parent — FixedSize uses ceil so ids on a partial trailing parent
    /// stay in range (production N is always an atom multiple; tests may not
    /// be).
    pub fn extra_n_levels_at(&self, g: usize, n: usize) -> usize {
        let rel = &self.extra_groupings[g].relation;
        match rel {
            GroupingRelation::Crossed { n_clusters } => (*n_clusters).max(1) as usize,
            GroupingRelation::NestedWithin { n_per_parent } => {
                let np = (*n_per_parent).max(1) as usize;
                let parents = match &self.sizing {
                    ClusterSizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
                    ClusterSizing::FixedSize { cluster_size } => {
                        n.div_ceil((*cluster_size).max(1) as usize)
                    }
                };
                parents * np
            }
        }
    }

    /// The primary random slopes (`q_p − 1` of them; empty for intercept-only).
    pub fn primary_slopes(&self) -> &[SlopeTerm] {
        &self.slopes
    }

    /// Number of diagonal variance components: one per primary RE dimension
    /// (intercept + each slope) plus one per extra grouping — the length of
    /// `boundary_rate_per_component` / `variance_components`. Slopes and extra
    /// groupings DO coexist (composition, Task 6).
    pub fn n_variance_components(&self) -> usize {
        1 + self.slopes.len() + self.extra_groupings.len()
    }

    /// The `q_p×q_p` RE *correlation* matrix `R` over `[intercept, slope_0, …]`,
    /// row-major (`q_p = 1 + slopes.len()`). Diagonal 1; `R[0][k+1]=R[k+1][0]=
    /// slopes[k].corr_with_intercept`; `R[i+1][k+1]=R[k+1][i+1]=slopes[k].corr_with[i]`
    /// for `i < k`. Multiply by `diag(τ)` on both sides for `D`.
    pub fn re_correlation_matrix(&self) -> (usize, Vec<f64>) {
        let q = 1 + self.slopes.len();
        let mut r = vec![0.0; q * q];
        for d in 0..q {
            r[d * q + d] = 1.0;
        }
        for (k, s) in self.slopes.iter().enumerate() {
            r[k + 1] = s.corr_with_intercept; // R[0][k+1]
            r[(k + 1) * q] = s.corr_with_intercept; // R[k+1][0]
            for (i, &cik) in s.corr_with.iter().enumerate() {
                r[(i + 1) * q + (k + 1)] = cik;
                r[(k + 1) * q + (i + 1)] = cik;
            }
        }
        (q, r)
    }

    /// True iff `D = diag(τ)·R·diag(τ)` is PSD (a tolerant Cholesky exists) —
    /// the data-gen draw needs `chol(D)`. `τ₀ = √tau_squared`,
    /// `τ_{k+1} = √slopes[k].variance`. Assumes `corr_with` lengths already
    /// validated (so `re_correlation_matrix` is well-formed).
    pub fn re_covariance_is_psd(&self) -> bool {
        let (q, r) = self.re_correlation_matrix();
        let mut tau = Vec::with_capacity(q);
        tau.push(self.tau_squared.max(0.0).sqrt());
        for s in &self.slopes {
            tau.push(s.variance.max(0.0).sqrt());
        }
        let mut d = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                d[i * q + j] = tau[i] * r[i * q + j] * tau[j];
            }
        }
        chol_exists(&d, q)
    }

    /// Level of extra grouping `g` that row `i` belongs to.
    ///
    /// ## The pinned within-block ordering (multi-factor row layout)
    ///
    /// One lexicographic stride chain, primary fastest, extras in declaration
    /// order (Regime A / `FixedClusters`):
    ///
    /// ```text
    /// primary(i)  = i % S                                   (unchanged)
    /// level_g(i)  = (i / stride_g) % block_levels_g
    /// stride_0    = S;  stride_{g+1} = stride_g · block_levels_g
    /// ```
    ///
    /// Nested levels are globalized as `parent·n_per + within` (child→parent
    /// = `id / n_per`). Regime B (`FixedSize`; crossed is rejected there):
    /// `parent = i / cs`, `within = (i % cs) % n_per` — round-robin inside the
    /// parent's contiguous block.
    ///
    /// Why this chain: each atom block enumerates every
    /// (primary, level_1, …, level_G) combination exactly once — so (i) every
    /// factor is balanced at atom multiples, (ii) no two factors' assignment
    /// rules can alias (the naive independent `i % S` / `i % I` chain
    /// degenerates catastrophically when counts share factors — S = I makes
    /// subject ≡ item; a `(i/2)%3` / `i%6` mix aliases a crossed factor to
    /// nested children), and (iii) prefix-nesting holds (levels are pure
    /// functions of the row index — truncation never reshuffles membership).
    /// Inside a partial block a later factor can sit constant — degenerate but
    /// not aliased; grid Ns are always atom multiples.
    ///
    /// Mirrors: `workspace.rs` builds `extra_grouping_ids` and `data_gen.rs`
    /// sizes the per-grouping draws from exactly these functions — change
    /// together.
    pub fn extra_level_of_row(&self, g: usize, i: usize) -> usize {
        let rel = &self.extra_groupings[g].relation;
        match &self.sizing {
            ClusterSizing::FixedClusters { n_clusters } => {
                let s = (*n_clusters).max(1) as usize;
                let mut stride = s;
                for h in &self.extra_groupings[..g] {
                    stride *= h.relation.block_levels();
                }
                let within = (i / stride) % rel.block_levels();
                match rel {
                    GroupingRelation::Crossed { .. } => within,
                    GroupingRelation::NestedWithin { n_per_parent } => {
                        (i % s) * (*n_per_parent).max(1) as usize + within
                    }
                }
            }
            ClusterSizing::FixedSize { cluster_size } => {
                // Crossed×FixedSize is rejected by validate(); only
                // NestedWithin reaches this arm.
                let cs = (*cluster_size).max(1) as usize;
                let np = rel.block_levels();
                (i / cs) * np + (i % cs) % np
            }
        }
    }
}

/// Lower-Cholesky existence test for a symmetric `q×q` row-major matrix,
/// tolerant of a merely PSD (not strictly PD) matrix: a pivot may sit at ~0 but
/// its column's off-diagonals must vanish; a negative pivot beyond −EPS rejects.
fn chol_exists(a: &[f64], q: usize) -> bool {
    const EPS: f64 = 1e-12;
    let mut l = vec![0.0f64; q * q];
    for j in 0..q {
        let mut diag = a[j * q + j];
        for k in 0..j {
            diag -= l[j * q + k] * l[j * q + k];
        }
        if diag < -EPS {
            return false;
        }
        let ljj = diag.max(0.0).sqrt();
        l[j * q + j] = ljj;
        for i in (j + 1)..q {
            let mut s = a[i * q + j];
            for k in 0..j {
                s -= l[i * q + k] * l[j * q + k];
            }
            if ljj <= EPS {
                if s.abs() > 1e-6 {
                    return false; // zero pivot but non-zero off-diagonal ⇒ indefinite
                }
                l[i * q + j] = 0.0;
            } else {
                l[i * q + j] = s / ljj;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_synthetic_sampled_roundtrips() {
        let spec = ColumnSpec::FactorSynthetic {
            n_levels: 3,
            proportions: vec![0.5, 0.3, 0.2],
            sampled_proportions: Some(false),
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: ColumnSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn synthetic_pinned_defaults_false_on_old_payload() {
        // Pre-pin payloads carry only `kind`; `#[serde(default)]` must fill
        // pinned = false (= swap-eligible when Normal).
        #[derive(serde::Serialize)]
        enum OldColumnSpec {
            Synthetic { kind: SyntheticKind },
        }
        let bytes = rmp_serde::to_vec_named(&OldColumnSpec::Synthetic {
            kind: SyntheticKind::Normal,
        })
        .unwrap();
        let back: ColumnSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert!(matches!(back, ColumnSpec::Synthetic { pinned: false, .. }));
    }

    #[test]
    fn factor_synthetic_sampled_defaults_to_none_when_absent() {
        // Older payloads predate `sampled_proportions`; `#[serde(default)]` must fill
        // None (inherit) so this stays an additive minor bump.
        #[derive(serde::Serialize)]
        enum OldColumnSpec {
            FactorSynthetic {
                n_levels: u32,
                proportions: Vec<f64>,
            },
        }
        let bytes = rmp_serde::to_vec_named(&OldColumnSpec::FactorSynthetic {
            n_levels: 2,
            proportions: vec![0.5, 0.5],
        })
        .unwrap();
        let back: ColumnSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert!(matches!(
            back,
            ColumnSpec::FactorSynthetic {
                sampled_proportions: None,
                ..
            }
        ));
    }

    #[test]
    fn generation_spec_msgpack_roundtrip_with_factor_and_cluster() {
        let spec = GenerationSpec {
            columns: vec![
                ColumnSpec::Synthetic {
                    kind: SyntheticKind::Normal,
                    pinned: false,
                },
                ColumnSpec::FactorSynthetic {
                    n_levels: 3,
                    proportions: vec![0.5, 0.3, 0.2],
                    sampled_proportions: None,
                },
                ColumnSpec::Resampled { frame_column: 0 },
                ColumnSpec::FactorFromFrame {
                    frame_column: 1,
                    n_levels: 2,
                    proportions: vec![0.4, 0.6],
                    sampled_proportions: None,
                },
            ],
            correlations: Correlations::Matrix {
                continuous_columns: vec![ColumnId(0), ColumnId(2)],
                values: vec![1.0, 0.3, 0.3, 1.0],
            },
            cluster: Some(ClusterSpec {
                sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
                tau_squared: 0.25,
                slopes: vec![],
                extra_groupings: vec![],
            }),
            uploaded_frame: Some(UploadedFrame {
                data: vec![1.0, 2.0, 3.0, 4.0],
                n_rows: 2,
                n_cols: 2,
                bootstrap: false,
            }),
            cluster_level_columns: vec![ColumnId(1)],
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: GenerationSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn fixed_clusters_atom_and_layout() {
        let s = ClusterSizing::FixedClusters { n_clusters: 4 };
        assert_eq!(s.atom(), 4);
        assert_eq!(s.n_clusters_at(20), 4); // constant regardless of N
        assert_eq!(s.n_clusters_at(8), 4);
        assert_eq!(s.cluster_of_row(0), 0);
        assert_eq!(s.cluster_of_row(5), 1); // 5 % 4
        assert_eq!(s.cluster_of_row(7), 3);
    }

    #[test]
    fn fixed_size_atom_and_layout() {
        let s = ClusterSizing::FixedSize { cluster_size: 5 };
        assert_eq!(s.atom(), 5);
        assert_eq!(s.n_clusters_at(20), 4); // 20 / 5
        assert_eq!(s.n_clusters_at(8), 1); // 8 / 5 floored
        assert_eq!(s.cluster_of_row(0), 0);
        assert_eq!(s.cluster_of_row(4), 0);
        assert_eq!(s.cluster_of_row(5), 1); // block boundary
        assert_eq!(s.cluster_of_row(12), 2);
    }

    /// growth: a payload WITHOUT the new keys (today's wire shape from every
    /// shipped host) must deserialize with empty growth fields — the serde-default
    /// byte-compat proof the M1 exit gate requires.
    #[test]
    fn cluster_spec_old_payload_deserializes_with_empty_growth_fields() {
        // Local mirror of the pre-growth struct: serializing it emits exactly the
        // keys an old host sends — no hand-built msgpack needed.
        #[derive(Serialize)]
        struct V0ClusterSpec {
            sizing: ClusterSizing,
            tau_squared: f64,
        }
        let old = V0ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
        };
        let bytes = rmp_serde::to_vec_named(&old).unwrap();
        let back: ClusterSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.sizing, ClusterSizing::FixedClusters { n_clusters: 20 });
        assert_eq!(back.tau_squared, 0.25);
        assert!(back.slopes.is_empty());
        assert!(back.extra_groupings.is_empty());
    }

    /// Populated growth fields roundtrip msgpack exactly (the L2 forward shape
    /// M2/M3 will send).
    #[test]
    fn cluster_spec_growth_fields_msgpack_roundtrip() {
        let spec = ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![SlopeTerm {
                column: ColumnId(1),
                variance: 0.4,
                corr_with_intercept: -0.3,
                corr_with: vec![],
            }],
            extra_groupings: vec![
                GroupingSpec {
                    relation: GroupingRelation::Crossed { n_clusters: 12 },
                    tau_squared: 0.1,
                    slopes: vec![],
                },
                GroupingSpec {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 3 },
                    tau_squared: 0.05,
                    slopes: vec![SlopeTerm {
                        column: ColumnId(2),
                        variance: 0.2,
                        corr_with_intercept: 0.0,
                        corr_with: vec![],
                    }],
                },
            ],
        };
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: ClusterSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec, back);
    }

    fn crossed(n: u32, tau: f64) -> GroupingSpec {
        GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: n },
            tau_squared: tau,
            slopes: vec![],
        }
    }

    fn nested(n_per: u32, tau: f64) -> GroupingSpec {
        GroupingSpec {
            relation: GroupingRelation::NestedWithin {
                n_per_parent: n_per,
            },
            tau_squared: tau,
            slopes: vec![],
        }
    }

    #[test]
    fn cluster_spec_atom_multiplies_extra_groupings() {
        let mut spec =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 20 }, 0.25);
        assert_eq!(spec.atom(), 20);
        spec.extra_groupings.push(crossed(12, 0.1));
        assert_eq!(spec.atom(), 240);
        spec.extra_groupings.push(nested(3, 0.05));
        assert_eq!(spec.atom(), 720);

        let mut b = ClusterSpec::intercept_only(ClusterSizing::FixedSize { cluster_size: 6 }, 0.25);
        b.extra_groupings.push(nested(3, 0.05));
        assert_eq!(b.atom(), 18); // primary atom · n_per_parent
    }

    /// The catastrophic naive case: S == I. Independent `i % S` / `i % I`
    /// assignment would alias subject ≡ item; the lexicographic stride must
    /// cover every (primary, crossed) pair exactly once per atom block.
    #[test]
    fn crossed_layout_full_factorial_when_s_equals_i() {
        let mut spec =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 4 }, 0.25);
        spec.extra_groupings.push(crossed(4, 0.1));
        let atom = spec.atom(); // 16
        let mut seen = std::collections::HashSet::new();
        for i in 0..atom {
            let pair = (spec.sizing.cluster_of_row(i), spec.extra_level_of_row(0, i));
            assert!(
                seen.insert(pair),
                "duplicate pair {pair:?} within one block"
            );
        }
        assert_eq!(seen.len(), atom);
    }

    /// Primary assignment is byte-identical with and without extras — adding
    /// a grouping never reshuffles existing cluster membership.
    #[test]
    fn primary_assignment_unchanged_by_extras() {
        let sizing = ClusterSizing::FixedClusters { n_clusters: 5 };
        let mut spec = ClusterSpec::intercept_only(sizing.clone(), 0.25);
        spec.extra_groupings.push(crossed(3, 0.1));
        spec.extra_groupings.push(nested(2, 0.05));
        for i in 0..120 {
            assert_eq!(spec.sizing.cluster_of_row(i), sizing.cluster_of_row(i));
        }
    }

    /// Regime A nested: child = parent·n_per + within; every child exactly
    /// once per atom block; balanced at atom multiples; child→parent = id/n_per.
    #[test]
    fn nested_regime_a_layout_and_balance() {
        let mut spec =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 2 }, 0.25);
        spec.extra_groupings.push(nested(3, 0.05));
        assert_eq!(spec.extra_n_levels_at(0, 60), 6);
        let atom = spec.atom(); // 6
        let n = 5 * atom;
        let mut counts = vec![0usize; 6];
        for i in 0..n {
            let c = spec.extra_level_of_row(0, i);
            assert_eq!(
                c / 3,
                spec.sizing.cluster_of_row(i),
                "child {c} not in its parent"
            );
            counts[c] += 1;
        }
        assert!(
            counts.iter().all(|&k| k == n / 6),
            "unbalanced at atom multiple: {counts:?}"
        );
    }

    /// Regime B nested: parent = i/cs, within round-robins inside the parent's
    /// contiguous block; balanced inside every complete parent.
    #[test]
    fn nested_regime_b_layout_and_balance() {
        let mut spec =
            ClusterSpec::intercept_only(ClusterSizing::FixedSize { cluster_size: 6 }, 0.25);
        spec.extra_groupings.push(nested(3, 0.05));
        assert_eq!(spec.extra_n_levels_at(0, 24), 12); // 4 parents × 3
        let mut counts = vec![0usize; 12];
        for i in 0..24 {
            let c = spec.extra_level_of_row(0, i);
            assert_eq!(c / 3, i / 6, "child {c} outside parent {}", i / 6);
            counts[c] += 1;
        }
        assert!(counts.iter().all(|&k| k == 2), "{counts:?}");
    }

    /// Crossed + nested together: the joint (primary, crossed, child-within)
    /// assignment is a full factorial per atom block — the regression case for
    /// the period-aliasing bug a per-factor `i % c` chain produces when a
    /// crossed count equals n_per_parent.
    #[test]
    fn crossed_plus_nested_no_confounding() {
        let mut spec =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 2 }, 0.25);
        spec.extra_groupings.push(crossed(3, 0.1));
        spec.extra_groupings.push(nested(3, 0.05));
        let atom = spec.atom(); // 18
        let mut seen = std::collections::HashSet::new();
        for i in 0..atom {
            let triple = (
                spec.sizing.cluster_of_row(i),
                spec.extra_level_of_row(0, i),
                spec.extra_level_of_row(1, i) % 3, // within-parent index
            );
            assert!(seen.insert(triple), "duplicate {triple:?} in one block");
        }
        assert_eq!(seen.len(), atom);
    }
}
