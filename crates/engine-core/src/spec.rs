//! Engine spec: error types, outcome kind / estimator axes, progress sink, POD types.
//!
//! These types form the wire format consumed by `engine-py` (msgpack) and
//! the eventual `engine-r`. No PyO3 / numpy types appear here — engine-core
//! is host-agnostic.

pub use engine_contract::{
    CorrectionMethod, Distribution, EstimatorSpec, LinkKind, OutcomeKind, ResidualDist, WaldSe,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors `run_batch` / `run_batch_st` can return. `Cancelled` is the
/// cooperative-stop path (host flipped the sink/token), not a fault.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid spec: {0}")]
    InvalidSpec(String),
    #[error("correlation matrix is not positive semi-definite")]
    CorrelationNotPSD,
    #[error("rank-deficient design at N={0}")]
    RankDeficient(u32),
    #[error("cancelled by host")]
    Cancelled,
}

// ---------------------------------------------------------------------------
// ProgressSink — host-language progress/cancel adapter trait
// ---------------------------------------------------------------------------

/// Implemented by host-language adapters (engine-py wraps a Python callable,
/// engine-r will wrap an R function). Return `false` from `report` to request
/// cancellation.
pub trait ProgressSink: Send + Sync {
    /// Progress notification, throttled to ~50 calls per run. Return `false` to
    /// request cancellation.
    fn report(&self, current: u64, total: u64) -> bool;

    /// Cheap cancellation poll, called once per sim — decoupled from `report`
    /// (which fires only ~50×/run) so cancel latency stays ~one sim even when
    /// `report` checkpoints are far apart. Coupling cancellation to the report
    /// cadence made latency = `progress_step × per-sim time`: unbounded for slow
    /// fits (mixed/GLMM), which are exactly the runs users cancel. Must be O(1)
    /// (an atomic load) — it runs in the per-sim hot loop. Default: never cancels.
    fn is_cancelled(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// CritValues — wire-format only. Runtime table lives in `critvals.rs`.
// ---------------------------------------------------------------------------

/// α levels carried by the spec. Wire-format only — the precomputed runtime
/// thresholds live in `CritValueTable`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CritValues {
    /// Uncorrected two-sided α; the engine computes `t_crit² = (t_ppf(1 - α/2, df))²`.
    pub alpha: f64,
    /// Posthoc α; defaults to `alpha` when None.
    pub posthoc_alpha: Option<f64>,
}

/// Precomputed population-statistic coefficients for the heteroskedasticity
/// scaler. Populated by `run_batch` from the full spec. `lp_pop_mean`/`lp_pop_std`
/// are the linear-predictor moments (used when `driver == None`); `col_mean`/
/// `col_std` are per-column moments in full-design layout
/// (`[intercept, continuous.., factor_dummies..]`) used when `driver == Some(idx)`.
/// Analytic (population) moments only — row-stable by construction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HeteroskedasticityCoeffs {
    /// Population mean of `lp = Σ βⱼ E[Xⱼ]`.
    pub lp_pop_mean: f64,
    /// Population standard deviation of `lp = sqrt(β' Σ β)`.
    pub lp_pop_std: f64,
    /// Per-column population means, full-design layout. Empty when uncomputed.
    pub col_mean: Vec<f64>,
    /// Per-column population standard deviations, full-design layout.
    pub col_std: Vec<f64>,
}

// ---------------------------------------------------------------------------
// PosthocSpec
// ---------------------------------------------------------------------------

/// Posthoc test specification for one factor family (e.g. Tukey HSD on factor
/// levels). `SimulationSpec.posthoc` is a Vec — one block per factor. When
/// the Vec is empty the engine skips posthoc passes entirely. The OLS hot path
/// only consumes `factor_index` and `target_indices`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PosthocSpec {
    /// Index into `factor_n_levels` identifying the factor on which posthoc
    /// pairwise comparisons run.
    pub factor_index: u32,
    /// Target indices into the global target list. Length = number of pairwise
    /// comparisons for the factor.
    pub target_indices: Vec<u32>,
}

// ---------------------------------------------------------------------------
// ClusterSpec and LmeScenarioPerturbations are byte-identical to their
// `engine_contract` twins, so the kernel re-exports them directly rather than
// maintaining a hand-translated copy (matching the `CorrectionMethod` /
// `Distribution` / `EstimatorSpec` pattern on line 7).
// ---------------------------------------------------------------------------

pub use engine_contract::{ClusterSizing, ClusterSpec, LmeScenarioPerturbations};

// ---------------------------------------------------------------------------
// ScenarioPerturbations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioPerturbations {
    /// Identifier used in result aggregation and logs ("optimistic",
    /// "realistic", "doomer", or a user-supplied custom name).
    pub name: String,

    /// β-jitter SD (v1's `heterogeneity` knob) — scenario-only, no model
    /// baseline. 0.0 = off.
    pub heterogeneity: f64,

    /// Variance ratio λ for residual heteroskedasticity — the only λ source
    /// (the model contributes the driver, never the magnitude). 1.0 =
    /// homoskedastic (no contribution).
    pub heteroskedasticity_ratio: f64,

    /// Symmetric Gaussian noise SD applied to `spec.correlation` per sim
    /// (off-diagonal, clipped to ±0.8, PSD-repaired). 0.0 → no perturbation.
    pub correlation_noise_sd: f64,

    /// Per-sim per-variable probability of swapping an unpinned
    /// continuous-normal var_type with one drawn uniformly from
    /// `new_distributions`.
    pub distribution_change_prob: f64,
    /// Distribution variants the scenario may swap in.
    pub new_distributions: Vec<Distribution>,

    /// Per-sim probability of replacing an unpinned default-normal residual
    /// distribution with one drawn from `residual_dists`.
    pub residual_change_prob: f64,
    pub residual_dists: Vec<ResidualDist>,
    /// df for any t-kernel residual this scenario realizes — both pool swaps
    /// and a pinned `HighKurtosis` spec residual read it (there is no
    /// model-level df).
    pub residual_df: f64,

    /// Factor-proportion sampling. `false` (default, optimistic) → the factor
    /// block assigns levels by the deterministic largest-remainder walk (exact
    /// requested proportions, no RNG consumed); `true` → per-row categorical
    /// draw (simple randomization; Multinomial count jitter). Read directly in
    /// the row loop; deliberately NOT part of `is_optimistic()` — see that
    /// method's docs. Additive serde-stable field: absent in older payloads,
    /// defaulting to `false` (exact), preserving v1 behaviour.
    #[serde(default)]
    pub sampled_factor_proportions: bool,

    /// Scenario assumption about the fitter's starting values, not a speed knob:
    /// `true` seeds the mixed-model optimizer at the DGP-truth θ (asserting
    /// well-behaved estimation); `false` starts blind, as a real analyst would.
    /// Read at the `fit_on` call sites (batch + introspect) to decide
    /// warm vs cold start. Deliberately NOT part of `is_optimistic()` — it
    /// changes only the optimizer's start, not the per-sim data, so it is
    /// orthogonal to the fast-path predicate. Additive serde-stable field: absent
    /// in older payloads, defaulting to `false` (cold start), preserving prior output.
    #[serde(default)]
    pub truth_start: bool,

    /// LME-only knobs. Rejected by the engine if set without `estimator == Mle`.
    /// Unused for OLS / Logit designs.
    pub lme: Option<LmeScenarioPerturbations>,
}

impl Default for ScenarioPerturbations {
    fn default() -> Self {
        // `heteroskedasticity_ratio` defaults to 1.0 (homoskedastic / off), NOT
        // the f64 zero a derived `Default` would give — a ratio of 0 is
        // nonsensical and would make `is_optimistic()` (which tests `== 1.0`)
        // reject every default-constructed baseline scenario.
        Self {
            name: String::new(),
            heterogeneity: 0.0,
            heteroskedasticity_ratio: 1.0,
            correlation_noise_sd: 0.0,
            distribution_change_prob: 0.0,
            new_distributions: Vec::new(),
            residual_change_prob: 0.0,
            residual_dists: Vec::new(),
            residual_df: 0.0,
            sampled_factor_proportions: false,
            truth_start: false,
            lme: None,
        }
    }
}

impl ScenarioPerturbations {
    /// Zero-perturbation baseline. Equivalent to v1's "optimistic" preset and
    /// to `find_power(scenarios=False)`.
    pub fn optimistic() -> Self {
        Self {
            name: "optimistic".into(),
            ..Default::default()
        }
    }

    /// Returns true if every perturbation knob is at its no-op value: all
    /// f64 noise knobs `== 0.0` and `lme` is None.
    ///
    /// The pool-of-candidates Vec fields (`new_distributions`, `residual_dists`)
    /// are NOT checked: they act as menus sampled with probability
    /// `distribution_change_prob` / `residual_change_prob`. When those
    /// probabilities are 0.0 the menus are never consulted, so a non-empty
    /// menu with zero probability is still an optimistic (no-perturbation) run.
    /// The Python frontend always populates the menus unconditionally; treating
    /// non-empty menus as "not optimistic" would reject every default Logit call.
    ///
    /// `sampled_factor_proportions` is also NOT consulted: the factor block
    /// reads it directly in the row loop, so the fast path (skip per-block
    /// scenario machinery) and the allocation mode are orthogonal. Folding it in
    /// would needlessly knock custom sampled-proportion scenarios off the fast path.
    ///
    /// Used as a fast-path predicate; the engine skips per-block scenario
    /// machinery entirely when this returns true.
    pub fn is_optimistic(&self) -> bool {
        self.heterogeneity == 0.0
            && self.heteroskedasticity_ratio == 1.0
            && self.correlation_noise_sd == 0.0
            && self.distribution_change_prob == 0.0
            && self.residual_change_prob == 0.0
            && self.lme.is_none()
    }
}

// ---------------------------------------------------------------------------
// SimulationSpec — full POD layout consumed by `run_batch`.
// ---------------------------------------------------------------------------

/// Serde default for `SimulationSpec.nagq` — Laplace.
fn default_nagq_spec() -> u8 {
    1
}

/// Full POD layout consumed by `run_batch` — the kernel-facing spec every
/// host contract lowers to. Carries no seed; hosts pass `base_seed` alongside.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationSpec {
    pub n_non_factor: u32,
    pub n_factor_dummies: u32,

    /// Flat `n_non_factor * n_non_factor`, column-major. Must be PSD on entry.
    pub correlation: Vec<f64>,
    pub var_types: Vec<Distribution>,
    /// Per-entry pin aligned with `var_types`: `true` = the user explicitly
    /// chose this distribution (incl. explicit normal), so scenario swaps
    /// leave it alone. Empty/short reads as unpinned (old payloads). Lowered
    /// by the contract adapter from `ColumnSpec::Synthetic.pinned`.
    #[serde(default)]
    pub var_pinned: Vec<bool>,
    pub var_params: Vec<f64>,
    pub upload_normal: Vec<f64>,
    pub upload_normal_shape: (u32, u32),
    pub upload_data: Vec<f64>,
    pub upload_data_shape: (u32, u32),

    /// Strict-mode (bootstrap) per-column frame map. Empty in NORTA/`none`/`partial`.
    /// Length `n_non_factor + n_factors`: indices `0..n_non_factor` map each non-factor
    /// column to its frame column (`Some(frame_col)` if uploaded, `None` if synthetic);
    /// indices `n_non_factor..` map each factor to its frame column (`Some` if
    /// `FactorFromFrame`, `None` if `FactorSynthetic`). Non-empty ⇒ kernel uses the
    /// bootstrap row-sampling arm. `frame_col` indexes `upload_data`'s columns.
    #[serde(default)]
    pub bootstrap_frame_map: Vec<Option<u32>>,

    pub between_var_indices: Vec<u32>,

    pub factor_n_levels: Vec<i32>,
    /// Flat, sized Σ factor_n_levels (concatenated per-factor proportions).
    pub factor_proportions: Vec<f64>,
    /// Per-factor proportion-sampling override, aligned with `factor_n_levels`
    /// (one entry per factor, NOT per dummy). `None` ⇒ inherit
    /// `scenario.sampled_factor_proportions`; `Some(true/false)` ⇒ force
    /// sampled / exact for that factor. Empty (default) ⇒ all inherit, preserving
    /// pre-feature output exactly. Lowered by the contract adapter; never seeded
    /// by a host directly. A short/absent entry is read as `None` (inherit).
    #[serde(default)]
    pub factor_sampled: Vec<Option<bool>>,

    pub effect_sizes: Vec<f64>,
    pub target_indices: Vec<u32>,
    /// Kernel column positions retained by the *fitted* (test) design, ascending.
    /// Empty ⇒ no reduction (fit == full generation design; current behaviour).
    /// Non-empty ⇒ fit only these columns; the complement is generated into `y`
    /// but omitted from the fit (omitted-variable / misspecification). Always
    /// includes column 0 (intercept). Derived from `design_test` by the contract
    /// adapter; hosts never set it directly.
    #[serde(default)]
    pub fit_columns: Vec<u32>,
    /// Pairwise contrasts `(positive_kernel_col, negative_kernel_col)` to test
    /// as `β_p − β_n`. The contract adapter translates `TestTarget::Contrast`
    /// pairs to these kernel column positions. Results are appended after the
    /// per-marginal target results in `BatchResult.uncorrected/corrected`, in
    /// the order they appear here.
    #[serde(default)]
    pub contrast_pairs: Vec<(u32, u32)>,
    /// One inner vec per appended interaction column, listing the **kernel**
    /// column indices to multiply (raw elementwise product). Appended after the
    /// factor dummies: column `1 + n_non_factor + n_factor_dummies + j` is the
    /// product of `interactions[j]`. Built by the contract adapter. Empty for
    /// designs without interactions.
    #[serde(default)]
    pub interactions: Vec<Vec<u32>>,
    pub correction_method: CorrectionMethod,
    pub crit_values: CritValues,
    /// Heteroskedasticity driver as a *full-design* (`x_full`) column index
    /// (intercept = 0, first continuous = 1), or `None` for the linear
    /// predictor Xβ — the contract adapter performs the ColumnId → x_full
    /// mapping. The variance ratio λ comes from `scenario` only; the
    /// generator applies `Var(εᵢ) = σ²·exp(γzᵢ)/exp(γ²/2)`, `γ = ln(λ)/4`,
    /// with z standardized via `HeteroskedasticityCoeffs`.
    #[serde(default)]
    pub heteroskedasticity_driver: Option<u32>,
    /// Full-design (`x_full`) column indices of the primary random slopes'
    /// covariates (intercept = 0, first continuous = 1), in `cluster.slopes`
    /// declaration order, resolved by the contract adapter. Empty for
    /// intercept-only / non-cluster specs. Read by data-gen (the `Σ_k u_{k+1}·x`
    /// terms) and the lmm solver (the slope `s`-row indices).
    #[serde(default)]
    pub cluster_slope_design_cols: Vec<u32>,
    /// Per-extra-grouping `x_full` column indices of that grouping's random-slope
    /// covariates (declaration order; intercept implicit at 0), resolved by the
    /// contract adapter — the crossed/nested analogue of
    /// `cluster_slope_design_cols`. Outer index = extra grouping (declaration
    /// order); inner = its slopes. Empty inner vecs (or empty outer) for
    /// intercept-only extras / non-cluster specs. Read by data-gen (the
    /// covariate-weighted extra-RE contribution) and the lmm suff-stats.
    #[serde(default)]
    pub extra_slope_cols: Vec<Vec<u32>>,
    pub residual_dist: ResidualDist,
    /// `true` = the user explicitly chose `residual_dist` (incl. explicit
    /// normal), so scenario residual swaps leave it alone. The df for a
    /// t-kernel residual always comes from `scenario.residual_df`.
    #[serde(default)]
    pub residual_pinned: bool,
    /// Consumed by data generation (Gaussian vs Bernoulli vs Poisson draw).
    pub outcome_kind: OutcomeKind,
    /// Non-canonical link override. `None` = canonical (Binary→logit,
    /// Count→log); `Some(Probit)` picks the probit link for `Binary`. Read by
    /// both data-gen (latent-normal vs logistic draw) and the fit dispatch
    /// (`glmm::Family` selection). Threaded from `outcome.link`.
    #[serde(default)]
    pub link: Option<LinkKind>,
    /// Consumed by the solver dispatch.
    pub estimator: EstimatorSpec,
    /// Fixed-effect Wald-SE mode for the clustered-binary/count GLMM (no-op
    /// elsewhere). Threaded from the contract field of the same name. (design §7.)
    #[serde(default)]
    pub wald_se: WaldSe,
    /// AGQ node count for the GLMM likelihood (1 = Laplace). Threaded from the
    /// contract `nagq`; the eligibility backstop already ran in `validate()`.
    #[serde(default = "default_nagq_spec")]
    pub nagq: u8,
    /// Baseline intercept used by GLM data-gen and surfaced to result
    /// formatters. The kernel does NOT consume this in the linear-predictor
    /// computation — that comes from `effect_sizes[0]` instead. It exists as
    /// metadata; OLS specs typically carry 0.0, Logit specs carry `logit(p)`.
    pub intercept: f64,
    #[serde(default)]
    pub posthoc: Vec<PosthocSpec>,
    /// Post-batch convergence threshold; evaluated by the frontend.
    pub max_failed_fraction: f64,

    /// Cluster spec for clustered designs. Present whenever cluster random
    /// intercepts are generated; independent of estimator.
    pub cluster: Option<ClusterSpec>,

    /// Scenario perturbation block. One `SimulationSpec` carries exactly one
    /// scenario; the orchestrator loops `run_batch` once per scenario.
    pub scenario: ScenarioPerturbations,

    /// Lazy lookup table for DIST_HIGH_KURTOSIS marginal. Populated by run_batch;
    /// callers from outside engine-core leave it at default and never touch it.
    /// Not serialised — derived value rebuilt per run.
    #[serde(skip)]
    #[serde(default)]
    pub t3_table: Option<std::sync::Arc<crate::marginals::T3PpfTable>>,

    /// Precomputed Option H2 heteroskedasticity coefficients. Populated by run_batch.
    #[serde(skip)]
    #[serde(default)]
    pub het_coeffs: HeteroskedasticityCoeffs,

    /// When `true`, `BatchResult.overall` carries an extra per-`(sim, n_idx)`
    /// boolean: OLS F-test for `estimator == Ols`, Logit LRT for `estimator == Glm`,
    /// always 0 for `estimator == Mle`. When `false`, `BatchResult.overall`
    /// is an empty `Vec` (no allocation). v1-parity overall significance —
    /// reaches the engine via the contract adapter's `TestTarget::Joint` path
    /// once that adapter is wired.
    #[serde(default)]
    pub report_overall: bool,

    /// Minimum observations every level of a factor must have (within the
    /// current (sim, N) row prefix) for the factor to enter the model. When any
    /// level falls below this, the whole factor — its dummy columns and every
    /// interaction built on them — is dropped from that round's fit and its
    /// targets count as non-significant. 0 disables exclusion (legacy behaviour:
    /// empty levels surface as converged=false). Serde-defaults to 0, so older
    /// payloads and bare test specs keep legacy behaviour; the only enabling
    /// source is configs `limits.factor_min_level_count`, which the
    /// orchestrator's contract lowering writes over whatever the host sent.
    #[serde(default)]
    pub factor_min_level_count: u32,
}

// ---------------------------------------------------------------------------
// SimulationSpec methods
// ---------------------------------------------------------------------------

impl SimulationSpec {
    /// Precompute population mean and std of the linear predictor under the
    /// spec's marginal assumptions. Used by the heteroskedasticity scaler
    /// (Option H2: residual variance scales with standardised linear predictor magnitude).
    ///
    /// Assumes: continuous non-factor columns are unit-variance zero-mean (true
    /// for DIST_NORMAL / skewed / t3 / uniform / DIST_HIGH_KURTOSIS by data_gen
    /// construction). DIST_BINARY columns use E = param, Var = param(1-param).
    /// Factor dummies use proportion[d+1] as mean.
    ///
    /// Caveat: binary columns in a correlated block have correlation matrix
    /// entries in standardised space, not in raw-marginal space; the cross-
    /// covariance approximation is exact only for the all-Gaussian case.
    pub fn compute_het_coeffs(&self) -> HeteroskedasticityCoeffs {
        let n_nf = self.n_non_factor as usize;
        let n_fd = self.n_factor_dummies as usize;
        let p = 1 + n_nf + n_fd;

        // Per-column means and variances of the *transformed* X.
        // Layout: index 0 = intercept (mean 1, var 0), 1..=n_nf = continuous,
        // then factor dummies.
        let mut mu = vec![0.0_f64; p];
        let mut sd = vec![0.0_f64; p];
        mu[0] = 1.0; // intercept
        sd[0] = 0.0;

        for j in 0..n_nf {
            let vt = self
                .var_types
                .get(j)
                .copied()
                .unwrap_or(Distribution::Normal);
            if matches!(vt, Distribution::Binary) {
                let pp = self.var_params.get(j).copied().unwrap_or(0.5);
                mu[1 + j] = pp;
                sd[1 + j] = (pp * (1.0 - pp)).max(0.0).sqrt();
            } else {
                mu[1 + j] = 0.0;
                sd[1 + j] = 1.0;
            }
        }

        let mut col = 1 + n_nf;
        let mut prop_off = 0usize;
        for &nl in &self.factor_n_levels {
            let n_levels = nl.max(0) as usize;
            for d in 0..n_levels.saturating_sub(1) {
                let pp = self
                    .factor_proportions
                    .get(prop_off + d + 1)
                    .copied()
                    .unwrap_or(0.0);
                mu[col + d] = pp;
                sd[col + d] = (pp * (1.0 - pp)).max(0.0).sqrt();
            }
            col += n_levels.saturating_sub(1);
            prop_off += n_levels;
        }

        // lp_pop_mean = Σⱼ βⱼ·μⱼ
        let mut lp_mean = 0.0;
        for (j, &mu_j) in mu.iter().enumerate().take(p) {
            let b = self.effect_sizes.get(j).copied().unwrap_or(0.0);
            lp_mean += b * mu_j;
        }

        // lp_pop_var = β' Σ β.
        let mut lp_var = 0.0;
        for (j, &sd_j) in sd.iter().enumerate().take(p) {
            let bj = self.effect_sizes.get(j).copied().unwrap_or(0.0);
            lp_var += bj * bj * sd_j * sd_j;
        }
        // Add 2 · βⱼ·βₖ·Cov(Xⱼ, Xₖ) for the continuous block.
        // self.correlation is flat column-major n_nf×n_nf with diag=1.
        for j in 0..n_nf {
            for k in (j + 1)..n_nf {
                let bj = self.effect_sizes.get(1 + j).copied().unwrap_or(0.0);
                let bk = self.effect_sizes.get(1 + k).copied().unwrap_or(0.0);
                let r = self.correlation[k * n_nf + j];
                lp_var += 2.0 * bj * bk * r * sd[1 + j] * sd[1 + k];
            }
        }
        let lp_std = lp_var.max(0.0).sqrt();

        HeteroskedasticityCoeffs {
            lp_pop_mean: lp_mean,
            lp_pop_std: lp_std,
            // Promote the per-column moments (full-design layout) so the
            // generator can standardize a chosen driver column directly.
            col_mean: mu,
            col_std: sd,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchResult — engine output. Not serialized (returned via numpy).
// ---------------------------------------------------------------------------

/// Per-factor posthoc block shape metadata, carried by `ResultShape` so
/// downstream consumers (aggregation, merge) can reconstruct how the
/// concatenated posthoc contrast buffer is sliced.
#[derive(Debug, Clone, PartialEq)]
pub struct PosthocBlockShape {
    /// Number of levels k for this factor (so consumers can verify C(k,2)).
    pub n_levels: u32,
    /// Number of pairwise contrasts: C(k,2) = k*(k-1)/2.
    pub n_contrasts: u32,
}

/// Tensor dimensions for a `BatchResult`'s flat buffers.
#[derive(Debug, Clone)]
pub struct ResultShape {
    pub n_sims: u32,
    pub n_sample_sizes: u32,
    pub n_targets: u32,
    /// One entry per posthoc block (factor), in block order.
    /// Empty when there are no posthoc tests.
    pub posthoc_blocks: Vec<PosthocBlockShape>,
    /// Number of factors in the spec. 0 for factor-free designs.
    pub n_factors: u32,
    /// Diagonal variance components for the Mle path (=
    /// `cluster.n_variance_components()`); 0 otherwise. Sizes
    /// `boundary_rate_per_component` in `EstimatorExtras::Mle`; component order is
    /// `[intercept, slope_0, …, extra_1, …]` per `LmmGroupings::diagonal_theta()`.
    pub n_variance_components: u32,
}

/// Raw per-sim significance tensors from one `run_batch` call, pre-aggregation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Flat, shape (n_sims, n_sample_sizes, n_targets); row-major.
    pub uncorrected: Vec<u8>,
    pub corrected: Vec<u8>,
    /// Empty if no posthoc.
    pub posthoc_unc: Vec<u8>,
    pub posthoc_cor: Vec<u8>,
    /// (n_sims, n_sample_sizes); always 1 for OLS except rank-deficient designs.
    pub converged: Vec<u8>,
    /// `(n_sims × n_sample_sizes)`, parallel to `converged`. 0 = no boundary,
    /// 1 = τ̂ ≈ 0 (OLS fallback), 2 = Brent failed near the upper θ bound.
    /// OLS / Logit branches always emit 0. Surfaced as a numpy array.
    pub boundary_hit: Vec<u8>,
    /// `(n_sims × n_sample_sizes)`, parallel to `boundary_hit`. Bit k set iff
    /// diagonal variance component k was pinned at 0 this fit — k indexes
    /// `LmmGroupings::diagonal_theta()` order `[intercept, slope_0, …, extra_1, …]`
    /// (general lmm path; Brent path writes bit 0 = `boundary_hit == 1`; OLS/Glm
    /// write 0). Unpacked into per-component pin rates by `EstimatorExtras::from_batch`.
    /// `u64`: mirrors `glmm::{LmmFit,GlmmFit}::pinned_components` (the sparse path's
    /// over-envelope component count can exceed the 32-bit ceiling).
    pub pinned_components: Vec<u64>,
    /// `(n_sims × n_sample_sizes)`, parallel to `converged`. Joint Wald-χ²
    /// significance: 1 if `joint_t_sq > χ²(k, 1-α)`, else 0. Written by the
    /// Mle branch only (Ols/Glm always emit 0). The Python bridge / `results.py`
    /// rewire to consume this has not yet landed; no in-tree consumer reads it yet.
    pub joint_unc: Vec<u8>,
    /// Same shape and semantics as `joint_unc`. The LME intercept lies outside
    /// the family-wise correction set, so the joint test is not Bonferroni-corrected
    /// and `joint_cor` always equals `joint_unc` for LME designs.
    pub joint_cor: Vec<u8>,
    /// `(n_sims × n_sample_sizes)` parallel to `converged`. v1-parity overall
    /// significance — 1 iff the OLS F-test (`estimator == Ols`) or Logit LRT
    /// (`estimator == Glm`) rejects at the spec's `alpha`. `estimator == Mle`
    /// always emits 0. Empty when `spec.report_overall == false` (no
    /// allocation — see batch.rs allocation gate).
    pub overall: Vec<u8>,
    /// `(n_sims × n_sample_sizes × n_factors)`, sim-major. Per-factor exclusion
    /// code: 0 = included, 1 = sparse-excluded (a level under
    /// `factor_min_level_count` in this round's rows), 2 = dropped by the GLM
    /// separation fallback. Empty when the spec has no factors.
    pub factor_excluded: Vec<u8>,
    /// Per-(sim, sample-size) estimated random-intercept variance τ̂² = D̂[0][0]
    /// (row-major `sim * n_ss + ss_idx`), written only by the Glm+cluster GLMM
    /// path; NaN on every other path and on non-converged fits. Harvested into
    /// `EstimatorExtras::Glm.tau_squared_hat_*` over converged fits.
    pub tau_squared_hat: Vec<f64>,
    pub shape: ResultShape,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but realistic OLS spec for round-trip testing.
    pub(crate) fn minimal_ols_spec() -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 2,
            n_factor_dummies: 0,
            correlation: vec![1.0, 0.3, 0.3, 1.0],
            var_types: vec![Distribution::Normal, Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0, 0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.5, 0.3],
            target_indices: vec![0, 1],
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
            estimator: EstimatorSpec::Ols,
            wald_se: WaldSe::default(),
            nagq: 1,
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            extra_slope_cols: Vec::new(),
            fit_columns: Vec::new(),
        }
    }

    #[test]
    fn spec_msgpack_roundtrip() {
        let spec = minimal_ols_spec();
        let bytes = rmp_serde::to_vec(&spec).expect("encode");
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(spec, back);
    }

    #[test]
    fn spec_msgpack_roundtrip_with_perturbed_scenario() {
        let mut spec = minimal_ols_spec();
        spec.scenario = ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.1,
            heteroskedasticity_ratio: 2.0,
            correlation_noise_sd: 0.15,
            distribution_change_prob: 0.2,
            new_distributions: vec![
                Distribution::RightSkewed,
                Distribution::LeftSkewed,
                Distribution::Uniform,
            ],
            residual_change_prob: 0.5,
            residual_dists: vec![ResidualDist::HighKurtosis, ResidualDist::RightSkewed],
            residual_df: 8.0,
            sampled_factor_proportions: true,
            truth_start: true,
            lme: None,
        };
        spec.heteroskedasticity_driver = Some(1);
        let bytes = rmp_serde::to_vec(&spec).expect("encode");
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(spec, back);
    }

    #[test]
    fn spec_uses_typed_enums_for_var_types_residual_correction() {
        let mut spec = minimal_ols_spec();
        spec.var_types = vec![Distribution::Normal, Distribution::Binary];
        spec.residual_dist = ResidualDist::Normal;
        spec.correction_method = CorrectionMethod::Bonferroni;
        // Use to_vec_named so the byte-window scan can find the snake_case
        // variant tags. Equality below works with either encoding.
        let bytes = rmp_serde::to_vec_named(&spec).expect("encode");
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(spec, back);
        assert!(bytes
            .windows(b"bonferroni".len())
            .any(|w| w == b"bonferroni"));
        assert!(bytes.windows(b"normal".len()).any(|w| w == b"normal"));
    }

    #[test]
    fn scenario_default_is_optimistic() {
        let s = ScenarioPerturbations::default();
        assert!(s.is_optimistic());
        let s2 = ScenarioPerturbations::optimistic();
        assert!(s2.is_optimistic());
        assert_eq!(s2.name, "optimistic");
    }

    #[test]
    fn sampled_factor_proportions_is_orthogonal_to_is_optimistic() {
        // The allocation mode (sampled_factor_proportions) and the per-block
        // scenario fast path (is_optimistic) are independent axes. A custom
        // scenario with sampled_factor_proportions flipped but zero perturbations
        // must stay on the fast path.
        assert!(!ScenarioPerturbations::default().sampled_factor_proportions);
        let s = ScenarioPerturbations {
            sampled_factor_proportions: true,
            ..Default::default()
        };
        assert!(
            s.is_optimistic(),
            "allocation mode must not affect the fast-path predicate"
        );
    }

    #[test]
    fn report_overall_roundtrips() {
        let mut spec = minimal_ols_spec();
        // helper explicitly sets report_overall = false; round-trip flips it.
        assert!(!spec.report_overall);
        spec.report_overall = true;
        let bytes = rmp_serde::to_vec(&spec).expect("encode");
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(spec, back);
        assert!(back.report_overall);
    }

    #[test]
    fn batch_result_has_overall_field() {
        // Compile-time check that the field exists with the documented type.
        let br = BatchResult {
            uncorrected: vec![],
            corrected: vec![],
            posthoc_unc: vec![],
            posthoc_cor: vec![],
            converged: vec![],
            boundary_hit: vec![],
            pinned_components: vec![],
            joint_unc: vec![],
            joint_cor: vec![],
            overall: vec![],
            factor_excluded: vec![],
            tau_squared_hat: vec![],
            shape: ResultShape {
                n_sims: 0,
                n_sample_sizes: 0,
                n_targets: 0,
                posthoc_blocks: vec![],
                n_factors: 0,
                n_variance_components: 0,
            },
        };
        assert!(br.overall.is_empty());
    }

    #[test]
    fn scenario_perturbed_is_not_optimistic() {
        let s = ScenarioPerturbations {
            correlation_noise_sd: 0.1,
            ..Default::default()
        };
        assert!(!s.is_optimistic());

        // Non-zero distribution_change_prob triggers perturbation; the pool
        // Vec alone (without the activating prob) does not.
        let s2 = ScenarioPerturbations {
            distribution_change_prob: 0.1,
            new_distributions: vec![Distribution::RightSkewed],
            ..Default::default()
        };
        assert!(!s2.is_optimistic());

        // A pool Vec with zero activation probability is still optimistic
        // (the Python frontend always sends the pool unconditionally).
        let s2_pool_only = ScenarioPerturbations {
            new_distributions: vec![
                Distribution::RightSkewed,
                Distribution::LeftSkewed,
                Distribution::Uniform,
            ],
            ..Default::default()
        };
        assert!(
            s2_pool_only.is_optimistic(),
            "pool-only (prob=0) must still be optimistic"
        );

        let s3 = ScenarioPerturbations {
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 5.0,
                icc_noise_sd: 0.1,
            }),
            ..Default::default()
        };
        assert!(!s3.is_optimistic());
    }

    #[test]
    fn factor_sampled_defaults_empty_and_roundtrips() {
        // Empty (default, absent in older payloads) ⇒ every factor inherits the
        // scenario-level sampled_factor_proportions ⇒ byte-identical to pre-feature.
        let mut spec = minimal_ols_spec();
        assert_eq!(spec.factor_sampled, Vec::<Option<bool>>::new());
        spec.factor_sampled = vec![Some(false), Some(true)];
        let bytes = rmp_serde::to_vec_named(&spec).unwrap();
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(spec.factor_sampled, back.factor_sampled);
    }
}
