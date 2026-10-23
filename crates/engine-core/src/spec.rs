//! Engine spec: error types, outcome kind / estimator axes, progress sink, POD types.
//!
//! Hosts never send these types: every port sends a msgpack
//! `SimulationContract`, which `contract_adapter` lowers to a `SimulationSpec`.
//! No PyO3 / numpy types appear here — engine-core is host-agnostic.

pub use engine_contract::{
    CorrectionMethod, DfMethod, Distribution, EstimatorSpec, LinkKind, OutcomeKind, ResidualDist,
    WaldSe,
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
/// (`[intercept, continuous.., factor_dummies.., interactions..]`) used when
/// `driver == Some(idx)`. Analytic (population) moments only — row-stable by
/// construction. See `SimulationSpec::compute_het_coeffs` for the model.
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
    /// Reference distribution for mixed-model fixed-effect Wald tests (no-op
    /// for `Ols` and unclustered `Glm`). Threaded from the contract field of
    /// the same name.
    #[serde(default)]
    pub df_method: DfMethod,
    /// AGQ node count for the GLMM likelihood (1 = Laplace). Threaded from the
    /// contract `nagq`; the eligibility backstop already ran in `validate()`.
    #[serde(default = "default_nagq_spec")]
    pub nagq: u8,
    /// Host-supplied curvature hints, one per grid point. Copied from the
    /// contract field of the same name, which owns the `None` / `Some(v)` /
    /// `Some([])` rules; `None` runs the pilot in `run_batch`.
    #[serde(default)]
    pub curvature_hints: Option<Vec<Option<engine_contract::CurvatureHintParts>>>,
    /// Baseline intercept as metadata only: no engine code reads it. The
    /// kernel takes the baseline from `effect_sizes[0]` (the adapter folds the
    /// contract intercept there). OLS specs typically carry 0.0, Logit specs
    /// carry `logit(p)`.
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
    /// Precompute the population moments of the heteroskedasticity driver: the
    /// linear predictor `lp = Σⱼ βⱼ·xⱼ` over every `x_full` column (intercept,
    /// continuous, factor dummies, interactions) and each column on its own.
    /// Deterministic (no RNG), computed once per batch from the base spec.
    ///
    /// Column model, matching what `generate_sim_data` writes: continuous
    /// columns are mean 0, variance 1, except `Binary`/`UploadedBinary` (0/1,
    /// mean = param); factor dummies are 0/1 with mean = level proportion, and
    /// two dummies of one factor are never 1 together; an interaction column is
    /// the raw product of its components. Factors are independent of each other
    /// and of the continuous block. Every mean and covariance reduces to a
    /// product moment `E[∏ xₐ]`, which factorizes over those independent
    /// blocks. Within the continuous block it is Isserlis' theorem over
    /// `correlation` (`E[x_a x_b] = r_ab`, odd products 0,
    /// `E[x_a x_b x_c x_d] = r_ab r_cd + r_ac r_bd + r_ad r_bc`), after
    /// writing each binary as `μ + σ·g` with `g` a standard normal carrying the
    /// binary's correlation entries (powers of a 0/1 column collapse first).
    ///
    /// Exact: all-`Normal` continuous columns with any correlation; columns of
    /// any synthetic marginal that are uncorrelated with the other components of a
    /// product (only E[x] and E[x²] enter then); factor dummies.
    /// Approximate: a binary or non-`Normal` column correlated with another
    /// component (the latent correlation is used as if it held on the observed
    /// scale); `UploadedData` columns in NORTA mode (the interpolated empirical
    /// quantile does not keep variance exactly 1); jointly resampled upload
    /// columns (bootstrap mode); cluster-level columns broadcast within
    /// clusters (the broadcast breaks their row-wise correlation with
    /// within-cluster columns); and the per-sim scenario
    /// perturbations, which move the realized moments away from the base spec.
    pub fn compute_het_coeffs(&self) -> HeteroskedasticityCoeffs {
        let n_nf = self.n_non_factor as usize;
        let n_fd = self.n_factor_dummies as usize;
        let n_main = 1 + n_nf + n_fd;
        let p = n_main + self.interactions.len();

        // Main-column marginals. `factor_of[c]` is the factor owning dummy
        // column c (usize::MAX for the intercept and continuous columns).
        let mut mu = vec![0.0_f64; n_main];
        let mut sd = vec![0.0_f64; n_main];
        let mut binary = vec![false; n_main];
        let mut factor_of = vec![usize::MAX; n_main];
        mu[0] = 1.0;
        for j in 0..n_nf {
            let vt = self
                .var_types
                .get(j)
                .copied()
                .unwrap_or(Distribution::Normal);
            if matches!(vt, Distribution::Binary | Distribution::UploadedBinary) {
                let pp = self.var_params.get(j).copied().unwrap_or(0.5);
                mu[1 + j] = pp;
                sd[1 + j] = (pp * (1.0 - pp)).max(0.0).sqrt();
                binary[1 + j] = true;
            } else {
                sd[1 + j] = 1.0;
            }
        }
        let mut col = 1 + n_nf;
        let mut prop_off = 0usize;
        for (f, &nl) in self.factor_n_levels.iter().enumerate() {
            let n_levels = nl.max(0) as usize;
            for d in 0..n_levels.saturating_sub(1) {
                let pp = self
                    .factor_proportions
                    .get(prop_off + d + 1)
                    .copied()
                    .unwrap_or(0.0);
                mu[col + d] = pp;
                sd[col + d] = (pp * (1.0 - pp)).max(0.0).sqrt();
                factor_of[col + d] = f;
            }
            col += n_levels.saturating_sub(1);
            prop_off += n_levels;
        }

        // Each x_full column as the multiset of main columns it multiplies.
        let mut atoms: Vec<Vec<usize>> = Vec::with_capacity(p);
        atoms.push(Vec::new());
        atoms.extend((1..n_main).map(|c| vec![c]));
        atoms.extend(
            self.interactions
                .iter()
                .map(|comps| comps.iter().map(|&c| c as usize).collect()),
        );

        let corr = |a: usize, b: usize| -> f64 {
            if a == b {
                1.0
            } else {
                self.correlation[(b - 1) * n_nf + (a - 1)]
            }
        };
        // E[∏ x_c] over a multiset of main columns.
        let product_moment = |cols: &mut Vec<usize>| -> f64 {
            cols.sort_unstable();
            let mut value = 1.0;
            // Factor dummies: one factor contributes p_d if it shows a single
            // distinct dummy d (any power), 0 if it shows two.
            let mut gauss: Vec<usize> = Vec::new();
            let mut bins: Vec<usize> = Vec::new();
            let mut last_dummy: Option<usize> = None;
            for &c in cols.iter() {
                if c == 0 {
                    continue;
                }
                if factor_of[c] != usize::MAX {
                    // Sorted, and one factor's dummies are adjacent columns, so
                    // only the previous dummy can share c's factor.
                    if let Some(o) = last_dummy {
                        if o == c {
                            continue;
                        }
                        if factor_of[o] == factor_of[c] {
                            return 0.0;
                        }
                    }
                    last_dummy = Some(c);
                    value *= mu[c];
                } else if binary[c] {
                    if bins.last() != Some(&c) {
                        bins.push(c);
                    }
                } else {
                    gauss.push(c);
                }
            }
            // Binaries as μ + σ·g: sum over the subsets S of binaries that
            // enter as σ·g, the rest as μ.
            let mut cont = 0.0;
            for mask in 0u64..(1u64 << bins.len()) {
                let mut coef = 1.0;
                let mut members = gauss.clone();
                for (k, &b) in bins.iter().enumerate() {
                    if mask & (1 << k) != 0 {
                        coef *= sd[b];
                        members.push(b);
                    } else {
                        coef *= mu[b];
                    }
                }
                if coef != 0.0 {
                    cont += coef * isserlis(&members, &corr);
                }
            }
            value * cont
        };

        let col_mean: Vec<f64> = atoms
            .iter()
            .map(|a| product_moment(&mut a.clone()))
            .collect();
        let cov = |a: usize, b: usize| -> f64 {
            let mut joint = atoms[a].clone();
            joint.extend_from_slice(&atoms[b]);
            product_moment(&mut joint) - col_mean[a] * col_mean[b]
        };
        // Main columns keep their closed-form marginal SD; interaction columns
        // take theirs from the product moments.
        let col_std: Vec<f64> = (0..p)
            .map(|c| {
                if c < n_main {
                    sd[c]
                } else {
                    cov(c, c).max(0.0).sqrt()
                }
            })
            .collect();

        let beta = |c: usize| self.effect_sizes.get(c).copied().unwrap_or(0.0);
        let mut lp_mean = 0.0;
        for (c, &mean_c) in col_mean.iter().enumerate() {
            lp_mean += beta(c) * mean_c;
        }
        // lp_pop_var = β' Σ β over the full design: diagonal first, then
        // 2·βₐ·β_b·Cov(Xₐ, X_b) for a < b. This is the main-effects-only
        // summation order, so a design without interactions keeps bit-identical
        // coefficients (cross terms that are exactly 0 add nothing).
        let mut lp_var = 0.0;
        for (a, &sd_a) in col_std.iter().enumerate() {
            let ba = beta(a);
            lp_var += ba * ba * sd_a * sd_a;
        }
        for a in 0..p {
            let ba = beta(a);
            if ba == 0.0 {
                continue;
            }
            for b in (a + 1)..p {
                let bb = beta(b);
                if bb != 0.0 {
                    lp_var += 2.0 * ba * bb * cov(a, b);
                }
            }
        }

        HeteroskedasticityCoeffs {
            lp_pop_mean: lp_mean,
            lp_pop_std: lp_var.max(0.0).sqrt(),
            col_mean,
            col_std,
        }
    }
}

/// Isserlis' theorem: `E[∏ g_c]` for zero-mean unit-variance jointly Gaussian
/// `g` with pairwise correlation `corr` — the sum over perfect pairings of the
/// product of the paired correlations (0 for an odd count). Repeats allowed.
fn isserlis(cols: &[usize], corr: &dyn Fn(usize, usize) -> f64) -> f64 {
    if cols.is_empty() {
        return 1.0;
    }
    if cols.len() % 2 == 1 {
        return 0.0;
    }
    let first = cols[0];
    let mut total = 0.0;
    for i in 1..cols.len() {
        let r = corr(first, cols[i]);
        if r == 0.0 {
            continue;
        }
        let rest: Vec<usize> = cols[1..]
            .iter()
            .enumerate()
            .filter(|&(k, _)| k + 1 != i)
            .map(|(_, &c)| c)
            .collect();
        total += r * isserlis(&rest, corr);
    }
    total
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
    /// Per-(sim, sample-size) realized event rate `mean(y[..N])` over the rows
    /// fitted at that grid point (row-major `sim * n_ss + ss_idx`). Written for a
    /// Binary outcome on every estimator path, converged or not; NaN for every
    /// other outcome. Harvested into `EstimatorExtras::Glm.baseline_prob_*`.
    pub event_rate: Vec<f64>,
    pub shape: ResultShape,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal but realistic OLS spec for round-trip testing.
    pub(crate) fn minimal_ols_spec() -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 2,
            correlation: vec![1.0, 0.3, 0.3, 1.0],
            var_types: vec![Distribution::Normal, Distribution::Normal],
            var_params: vec![0.0, 0.0],
            effect_sizes: vec![0.5, 0.3],
            target_indices: vec![0, 1],
            ..crate::test_support::base_spec()
        }
    }

    /// Lossless serde roundtrip of a spec with every `ScenarioPerturbations`
    /// field and the optional flags set. JSON is the format this type actually
    /// travels in: `bin/throughput.rs --dump-cases` writes specs that the wasm
    /// bench (`engine-wasm` `bench_run_pass`) reads back with `serde_json`.
    #[test]
    fn spec_json_roundtrip_with_every_scenario_field() {
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
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::HighKurtosis,
                random_effect_df: 5.0,
                icc_noise_sd: 0.1,
            }),
        };
        spec.heteroskedasticity_driver = Some(1);
        spec.report_overall = true;
        spec.factor_sampled = vec![Some(false), Some(true)];
        let json = serde_json::to_string(&spec).expect("encode");
        let back: SimulationSpec = serde_json::from_str(&json).expect("decode");
        assert_eq!(spec, back);
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
}
