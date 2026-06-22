//! Public result types returned by the orchestrator. Estimator-generic.
//!
//! Evolution rules: new fields with serde defaults are non-breaking; new
//! `EstimatorExtras` variants are non-breaking for `match _ => {}` consumers;
//! renaming/removing a field is a major bump.
//!
//! Deliberate redundancy: `PowerResult` ships raw counters *and* derived
//! rates/CIs for the same quantities. The counters are load-bearing for
//! `merge_power_results` (pooling then re-deriving); the rates/CIs are the
//! host-facing convenience. Both are accepted contract surface — do not slim.

use engine_core::{BatchResult, EstimatorSpec, SimulationSpec};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Confidence-interval bounds — a named pair, not a tuple, so hosts
/// serialize `{lo, hi}` keys. Wilson proportion bounds: both in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ci {
    pub lo: f64,
    pub hi: f64,
}

/// One scenario passed into the orchestrator. `label` is opaque to Rust; the
/// orchestrator threads it through to `ProgressEvent::ScenarioStarted/Completed`
/// and into the returned `ScenarioResult` keys. `base_seed` is forwarded as-is
/// to `engine_core::run_batch(..., base_seed, ...)` — `SimulationSpec` itself
/// carries no seed field.
///
/// Internal lowering target: hosts pass `&[engine_contract::SimulationContract]`
/// and `base_seed: u64` directly to `find_power` / `find_sample_size`; the
/// orchestrator builds `Scenario` values internally via
/// `engine_core::contract_adapter::contract_to_simulation_spec`.
#[derive(Debug, Clone)]
pub(crate) struct Scenario {
    pub label: String,
    pub spec: SimulationSpec,
    pub base_seed: u64,
    /// Host-facing report indices (design_test term-position space) for this
    /// scenario's marginal targets and contrast pairs — echoed into
    /// `PowerResult.{target_indices, contrast_pairs}` so they index the effect
    /// skeleton. Lockstep with `spec.{target_indices, contrast_pairs}` (the
    /// generation-kernel space the engine reads), which stay as-is for the
    /// kernel read path. Equal to the spec values unless a `test_formula`
    /// drops/reorders a term. Source: `report_targets_and_contrasts`.
    pub report_target_indices: Vec<u32>,
    pub report_contrast_pairs: Vec<(u32, u32)>,
}

/// Power at one sample size for one scenario: per-target rates (uncorrected +
/// corrected), Wilson CIs, convergence, and estimator-specific extras. Raw
/// counters ride along so `merge_power_results` can re-derive the rates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowerResult {
    pub n: usize,
    pub n_sims: u64,
    pub target_indices: Vec<usize>,
    /// Pairwise contrast identities `(positive, negative)` as β-column indices
    /// in the same 1-based space as `target_indices`. The per-target vectors
    /// (`power_*`, `ci_*`, `success_counts_*`) hold the marginals first (one
    /// per `target_indices` entry) and then one entry per contrast pair, in
    /// this order — their length is `target_indices.len() +
    /// contrast_pairs.len()`. Hosts render names from the effect skeleton
    /// (`skeleton[p]` vs `skeleton[n]`). Empty when no contrasts were
    /// requested; serde-default for older payloads. Additive.
    #[serde(default)]
    pub contrast_pairs: Vec<(u32, u32)>,
    pub power_uncorrected: Vec<f64>,
    pub power_corrected: Vec<f64>,
    pub ci_uncorrected: Vec<Ci>,
    pub ci_corrected: Vec<Ci>,
    pub convergence_rate: f64,
    /// Per-simulation boundary_hit values for this sample size. Length == n_sims.
    /// 0 = no boundary, 1 = τ̂≈0 (OLS fallback), 2 = Brent failed.
    /// Always 0 for Ols/Glm estimators.
    pub boundary_hit: Vec<u8>,
    /// Per-estimator numerics surfaced by aggregation. Keyed on the estimator axis.
    pub estimator_extras: EstimatorExtras,
    /// `Some(rate)` when `spec.report_overall == true`, else `None`. Folded
    /// from `BatchResult.overall` the same way `power_uncorrected` is folded.
    #[serde(default)]
    pub overall_significant_rate: Option<f64>,
    /// Raw uncorrected success counts per target (marginals then contrasts —
    /// same length and order as `power_uncorrected`).
    /// `success_counts_uncorrected[t] / n_sims` reproduces `power_uncorrected[t]`.
    /// Load-bearing for `merge_power_results`. Serde-default for older hosts.
    #[serde(default)]
    pub success_counts_uncorrected: Vec<u64>,
    /// Raw corrected success counts per target — same length/order as above.
    #[serde(default)]
    pub success_counts_corrected: Vec<u64>,
    /// Number of sims where the model converged. `convergence_count / n_sims_used`
    /// reproduces `convergence_rate`.
    #[serde(default)]
    pub convergence_count: u64,
    /// Number of sims where the overall-F (or estimator-equivalent overall test) was significant.
    /// `Some(overall_significant_count / n_sims_used)` reproduces `overall_significant_rate`
    /// when the result requested an overall test, else `0`.
    #[serde(default)]
    pub overall_significant_count: u64,
    /// Wilson 95% CI for `overall_significant_rate`. `None` whenever that rate
    /// is `None` (i.e. no overall test was requested). Recomputed from pooled
    /// `overall_significant_count / n_sims` on merge. Singular — the overall
    /// test is one joint test, so no `_corrected` variant.
    #[serde(default)]
    pub overall_significant_ci: Option<Ci>,
    /// Distribution over the number of *marginal tests* significant per sim —
    /// the main targets (marginals + contrast_pairs) AND every post-hoc
    /// pairwise contrast. Excludes only the overall omnibus test.
    /// `len() == n_targets + Σ(post-hoc contrasts) + 1`; bucket `k` counts sims
    /// with exactly `k` tests significant (each counted at its own corrected/
    /// uncorrected threshold). Merged by elementwise sum.
    #[serde(default)]
    pub success_count_histogram_uncorrected: Vec<u64>,
    #[serde(default)]
    pub success_count_histogram_corrected: Vec<u64>,
    /// Pre-run advisory warnings for this scenario's result: cluster snap
    /// (N floored to a multiple of the cluster atom) and sparse-factor
    /// preflight (a factor level too sparse at this N under fixed allocation).
    /// Empty for the per-N trace results inside a `SampleSizeResult`. Additive
    /// (serde-default).
    /// Deliberate exception — the engine normally emits no host-facing prose,
    /// but composes these English strings so the wording is single-sourced
    /// across all four ports. Hosts re-emit them verbatim and must never parse
    /// them; the text may change at any Z bump.
    /// Mirrors SampleSizeResult.grid_warnings — change together.
    #[serde(default)]
    pub grid_warnings: Vec<String>,
    /// One post-hoc pairwise family per requested factor, in request order.
    /// Empty when no post-hoc was requested. Additive.
    #[serde(default)]
    pub posthoc: Vec<PosthocPower>,
    /// Sims in which factor f was sparse-excluded (a level under the configured
    /// minimum), per factor in spec order. Length == n_factors; empty for
    /// factor-free designs and older payloads. count / n_sims = exclusion rate.
    /// Pooled by merge_power_results via elementwise sum.
    #[serde(default)]
    pub factor_exclusion_counts: Vec<u64>,
    /// Sims in which the GLM separation fallback dropped factor f — counted when
    /// the drop is attempted, whether or not the one-shot refit then converged.
    /// Same shape and merge rule as factor_exclusion_counts.
    #[serde(default)]
    pub factor_separation_counts: Vec<u64>,
}

/// Post-hoc pairwise contrast results for ONE factor's family. Arrays are length
/// C(k,2) in canonical pair order. Labels are owned by the host port. Blocks are
/// in the host's posthoc_requests order, so the host maps block→factor by position.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PosthocPower {
    /// Factor's level count k (host sanity-checks C(k,2)).
    pub n_levels: usize,
    pub power_uncorrected: Vec<f64>,
    pub power_corrected: Vec<f64>,
    pub ci_uncorrected: Vec<Ci>,
    pub ci_corrected: Vec<Ci>,
    /// Raw counts for merge_power_results (WASM). count / n_sims = power.
    pub success_counts_uncorrected: Vec<u64>,
    pub success_counts_corrected: Vec<u64>,
}

fn nan_default() -> f64 {
    f64::NAN
}

/// Accept a JSON `null` (or msgpack nil) as `NaN` when reading a placeholder
/// `f64`. `serde_json` renders `f64::NAN` as `null` on the way out and then
/// *refuses* to parse `null` back into an `f64` — which crashes the WASM merge
/// path, since per-worker results round-trip through JSON and the
/// `baseline_prob_realized` / `tau_squared_hat_mean` / `tau_estimate` fields
/// carry NaN until the kernel surfaces the per-sim signal. Only deserialize is
/// overridden: serialize stays default, so msgpack hosts (Py/R) keep carrying
/// NaN natively and surface it unchanged — this codec is transparent for finite
/// values and for the native float-NaN msgpack already writes.
fn nan_tolerant<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// Per-estimator numerics surfaced by aggregation. Tagged enum so JSON envelope
/// is self-describing. Append-only; do not reorder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "estimator", rename_all = "snake_case")]
pub enum EstimatorExtras {
    Ols {},
    Glm {
        #[serde(deserialize_with = "nan_tolerant")]
        baseline_prob_realized: f64,
        /// Sum of per-sim realised baseline probabilities. Paired with
        /// `baseline_prob_n` so `merge_power_results` can pool means via Σ/n.
        /// Engine kernel does not yet surface per-sim baseline probs:
        /// `aggregate_batch` populates 0.0 / 0 until the signal lands.
        #[serde(default)]
        baseline_prob_sum: f64,
        #[serde(default)]
        baseline_prob_n: u64,
        /// Fraction of CONVERGED GLMM fits pinned at a variance-component
        /// boundary (`boundary_hit == 1`; lme4's "singular fit"). 0.0 when
        /// nothing converged / no cluster. Mirrors the Mle field.
        #[serde(default)]
        singular_fit_rate: f64,
        /// Pinned-and-converged count. Paired with `singular_n` so
        /// `merge_power_results` can pool the rate via Σ/Σ.
        #[serde(default)]
        singular_count: u64,
        /// Converged-fit count — the rate's denominator.
        #[serde(default)]
        singular_n: u64,
        /// Mean estimated random-intercept variance D̂[0][0] across converged
        /// fits — the engine half of the Laplace-bias warning (the host
        /// pairs it with its own cluster sizes). NaN when nothing converged.
        #[serde(default = "nan_default", deserialize_with = "nan_tolerant")]
        tau_squared_hat_mean: f64,
        /// Σ of per-sim τ̂² over converged fits, paired with `tau_squared_hat_n`
        /// so `merge_power_results` can pool the mean via Σ/n.
        #[serde(default)]
        tau_squared_hat_sum: f64,
        #[serde(default)]
        tau_squared_hat_n: u64,
    },
    Mle {
        #[serde(deserialize_with = "nan_tolerant")]
        tau_estimate: f64,
        boundary_hits: u64,
        joint_uncorrected_rate: f64,
        joint_corrected_rate: f64,
        /// Sum of per-sim τ̂ values. Paired with `tau_n` so
        /// `merge_power_results` can pool means via Σ/n. Engine kernel does
        /// not yet surface per-sim τ̂: `aggregate_batch` populates
        /// 0.0 / 0 until the signal lands.
        #[serde(default)]
        tau_sum: f64,
        #[serde(default)]
        tau_n: u64,
        /// Raw joint Wald-χ² counts paired with `joint_*_rate` so
        /// `merge_power_results` can pool counts then divide.
        #[serde(default)]
        joint_uncorrected_count: u64,
        #[serde(default)]
        joint_corrected_count: u64,
        /// Fraction of CONVERGED fits pinned at a variance-component boundary
        /// (`boundary_hit == 1`; lme4's "singular fit"). 0.0 when nothing
        /// converged.
        #[serde(default)]
        singular_fit_rate: f64,
        /// Pinned-and-converged count. Paired with `singular_n` so
        /// `merge_power_results` can pool the rate via Σ/Σ.
        #[serde(default)]
        singular_count: u64,
        /// Converged-fit count — the rate's denominator.
        #[serde(default)]
        singular_n: u64,
        /// Per-diagonal-component pin rate (fraction of CONVERGED fits that
        /// pinned component k), ordered [intercept, slope_0, …, extra_1, …]
        /// (= `n_variance_components`). Empty for non-cluster or
        /// intercept-only Mle (n_variance_components == 0).
        #[serde(default)]
        boundary_rate_per_component: Vec<f64>,
        /// Per-component pinned counts, paired with `singular_n` (the converged-fit
        /// count, NOT n_sims — bh == 2 optimizer-failure fits are excluded, matching
        /// `singular_fit_rate`) so `merge_power_results` pools via Σcounts / Σsingular_n.
        #[serde(default)]
        boundary_component_counts: Vec<u64>,
    },
}

impl EstimatorExtras {
    /// Build the per-estimator extras for one sample-size slot directly from a
    /// raw `BatchResult` (the `aggregate_batch` path). The merge path
    /// (`merge_power_results`) folds already-aggregated fields and owns its own
    /// constructor — the two inputs are structurally different, but both
    /// enumerate the same variants and fields defined above.
    ///
    /// `tau_estimate` / `baseline_prob_realized` carry `NaN` and the merge-support
    /// sums carry `0` until the kernel surfaces the per-sim signals.
    pub fn from_batch(estimator: &EstimatorSpec, batch: &BatchResult, ss_idx: usize) -> Self {
        let n_sims = batch.shape.n_sims as usize;
        let n_ss = batch.shape.n_sample_sizes as usize;
        let rate = |k: u64| {
            if n_sims == 0 {
                0.0
            } else {
                k as f64 / n_sims as f64
            }
        };
        match estimator {
            EstimatorSpec::Ols => EstimatorExtras::Ols {},
            EstimatorSpec::Glm => {
                let mut singular_count = 0u64;
                let mut singular_n = 0u64;
                let mut tau_sum = 0.0f64;
                let mut tau_n = 0u64;
                for sim in 0..n_sims {
                    let idx = sim * n_ss + ss_idx;
                    if batch.converged[idx] != 0 {
                        singular_n += 1;
                        if batch.boundary_hit[idx] == 1 {
                            singular_count += 1;
                        }
                        let th = batch.tau_squared_hat.get(idx).copied().unwrap_or(f64::NAN);
                        if th.is_finite() {
                            tau_sum += th;
                            tau_n += 1;
                        }
                    }
                }
                EstimatorExtras::Glm {
                    baseline_prob_realized: f64::NAN,
                    baseline_prob_sum: 0.0,
                    baseline_prob_n: 0,
                    singular_fit_rate: if singular_n > 0 {
                        singular_count as f64 / singular_n as f64
                    } else {
                        0.0
                    },
                    singular_count,
                    singular_n,
                    tau_squared_hat_mean: if tau_n > 0 {
                        tau_sum / tau_n as f64
                    } else {
                        f64::NAN
                    },
                    tau_squared_hat_sum: tau_sum,
                    tau_squared_hat_n: tau_n,
                }
            }
            EstimatorSpec::Mle => {
                // boundary_hits = count of nonzero entries in this ss slot;
                // joint_*_count = per-sim joint-significance sums for this slot.
                let bh_count: u64 = (0..n_sims)
                    .map(|sim| (batch.boundary_hit[sim * n_ss + ss_idx] != 0) as u64)
                    .sum();
                let mut k_ju = 0u64;
                let mut k_jc = 0u64;
                for sim in 0..n_sims {
                    k_ju += batch.joint_unc[sim * n_ss + ss_idx] as u64;
                    k_jc += batch.joint_cor[sim * n_ss + ss_idx] as u64;
                }
                // singular = pinned (bh == 1) among converged fits; bh == 2 is optimizer
                // failure (non-converged by kernel policy) and must not count.
                let mut singular_count = 0u64;
                let mut singular_n = 0u64;
                for sim in 0..n_sims {
                    let idx = sim * n_ss + ss_idx;
                    if batch.converged[idx] != 0 {
                        singular_n += 1;
                        if batch.boundary_hit[idx] == 1 {
                            singular_count += 1;
                        }
                    }
                }
                let singular_fit_rate = if singular_n > 0 {
                    singular_count as f64 / singular_n as f64
                } else {
                    0.0
                };
                // Per-component pin rates: unpack the u32 bitmask for every
                // converged fit in this ss slot, count bit-k hits, then
                // divide by the same singular_n denominator.
                let n_comp = batch.shape.n_variance_components as usize;
                let mut comp_counts = vec![0u64; n_comp];
                for sim in 0..n_sims {
                    let idx = sim * n_ss + ss_idx;
                    if batch.converged[idx] != 0 {
                        let bits = batch.pinned_components[idx];
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..n_comp {
                            if bits & (1 << k) != 0 {
                                comp_counts[k] += 1;
                            }
                        }
                    }
                }
                let comp_rates: Vec<f64> = comp_counts
                    .iter()
                    .map(|&c| {
                        if singular_n > 0 {
                            c as f64 / singular_n as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();
                EstimatorExtras::Mle {
                    tau_estimate: f64::NAN,
                    boundary_hits: bh_count,
                    joint_uncorrected_rate: rate(k_ju),
                    joint_corrected_rate: rate(k_jc),
                    tau_sum: 0.0,
                    tau_n: 0,
                    joint_uncorrected_count: k_ju,
                    joint_corrected_count: k_jc,
                    singular_fit_rate,
                    singular_count,
                    singular_n,
                    boundary_rate_per_component: comp_rates,
                    boundary_component_counts: comp_counts,
                }
            }
        }
    }
}

/// Model-based crossing estimate for one power-vs-N curve: an isotonic (PAVA)
/// fit of the corrected series over the whole grid, read off at the target
/// power. The headline required-N in every host; `first_achieved` stays the
/// grid-empirical raw/fallback record. Tagged like `EstimatorExtras`
/// (self-describing JSON envelope); append-only.
///
/// No `f64` infinity ever appears in these fields (`Option` instead) — the
/// wire format stays JSON-safe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CrossingFit {
    /// The fit crosses the target within the grid.
    Fitted {
        /// Interpolated crossing of the isotonic fit with the target —
        /// continuous; plot markers use it, hosts never print it.
        n_star: f64,
        /// `ceil(n_star)` rounded up to the cluster atom — the headline
        /// integer hosts display. Computed engine-side so all four ports
        /// agree on the atom ceiling.
        n_achievable: usize,
        /// 95% CI on required N by Wilson band inversion: crossing of the
        /// PAVA-fitted per-point Wilson *hi* band (the optimistic band
        /// crosses earlier). `None` = below the search floor.
        ci_lo: Option<f64>,
        /// Crossing of the PAVA-fitted Wilson *lo* band. `None` = above the
        /// search ceiling.
        ci_hi: Option<f64>,
    },
    /// Fitted power already >= target at the first grid point; the true
    /// required N is at or below `n_min` (the grid floor).
    AtOrBelowMin { n_min: usize },
    /// Fitted power never reaches the target within the grid. `n_approx` is
    /// the probit-in-√N extrapolation hint (atom-ceiled); `None` when the
    /// hint is suppressed (fitted endpoint power below the low-power gate,
    /// non-positive/degenerate WLS slope, or crossing beyond the
    /// extrapolation cap).
    NotReached { n_approx: Option<usize> },
    /// The monotonicity gate fired: the largest decrease between two grid
    /// points exceeded the 2-SE Monte-Carlo noise band, i.e. the power curve
    /// has real non-monotone structure (model misspecification, e.g. an
    /// interaction flipping sign with N). The fit is suppressed; hosts fall
    /// back to the grid-empirical value and warn. `max_violation` is that
    /// largest decrease, in proportion points.
    NonMonotone { max_violation: f64 },
}

fn default_cluster_atom() -> usize {
    1
}

/// Sample-size search result for one scenario: the evaluated grid plus the
/// per-target and joint first-N-reaching-target-power derivations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleSizeResult {
    /// Per-N evaluated, in grid order.
    pub grid_or_trace: Vec<PowerResult>,
    /// Smallest N at which target `t` independently meets `target_power`,
    /// or `None` if never achieved. One entry per power-vector slot —
    /// marginals first, then contrast pairs (`len() == target_indices.len()
    /// + contrast_pairs.len()` of the grid points).
    pub first_achieved: Vec<Option<usize>>,
    /// Smallest N at which *at least k* tests are jointly significant with
    /// probability >= `target_power`, derived from the per-N **corrected**
    /// success-count histogram. `len() == n_targets + Σ(post-hoc contrasts)`
    /// (== `success_count_histogram` length − 1); index `j` is k = j+1
    /// (`[0]` = ">=1 test", last = "all tests"). `None` = never reached.
    /// Additive contract field (serde default) — minor version bump.
    #[serde(default)]
    pub first_joint_achieved: Vec<Option<usize>>,
    /// Model-based crossing per target, parallel to `first_achieved`
    /// (same length and marginals-then-contrasts order). Empty in older
    /// payloads — hosts then fall back to `first_achieved`, the same path
    /// `NonMonotone` uses. Additive (serde default) — minor version bump.
    #[serde(default)]
    pub fitted: Vec<CrossingFit>,
    /// Model-based crossing for the joint P(>=k) curves, parallel to
    /// `first_joint_achieved` (index `j` is k = j+1). Empty when histograms
    /// are absent, mirroring `first_joint_achieved`. Additive (serde default).
    #[serde(default)]
    pub fitted_joint: Vec<CrossingFit>,
    /// Grid-empirical first N at which the overall/omnibus test reaches
    /// `target_power`, or `None` when no overall test was requested (mirrors
    /// `PowerResult.overall_significant_rate`). Singular — the omnibus is one
    /// test, so no per-target vector. `None` also when the overall test was
    /// requested but never crossed in-grid; `fitted_overall` carries the
    /// requested-vs-absent distinction. Additive (serde default).
    #[serde(default)]
    pub first_overall_achieved: Option<usize>,
    /// Model-based crossing for the overall-test power-vs-N curve, the same
    /// isotonic (PAVA) + Wilson-band fit `fitted` uses per target. `None`
    /// whenever `first_overall_achieved`'s source rate is `None` (no overall
    /// test requested). Additive (serde default).
    #[serde(default)]
    pub fitted_overall: Option<CrossingFit>,
    /// Cluster atom the grid was built on (1 when unclustered). Merge needs
    /// it to recompute `n_achievable` from pooled counts. Additive
    /// (serde default = 1).
    #[serde(default = "default_cluster_atom")]
    pub cluster_atom: usize,
    pub target_power: f64,
    pub method: SampleSizeMethod,
    /// Pre-run advisory warnings: cluster-aware grid construction (bounds
    /// raised/lowered to the atom, coarse-crossing, regime-A too-few-clusters)
    /// and sparse-factor preflight (a factor level stays under the minimum
    /// across part or all of the grid under fixed allocation). Per-scenario
    /// (factor allocation mode is a scenario knob). Additive (serde-default).
    /// Hosts re-emit these as user warnings.
    /// String-not-enum rationale: see PowerResult.grid_warnings (the owning
    /// comment) — change together.
    #[serde(default)]
    pub grid_warnings: Vec<String>,
}

/// The additive grid step/count variant.
/// `Fixed`: Linear → step (rounded up to a multiple of the atom); Log → point count.
/// `Auto { count }`: ~`count` points placed between the snapped bounds, auto-capped
/// by the feasible count when the atom is coarse. Default `count` is
/// `config().simulation.cluster_auto_count` (12), applied host-side.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ByValue {
    Fixed(usize),
    Auto { count: usize },
}

/// Host-exposed search method. Grid is deliberately the only variant: it is
/// the only method whose per-worker results merge (a bisection trace cannot
/// be pooled across workers).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SampleSizeMethod {
    Grid { by: ByValue, mode: GridMode },
}

/// Grid spacing: `Linear` steps by a fixed increment; `Log` places points
/// geometrically between the bounds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GridMode {
    Linear,
    Log,
}

impl SampleSizeMethod {
    /// Build from the stringly keyword-arg encoding shared by the Python and
    /// R FFI adapters (`method` / `by` / `by_kind` / `mode`). Error strings
    /// are host-facing as-is; each adapter wraps them in its native error
    /// type. Defaults mirror the adapters' historic behaviour: `by_kind` →
    /// `"fixed"`, `mode` → `"linear"`.
    pub fn from_host_args(
        method: &str,
        by: Option<usize>,
        by_kind: Option<&str>,
        mode: Option<&str>,
    ) -> Result<Self, String> {
        match method {
            "grid" => {
                let n = by.ok_or_else(|| "grid method requires `by`".to_string())?;
                let by = match by_kind.unwrap_or("fixed") {
                    "fixed" => ByValue::Fixed(n),
                    "auto" => ByValue::Auto { count: n },
                    other => return Err(format!("unknown by_kind {other:?}")),
                };
                let mode = match mode.unwrap_or("linear") {
                    "linear" => GridMode::Linear,
                    "log" => GridMode::Log,
                    other => return Err(format!("unknown mode {other:?}")),
                };
                Ok(SampleSizeMethod::Grid { by, mode })
            }
            other => Err(format!("unknown method {other:?}")),
        }
    }
}

/// Multi-scenario result wrapper. Always returned, even for a single
/// scenario; single-scenario callers see a one-element vec. Order matches
/// the input slice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioResult<T> {
    pub scenarios: Vec<(String, T)>,
}

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("engine error: {0}")]
    Engine(#[from] engine_core::EngineError),
    #[error("invalid scenarios input: {0}")]
    InvalidScenarios(String),
    #[error("invalid grid bounds: from={from}, to={to}, mode={mode:?}")]
    InvalidGridBounds {
        from: usize,
        to: usize,
        mode: GridMode,
    },
    #[error("invalid cluster atom: atom must be >= 1")]
    InvalidClusterAtom,
    #[error("after cluster snapping the [{from},{to}] range is empty (atom={atom}); widen bounds or reduce n_clusters/cluster_size")]
    ClusterGridEmpty { from: usize, to: usize, atom: usize },
    #[error(
        "range admits one sample size (snapped [{from},{to}], atom={atom}); nothing to search"
    )]
    ClusterGridSinglePoint { from: usize, to: usize, atom: usize },
    #[error("scenarios in this call disagree on the cluster atom ({a} vs {b}); a single shared grid cannot be a multiple of every atom")]
    MixedClusterAtoms { a: usize, b: usize },
    #[error("FixedSize cluster regime needs cluster_size >= {min} (no within-cluster information below that); got {got}")]
    ClusterSizeTooSmall { got: usize, min: usize },
    #[error("at N={n} the FixedSize regime yields only {got} cluster(s) (cluster_size={cluster_size}); need >= {min}. Raise sample_size to at least {min}*{cluster_size}.")]
    ClusterTooFewAtN {
        n: usize,
        cluster_size: usize,
        got: usize,
        min: usize,
    },
    #[error("cancelled at scenario {scenario_idx}, n={n:?}")]
    Cancelled {
        scenario_idx: usize,
        n: Option<usize>,
    },
    #[error("cannot merge incompatible results: {0}")]
    IncompatibleMerge(String),
}

#[cfg(test)]
mod result_counter_tests {
    use super::*;

    #[test]
    fn power_result_serde_roundtrip_with_counters() {
        let pr = PowerResult {
            n: 100,
            n_sims: 200,
            target_indices: vec![0, 1],
            contrast_pairs: vec![],
            power_uncorrected: vec![0.8, 0.6],
            power_corrected: vec![0.75, 0.55],
            ci_uncorrected: vec![Ci { lo: 0.7, hi: 0.9 }, Ci { lo: 0.5, hi: 0.7 }],
            ci_corrected: vec![Ci { lo: 0.65, hi: 0.85 }, Ci { lo: 0.45, hi: 0.65 }],
            convergence_rate: 1.0,
            boundary_hit: vec![0u8; 200],
            estimator_extras: EstimatorExtras::Ols {},
            overall_significant_rate: None,
            success_counts_uncorrected: vec![160, 120],
            success_counts_corrected: vec![150, 110],
            convergence_count: 200,
            overall_significant_count: 0,
            overall_significant_ci: None,
            success_count_histogram_uncorrected: vec![],
            success_count_histogram_corrected: vec![],
            grid_warnings: vec![],
            posthoc: vec![],
            factor_exclusion_counts: vec![],
            factor_separation_counts: vec![],
        };
        let json = serde_json::to_string(&pr).unwrap();
        let back: PowerResult = serde_json::from_str(&json).unwrap();
        assert_eq!(pr, back);
        assert_eq!(pr.grid_warnings, Vec::<String>::new());
    }

    #[test]
    fn power_result_deserializes_older_payload_without_counters() {
        // Legacy payload — no counter fields. Defaults must kick in.
        let legacy = r#"{
            "n": 100, "n_sims": 200, "target_indices": [0],
            "power_uncorrected": [0.8], "power_corrected": [0.75],
            "ci_uncorrected": [{"lo":0.7,"hi":0.9}], "ci_corrected": [{"lo":0.65,"hi":0.85}],
            "convergence_rate": 1.0, "boundary_hit": [],
            "estimator_extras": {"estimator":"ols"}
        }"#;
        let pr: PowerResult = serde_json::from_str(legacy).unwrap();
        assert_eq!(pr.success_counts_uncorrected, Vec::<u64>::new());
        assert_eq!(pr.convergence_count, 0);
        assert_eq!(pr.factor_exclusion_counts, Vec::<u64>::new());
        assert_eq!(pr.factor_separation_counts, Vec::<u64>::new());
    }

    #[test]
    fn power_result_roundtrips_histogram_and_overall_ci() {
        let mut pr = PowerResult {
            n: 100,
            n_sims: 200,
            target_indices: vec![0, 1],
            contrast_pairs: vec![],
            power_uncorrected: vec![0.8, 0.6],
            power_corrected: vec![0.75, 0.55],
            ci_uncorrected: vec![Ci { lo: 0.7, hi: 0.9 }, Ci { lo: 0.5, hi: 0.7 }],
            ci_corrected: vec![Ci { lo: 0.65, hi: 0.85 }, Ci { lo: 0.45, hi: 0.65 }],
            convergence_rate: 1.0,
            boundary_hit: vec![0u8; 200],
            estimator_extras: EstimatorExtras::Ols {},
            overall_significant_rate: Some(0.92),
            success_counts_uncorrected: vec![160, 120],
            success_counts_corrected: vec![150, 110],
            convergence_count: 200,
            overall_significant_count: 184,
            overall_significant_ci: None,
            success_count_histogram_uncorrected: vec![],
            success_count_histogram_corrected: vec![],
            grid_warnings: vec![],
            posthoc: vec![],
            factor_exclusion_counts: vec![],
            factor_separation_counts: vec![],
        };
        pr.overall_significant_ci = Some(Ci { lo: 0.91, hi: 0.94 });
        pr.success_count_histogram_uncorrected = vec![10, 30, 60];
        pr.success_count_histogram_corrected = vec![15, 35, 50];

        let json = serde_json::to_string(&pr).unwrap();
        let back: PowerResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pr);
    }

    #[test]
    fn power_result_older_payload_defaults_new_fields() {
        // A payload emitted before these fields existed: they are simply absent.
        let json = r#"{
            "n": 100, "n_sims": 5, "target_indices": [0],
            "power_uncorrected": [0.8], "power_corrected": [0.8],
            "ci_uncorrected": [{"lo":0.7,"hi":0.9}], "ci_corrected": [{"lo":0.7,"hi":0.9}],
            "convergence_rate": 1.0, "boundary_hit": [0,0,0,0,0],
            "estimator_extras": {"estimator":"ols"}
        }"#;
        let pr: PowerResult = serde_json::from_str(json).unwrap();
        assert_eq!(pr.overall_significant_ci, None);
        assert!(pr.success_count_histogram_uncorrected.is_empty());
        assert!(pr.success_count_histogram_corrected.is_empty());
    }

    #[test]
    fn crossing_fit_serde_roundtrips_all_variants() {
        let variants = vec![
            CrossingFit::Fitted {
                n_star: 84.3,
                n_achievable: 90,
                ci_lo: Some(76.2),
                ci_hi: None,
            },
            CrossingFit::AtOrBelowMin { n_min: 30 },
            CrossingFit::NotReached {
                n_approx: Some(330),
            },
            CrossingFit::NotReached { n_approx: None },
            CrossingFit::NonMonotone {
                max_violation: 0.054,
            },
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: CrossingFit = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
        // The tag is `status` in snake_case, like EstimatorExtras' `estimator`.
        let json = serde_json::to_string(&variants[1]).unwrap();
        assert!(json.contains(r#""status":"at_or_below_min""#), "{json}");
    }

    #[test]
    fn sample_size_result_older_payload_defaults_fit_fields() {
        // A payload emitted before the model-based crossing fields existed.
        let json = r#"{
            "grid_or_trace": [], "first_achieved": [null],
            "target_power": 0.8,
            "method": {"Grid": {"by": {"Fixed": 10}, "mode": "Linear"}}
        }"#;
        let ssr: SampleSizeResult = serde_json::from_str(json).unwrap();
        assert!(ssr.fitted.is_empty());
        assert!(ssr.fitted_joint.is_empty());
        assert_eq!(ssr.cluster_atom, 1);
        // Overall crossing fields absent in a pre-overall payload default to None.
        assert_eq!(ssr.first_overall_achieved, None);
        assert_eq!(ssr.fitted_overall, None);
    }

    /// singular_fit_rate counts boundary_hit == 1 (pinned) AND converged only —
    /// a bh == 2 sim (optimizer failure) must not count even if a buggy kernel
    /// marked it converged, and the denominator is converged sims, not n_sims.
    #[test]
    fn mle_extras_singular_fit_rate_counts_pinned_converged_only() {
        use engine_core::spec::{BatchResult, ResultShape};
        // 4 sims × 1 sample size × 1 target:
        // converged    = [1, 1, 1, 0]
        // boundary_hit = [1, 0, 2, 2]
        // ⇒ singular_count = 1 (sim 0), singular_n = 3, rate = 1/3.
        let batch = BatchResult {
            uncorrected: vec![0; 4],
            corrected: vec![0; 4],
            posthoc_unc: vec![],
            posthoc_cor: vec![],
            converged: vec![1, 1, 1, 0],
            boundary_hit: vec![1, 0, 2, 2],
            pinned_components: vec![0u32; 4],
            joint_unc: vec![0; 4],
            joint_cor: vec![0; 4],
            overall: vec![],
            factor_excluded: vec![],
            tau_squared_hat: vec![],
            shape: ResultShape {
                n_sims: 4,
                n_sample_sizes: 1,
                n_targets: 1,
                posthoc_blocks: vec![],
                n_factors: 0,
                n_variance_components: 0,
            },
        };
        let x = EstimatorExtras::from_batch(&EstimatorSpec::Mle, &batch, 0);
        let EstimatorExtras::Mle {
            singular_fit_rate,
            singular_count,
            singular_n,
            ..
        } = x
        else {
            panic!("expected Mle extras");
        };
        assert_eq!(singular_count, 1);
        assert_eq!(singular_n, 3);
        assert!((singular_fit_rate - 1.0 / 3.0).abs() < 1e-15);
    }

    /// Two components, two sims: sim0 pins component 0 (bit 0 set),
    /// sim1 pins component 1 (bit 1 set), both converged ⇒ each rate 0.5.
    #[test]
    fn boundary_rate_per_component_unpacks_bits() {
        use engine_core::spec::{BatchResult, ResultShape};
        let batch = BatchResult {
            uncorrected: vec![0; 2],
            corrected: vec![0; 2],
            posthoc_unc: vec![],
            posthoc_cor: vec![],
            converged: vec![1, 1],
            boundary_hit: vec![1, 1],
            pinned_components: vec![0b01, 0b10],
            joint_unc: vec![0; 2],
            joint_cor: vec![0; 2],
            overall: vec![],
            factor_excluded: vec![],
            tau_squared_hat: vec![],
            shape: ResultShape {
                n_sims: 2,
                n_sample_sizes: 1,
                n_targets: 1,
                posthoc_blocks: vec![],
                n_factors: 0,
                n_variance_components: 2,
            },
        };
        let ex = EstimatorExtras::from_batch(&EstimatorSpec::Mle, &batch, 0);
        if let EstimatorExtras::Mle {
            boundary_rate_per_component,
            boundary_component_counts,
            singular_n,
            ..
        } = ex
        {
            assert_eq!(singular_n, 2);
            assert_eq!(boundary_component_counts, vec![1u64, 1]);
            assert_eq!(boundary_rate_per_component, vec![0.5, 0.5]);
        } else {
            panic!("expected Mle");
        }
    }

    #[test]
    fn sample_size_result_roundtrips_fit_fields() {
        let ssr = SampleSizeResult {
            grid_or_trace: vec![],
            first_achieved: vec![Some(90), None],
            first_joint_achieved: vec![Some(120), None],
            fitted: vec![
                CrossingFit::Fitted {
                    n_star: 84.3,
                    n_achievable: 90,
                    ci_lo: Some(76.2),
                    ci_hi: Some(95.8),
                },
                CrossingFit::NotReached { n_approx: None },
            ],
            fitted_joint: vec![
                CrossingFit::Fitted {
                    n_star: 118.0,
                    n_achievable: 120,
                    ci_lo: None,
                    ci_hi: None,
                },
                CrossingFit::NonMonotone {
                    max_violation: 0.07,
                },
            ],
            first_overall_achieved: Some(100),
            fitted_overall: Some(CrossingFit::Fitted {
                n_star: 92.5,
                n_achievable: 100,
                ci_lo: Some(80.0),
                ci_hi: Some(110.0),
            }),
            cluster_atom: 30,
            target_power: 0.8,
            method: SampleSizeMethod::Grid {
                by: ByValue::Auto { count: 12 },
                mode: GridMode::Linear,
            },
            grid_warnings: vec![],
        };
        let json = serde_json::to_string(&ssr).unwrap();
        let back: SampleSizeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ssr);
    }

    #[test]
    fn glm_cluster_extras_harvest_singular_and_tau() {
        use engine_core::spec::{BatchResult, ResultShape};
        // 2 sims × 1 ss, n_variance_components = 1 (random intercept only).
        // sim0: converged, pinned (bh=1), τ̂²=0.20.
        // sim1: converged, interior (bh=0), τ̂²=0.30.
        let batch = BatchResult {
            uncorrected: vec![1, 1],
            corrected: vec![1, 1],
            posthoc_unc: vec![],
            posthoc_cor: vec![],
            converged: vec![1, 1],
            boundary_hit: vec![1, 0],
            pinned_components: vec![0b1, 0b0],
            joint_unc: vec![0, 0],
            joint_cor: vec![0, 0],
            overall: vec![0, 0],
            factor_excluded: vec![],
            tau_squared_hat: vec![0.20, 0.30],
            shape: ResultShape {
                n_sims: 2,
                n_sample_sizes: 1,
                n_targets: 1,
                posthoc_blocks: vec![],
                n_factors: 0,
                n_variance_components: 1,
            },
        };
        let ex = EstimatorExtras::from_batch(&EstimatorSpec::Glm, &batch, 0);
        let EstimatorExtras::Glm {
            singular_fit_rate,
            singular_count,
            singular_n,
            tau_squared_hat_mean,
            tau_squared_hat_n,
            ..
        } = ex
        else {
            panic!("expected Glm")
        };
        assert_eq!((singular_count, singular_n), (1, 2));
        assert!((singular_fit_rate - 0.5).abs() < 1e-12);
        assert_eq!(tau_squared_hat_n, 2);
        assert!((tau_squared_hat_mean - 0.25).abs() < 1e-12);
    }

    #[test]
    fn glm_extras_msgpack_roundtrip_with_new_fields() {
        let g = EstimatorExtras::Glm {
            baseline_prob_realized: 0.3,
            baseline_prob_sum: 0.6,
            baseline_prob_n: 2,
            singular_fit_rate: 0.25,
            singular_count: 1,
            singular_n: 4,
            tau_squared_hat_mean: 0.21,
            tau_squared_hat_sum: 0.84,
            tau_squared_hat_n: 4,
        };
        let bytes = rmp_serde::to_vec_named(&g).unwrap();
        let back: EstimatorExtras = rmp_serde::from_slice(&bytes).unwrap();
        // NaN equality: instead compare field by field for this test.
        let EstimatorExtras::Glm {
            baseline_prob_realized,
            baseline_prob_sum,
            baseline_prob_n,
            singular_fit_rate,
            singular_count,
            singular_n,
            tau_squared_hat_mean,
            tau_squared_hat_sum,
            tau_squared_hat_n,
        } = back
        else {
            panic!("expected Glm after roundtrip")
        };
        assert_eq!(baseline_prob_realized, 0.3);
        assert_eq!(baseline_prob_sum, 0.6);
        assert_eq!(baseline_prob_n, 2);
        assert_eq!(singular_fit_rate, 0.25);
        assert_eq!(singular_count, 1);
        assert_eq!(singular_n, 4);
        assert_eq!(tau_squared_hat_mean, 0.21);
        assert_eq!(tau_squared_hat_sum, 0.84);
        assert_eq!(tau_squared_hat_n, 4);
    }

    #[test]
    fn from_host_args_parses_grid_variants() {
        assert_eq!(
            SampleSizeMethod::from_host_args("grid", Some(10), None, None).unwrap(),
            SampleSizeMethod::Grid {
                by: ByValue::Fixed(10),
                mode: GridMode::Linear
            }
        );
        assert_eq!(
            SampleSizeMethod::from_host_args("grid", Some(12), Some("auto"), Some("log")).unwrap(),
            SampleSizeMethod::Grid {
                by: ByValue::Auto { count: 12 },
                mode: GridMode::Log
            }
        );
    }

    #[test]
    fn from_host_args_rejects_unknown_tokens() {
        assert!(SampleSizeMethod::from_host_args("bisect", Some(1), None, None).is_err());
        assert!(SampleSizeMethod::from_host_args("grid", None, None, None).is_err());
        assert!(SampleSizeMethod::from_host_args("grid", Some(1), Some("step"), None).is_err());
        assert!(
            SampleSizeMethod::from_host_args("grid", Some(1), None, Some("geometric")).is_err()
        );
    }
}
